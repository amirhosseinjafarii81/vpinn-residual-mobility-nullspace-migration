#!/usr/bin/env python3
"""
VPINN Test-Function Gradient Conflict — Stage 1
================================================

A controlled diagnostic for intra-test gradient conflict in a 1D VPINN.

Model problem
-------------
    -u''(x) = f(x),  x in (0,1),
     u(0) = u(1) = 0,
with exact solution
    u*(x) = sin(pi x) + 0.15 sin(7 pi x).

Weak residuals
--------------
    R_k(theta) = ∫ [u_theta'(x) v_k'(x) - f(x) v_k(x)] dx,
using H_0^1-orthonormal tests
    v_k(x) = sqrt(2)/(k*pi) sin(k*pi*x),
so ∫ v_i' v_j' = delta_ij.

Per-test gradients
------------------
    ell_k = R_k^2,
    g_k   = grad_theta ell_k.

Primary metric
--------------
    Gamma = ||sum_k g_k|| / sum_k ||g_k||  in [0,1].

Gamma ~ 1: aligned per-test gradients.
Gamma ~ 0: strong cancellation among per-test gradients.

Design guardrails
-----------------
* Dirichlet BCs are enforced exactly by u_theta=x(1-x)N_theta.
* High-order Gauss-Legendre quadrature is used.
* Test basis orthonormality is verified numerically.
* float64 is used throughout.
* No adaptive loss weighting, scheduler, clipping, or regularization is used.
* Diagnostics are sparse in time to keep overhead low.

This is a Stage-1 diagnostic, not a publication-level robustness study.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True)
class Config:
    seed: int = 0
    device: str = "auto"
    epochs: int = 2500
    lr: float = 1.0e-3
    width: int = 32
    depth: int = 3
    n_test: int = 24
    n_quad: int = 256
    n_eval: int = 4001
    checkpoints: Tuple[int, ...] = (0, 10, 50, 100, 250, 500, 1000, 2500)
    active_relative_tol: float = 1.0e-8
    grad_eps: float = 1.0e-300
    output_dir: str = "vpinn_gradient_conflict_stage1"
    dpi: int = 220


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Controlled VPINN per-test gradient-conflict diagnostic."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n-test", type=int, default=24)
    p.add_argument("--n-quad", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=4001)
    p.add_argument("--output-dir", default="vpinn_gradient_conflict_stage1")
    a = p.parse_args()

    if a.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if a.width < 1 or a.depth < 1:
        raise ValueError("--width and --depth must be >= 1")
    if a.n_test < 2:
        raise ValueError("--n-test must be >= 2")
    if a.n_quad < max(32, 2 * a.n_test):
        raise ValueError("Use --n-quad >= max(32, 2*n_test) for this diagnostic")
    if a.n_eval < 101:
        raise ValueError("--n-eval must be >= 101")
    if a.lr <= 0:
        raise ValueError("--lr must be positive")

    cps = tuple(sorted({e for e in (0, 10, 50, 100, 250, 500, 1000, a.epochs)
                        if 0 <= e <= a.epochs}))

    return Config(
        seed=a.seed,
        device=a.device,
        epochs=a.epochs,
        lr=a.lr,
        width=a.width,
        depth=a.depth,
        n_test=a.n_test,
        n_quad=a.n_quad,
        n_eval=a.n_eval,
        checkpoints=cps,
        output_dir=a.output_dir,
    )


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def exact_solution(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * x) + 0.15 * torch.sin(7.0 * math.pi * x)


def forcing(x: torch.Tensor) -> torch.Tensor:
    return ((math.pi ** 2) * torch.sin(math.pi * x)
            + 0.15 * (7.0 * math.pi) ** 2 * torch.sin(7.0 * math.pi * x))


class VPINNNetwork(torch.nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        layers: List[torch.nn.Module] = []
        in_features = 1
        for _ in range(depth):
            lin = torch.nn.Linear(in_features, width)
            torch.nn.init.xavier_normal_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
            layers += [lin, torch.nn.Tanh()]
            in_features = width
        out = torch.nn.Linear(in_features, 1)
        torch.nn.init.xavier_normal_(out.weight)
        torch.nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 - x) * self.net(x)


class VPINNExperiment:
    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = torch.float64
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = VPINNNetwork(cfg.width, cfg.depth).to(device=device, dtype=self.dtype)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self._build_quadrature_and_tests()

    def _build_quadrature_and_tests(self) -> None:
        xi_np, wi_np = leggauss(self.cfg.n_quad)
        x = torch.as_tensor((xi_np + 1.0) / 2.0, dtype=self.dtype,
                            device=self.device).reshape(-1, 1)
        w = torch.as_tensor(wi_np / 2.0, dtype=self.dtype,
                            device=self.device).reshape(-1, 1)
        self.x_quad = x.detach().clone().requires_grad_(True)
        self.w_quad = w

        k = torch.arange(1, self.cfg.n_test + 1,
                         dtype=self.dtype, device=self.device).reshape(1, -1)
        with torch.no_grad():
            phase = math.pi * (self.x_quad.detach() @ k)
            self.test_values = math.sqrt(2.0) / (math.pi * k) * torch.sin(phase)
            self.test_derivatives = math.sqrt(2.0) * torch.cos(phase)
            self.forcing_values = forcing(self.x_quad.detach())

        with torch.no_grad():
            weighted = self.test_derivatives * torch.sqrt(self.w_quad)
            gram = weighted.T @ weighted
            eye = torch.eye(self.cfg.n_test, dtype=self.dtype, device=self.device)
            self.gram_error = float(torch.max(torch.abs(gram - eye)).item())
        if not np.isfinite(self.gram_error) or self.gram_error > 1.0e-10:
            raise RuntimeError(
                f"Test-space orthonormality check failed: max|G-I|={self.gram_error:.3e}"
            )

    def weak_residuals(self) -> torch.Tensor:
        u = self.model(self.x_quad)
        du_dx = torch.autograd.grad(
            outputs=u,
            inputs=self.x_quad,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        integrand = (du_dx * self.test_derivatives
                     - self.forcing_values * self.test_values)
        return torch.sum(self.w_quad * integrand, dim=0)

    @torch.no_grad()
    def relative_l2_error(self) -> float:
        x = torch.linspace(0.0, 1.0, self.cfg.n_eval,
                           dtype=self.dtype, device=self.device).reshape(-1, 1)
        pred = self.model(x)
        truth = exact_solution(x)
        return float((torch.linalg.vector_norm(pred - truth)
                      / torch.linalg.vector_norm(truth)).item())

    @staticmethod
    def _flatten(grads: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.reshape(-1) for g in grads], dim=0)

    def gradient_diagnostics(self):
        residuals = self.weak_residuals()
        params = tuple(p for p in self.model.parameters() if p.requires_grad)
        rows: List[torch.Tensor] = []

        # Exact autograd per test. This loop is intentionally kept because N_TEST
        # is small and diagnostics are sparse; it is more robust than a more
        # fragile higher-order torch.func construction for this experiment.
        for k in range(self.cfg.n_test):
            ell_k = residuals[k].square()
            grads = torch.autograd.grad(
                ell_k, params, retain_graph=True, create_graph=False,
                allow_unused=False
            )
            rows.append(self._flatten(grads).detach())

        G_all = torch.stack(rows, dim=0)
        norms_all = torch.linalg.vector_norm(G_all, dim=1)
        max_norm = norms_all.max().clamp_min(self.cfg.grad_eps)
        active = norms_all > self.cfg.active_relative_tol * max_norm
        active_idx = torch.nonzero(active, as_tuple=False).reshape(-1)

        cosine_full = torch.full(
            (self.cfg.n_test, self.cfg.n_test), float("nan"),
            dtype=self.dtype, device=self.device
        )

        d: Dict[str, float] = {
            "active_tests": int(active_idx.numel()),
            "conflict_fraction": float("nan"),
            "mean_pairwise_cosine": float("nan"),
            "minimum_pairwise_cosine": float("nan"),
            "gradient_coherence": float("nan"),
            "weighted_negative_cosine": float("nan"),
            "worst_test_i": -1,
            "worst_test_j": -1,
        }

        if active_idx.numel() >= 2:
            G = G_all.index_select(0, active_idx)
            norms = norms_all.index_select(0, active_idx).clamp_min(self.cfg.grad_eps)
            C = (G @ G.T) / (norms[:, None] * norms[None, :])
            C = C.clamp(-1.0, 1.0)
            cosine_full[active_idx[:, None], active_idx[None, :]] = C

            m = int(active_idx.numel())
            upper = torch.triu(torch.ones((m, m), dtype=torch.bool,
                                          device=self.device), diagonal=1)
            pairwise = C[upper]
            weights = (norms[:, None] * norms[None, :])[upper]

            min_cos, min_idx = torch.min(pairwise, dim=0)
            pair_indices = torch.nonzero(upper, as_tuple=False)
            worst_local = pair_indices[min_idx]
            worst_i = int(active_idx[worst_local[0]].item()) + 1
            worst_j = int(active_idx[worst_local[1]].item()) + 1

            coherence = (torch.linalg.vector_norm(G.sum(dim=0))
                         / norms.sum().clamp_min(self.cfg.grad_eps))
            neg = torch.clamp(-pairwise, min=0.0)
            weighted_neg = ((weights * neg).sum()
                            / weights.sum().clamp_min(self.cfg.grad_eps))

            d.update({
                "conflict_fraction": float((pairwise < 0).to(self.dtype).mean().item()),
                "mean_pairwise_cosine": float(pairwise.mean().item()),
                "minimum_pairwise_cosine": float(min_cos.item()),
                "gradient_coherence": float(coherence.item()),
                "weighted_negative_cosine": float(weighted_neg.item()),
                "worst_test_i": worst_i,
                "worst_test_j": worst_j,
            })

        return residuals.detach(), norms_all.detach(), cosine_full.detach(), d

    def train_step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        residuals = self.weak_residuals()
        loss = torch.mean(residuals.square())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_cosine_heatmap(matrix: torch.Tensor, epoch: int,
                        outdir: Path, dpi: int) -> None:
    arr = matrix.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(arr, vmin=-1.0, vmax=1.0, cmap="coolwarm",
                   origin="lower", interpolation="nearest", aspect="equal")
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xlabel("Test function index")
    ax.set_ylabel("Test function index")
    ax.set_title(f"Per-test VPINN gradient cosine matrix | epoch {epoch}")
    fig.tight_layout()
    fig.savefig(outdir / f"gradient_cosine_epoch_{epoch:04d}.png",
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_curve(epochs, values, ylabel, title, filename, dpi, log_y=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y and any(np.isfinite(v) and v > 0 for v in values):
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_final_plots(exp: VPINNExperiment, final_residuals: np.ndarray) -> None:
    cfg = exp.cfg
    x = torch.linspace(0.0, 1.0, 3000, dtype=torch.float64,
                       device=exp.device).reshape(-1, 1)
    with torch.no_grad():
        pred = exp.model(x).cpu().numpy().reshape(-1)
        truth = exact_solution(x).cpu().numpy().reshape(-1)
    x_np = x.cpu().numpy().reshape(-1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_np, truth, label="Exact solution", linewidth=2.5)
    ax.plot(x_np, pred, "--", label="VPINN", linewidth=2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("VPINN solution after training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(exp.output_dir / "final_solution.png", dpi=cfg.dpi,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_np, np.abs(pred - truth), linewidth=2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("|u_VPINN - u_exact|")
    ax.set_title("Pointwise solution error")
    fig.tight_layout()
    fig.savefig(exp.output_dir / "final_pointwise_error.png", dpi=cfg.dpi,
                bbox_inches="tight")
    plt.close(fig)

    k = np.arange(1, final_residuals.size + 1)
    floor = np.finfo(float).tiny
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(k, np.maximum(np.abs(final_residuals), floor))
    ax.set_yscale("log")
    ax.set_xlabel("Test function index k")
    ax.set_ylabel("|R_k|")
    ax.set_title("Final weak residual by test function")
    fig.tight_layout()
    fig.savefig(exp.output_dir / "final_weak_residuals.png", dpi=cfg.dpi,
                bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cfg = parse_args()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    exp = VPINNExperiment(cfg, device)
    metadata = {
        "config": asdict(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gram_max_abs_error": exp.gram_error,
        "parameter_count": sum(p.numel() for p in exp.model.parameters()),
    }
    save_json(exp.output_dir / "run_metadata.json", metadata)

    print("=" * 78)
    print("VPINN TEST-FUNCTION GRADIENT CONFLICT — STAGE 1")
    print("=" * 78)
    print(f"device                 : {device}")
    print("dtype                  : torch.float64")
    print(f"seed                   : {cfg.seed}")
    print(f"epochs                 : {cfg.epochs}")
    print(f"network                : depth={cfg.depth}, width={cfg.width}")
    print(f"trainable parameters   : {metadata['parameter_count']}")
    print(f"number of test funcs   : {cfg.n_test}")
    print(f"quadrature points      : {cfg.n_quad}")
    print(f"max |Gram-I|           : {exp.gram_error:.3e}")
    print(f"output directory       : {exp.output_dir.resolve()}")
    print("=" * 78)

    checkpoint_set = set(cfg.checkpoints)
    history: List[dict] = []
    checkpoint_arrays: Dict[int, Dict[str, np.ndarray]] = {}
    start = time.perf_counter()

    for epoch in range(cfg.epochs + 1):
        if epoch in checkpoint_set:
            residuals, grad_norms, cosine, diag = exp.gradient_diagnostics()
            loss_value = float(torch.mean(residuals.square()).item())
            rel_error = exp.relative_l2_error()
            row = {
                "epoch": epoch,
                "vpinn_loss": loss_value,
                "relative_l2_error": rel_error,
                **diag,
            }
            history.append(row)

            r_np = residuals.cpu().numpy()
            gn_np = grad_norms.cpu().numpy()
            c_np = cosine.cpu().numpy()
            checkpoint_arrays[epoch] = {
                "residuals": r_np,
                "gradient_norms": gn_np,
                "cosine": c_np,
            }
            np.savez_compressed(
                exp.output_dir / f"checkpoint_{epoch:04d}.npz",
                residuals=r_np,
                gradient_norms=gn_np,
                cosine_matrix=c_np,
            )
            save_cosine_heatmap(cosine, epoch, exp.output_dir, cfg.dpi)

            print("-" * 78)
            print(f"epoch                  : {epoch}")
            print(f"VPINN loss             : {loss_value:.6e}")
            print(f"relative L2 error      : {rel_error:.6e}")
            print(f"active tests           : {diag['active_tests']}")
            print(f"conflict fraction      : {diag['conflict_fraction']:.6f}")
            print(f"mean pairwise cosine   : {diag['mean_pairwise_cosine']:.6f}")
            print(f"minimum cosine         : {diag['minimum_pairwise_cosine']:.6f}")
            print(f"gradient coherence Γ   : {diag['gradient_coherence']:.6e}")
            print(f"weighted neg. cosine   : {diag['weighted_negative_cosine']:.6f}")
            print(f"worst conflicting pair : v_{diag['worst_test_i']} vs v_{diag['worst_test_j']}")

        if epoch == cfg.epochs:
            break
        exp.train_step()

    elapsed = time.perf_counter() - start
    save_csv(exp.output_dir / "checkpoint_metrics.csv", history)

    final_epoch = cfg.checkpoints[-1]
    save_final_plots(exp, checkpoint_arrays[final_epoch]["residuals"])

    epochs = [int(r["epoch"]) for r in history]
    save_curve(
        epochs,
        [float(r["gradient_coherence"]) for r in history],
        "Gradient coherence Γ",
        "Cancellation among VPINN test-function gradients",
        exp.output_dir / "gradient_coherence_history.png",
        cfg.dpi,
        log_y=True,
    )
    save_curve(
        epochs,
        [float(r["relative_l2_error"]) for r in history],
        "Relative L2 error",
        "VPINN solution error during training",
        exp.output_dir / "relative_l2_history.png",
        cfg.dpi,
        log_y=True,
    )
    save_curve(
        epochs,
        [float(r["conflict_fraction"]) for r in history],
        "Fraction of active test pairs with cosine < 0",
        "Pairwise VPINN gradient conflict during training",
        exp.output_dir / "conflict_fraction_history.png",
        cfg.dpi,
        log_y=False,
    )

    summary = {
        "elapsed_seconds": elapsed,
        "final_checkpoint": history[-1],
        "interpretation_guardrail": (
            "Stage 1 is diagnostic only. Replicate across seeds, architectures, "
            "test counts, and test families before making a strong scientific claim."
        ),
    }
    save_json(exp.output_dir / "run_summary.json", summary)

    print("=" * 78)
    print("RUN COMPLETE")
    print("=" * 78)
    print(f"elapsed seconds        : {elapsed:.2f}")
    print(f"metrics CSV            : {exp.output_dir / 'checkpoint_metrics.csv'}")
    print(f"metadata               : {exp.output_dir / 'run_metadata.json'}")
    print(f"summary                : {exp.output_dir / 'run_summary.json'}")
    print(f"plots/checkpoints      : {exp.output_dir.resolve()}")
    print("=" * 78)
    print("Scientific guardrail: one seed is not enough for a publication-level claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
