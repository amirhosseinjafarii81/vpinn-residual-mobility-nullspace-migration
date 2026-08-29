#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 3: Matched-Energy Frequency Transfer
===================================================================

Scientific question
-------------------
Stage 2 showed that, for

    u*(x) = sin(pi x) + 0.15 sin(7 pi x),

five independent seeds reproducibly entered a regime where:
  * the v_7 weak residual dominated,
  * per-test parameter gradients strongly cancelled,
  * training eventually escaped and converged.

Stage 3 asks a sharper question:

    Does the unresolved weak-residual mode TRACK the target frequency
    when that frequency is changed, after controlling for weak-residual
    amplitude?

To remove an important confound, the high-frequency amplitude is chosen as

    a_m = 0.15 * 7 / m,

so that

    a_m * m = 1.05

is constant. Because the test basis is H_0^1-orthonormal, this holds the
target-mode derivative/weak-residual scale fixed across frequencies.

Paired design
-------------
Modes: m in {3, 5, 7, 9}
Seed : 0 for every mode
Network initialization is reset to the SAME seed before every mode.
Thus all four experiments start from the same parameter initialization.

This stage is a route-feasibility / frequency-transfer audit, not a
publication-level multi-seed claim.

Model problem
-------------
    -u''(x) = f(x),  x in (0,1)
     u(0) = u(1) = 0

Exact solution
--------------
    u*_m(x) = sin(pi x) + a_m sin(m pi x)
    a_m     = 0.15 * 7 / m

Test basis
----------
    v_k(x) = sqrt(2)/(k*pi) sin(k*pi*x), k=1,...,24

which satisfies
    int_0^1 v_i'(x) v_j'(x) dx = delta_ij.

Primary precommitted gate
-------------------------
For every tested frequency m:
  1) target residual mode m must become dominant before convergence;
  2) max target residual-energy share must be >= 0.80.

Conflict metrics are measured as outcomes, not forced as a universal
pass condition across all frequencies. This distinction is intentional:
lower frequencies may be learned too rapidly to develop a long conflict
plateau.

Additional controls
-------------------
  * hard Dirichlet boundary enforcement;
  * float64;
  * high-order Gauss-Legendre quadrature;
  * same architecture and optimizer as Stages 1-2;
  * all modes executed unconditionally;
  * no early stopping;
  * dense low-cost tracking plus sparse exact gradient diagnostics;
  * optional automatic reproduction check against Stage-2 seed 0 at m=7.

Example
-------
    python vpinn_gradient_conflict_stage3_frequency_transfer.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    seed: int = 0
    device: str = "cpu"
    epochs: int = 2500
    learning_rate: float = 1.0e-3
    width: int = 32
    depth: int = 3
    n_test: int = 24
    n_quad: int = 256
    n_eval: int = 4001

    modes: Tuple[int, ...] = (3, 5, 7, 9)

    # Exact Stage-1 anchor is recovered at m=7:
    reference_mode: int = 7
    reference_amplitude: float = 0.15

    track_interval: int = 25
    diagnostic_epochs: Tuple[int, ...] = (
        0, 10, 25, 50, 100, 150, 250, 400, 500,
        750, 1000, 1250, 1500, 2000, 2500
    )

    convergence_error_threshold: float = 1.0e-2
    localization_share_threshold: float = 0.80
    resolved_share_threshold: float = 0.20

    conflict_gamma_threshold: float = 0.20
    conflict_weighted_negative_threshold: float = 0.50

    active_relative_tol: float = 1.0e-8
    grad_absolute_eps: float = 1.0e-300

    output_dir: str = "vpinn_gradient_conflict_stage3_frequency_transfer"
    dpi: int = 220


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Matched-energy frequency-transfer audit for VPINN gradient geometry."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n-test", type=int, default=24)
    p.add_argument("--n-quad", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=4001)
    p.add_argument("--track-interval", type=int, default=25)
    p.add_argument(
        "--modes",
        type=int,
        nargs="+",
        default=[3, 5, 7, 9],
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="vpinn_gradient_conflict_stage3_frequency_transfer",
    )
    p.add_argument(
        "--stage2-anchor",
        type=str,
        default=(
            "vpinn_gradient_conflict_stage2_seedreplication/"
            "runs/seed_000/checkpoint_metrics.csv"
        ),
        help=(
            "Optional Stage-2 seed-0 checkpoint CSV. If present, m=7 is "
            "automatically checked for exact numerical reproduction."
        ),
    )
    args = p.parse_args()

    modes = tuple(args.modes)
    if len(modes) < 2 or len(set(modes)) != len(modes):
        raise ValueError("--modes must contain at least two unique integers.")
    if min(modes) < 2:
        raise ValueError("All target modes must be >= 2.")
    if max(modes) > args.n_test:
        raise ValueError("Every target mode must be <= n_test.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.track_interval < 1:
        raise ValueError("--track-interval must be >= 1.")
    if args.n_quad < max(32, 2 * args.n_test):
        raise ValueError("Use n_quad >= max(32, 2*n_test).")

    diag = tuple(
        sorted(
            set(
                e for e in (
                    0, 10, 25, 50, 100, 150, 250, 400, 500,
                    750, 1000, 1250, 1500, 2000, args.epochs
                )
                if 0 <= e <= args.epochs
            )
        )
    )

    cfg = Config(
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.lr,
        width=args.width,
        depth=args.depth,
        n_test=args.n_test,
        n_quad=args.n_quad,
        n_eval=args.n_eval,
        modes=modes,
        track_interval=args.track_interval,
        diagnostic_epochs=diag,
        output_dir=args.output_dir,
    )
    # Keep the anchor path outside the immutable dataclass.
    object.__setattr__(cfg, "_stage2_anchor", args.stage2_anchor)
    return cfg


# =============================================================================
# Helpers
# =============================================================================

def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =============================================================================
# PDE family
# =============================================================================

def amplitude_for_mode(cfg: Config, mode: int) -> float:
    # H1/weak-scale matching:
    # a_m * m = reference_amplitude * reference_mode = 1.05.
    return cfg.reference_amplitude * cfg.reference_mode / mode


def exact_solution(
    x: torch.Tensor,
    mode: int,
    amplitude: float,
) -> torch.Tensor:
    return (
        torch.sin(math.pi * x)
        + amplitude * torch.sin(mode * math.pi * x)
    )


def forcing(
    x: torch.Tensor,
    mode: int,
    amplitude: float,
) -> torch.Tensor:
    return (
        (math.pi ** 2) * torch.sin(math.pi * x)
        + amplitude
        * (mode * math.pi) ** 2
        * torch.sin(mode * math.pi * x)
    )


def theoretical_low_mode_only_rel_l2(amplitude: float) -> float:
    # Orthogonality of sine modes gives:
    # ||a sin(m*pi*x)|| / ||sin(pi*x)+a sin(m*pi*x)||
    return abs(amplitude) / math.sqrt(1.0 + amplitude * amplitude)


# =============================================================================
# Network
# =============================================================================

class VPINNNetwork(torch.nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        layers: List[torch.nn.Module] = []
        in_features = 1

        for _ in range(depth):
            linear = torch.nn.Linear(in_features, width)
            torch.nn.init.xavier_normal_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
            layers.extend((linear, torch.nn.Tanh()))
            in_features = width

        out = torch.nn.Linear(in_features, 1)
        torch.nn.init.xavier_normal_(out.weight)
        torch.nn.init.zeros_(out.bias)
        layers.append(out)

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hard homogeneous Dirichlet condition.
        return x * (1.0 - x) * self.net(x)


# =============================================================================
# Single-mode experiment
# =============================================================================

class ModeExperiment:
    def __init__(
        self,
        cfg: Config,
        device: torch.device,
        mode: int,
        out_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = torch.float64
        self.mode = mode
        self.amplitude = amplitude_for_mode(cfg, mode)
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Critical paired-design control: reset before EACH mode.
        seed_everything(cfg.seed)

        self.model = VPINNNetwork(cfg.width, cfg.depth).to(
            device=device, dtype=self.dtype
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
        )

        self._build_quadrature_and_tests()

    def _build_quadrature_and_tests(self) -> None:
        xi, wi = leggauss(self.cfg.n_quad)
        x = torch.as_tensor(
            (xi + 1.0) / 2.0,
            dtype=self.dtype,
            device=self.device,
        ).reshape(-1, 1)
        w = torch.as_tensor(
            wi / 2.0,
            dtype=self.dtype,
            device=self.device,
        ).reshape(-1, 1)

        self.x_quad = x.detach().clone().requires_grad_(True)
        self.w_quad = w

        k = torch.arange(
            1,
            self.cfg.n_test + 1,
            dtype=self.dtype,
            device=self.device,
        ).reshape(1, -1)

        with torch.no_grad():
            phase = math.pi * (self.x_quad.detach() @ k)
            self.test_values = (
                math.sqrt(2.0) / (math.pi * k) * torch.sin(phase)
            )
            self.test_derivatives = math.sqrt(2.0) * torch.cos(phase)
            self.forcing_values = forcing(
                self.x_quad.detach(),
                self.mode,
                self.amplitude,
            )

            weighted = self.test_derivatives * torch.sqrt(self.w_quad)
            gram = weighted.T @ weighted
            identity = torch.eye(
                self.cfg.n_test,
                dtype=self.dtype,
                device=self.device,
            )
            self.gram_error = float(
                torch.max(torch.abs(gram - identity)).item()
            )

        if not np.isfinite(self.gram_error) or self.gram_error > 1.0e-10:
            raise RuntimeError(
                f"Gram check failed for mode {self.mode}: "
                f"max|G-I|={self.gram_error:.3e}"
            )

    def weak_residuals(self) -> torch.Tensor:
        u = self.model(self.x_quad)
        du = torch.autograd.grad(
            u,
            self.x_quad,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        integrand = (
            du * self.test_derivatives
            - self.forcing_values * self.test_values
        )
        return torch.sum(self.w_quad * integrand, dim=0)

    @torch.no_grad()
    def relative_l2_error(self) -> float:
        x = torch.linspace(
            0.0,
            1.0,
            self.cfg.n_eval,
            dtype=self.dtype,
            device=self.device,
        ).reshape(-1, 1)
        pred = self.model(x)
        truth = exact_solution(x, self.mode, self.amplitude)
        return float(
            (
                torch.linalg.vector_norm(pred - truth)
                / torch.linalg.vector_norm(truth)
            ).item()
        )

    def residual_metrics(self) -> Dict[str, float]:
        residuals = self.weak_residuals().detach()
        energy = residuals.square()
        total = energy.sum().clamp_min(1.0e-300)

        target_index = self.mode - 1
        target_share = energy[target_index] / total
        dominant_index = int(torch.argmax(energy).item())

        return {
            "vpinn_loss": float(torch.mean(energy).item()),
            "residual_l2_norm": float(
                torch.linalg.vector_norm(residuals).item()
            ),
            "dominant_residual_mode": dominant_index + 1,
            "dominant_residual_energy_share": float(
                (energy[dominant_index] / total).item()
            ),
            "target_mode_residual_energy_share": float(
                target_share.item()
            ),
            "target_mode_abs_residual": float(
                torch.abs(residuals[target_index]).item()
            ),
        }

    def gradient_diagnostics(self) -> Dict[str, float]:
        residuals = self.weak_residuals()
        params = tuple(
            p for p in self.model.parameters() if p.requires_grad
        )

        rows: List[torch.Tensor] = []
        for k in range(self.cfg.n_test):
            loss_k = residuals[k].square()
            grads = torch.autograd.grad(
                loss_k,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            rows.append(
                torch.cat([g.reshape(-1) for g in grads]).detach()
            )

        G_all = torch.stack(rows, dim=0)
        norms_all = torch.linalg.vector_norm(G_all, dim=1)
        max_norm = norms_all.max().clamp_min(self.cfg.grad_absolute_eps)
        active = norms_all > self.cfg.active_relative_tol * max_norm
        idx = torch.nonzero(active, as_tuple=False).reshape(-1)

        result: Dict[str, float] = {
            "active_tests": int(idx.numel()),
            "conflict_fraction": float("nan"),
            "mean_pairwise_cosine": float("nan"),
            "minimum_pairwise_cosine": float("nan"),
            "gradient_coherence": float("nan"),
            "weighted_negative_cosine": float("nan"),
            "worst_test_i": -1,
            "worst_test_j": -1,
            "target_mode_in_worst_pair": False,
        }

        cosine_full = torch.full(
            (self.cfg.n_test, self.cfg.n_test),
            float("nan"),
            dtype=self.dtype,
            device=self.device,
        )

        if idx.numel() >= 2:
            G = G_all.index_select(0, idx)
            norms = norms_all.index_select(0, idx).clamp_min(
                self.cfg.grad_absolute_eps
            )
            cosine = (G @ G.T) / (norms[:, None] * norms[None, :])
            cosine = cosine.clamp(-1.0, 1.0)
            cosine_full[idx[:, None], idx[None, :]] = cosine

            n = idx.numel()
            upper = torch.triu(
                torch.ones((n, n), dtype=torch.bool, device=self.device),
                diagonal=1,
            )
            pairs = cosine[upper]
            pair_weights = (norms[:, None] * norms[None, :])[upper]

            min_cos, min_flat = torch.min(pairs, dim=0)
            pair_idx = torch.nonzero(upper, as_tuple=False)
            worst_local = pair_idx[min_flat]
            wi = int(idx[worst_local[0]].item()) + 1
            wj = int(idx[worst_local[1]].item()) + 1

            result.update(
                {
                    "conflict_fraction": float(
                        (pairs < 0).to(self.dtype).mean().item()
                    ),
                    "mean_pairwise_cosine": float(pairs.mean().item()),
                    "minimum_pairwise_cosine": float(min_cos.item()),
                    "gradient_coherence": float(
                        (
                            torch.linalg.vector_norm(G.sum(dim=0))
                            / norms.sum().clamp_min(
                                self.cfg.grad_absolute_eps
                            )
                        ).item()
                    ),
                    "weighted_negative_cosine": float(
                        (
                            torch.sum(
                                pair_weights * torch.clamp(-pairs, min=0.0)
                            )
                            / pair_weights.sum().clamp_min(
                                self.cfg.grad_absolute_eps
                            )
                        ).item()
                    ),
                    "worst_test_i": wi,
                    "worst_test_j": wj,
                    "target_mode_in_worst_pair": bool(
                        self.mode in (wi, wj)
                    ),
                }
            )

        result["_cosine_matrix"] = cosine_full.detach().cpu().numpy()
        result["_gradient_norms"] = norms_all.detach().cpu().numpy()
        result["_residuals"] = residuals.detach().cpu().numpy()
        return result

    def train_step(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        residuals = self.weak_residuals()
        loss = torch.mean(residuals.square())
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at mode {self.mode}: {loss.item()}"
            )
        loss.backward()
        self.optimizer.step()


# =============================================================================
# Plotting
# =============================================================================

def save_cosine_heatmap(
    matrix: np.ndarray,
    mode: int,
    epoch: int,
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(
        matrix,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xlabel("Test function index")
    ax.set_ylabel("Test function index")
    ax.set_title(
        f"Per-test gradient cosine | target mode m={mode} | epoch {epoch}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_final_solution(
    exp: ModeExperiment,
) -> None:
    x = torch.linspace(
        0.0, 1.0, 3000,
        dtype=torch.float64,
        device=exp.device,
    ).reshape(-1, 1)

    with torch.no_grad():
        pred = exp.model(x).cpu().numpy().reshape(-1)
        truth = exact_solution(
            x, exp.mode, exp.amplitude
        ).cpu().numpy().reshape(-1)

    x_np = x.cpu().numpy().reshape(-1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_np, truth, label="Exact solution", linewidth=2.5)
    ax.plot(x_np, pred, "--", label="VPINN", linewidth=2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(
        f"Final VPINN solution | target mode m={exp.mode}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        exp.out_dir / "final_solution.png",
        dpi=exp.cfg.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_group_trajectories(
    tracking: List[dict],
    cfg: Config,
    out_dir: Path,
) -> None:
    for key, ylabel, title, filename, log_y in [
        (
            "relative_l2_error",
            "Relative L2 error",
            "Frequency-transfer solution error",
            "error_trajectories.png",
            True,
        ),
        (
            "target_mode_residual_energy_share",
            "Target-mode residual energy share",
            "Does residual localization track the target frequency?",
            "target_mode_share_trajectories.png",
            False,
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for mode in cfg.modes:
            rows = [r for r in tracking if r["mode"] == mode]
            ax.plot(
                [r["epoch"] for r in rows],
                [r[key] for r in rows],
                marker=None,
                label=f"m={mode}",
            )
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            out_dir / filename,
            dpi=cfg.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_diagnostic_trajectories(
    diagnostics: List[dict],
    cfg: Config,
    out_dir: Path,
) -> None:
    for key, ylabel, title, filename, log_y in [
        (
            "gradient_coherence",
            "Gradient coherence Γ",
            "Gradient cancellation across target frequencies",
            "gradient_coherence_by_frequency.png",
            True,
        ),
        (
            "weighted_negative_cosine",
            "Weighted negative cosine",
            "Magnitude-weighted gradient conflict",
            "weighted_negative_cosine_by_frequency.png",
            False,
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for mode in cfg.modes:
            rows = [r for r in diagnostics if r["mode"] == mode]
            ax.plot(
                [r["epoch"] for r in rows],
                [r[key] for r in rows],
                marker="o",
                label=f"m={mode}",
            )
        if log_y:
            positive = [
                r[key] for r in diagnostics
                if np.isfinite(r[key]) and r[key] > 0
            ]
            if positive:
                ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            out_dir / filename,
            dpi=cfg.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# Stage-2 m=7 anchor reproduction
# =============================================================================

def check_stage2_anchor(
    anchor_path: Path,
    diagnostic_rows: List[dict],
    reference_mode: int = 7,
) -> dict:
    result = {
        "anchor_found": anchor_path.is_file(),
        "anchor_path": str(anchor_path),
        "shared_epoch_count": 0,
        "max_abs_vpinn_loss_difference": None,
        "max_abs_relative_l2_difference": None,
        "pass": None,
        "tolerance": 1.0e-10,
    }

    if not anchor_path.is_file():
        return result

    old_rows = read_csv_rows(anchor_path)
    old = {int(r["epoch"]): r for r in old_rows}
    new = {
        int(r["epoch"]): r
        for r in diagnostic_rows
        if int(r["mode"]) == reference_mode
    }

    shared = sorted(set(old) & set(new))
    result["shared_epoch_count"] = len(shared)

    if not shared:
        result["pass"] = False
        return result

    loss_diffs = []
    err_diffs = []
    for e in shared:
        loss_diffs.append(
            abs(
                float(old[e]["vpinn_loss"])
                - float(new[e]["vpinn_loss"])
            )
        )
        err_diffs.append(
            abs(
                float(old[e]["relative_l2_error"])
                - float(new[e]["relative_l2_error"])
            )
        )

    result["max_abs_vpinn_loss_difference"] = max(loss_diffs)
    result["max_abs_relative_l2_difference"] = max(err_diffs)
    result["pass"] = bool(
        max(loss_diffs) <= result["tolerance"]
        and max(err_diffs) <= result["tolerance"]
    )
    return result


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    cfg = parse_args()
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    anchor_path = Path(getattr(cfg, "_stage2_anchor"))

    precommitment = {
        "modes": list(cfg.modes),
        "paired_seed": cfg.seed,
        "all_modes_unconditional": True,
        "no_early_stop": True,
        "amplitude_rule": "a_m = 0.15 * 7 / m",
        "matched_quantity": "a_m * m = 1.05",
        "primary_localization_gate": {
            "per_mode": (
                "target mode becomes dominant before convergence AND "
                "max target residual-energy share >= 0.80"
            ),
            "group_requirement": "all tested modes pass",
        },
        "conflict_metrics_are_outcomes_not_primary_gate": True,
        "convergence_threshold_relative_l2": cfg.convergence_error_threshold,
        "conflict_signature_definition": {
            "min_preconvergence_gamma_le": cfg.conflict_gamma_threshold,
            "max_preconvergence_weighted_negative_cosine_ge":
                cfg.conflict_weighted_negative_threshold,
        },
    }

    manifest = {
        "config": {
            k: v for k, v in asdict(cfg).items()
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),
        "script_sha256": sha256_file(script_path),
        "precommitment": precommitment,
    }
    write_json(out_dir / "manifest.json", manifest)

    tracking_rows: List[dict] = []
    diagnostic_rows: List[dict] = []
    summary_rows: List[dict] = []

    print("=" * 94)
    print("VPINN GRADIENT GEOMETRY — STAGE 3 MATCHED-ENERGY FREQUENCY TRANSFER")
    print("=" * 94)
    print(f"device                 : {device}")
    print(f"paired seed            : {cfg.seed}")
    print(f"modes                  : {list(cfg.modes)}")
    print(
        "amplitude rule         : "
        "a_m = 0.15*7/m  ->  a_m*m = 1.05 for every mode"
    )
    print(f"epochs per mode        : {cfg.epochs}")
    print(f"all modes unconditional: True")
    print("=" * 94)

    global_start = time.perf_counter()

    track_epochs = set(range(0, cfg.epochs + 1, cfg.track_interval))
    track_epochs.add(cfg.epochs)
    diagnostic_epochs = set(cfg.diagnostic_epochs)

    for mode in cfg.modes:
        mode_dir = out_dir / f"mode_{mode:02d}"
        exp = ModeExperiment(cfg, device, mode, mode_dir)
        amp = exp.amplitude
        theory_plateau = theoretical_low_mode_only_rel_l2(amp)

        print()
        print("-" * 94)
        print(
            f"MODE m={mode} | amplitude={amp:.12g} | "
            f"a_m*m={amp*mode:.12g} | "
            f"low-mode-only relL2={theory_plateau:.6e}"
        )

        mode_tracking: List[dict] = []
        mode_diagnostics: List[dict] = []

        for epoch in range(cfg.epochs + 1):
            if epoch in track_epochs:
                rm = exp.residual_metrics()
                rel = exp.relative_l2_error()
                row = {
                    "mode": mode,
                    "amplitude": amp,
                    "matched_weak_scale_a_times_m": amp * mode,
                    "theoretical_low_mode_only_relative_l2": theory_plateau,
                    "epoch": epoch,
                    "relative_l2_error": rel,
                    **rm,
                    "preconvergence": int(
                        rel > cfg.convergence_error_threshold
                    ),
                }
                mode_tracking.append(row)
                tracking_rows.append(row)

            if epoch in diagnostic_epochs:
                rm = exp.residual_metrics()
                rel = exp.relative_l2_error()
                gd = exp.gradient_diagnostics()

                cosine = gd.pop("_cosine_matrix")
                grad_norms = gd.pop("_gradient_norms")
                residuals = gd.pop("_residuals")

                row = {
                    "mode": mode,
                    "amplitude": amp,
                    "epoch": epoch,
                    "relative_l2_error": rel,
                    **rm,
                    **gd,
                    "preconvergence": int(
                        rel > cfg.convergence_error_threshold
                    ),
                }
                mode_diagnostics.append(row)
                diagnostic_rows.append(row)

                np.savez_compressed(
                    mode_dir / f"diagnostic_{epoch:04d}.npz",
                    cosine_matrix=cosine,
                    gradient_norms=grad_norms,
                    residuals=residuals,
                )

                save_cosine_heatmap(
                    cosine,
                    mode,
                    epoch,
                    mode_dir / f"gradient_cosine_epoch_{epoch:04d}.png",
                    cfg.dpi,
                )

                print(
                    f"epoch={epoch:4d} "
                    f"relL2={rel:.6e} "
                    f"target_share={rm['target_mode_residual_energy_share']:.6f} "
                    f"Gamma={gd['gradient_coherence']:.6e} "
                    f"WNC={gd['weighted_negative_cosine']:.6f} "
                    f"worst=({gd['worst_test_i']},{gd['worst_test_j']})"
                )

            if epoch == cfg.epochs:
                break

            exp.train_step()

        write_csv(mode_dir / "tracking_metrics.csv", mode_tracking)
        write_csv(mode_dir / "diagnostic_metrics.csv", mode_diagnostics)
        save_final_solution(exp)

        pre_track = [
            r for r in mode_tracking if r["preconvergence"] == 1
        ]
        pre_diag = [
            r for r in mode_diagnostics if r["preconvergence"] == 1
        ]

        max_target_row = max(
            pre_track,
            key=lambda r: r["target_mode_residual_energy_share"],
        )
        target_dominant_any = any(
            int(r["dominant_residual_mode"]) == mode
            for r in pre_track
        )

        localization_pass = bool(
            target_dominant_any
            and max_target_row["target_mode_residual_energy_share"]
            >= cfg.localization_share_threshold
        )

        min_gamma_row = min(
            pre_diag,
            key=lambda r: r["gradient_coherence"],
        )
        max_wnc_row = max(
            pre_diag,
            key=lambda r: r["weighted_negative_cosine"],
        )

        conflict_signature_pass = bool(
            min_gamma_row["gradient_coherence"]
            <= cfg.conflict_gamma_threshold
            and max_wnc_row["weighted_negative_cosine"]
            >= cfg.conflict_weighted_negative_threshold
        )

        target_in_worst_pair_any = any(
            bool(r["target_mode_in_worst_pair"])
            for r in pre_diag
        )

        convergence_rows = [
            r for r in mode_tracking
            if r["relative_l2_error"]
            <= cfg.convergence_error_threshold
        ]
        first_convergence_epoch = (
            int(convergence_rows[0]["epoch"])
            if convergence_rows
            else -1
        )

        # Resolution transition: after target share has first crossed 0.8,
        # identify first tracked epoch at which it falls to <= 0.2.
        first_high_idx = next(
            (
                i for i, r in enumerate(mode_tracking)
                if r["target_mode_residual_energy_share"]
                >= cfg.localization_share_threshold
            ),
            None,
        )
        target_resolution_epoch = -1
        if first_high_idx is not None:
            for r in mode_tracking[first_high_idx + 1:]:
                if (
                    r["target_mode_residual_energy_share"]
                    <= cfg.resolved_share_threshold
                ):
                    target_resolution_epoch = int(r["epoch"])
                    break

        final_row = mode_tracking[-1]

        summary = {
            "mode": mode,
            "amplitude": amp,
            "matched_weak_scale_a_times_m": amp * mode,
            "theoretical_low_mode_only_relative_l2": theory_plateau,
            "final_relative_l2_error": final_row["relative_l2_error"],
            "final_vpinn_loss": final_row["vpinn_loss"],
            "max_preconvergence_target_mode_energy_share":
                max_target_row["target_mode_residual_energy_share"],
            "epoch_max_target_mode_energy_share":
                int(max_target_row["epoch"]),
            "target_mode_dominant_preconvergence":
                target_dominant_any,
            "localization_pass": localization_pass,
            "min_preconvergence_gamma":
                min_gamma_row["gradient_coherence"],
            "epoch_min_preconvergence_gamma":
                int(min_gamma_row["epoch"]),
            "max_preconvergence_weighted_negative_cosine":
                max_wnc_row["weighted_negative_cosine"],
            "epoch_max_preconvergence_weighted_negative_cosine":
                int(max_wnc_row["epoch"]),
            "conflict_signature_pass": conflict_signature_pass,
            "target_mode_in_worst_pair_preconvergence":
                target_in_worst_pair_any,
            "first_relative_l2_le_1e-2_epoch":
                first_convergence_epoch,
            "target_residual_share_le_0p2_after_localization_epoch":
                target_resolution_epoch,
            "gram_max_abs_error": exp.gram_error,
        }
        summary_rows.append(summary)

    write_csv(out_dir / "aggregate_tracking_metrics.csv", tracking_rows)
    write_csv(out_dir / "aggregate_diagnostic_metrics.csv", diagnostic_rows)
    write_csv(out_dir / "frequency_summary.csv", summary_rows)

    plot_group_trajectories(tracking_rows, cfg, out_dir)
    plot_diagnostic_trajectories(diagnostic_rows, cfg, out_dir)

    anchor_result = check_stage2_anchor(
        anchor_path,
        diagnostic_rows,
        reference_mode=cfg.reference_mode,
    )

    localization_pass_count = sum(
        bool(r["localization_pass"]) for r in summary_rows
    )
    conflict_pass_count = sum(
        bool(r["conflict_signature_pass"]) for r in summary_rows
    )

    matched_values = [
        float(r["matched_weak_scale_a_times_m"])
        for r in summary_rows
    ]
    matched_scale_spread = max(matched_values) - min(matched_values)
    matched_scale_pass = bool(matched_scale_spread <= 1.0e-12)

    localization_transfer_pass = bool(
        localization_pass_count == len(summary_rows)
    )

    anchor_gate_ok = bool(
        anchor_result["pass"] is not False
    )

    route_to_edge_mode_replication = bool(
        matched_scale_pass
        and localization_transfer_pass
        and anchor_gate_ok
    )

    decision = {
        "n_modes": len(summary_rows),
        "modes": list(cfg.modes),
        "paired_seed": cfg.seed,
        "matched_scale_spread": matched_scale_spread,
        "matched_scale_pass": matched_scale_pass,
        "localization_pass_count": localization_pass_count,
        "localization_transfer_group_pass":
            localization_transfer_pass,
        "conflict_signature_pass_count": conflict_pass_count,
        "conflict_signature_is_not_primary_stage3_gate": True,
        "stage2_anchor_reproduction": anchor_result,
        "route_to_edge_mode_replication":
            route_to_edge_mode_replication,
        "scientific_interpretation": (
            "A Stage-3 localization PASS supports frequency transfer of "
            "weak-residual localization under matched target weak scale for "
            "this paired single-seed experiment. It does not by itself "
            "establish multi-seed frequency generality."
        ),
    }
    write_json(out_dir / "decision.json", decision)

    elapsed = time.perf_counter() - global_start

    lines = []
    lines.append("=" * 118)
    lines.append("VPINN GRADIENT GEOMETRY — STAGE 3 FREQUENCY-TRANSFER SUMMARY")
    lines.append("=" * 118)
    lines.append(
        "mode | amplitude | a*m  | final relL2 | max target share | min Gamma | "
        "max WNC | localization | conflict | conv epoch"
    )
    lines.append("-" * 118)

    for r in summary_rows:
        lines.append(
            f"{int(r['mode']):4d} | "
            f"{r['amplitude']:9.6f} | "
            f"{r['matched_weak_scale_a_times_m']:4.2f} | "
            f"{r['final_relative_l2_error']:11.4e} | "
            f"{r['max_preconvergence_target_mode_energy_share']:16.6f} | "
            f"{r['min_preconvergence_gamma']:9.4e} | "
            f"{r['max_preconvergence_weighted_negative_cosine']:7.4f} | "
            f"{'PASS' if r['localization_pass'] else 'FAIL':12s} | "
            f"{'PASS' if r['conflict_signature_pass'] else 'FAIL':8s} | "
            f"{int(r['first_relative_l2_le_1e-2_epoch']):10d}"
        )

    lines.append("-" * 118)
    lines.append(
        f"matched weak-scale control       : "
        f"{'PASS' if matched_scale_pass else 'FAIL'}"
    )
    lines.append(
        f"frequency localization transfer  : "
        f"{localization_pass_count}/{len(summary_rows)} -> "
        f"{'PASS' if localization_transfer_pass else 'FAIL'}"
    )
    lines.append(
        f"conflict signature (descriptive) : "
        f"{conflict_pass_count}/{len(summary_rows)}"
    )
    if anchor_result["anchor_found"]:
        lines.append(
            f"Stage-2 m=7 anchor reproduction  : "
            f"{'PASS' if anchor_result['pass'] else 'FAIL'}"
        )
        lines.append(
            f"  max |loss difference|          : "
            f"{anchor_result['max_abs_vpinn_loss_difference']:.3e}"
        )
        lines.append(
            f"  max |relL2 difference|         : "
            f"{anchor_result['max_abs_relative_l2_difference']:.3e}"
        )
    else:
        lines.append(
            "Stage-2 m=7 anchor reproduction  : NOT CHECKED "
            "(anchor CSV not found)"
        )

    lines.append(
        f"route to edge-mode replication   : "
        f"{'AUTHORIZED' if route_to_edge_mode_replication else 'NOT AUTHORIZED'}"
    )
    lines.append(f"elapsed seconds                  : {elapsed:.2f}")
    lines.append("=" * 118)
    lines.append(
        "Guardrail: this is a paired single-seed frequency-transfer audit, "
        "not a universal VPINN claim."
    )
    lines.append("=" * 118)

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    (out_dir / "console_summary.txt").write_text(
        summary_text, encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
