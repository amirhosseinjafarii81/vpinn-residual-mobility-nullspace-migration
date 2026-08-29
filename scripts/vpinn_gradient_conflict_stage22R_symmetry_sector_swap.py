#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 22R
Midpoint-Reflection Symmetry-Sector Swap Audit
==============================================

Scientific motivation
---------------------
Stages 19-20 established a strong m=9 lock/mobility-collapse phenomenon for
odd target modes with the fixed base component sin(pi*x).

Stage 21 then used completely new EVEN target modes m={6,8,10} and new seeds
{10,...,14}. The proposed frequency-only transfer FAILED:

    conflict counts:
        N6 = 0/5
        N8 = 0/5
        N10 = 1/5

and all 15 even-mode runs certified escape.

The single m=10 conflict had

    mu_conflict ~ 7.19e-6,

not the ~1e-9 to 1e-8 collapse observed in the m=9 lock, and its mobility
recovery was only ~11.7x.

This falsifies a simple "higher frequency => mobility collapse" explanation.

A sharper structural hypothesis
-------------------------------
On (0,1), midpoint reflection R:x->1-x acts on sine modes as

    sin(k*pi*(1-x)) = (-1)^(k+1) sin(k*pi*x).

Thus:
    odd k  : midpoint-symmetric sector    (+)
    even k : midpoint-antisymmetric sector (-)

All previous experiments used base mode b=1, which is symmetric.

Therefore:
    odd targets  (3,5,7,9) share the base reflection sector;
    even targets (6,8,10) occupy the opposite sector.

Stage 22R performs a PRECOMMITTED 2x2 SYMMETRY SWAP:

    base b=1, target m=9   : SAME sector
    base b=1, target m=10  : DIFFERENT sector

    base b=2, target m=9   : DIFFERENT sector
    base b=2, target m=10  : SAME sector

with completely new paired seeds

    {15,16,17,18,19}.

If conflict/mobility collapse follows SAME-SECTOR status rather than target
frequency or odd/even target label, flipping the base parity should flip the
conflict phenotype:

    (b=1,m=9)   conflict
    (b=1,m=10)  no conflict

    (b=2,m=9)   no conflict
    (b=2,m=10)  conflict

This is a substantially stronger mechanistic test than another frequency
sweep.

Weak-scale matching
-------------------
To avoid changing absolute weak/derivative scales when the base frequency is
changed, define

    u*_{b,m}(x)
      = c_b sin(b*pi*x) + a_m sin(m*pi*x),

with

    c_b = 1/b,
    a_m = 1.05/m.

Hence

    c_b * b = 1,
    a_m * m = 1.05

for EVERY cell.

The derivative amplitudes of the base and target components are therefore
identical across the entire 2x2 experiment.

The original b=1 family is reproduced exactly.

Paired initialization
---------------------
For each seed, every one of the four cells resets the RNG to that same seed
before model construction. The initial model parameter vector must therefore
be EXACTLY identical across all four cells. Stage 22R explicitly checks this.

Ordinary Adam only.
No optimizer intervention.

Tracking and certification
--------------------------
Track every 25 epochs to at most epoch 4000.

Mechanism-active:
    target share >=0.80 AND relL2>1e-2.

Certified conflict:
    <g_T,Delta_Adam> > 0
for THREE consecutive mechanism-active probes.

Certified escape:
    relL2<=1e-2 AND target share<=0.20
for THREE consecutive observations.

For efficiency, a run stops immediately after certified conflict or certified
escape.

Full J/K/Pareto/invariant audit is computed once at:
    * certified conflict onset, or
    * certified escape onset when no conflict occurs.

Primary cell gates
------------------
Let N_{b,m} be the number of certified-conflict seeds in a cell.

G1 — ORIGINAL-SECTOR REPLICATION:
    N_{1,9} >= 4
    N_{1,10} <= 1

G2 — PARITY SWAP:
    N_{2,9} <= 1
    N_{2,10} >= 4

G3 — PAIRED WITHIN-SEED REVERSAL:
    For target 9:
        conflict(b=1,m=9)=True
        conflict(b=2,m=9)=False
    in >=4/5 paired seeds.

    For target 10:
        conflict(b=1,m=10)=False
        conflict(b=2,m=10)=True
    in >=4/5 paired seeds.

G4 — SAME-SECTOR MOBILITY COLLAPSE:
    Across all certified-conflict states in SAME-sector cells,

        mu_raw <= 1e-6

    in >=80%.

G5 — DIFFERENT-SECTOR ESCAPE:
    In each different-sector cell, certified escape must occur in >=4/5.

G6 — EXACT PAIRED INITIALIZATION:
    all four cells for every seed must have max initial parameter difference
    exactly 0 within 1e-15.

STRONG SYMMETRY-SECTOR SUPPORT:
    G1 & G2 & G3 & G4 & G5 & G6.

Reflection-sector diagnostics
-----------------------------
At each full audit, additionally compute the physical reflection-sector
structure of K.

Let

    S = diag((-1)^(k+1)), k=1,...,M.

Define the normalized commutator defect

    delta_R = ||S K - K S||_F / (2 ||K||_F).

delta_R=0 means K exactly preserves the midpoint-reflection sectors.

Also record residual energy fractions in the symmetric (odd-k) and
antisymmetric (even-k) sectors.

These are descriptive in Stage 22R; no threshold is post-hoc invented.

Decision
--------
A) STRONG symmetry-sector support:
       Stage 23R = second PDE / symmetry-broken domain or test-basis robustness.

B) Original b=1 pattern replicates but parity swap fails:
       Stage 23R = base-frequency/scaling control; do NOT claim symmetry sector.

C) Parity swap occurs but mobility collapse fails:
       Stage 23R = symmetry-sector kernel-spectrum mechanism audit.

D) Original b=1 pattern itself fails:
       Stage 23R = initialization/seed heterogeneity audit.

No universal or novelty claim is authorized by Stage 22R alone.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss


BASE_MODES = (1, 2)
TARGET_MODES = (9, 10)
SEEDS = (15, 16, 17, 18, 19)

MAX_EPOCH = 4000
TRACK_INTERVAL = 25

ACTIVE_TARGET_SHARE = 0.80
CONVERGENCE_REL_L2 = 1.0e-2
CONVERGENCE_TARGET_SHARE = 0.20
CERTIFY_POINTS = 3

MOBILITY_COLLAPSE_THRESHOLD = 1.0e-6
INITIAL_CLONE_TOL = 1.0e-15


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-22R midpoint-reflection symmetry-sector swap audit."
    )

    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")

    p.add_argument(
        "--stage3-script",
        default="vpinn_gradient_conflict_stage3_frequency_transfer.py",
    )

    p.add_argument(
        "--stage18-script",
        default="vpinn_gradient_conflict_stage18R_frequency_transfer.py",
    )

    p.add_argument(
        "--stage19-script",
        default="vpinn_gradient_conflict_stage19R_temporal_conflict_parity.py",
    )

    p.add_argument(
        "--stage20-script",
        default="vpinn_gradient_conflict_stage20R_heldout_mobility_unlock.py",
    )

    p.add_argument(
        "--stage21-script",
        default="vpinn_gradient_conflict_stage21R_even_frequency_mobility_transfer.py",
    )

    p.add_argument(
        "--stage21-dir",
        default="vpinn_gradient_conflict_stage21R_even_frequency_mobility_transfer",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage22R_symmetry_sector_swap",
    )

    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(
            [{k: row.get(k, None) for k in fields} for row in rows]
        )


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    return mod


def flatten_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [p.detach().reshape(-1) for p in model.parameters()],
        dim=0,
    )


# =============================================================================
# Preflight
# =============================================================================

def preflight(
    stage3_script: Path,
    stage18_script: Path,
    stage19_script: Path,
    stage20_script: Path,
    stage21_script: Path,
    stage21_dir: Path,
) -> dict:

    manifest_path = stage21_dir / "manifest.json"
    decision_path = stage21_dir / "decision.json"

    if not manifest_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError("Stage-21 manifest/decision missing.")

    manifest = read_json(manifest_path)
    decision = read_json(decision_path)

    s3 = sha256_file(stage3_script)
    s18 = sha256_file(stage18_script)
    s19 = sha256_file(stage19_script)
    s20 = sha256_file(stage20_script)
    s21 = sha256_file(stage21_script)

    if manifest.get("stage3_solver_sha256") != s3:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 21.")

    if manifest.get("stage18_script_sha256") != s18:
        raise RuntimeError("Stage-18 SHA mismatch against Stage 21.")

    if manifest.get("stage19_script_sha256") != s19:
        raise RuntimeError("Stage-19 SHA mismatch against Stage 21.")

    if manifest.get("stage20_script_sha256") != s20:
        raise RuntimeError("Stage-20 SHA mismatch against Stage 21.")

    if manifest.get("stage21r_script_sha256") != s21:
        raise RuntimeError(
            "Stage-21 source SHA mismatch against executed manifest."
        )

    expected_counts = {"6": 0, "8": 0, "10": 1}

    if decision.get("conflict_counts_by_mode") != expected_counts:
        raise RuntimeError(
            "Unexpected Stage-21 conflict counts; symmetry-swap stage was "
            "designed after observing {6:0,8:0,10:1}."
        )

    if bool(decision.get("G1_interleaved_frequency_transition", True)):
        raise RuntimeError("Stage-21 frequency transition unexpectedly passed.")

    if bool(decision.get("G2_invariant_mobility_collapse_transfer", True)):
        raise RuntimeError("Stage-21 mobility-collapse transfer unexpectedly passed.")

    if bool(decision.get("secondary_even_parity_ladder_transfer", True)):
        raise RuntimeError("Stage-21 even-parity ladder unexpectedly passed.")

    if not bool(decision.get("G5_rotation_invariance", False)):
        raise RuntimeError("Stage-21 rotation invariance did not pass.")

    return {
        "stage3_sha256": s3,
        "stage18_sha256": s18,
        "stage19_sha256": s19,
        "stage20_sha256": s20,
        "stage21_sha256": s21,
        "stage21_decision": decision,
    }


# =============================================================================
# Custom symmetry-swap experiment
# =============================================================================

class SymmetrySwapExperiment:
    """
    Same network, quadrature, test space and optimizer as Stage 3.

    Only the manufactured solution / forcing is generalized from base mode 1
    to base mode b in {1,2}, with weak-scale-matched amplitude 1/b.
    """

    def __init__(
        self,
        stage3,
        cfg,
        device: torch.device,
        base_mode: int,
        target_mode: int,
        out_dir: Path,
    ):
        self.stage3 = stage3
        self.cfg = cfg
        self.device = device
        self.dtype = torch.float64

        self.base_mode = int(base_mode)
        self.mode = int(target_mode)

        self.base_amplitude = 1.0 / self.base_mode
        self.amplitude = (
            cfg.reference_amplitude
            * cfg.reference_mode
            / self.mode
        )

        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        stage3.seed_everything(cfg.seed)

        self.model = stage3.VPINNNetwork(
            cfg.width,
            cfg.depth,
        ).to(
            device=device,
            dtype=self.dtype,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
        )

        self._build_quadrature_and_tests()

    @property
    def same_reflection_sector(self) -> bool:
        return (self.base_mode % 2) == (self.mode % 2)

    @property
    def base_reflection_sign(self) -> int:
        return (-1) ** (self.base_mode + 1)

    @property
    def target_reflection_sign(self) -> int:
        return (-1) ** (self.mode + 1)

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * torch.sin(self.base_mode * math.pi * x)
            +
            self.amplitude
            * torch.sin(self.mode * math.pi * x)
        )

    def forcing(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * (self.base_mode * math.pi) ** 2
            * torch.sin(self.base_mode * math.pi * x)
            +
            self.amplitude
            * (self.mode * math.pi) ** 2
            * torch.sin(self.mode * math.pi * x)
        )

    def _build_quadrature_and_tests(self):
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
            phase = math.pi * (
                self.x_quad.detach() @ k
            )

            self.test_values = (
                math.sqrt(2.0)
                / (math.pi * k)
                * torch.sin(phase)
            )

            self.test_derivatives = (
                math.sqrt(2.0)
                * torch.cos(phase)
            )

            self.forcing_values = self.forcing(
                self.x_quad.detach()
            )

            weighted = (
                self.test_derivatives
                * torch.sqrt(self.w_quad)
            )

            gram = weighted.T @ weighted

            identity = torch.eye(
                self.cfg.n_test,
                dtype=self.dtype,
                device=self.device,
            )

            self.gram_error = float(
                torch.max(
                    torch.abs(gram - identity)
                ).item()
            )

        if (
            not np.isfinite(self.gram_error)
            or self.gram_error > 1.0e-10
        ):
            raise RuntimeError(
                f"Gram check failed: "
                f"b={self.base_mode}, m={self.mode}, "
                f"error={self.gram_error:.3e}"
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
            -
            self.forcing_values * self.test_values
        )

        return torch.sum(
            self.w_quad * integrand,
            dim=0,
        )

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
        truth = self.exact_solution(x)

        return float(
            (
                torch.linalg.vector_norm(pred - truth)
                /
                torch.linalg.vector_norm(truth)
            ).item()
        )

    def residual_metrics(self) -> Dict[str, float]:
        residuals = self.weak_residuals().detach()
        energy = residuals.square()
        total = energy.sum().clamp_min(1.0e-300)

        t = self.mode - 1
        dominant = int(
            torch.argmax(energy).item()
        )

        return {
            "vpinn_loss":
                float(torch.mean(energy).item()),

            "residual_l2_norm":
                float(
                    torch.linalg.vector_norm(residuals).item()
                ),

            "dominant_residual_mode":
                dominant + 1,

            "dominant_residual_energy_share":
                float(
                    (energy[dominant] / total).item()
                ),

            "target_mode_residual_energy_share":
                float(
                    (energy[t] / total).item()
                ),

            "target_mode_abs_residual":
                float(
                    torch.abs(residuals[t]).item()
                ),
        }

    def train_step(self) -> float:
        self.model.train()

        self.optimizer.zero_grad(set_to_none=True)

        residuals = self.weak_residuals()
        loss = residuals.square().mean()

        loss.backward()
        self.optimizer.step()

        return float(loss.detach().item())


def make_config(stage3, seed: int, device: torch.device, out_dir: Path):
    return stage3.Config(
        seed=seed,
        device=str(device),
        epochs=MAX_EPOCH,
        learning_rate=1.0e-3,
        width=32,
        depth=3,
        n_test=24,
        n_quad=256,
        n_eval=4001,
        modes=(9,),
        reference_mode=7,
        reference_amplitude=0.15,
        track_interval=TRACK_INTERVAL,
        diagnostic_epochs=(0,),
        convergence_error_threshold=CONVERGENCE_REL_L2,
        localization_share_threshold=ACTIVE_TARGET_SHARE,
        resolved_share_threshold=CONVERGENCE_TARGET_SHARE,
        conflict_gamma_threshold=0.20,
        conflict_weighted_negative_threshold=0.50,
        active_relative_tol=1.0e-8,
        grad_absolute_eps=1.0e-300,
        output_dir=str(out_dir),
        dpi=220,
    )


# =============================================================================
# Reflection-sector diagnostics
# =============================================================================

def reflection_sector_diagnostics(
    r: np.ndarray,
    K: np.ndarray,
) -> dict:

    M = r.size

    modes = np.arange(1, M + 1)

    # +1 for odd modes (midpoint symmetric), -1 for even modes.
    signs = np.where(
        modes % 2 == 1,
        1.0,
        -1.0,
    )

    S = np.diag(signs)

    Ksym = 0.5 * (K + K.T)

    denom = max(
        np.linalg.norm(Ksym, ord="fro"),
        1.0e-300,
    )

    comm = S @ Ksym - Ksym @ S

    commutator_defect = float(
        np.linalg.norm(comm, ord="fro")
        /
        (2.0 * denom)
    )

    e = r * r
    et = max(float(e.sum()), 1.0e-300)

    symmetric_share = float(
        e[modes % 2 == 1].sum() / et
    )

    antisymmetric_share = float(
        e[modes % 2 == 0].sum() / et
    )

    K_cross = Ksym[
        np.ix_(
            modes % 2 == 1,
            modes % 2 == 0,
        )
    ]

    cross_block_frobenius_ratio = float(
        np.linalg.norm(K_cross, ord="fro")
        /
        denom
    )

    return {
        "reflection_commutator_defect":
            commutator_defect,

        "reflection_cross_block_frobenius_ratio":
            cross_block_frobenius_ratio,

        "symmetric_residual_energy_share":
            symmetric_share,

        "antisymmetric_residual_energy_share":
            antisymmetric_share,
    }


# =============================================================================
# Full audit
# =============================================================================

def full_audit(
    stage18,
    stage20,
    exp: SymmetrySwapExperiment,
    seed: int,
    base_mode: int,
    target_mode: int,
    epoch: int,
    kind: str,
    out_dir: Path,
) -> dict:

    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    kernel = stage18.residual_jacobian(exp)

    raw = stage18.kernel_summary(
        r=kernel["r"],
        K=kernel["K"],
        Kcorr=kernel["Kcorr"],
        target_index=target_mode - 1,
    )

    adam = stage18.predict_and_decompose_adam(
        exp=exp,
        J=kernel["J"],
        r=kernel["r"],
        params=kernel["params"],
        target_index=target_mode - 1,
    )

    formula_check = stage18.verify_predicted_adam_step(
        exp=exp,
        predicted=adam["delta_candidate"],
    )

    if not formula_check["pass"]:
        raise RuntimeError(
            f"Adam formula check failed "
            f"seed={seed}, b={base_mode}, m={target_mode}, "
            f"epoch={epoch}."
        )

    pareto = stage18.pareto_curvature_audit(
        exp=exp,
        kernel=kernel,
        adam=adam,
        target_index=target_mode - 1,
    )

    r_np = kernel["r"].cpu().numpy()
    K_np = kernel["K"].cpu().numpy()
    KD_np = adam["K_D"].cpu().numpy()

    inv = stage20.kernel_invariants(
        r=r_np,
        K=K_np,
        KD=KD_np,
        rotation_seed=(
            220000
            + 10000 * base_mode
            + 1000 * target_mode
            + seed
        ),
    )

    sector = reflection_sector_diagnostics(
        r=r_np,
        K=K_np,
    )

    np.savez_compressed(
        out_dir / f"{kind.lower()}_kernels.npz",
        residuals=r_np,
        raw_kernel=K_np,
        raw_kernel_corr=
            kernel["Kcorr"].cpu().numpy(),
        adam_current_kernel=KD_np,
        adam_current_kernel_corr=
            adam["K_D_corr"].cpu().numpy(),
    )

    return {
        "seed": seed,
        "base_mode": base_mode,
        "target_mode": target_mode,
        "same_reflection_sector":
            exp.same_reflection_sector,

        "base_amplitude":
            exp.base_amplitude,

        "target_amplitude":
            exp.amplitude,

        "base_weak_scale":
            exp.base_amplitude * base_mode,

        "target_weak_scale":
            exp.amplitude * target_mode,

        "epoch":
            epoch,

        "audit_kind":
            kind,

        "relative_l2_error":
            rel,

        **rm,
        **raw,

        "adam_target_uphill_cosine":
            adam["adam_target_uphill_cosine"],

        "adam_candidate_target_uphill":
            adam["adam_candidate_target_uphill"],

        "target_dot_current":
            adam["target_dot_current"],

        "target_dot_history":
            adam["target_dot_history"],

        "target_dot_candidate":
            adam["target_dot_candidate"],

        "adam_formula_max_abs_difference":
            formula_check["max_abs_difference"],

        "pareto_status":
            pareto.get("pareto_status"),

        "pareto_active":
            pareto.get("pareto_active"),

        **inv,
        **sector,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_cell_conflicts(cell_summary: List[dict], path: Path) -> None:

    labels = [
        f"b={int(r['base_mode'])}, m={int(r['target_mode'])}"
        for r in cell_summary
    ]

    vals = [
        int(r["certified_conflict_count"])
        for r in cell_summary
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    ax.bar(labels, vals)

    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Certified-conflict seeds out of 5")
    ax.set_xlabel("Base/target cell")
    ax.set_title("Does conflict follow reflection-sector match rather than frequency?")

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_mobility(audits: List[dict], path: Path) -> None:

    conflict = [
        r for r in audits
        if r["audit_kind"] == "CERTIFIED_CONFLICT_ONSET"
    ]

    if not conflict:
        return

    x = np.arange(len(conflict))

    fig, ax = plt.subplots(figsize=(10.0, 5.4))

    ax.scatter(
        x,
        [float(r["mu_raw"]) for r in conflict],
    )

    ax.axhline(
        MOBILITY_COLLAPSE_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
    )

    labels = [
        f"s{int(r['seed'])}\nb{int(r['base_mode'])}m{int(r['target_mode'])}"
        for r in conflict
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Basis-invariant residual mobility μ")
    ax.set_title("Mobility at certified conflict in the symmetry-swap experiment")

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> int:

    args = parse_args()

    root = Path(__file__).resolve().parent

    def resolve(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else root / p

    stage3_script = resolve(args.stage3_script)
    stage18_script = resolve(args.stage18_script)
    stage19_script = resolve(args.stage19_script)
    stage20_script = resolve(args.stage20_script)
    stage21_script = resolve(args.stage21_script)
    stage21_dir = resolve(args.stage21_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage18_script=stage18_script,
        stage19_script=stage19_script,
        stage20_script=stage20_script,
        stage21_script=stage21_script,
        stage21_dir=stage21_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage22R",
    )

    stage18 = load_module(
        stage18_script,
        "vpinn_stage18_stage22R",
    )

    stage19 = load_module(
        stage19_script,
        "vpinn_stage19_stage22R",
    )

    stage20 = load_module(
        stage20_script,
        "vpinn_stage20_stage22R",
    )

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),

        "stage3_solver_sha256":
            pf["stage3_sha256"],

        "stage18_script_sha256":
            pf["stage18_sha256"],

        "stage19_script_sha256":
            pf["stage19_sha256"],

        "stage20_script_sha256":
            pf["stage20_sha256"],

        "stage21_script_sha256":
            pf["stage21_sha256"],

        "stage22r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "midpoint_reflection_symmetry_sector_swap",

            "base_modes":
                list(BASE_MODES),

            "target_modes":
                list(TARGET_MODES),

            "new_seeds":
                list(SEEDS),

            "exact_solution":
                "u=(1/b)sin(b*pi*x)+(1.05/m)sin(m*pi*x)",

            "base_weak_scale":
                1.0,

            "target_weak_scale":
                1.05,

            "paired_initialization":
                True,

            "conflict":
                "3 consecutive active target-uphill Adam probes",

            "escape":
                "3 consecutive relL2<=1e-2 and target share<=0.20",

            "G1":
                "N_1,9>=4 and N_1,10<=1",

            "G2":
                "N_2,9<=1 and N_2,10>=4",

            "G3":
                "paired reversal >=4/5 for target9 and >=4/5 for target10",

            "G4":
                "mu<=1e-6 in >=80% same-sector conflict states",

            "G5":
                "different-sector escape >=4/5 in each cell",

            "G6":
                "all paired initial parameter max differences <=1e-15",

            "no_optimizer_intervention":
                True,
        },

        "stage21_result_trigger": {
            "conflict_counts":
                pf["stage21_decision"][
                    "conflict_counts_by_mode"
                ],

            "frequency_transition":
                pf["stage21_decision"][
                    "G1_interleaved_frequency_transition"
                ],

            "mobility_transfer":
                pf["stage21_decision"][
                    "G2_invariant_mobility_collapse_transfer"
                ],
        },
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 176)
    print(
        "VPINN — STAGE 22R MIDPOINT-REFLECTION SYMMETRY-SECTOR SWAP AUDIT"
    )
    print("=" * 176)

    print(f"device                    : {device}")
    print(f"base modes                : {list(BASE_MODES)}")
    print(f"target modes              : {list(TARGET_MODES)}")
    print(f"new paired seeds          : {list(SEEDS)}")
    print("base weak scale           : 1.00")
    print("target weak scale         : 1.05")
    print("optimizer intervention    : NONE")
    print("=" * 176)

    tracking_rows = []
    run_rows = []
    audit_rows = []
    init_rows = []

    global_start = time.perf_counter()

    for seed in SEEDS:

        initial_vectors = {}

        for base_mode in BASE_MODES:
            for target_mode in TARGET_MODES:

                cell_dir = (
                    out_dir
                    / f"seed_{seed:03d}"
                    / f"base_{base_mode:02d}_target_{target_mode:02d}"
                )

                cfg = make_config(
                    stage3=stage3,
                    seed=seed,
                    device=device,
                    out_dir=cell_dir,
                )

                exp = SymmetrySwapExperiment(
                    stage3=stage3,
                    cfg=cfg,
                    device=device,
                    base_mode=base_mode,
                    target_mode=target_mode,
                    out_dir=cell_dir,
                )

                initial_vectors[
                    (base_mode, target_mode)
                ] = flatten_params(exp.model).cpu()

                conflict_streak = 0
                conflict_candidate_epoch = None
                conflict_candidate_state = None
                conflict_onset = -1
                conflict_confirmation = -1
                conflict_state = None

                escape_streak = 0
                escape_candidate_epoch = None
                escape_candidate_state = None
                escape_onset = -1
                escape_confirmation = -1
                escape_state = None

                print()
                print("-" * 176)
                print(
                    f"seed={seed} "
                    f"base={base_mode} target={target_mode} "
                    f"same_sector={exp.same_reflection_sector} "
                    f"c_b={exp.base_amplitude:.12g} "
                    f"a_m={exp.amplitude:.12g}"
                )

                for epoch in range(MAX_EPOCH + 1):

                    if epoch % TRACK_INTERVAL == 0:

                        rm = exp.residual_metrics()
                        rel = exp.relative_l2_error()

                        mechanism_active = bool(
                            rm[
                                "target_mode_residual_energy_share"
                            ]
                            >= ACTIVE_TARGET_SHARE
                            and
                            rel > CONVERGENCE_REL_L2
                        )

                        probe = None

                        if mechanism_active:
                            probe = stage19.cheap_probe(
                                exp=exp,
                                mode=target_mode,
                            )

                            state = stage18.capture_state(exp)

                            condition = bool(
                                probe[
                                    "adam_candidate_target_uphill"
                                ]
                            )

                            if condition:
                                if conflict_streak == 0:
                                    conflict_candidate_epoch = epoch
                                    conflict_candidate_state = copy.deepcopy(
                                        state
                                    )

                                conflict_streak += 1

                            else:
                                conflict_streak = 0
                                conflict_candidate_epoch = None
                                conflict_candidate_state = None

                            if (
                                conflict_onset < 0
                                and
                                conflict_streak >= CERTIFY_POINTS
                            ):
                                conflict_onset = int(
                                    conflict_candidate_epoch
                                )
                                conflict_confirmation = epoch
                                conflict_state = copy.deepcopy(
                                    conflict_candidate_state
                                )

                        qualifies_escape = bool(
                            rel <= CONVERGENCE_REL_L2
                            and
                            rm[
                                "target_mode_residual_energy_share"
                            ]
                            <= CONVERGENCE_TARGET_SHARE
                        )

                        if qualifies_escape:

                            state = stage18.capture_state(exp)

                            if escape_streak == 0:
                                escape_candidate_epoch = epoch
                                escape_candidate_state = copy.deepcopy(
                                    state
                                )

                            escape_streak += 1

                        else:
                            escape_streak = 0
                            escape_candidate_epoch = None
                            escape_candidate_state = None

                        if (
                            escape_onset < 0
                            and
                            escape_streak >= CERTIFY_POINTS
                        ):
                            escape_onset = int(
                                escape_candidate_epoch
                            )
                            escape_confirmation = epoch
                            escape_state = copy.deepcopy(
                                escape_candidate_state
                            )

                        tracking_rows.append(
                            {
                                "seed": seed,
                                "base_mode": base_mode,
                                "target_mode": target_mode,
                                "same_reflection_sector":
                                    exp.same_reflection_sector,

                                "epoch": epoch,
                                "relative_l2_error": rel,
                                **rm,

                                "mechanism_active":
                                    mechanism_active,

                                "adam_target_uphill_cosine": (
                                    probe[
                                        "adam_target_uphill_cosine"
                                    ]
                                    if probe is not None
                                    else None
                                ),

                                "adam_candidate_target_uphill": (
                                    probe[
                                        "adam_candidate_target_uphill"
                                    ]
                                    if probe is not None
                                    else None
                                ),

                                "signed_sqgrad_cos_m_plus_2": (
                                    probe[
                                        "signed_sqgrad_cos_m_plus_2"
                                    ]
                                    if probe is not None
                                    else None
                                ),

                                "conflict_streak":
                                    conflict_streak,
                            }
                        )

                        if conflict_onset >= 0:
                            break

                        if escape_onset >= 0:
                            break

                    if epoch < MAX_EPOCH:
                        exp.train_step()

                # -------------------------------------------------------------
                # One full audit per run.
                # -------------------------------------------------------------
                if conflict_state is not None:
                    audit_kind = "CERTIFIED_CONFLICT_ONSET"
                    audit_epoch = conflict_onset
                    audit_state = conflict_state
                elif escape_state is not None:
                    audit_kind = "CERTIFIED_ESCAPE_ONSET"
                    audit_epoch = escape_onset
                    audit_state = escape_state
                else:
                    audit_kind = None
                    audit_epoch = None
                    audit_state = None

                audit = None

                if audit_state is not None:

                    audit_dir = (
                        cell_dir
                        / audit_kind.lower()
                    )

                    audit_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    audit_exp = SymmetrySwapExperiment(
                        stage3=stage3,
                        cfg=cfg,
                        device=device,
                        base_mode=base_mode,
                        target_mode=target_mode,
                        out_dir=audit_dir,
                    )

                    stage18.restore_state(
                        audit_exp,
                        audit_state,
                    )

                    audit = full_audit(
                        stage18=stage18,
                        stage20=stage20,
                        exp=audit_exp,
                        seed=seed,
                        base_mode=base_mode,
                        target_mode=target_mode,
                        epoch=audit_epoch,
                        kind=audit_kind,
                        out_dir=audit_dir,
                    )

                    audit_rows.append(audit)

                run_rows.append(
                    {
                        "seed": seed,
                        "base_mode": base_mode,
                        "target_mode": target_mode,

                        "same_reflection_sector":
                            exp.same_reflection_sector,

                        "base_reflection_sign":
                            exp.base_reflection_sign,

                        "target_reflection_sign":
                            exp.target_reflection_sign,

                        "certified_conflict":
                            conflict_onset >= 0,

                        "conflict_onset_epoch":
                            conflict_onset,

                        "conflict_confirmation_epoch":
                            conflict_confirmation,

                        "certified_escape":
                            escape_onset >= 0,

                        "escape_onset_epoch":
                            escape_onset,

                        "escape_confirmation_epoch":
                            escape_confirmation,

                        "audit_kind":
                            audit_kind,

                        "audit_mu_raw": (
                            float(audit["mu_raw"])
                            if audit is not None
                            else None
                        ),

                        "audit_effective_rank": (
                            float(audit["effective_rank"])
                            if audit is not None
                            else None
                        ),

                        "audit_reflection_commutator_defect": (
                            float(
                                audit[
                                    "reflection_commutator_defect"
                                ]
                            )
                            if audit is not None
                            else None
                        ),
                    }
                )

                print(
                    f"  conflict={conflict_onset} "
                    f"escape={escape_onset} "
                    f"audit={audit_kind} "
                    f"mu="
                    f"{run_rows[-1]['audit_mu_raw']}"
                )

        # ---------------------------------------------------------------------
        # Exact paired initialization checks for this seed.
        # ---------------------------------------------------------------------
        reference_key = (1, 9)
        ref = initial_vectors[reference_key]

        for key, vec in initial_vectors.items():
            gap = float(
                torch.max(
                    torch.abs(vec - ref)
                ).item()
            )

            init_rows.append(
                {
                    "seed": seed,
                    "reference_base_mode":
                        reference_key[0],

                    "reference_target_mode":
                        reference_key[1],

                    "base_mode":
                        key[0],

                    "target_mode":
                        key[1],

                    "max_abs_initial_parameter_gap":
                        gap,

                    "pass":
                        bool(gap <= INITIAL_CLONE_TOL),
                }
            )

            if gap > INITIAL_CLONE_TOL:
                raise RuntimeError(
                    f"Paired initialization failed "
                    f"seed={seed}, cell={key}, gap={gap:.3e}"
                )

    # =========================================================================
    # Persist
    # =========================================================================
    write_csv(
        out_dir / "tracking_metrics.csv",
        tracking_rows,
    )

    write_csv(
        out_dir / "run_summary.csv",
        run_rows,
    )

    write_csv(
        out_dir / "event_kernel_audits.csv",
        audit_rows,
    )

    write_csv(
        out_dir / "paired_initialization_checks.csv",
        init_rows,
    )

    # =========================================================================
    # Aggregate cells
    # =========================================================================
    cell_summary = []

    for base_mode in BASE_MODES:
        for target_mode in TARGET_MODES:

            rr = [
                r for r in run_rows
                if (
                    int(r["base_mode"]) == base_mode
                    and
                    int(r["target_mode"]) == target_mode
                )
            ]

            conflicts = [
                r for r in rr
                if bool(r["certified_conflict"])
            ]

            escapes = [
                r for r in rr
                if bool(r["certified_escape"])
            ]

            cell_summary.append(
                {
                    "base_mode":
                        base_mode,

                    "target_mode":
                        target_mode,

                    "same_reflection_sector":
                        (base_mode % 2)
                        == (target_mode % 2),

                    "certified_conflict_count":
                        len(conflicts),

                    "certified_escape_count":
                        len(escapes),

                    "median_conflict_onset_epoch": (
                        float(
                            np.median([
                                int(
                                    r[
                                        "conflict_onset_epoch"
                                    ]
                                )
                                for r in conflicts
                            ])
                        )
                        if conflicts
                        else None
                    ),

                    "median_conflict_mu_raw": (
                        float(
                            np.median([
                                float(r["audit_mu_raw"])
                                for r in conflicts
                            ])
                        )
                        if conflicts
                        else None
                    ),
                }
            )

    write_csv(
        out_dir / "cell_summary.csv",
        cell_summary,
    )

    counts = {
        (int(r["base_mode"]), int(r["target_mode"])):
            int(r["certified_conflict_count"])
        for r in cell_summary
    }

    # =========================================================================
    # Gates
    # =========================================================================
    G1 = bool(
        counts[(1, 9)] >= 4
        and
        counts[(1, 10)] <= 1
    )

    G2 = bool(
        counts[(2, 9)] <= 1
        and
        counts[(2, 10)] >= 4
    )

    by = {
        (
            int(r["seed"]),
            int(r["base_mode"]),
            int(r["target_mode"]),
        ): bool(r["certified_conflict"])
        for r in run_rows
    }

    target9_reversal = sum(
        int(
            by[(seed, 1, 9)]
            and
            not by[(seed, 2, 9)]
        )
        for seed in SEEDS
    )

    target10_reversal = sum(
        int(
            not by[(seed, 1, 10)]
            and
            by[(seed, 2, 10)]
        )
        for seed in SEEDS
    )

    G3 = bool(
        target9_reversal >= 4
        and
        target10_reversal >= 4
    )

    same_sector_conflict_audits = [
        r for r in audit_rows
        if (
            bool(r["same_reflection_sector"])
            and
            r["audit_kind"]
            == "CERTIFIED_CONFLICT_ONSET"
        )
    ]

    collapse_count = sum(
        int(
            float(r["mu_raw"])
            <= MOBILITY_COLLAPSE_THRESHOLD
        )
        for r in same_sector_conflict_audits
    )

    G4 = bool(
        same_sector_conflict_audits
        and
        collapse_count
        >= math.ceil(
            0.80
            * len(same_sector_conflict_audits)
        )
    )

    different_cells = [
        (1, 10),
        (2, 9),
    ]

    different_escape_counts = {
        cell: sum(
            int(
                bool(r["certified_escape"])
            )
            for r in run_rows
            if (
                int(r["base_mode"]),
                int(r["target_mode"]),
            ) == cell
        )
        for cell in different_cells
    }

    G5 = all(
        different_escape_counts[cell] >= 4
        for cell in different_cells
    )

    G6 = all(
        bool(r["pass"])
        for r in init_rows
    )

    factorial_interaction_score = (
        counts[(1, 9)]
        + counts[(2, 10)]
        - counts[(1, 10)]
        - counts[(2, 9)]
    )

    strong = bool(
        G1 and G2 and G3 and G4 and G5 and G6
    )

    if strong:
        route_class = (
            "midpoint_reflection_sector_swap_strongly_supported"
        )

        next_route = (
            "stage23R_second_problem_or_symmetry_broken_robustness"
        )

    elif G1 and not G2:
        route_class = (
            "original_odd_even_pattern_replicates_but_base_parity_does_not_swap"
        )

        next_route = (
            "stage23R_base_mode_scaling_control"
        )

    elif G2 and not G4:
        route_class = (
            "symmetry_swap_occurs_without_deep_mobility_collapse"
        )

        next_route = (
            "stage23R_symmetry_sector_kernel_spectrum_audit"
        )

    else:
        route_class = (
            "symmetry_sector_hypothesis_not_cleanly_supported"
        )

        next_route = (
            "stage23R_seed_architecture_heterogeneity_audit"
        )

    decision = {
        "new_seeds":
            list(SEEDS),

        "conflict_counts": {
            "b1_m9": counts[(1, 9)],
            "b1_m10": counts[(1, 10)],
            "b2_m9": counts[(2, 9)],
            "b2_m10": counts[(2, 10)],
        },

        "factorial_interaction_score":
            factorial_interaction_score,

        "G1_original_sector_replication":
            G1,

        "G2_base_parity_swap":
            G2,

        "paired_target9_reversal_count":
            target9_reversal,

        "paired_target10_reversal_count":
            target10_reversal,

        "G3_paired_within_seed_reversal":
            G3,

        "same_sector_conflict_state_count":
            len(same_sector_conflict_audits),

        "same_sector_mobility_collapse_count":
            collapse_count,

        "G4_same_sector_mobility_collapse":
            G4,

        "different_sector_escape_counts": {
            "b1_m10":
                different_escape_counts[(1, 10)],

            "b2_m9":
                different_escape_counts[(2, 9)],
        },

        "G5_different_sector_escape":
            G5,

        "G6_exact_paired_initialization":
            G6,

        "strong_symmetry_sector_support":
            strong,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A strong 2x2 parity swap would support a midpoint-reflection "
            "sector-selective mechanism in this manufactured 1D VPINN family. "
            "It would be much stronger than a frequency-only claim, but still "
            "requires a second PDE, symmetry-broken control, or independent "
            "test-space robustness experiment before broad VPINN claims."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    plot_cell_conflicts(
        cell_summary,
        out_dir / "symmetry_swap_conflict_counts.png",
    )

    plot_mobility(
        audit_rows,
        out_dir / "same_sector_conflict_mobility.png",
    )

    # =========================================================================
    # Console summary
    # =========================================================================
    elapsed = time.perf_counter() - global_start

    lines = []

    lines.append("=" * 180)
    lines.append(
        "VPINN — STAGE 22R MIDPOINT-REFLECTION SYMMETRY-SECTOR SWAP SUMMARY"
    )
    lines.append("=" * 180)

    lines.append(
        "base | target | same sector | conflict | escape | median conflict epoch | median mu(conflict)"
    )

    lines.append("-" * 180)

    for r in cell_summary:
        lines.append(
            f"{int(r['base_mode']):4d} | "
            f"{int(r['target_mode']):6d} | "
            f"{str(r['same_reflection_sector']):11s} | "
            f"{int(r['certified_conflict_count']):4d}/5   | "
            f"{int(r['certified_escape_count']):4d}/5 | "
            f"{str(r['median_conflict_onset_epoch']):21s} | "
            f"{str(r['median_conflict_mu_raw'])}"
        )

    lines.append("-" * 180)

    lines.append(
        f"conflict counts                        : "
        f"{decision['conflict_counts']}"
    )

    lines.append(
        f"factorial interaction score            : "
        f"{factorial_interaction_score}"
    )

    lines.append(
        f"G1 original b=1 odd/even replication   : {G1}"
    )

    lines.append(
        f"G2 base-parity conflict swap            : {G2}"
    )

    lines.append(
        f"G3 paired within-seed reversal           : "
        f"target9={target9_reversal}/5, "
        f"target10={target10_reversal}/5 -> {G3}"
    )

    lines.append(
        f"G4 same-sector mobility collapse         : "
        f"{collapse_count}/"
        f"{len(same_sector_conflict_audits)} -> {G4}"
    )

    lines.append(
        f"G5 different-sector escape               : "
        f"{different_escape_counts} -> {G5}"
    )

    lines.append(
        f"G6 exact paired initialization            : "
        f"{sum(int(r['pass']) for r in init_rows)}/"
        f"{len(init_rows)} -> {G6}"
    )

    lines.append(
        f"STRONG SYMMETRY-SECTOR SUPPORT           : {strong}"
    )

    lines.append(
        f"route class                               : {route_class}"
    )

    lines.append(
        f"next route                                : {next_route}"
    )

    lines.append(
        f"elapsed seconds                           : {elapsed:.2f}"
    )

    lines.append("=" * 180)

    lines.append(
        "Guardrail: this factorial swap is designed to distinguish frequency "
        "from midpoint-reflection sector membership. If it fails, do not "
        "rescue the symmetry story by changing the gates."
    )

    lines.append("=" * 180)

    summary = "\n".join(lines)

    print()
    print(summary)

    (out_dir / "console_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
