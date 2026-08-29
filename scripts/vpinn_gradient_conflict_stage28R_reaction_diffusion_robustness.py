#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 28R
Independent Reaction-Diffusion Robustness of Deep Residual-Mobility Lock
=========================================================================

Scientific motivation
---------------------
The Poisson-family chain has now established:

1) target-uphill Adam conflict can be transient and is NOT sufficient for lock;
2) the pathological b=1,m=9 plateau is distinguished by persistent
   basis-invariant residual-mobility collapse;
3) that collapse precedes certified Adam conflict by ~1.3k-1.9k epochs;
4) strong pair-specific K/r interaction develops later and is large by conflict.

The next scientifically useful question is whether the DEEP-LOCK phenotype
survives a genuine change of PDE operator.

Stage 28R changes

    -u'' = f

to the reaction-diffusion problem

    -u'' + sigma u = f,   x in (0,1),
    u(0)=u(1)=0,

with

    sigma = (5*pi)^2.

Why this sigma?
---------------
The reaction scale k_sigma=5 lies between the low background modes b=1,2
and the hard target m=9.

Therefore the reaction term is:
    * not a tiny perturbation of the Poisson operator;
    * not so large that it completely dominates the m=9 diffusion scale.

No Stage-28 result is used to choose sigma.

Energy-orthonormal weak tests
-----------------------------
For

    lambda_k = (k*pi)^2 + sigma,

use

    v_k(x) = sqrt(2)/sqrt(lambda_k) * sin(k*pi*x).

Then

    a(v_i,v_j)
      = int v_i' v_j' + sigma v_i v_j
      = delta_ij

up to quadrature precision.

Matched weak scales
-------------------
Use the manufactured family

    u*_{b,9}(x)
      = c_b sin(b*pi*x) + a_9 sin(9*pi*x),

where

    c_b = pi / sqrt(lambda_b),
    a_9 = 1.05*pi / sqrt(lambda_9).

For an energy-normalized test, a missing sine component A sin(k*pi*x)
produces weak coefficient

    A * sqrt(lambda_k/2).

Hence the base and target weak scales are EXACTLY

    base   = pi/sqrt(2),
    target = 1.05*pi/sqrt(2)

for BOTH b=1 and b=2.

In the Poisson limit sigma->0, these amplitudes reduce exactly to

    c_b = 1/b,
    a_9 = 1.05/9,

the prior Stage-22 family.

Paired design
-------------
New seeds:

    {20,21,22,23,24}

Two cells per seed:

    b=1,m=9
    b=2,m=9

The model initialization must be exactly paired across b=1 and b=2.

Ordinary Adam only.
Same network depth/width, quadrature count, test count, dtype and learning rate.

Analytic preflight
------------------
For every constructed cell require:

    energy-test Gram max error <= 1e-10

and the exact manufactured weak residual evaluated directly on quadrature

    max_k |a(u*,v_k)-l(v_k)| <= 1e-10.

Failure aborts before training.

Tracking
--------
Cheap residual / Adam-target geometry every 25 epochs.

After target localization:

    target residual share >=0.80
    AND relL2>1e-2,

compute full J/K geometry at:
    * localization onset;
    * every global 250 epochs.

Persistent deep lock
--------------------
Inherited unchanged:

while unresolved and target-localized, TWO consecutive full audits satisfy

    mu = r^T K r / (||r||^2 tr K) <= 1e-6.

The event onset is the first of those two audits.

Certified escape
----------------
Inherited unchanged:

    relL2<=1e-2
    AND target residual share<=0.20

for THREE consecutive 25-epoch observations.

Efficiency stop
---------------
A cell stops at the earliest of:

    * persistent deep-lock certification;
    * certified escape;
    * epoch 3000.

This stage is a robustness pilot. Once deep lock is certified, no expensive
post-lock continuation is needed.

Temporal factor decomposition
-----------------------------
At each full audit t relative to localization baseline (K0,r0), compute

    mu_00 = mu(K0,r0)
    mu_tt = mu(Kt,rt)
    mu_t0 = mu(Kt,r0)
    mu_0t = mu(K0,rt)

and exact log-Shapley changes:

    phi_K, phi_r,
    phi_K + phi_r = log(mu_tt/mu_00).

For a mobility drop define

    D_K = -phi_K,
    D_r = -phi_r.

The Poisson Stage-27 discovery suggested the EARLY collapse is
residual-direction-heavy even though the later deep endpoint requires joint
K/r geometry.

Stage 28 therefore precommits the independent test:

    D_r > D_K

at persistent deep-lock onset.

This is a newly confirmatory use on a different PDE.

Primary gates
-------------

R1 — ANALYTIC / PAIRED PREFLIGHT
    10/10 cells:
        Gram error <=1e-10
        manufactured weak residual <=1e-10
    and 5/5 paired initializations have max parameter gap <=1e-15.

R2 — MOBILE LOCALIZED BASELINE
    localized baseline mu>1e-6 in >=4/5 b=1 cells.

R3 — B=1 DEEP-LOCK TRANSFER
    persistent deep mobility lock in >=4/5 b=1 cells.

R4 — PAIRED B=2 NON-LOCK CONTROL
    persistent deep lock in <=1/5 b=2 cells
    AND certified escape in >=4/5 b=2 cells.

R5 — EARLY COLLAPSE REMAINS RESIDUAL-DROP DOMINATED
    among b=1 cells with certified deep lock,
    D_r > D_K at deep-lock onset in >=80%.

INDEPENDENT-PDE DEEP-LOCK ROBUSTNESS:
    R1 & R2 & R3 & R4.

TWO-STAGE GEOMETRY ROBUSTNESS:
    independent-PDE robustness & R5.

Decision
--------
A) Independent-PDE robustness + R5:
       Stage 29R = independent TEST-SPACE robustness using a non-Fourier
                   orthonormalized weak-test family.

B) R3 passes but R4 fails:
       operator transfer exists but base-mode specificity is lost;
       route to reaction-strength localization, not test-space claim.

C) R3 fails:
       the deep-lock mechanism is Poisson/operator-family specific;
       do not broaden the claim.

Guardrail
---------
A PASS supports transfer from Poisson to this reaction-diffusion operator.
It is still one architecture and one manufactured 1D solution family.
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
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss


SEEDS = (20, 21, 22, 23, 24)
BASE_MODES = (1, 2)
TARGET_MODE = 9

REACTION_WAVENUMBER = 5.0
SIGMA = (REACTION_WAVENUMBER * math.pi) ** 2

MAX_EPOCH = 3000
TRACK_INTERVAL = 25
FULL_AUDIT_INTERVAL = 250

LOCALIZE_SHARE = 0.80
CONVERGENCE_REL_L2 = 1.0e-2
CONVERGENCE_TARGET_SHARE = 0.20
ESCAPE_CERTIFY_POINTS = 3

MOBILITY_COLLAPSE_THRESHOLD = 1.0e-6
DEEP_LOCK_CERTIFY_POINTS = 2

PREFLIGHT_TOL = 1.0e-10
INITIAL_CLONE_TOL = 1.0e-15


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-28R reaction-diffusion independent-PDE robustness."
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
        "--stage22-script",
        default="vpinn_gradient_conflict_stage22R_symmetry_sector_swap.py",
    )
    p.add_argument(
        "--stage27-script",
        default="vpinn_gradient_conflict_stage27R_joint_nearnull_precursor.py",
    )
    p.add_argument(
        "--stage27-dir",
        default="vpinn_gradient_conflict_stage27R_joint_nearnull_precursor",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage28R_reaction_diffusion_robustness",
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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{key: row.get(key, None) for key in fields} for row in rows]
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
# Provenance
# =============================================================================

def preflight_provenance(
    stage3_script: Path,
    stage18_script: Path,
    stage19_script: Path,
    stage20_script: Path,
    stage22_script: Path,
    stage27_script: Path,
    stage27_dir: Path,
) -> dict:

    manifest_path = stage27_dir / "manifest.json"
    decision_path = stage27_dir / "decision.json"

    if not manifest_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError("Stage-27 manifest/decision missing.")

    manifest = read_json(manifest_path)
    decision = read_json(decision_path)

    shas = {
        "s3": sha256_file(stage3_script),
        "s18": sha256_file(stage18_script),
        "s19": sha256_file(stage19_script),
        "s20": sha256_file(stage20_script),
        "s22": sha256_file(stage22_script),
        "s27": sha256_file(stage27_script),
    }

    checks = (
        ("stage3_solver_sha256", "s3"),
        ("stage18_script_sha256", "s18"),
        ("stage19_script_sha256", "s19"),
        ("stage20_script_sha256", "s20"),
        ("stage22_script_sha256", "s22"),
        ("stage27r_script_sha256", "s27"),
    )

    for key, skey in checks:
        if manifest.get(key) != shas[skey]:
            raise RuntimeError(f"Stage-27 provenance mismatch: {key}")

    if not bool(decision.get("joint_nearnull_precursor_supported", False)):
        raise RuntimeError(
            "Stage 27 did not authorize independent robustness."
        )

    if decision.get("next_route") != (
        "stage28R_independent_problem_testspace_robustness"
    ):
        raise RuntimeError("Unexpected Stage-27 next route.")

    return {
        **shas,
        "stage27_decision": decision,
    }


# =============================================================================
# Reaction-diffusion experiment
# =============================================================================

class ReactionDiffusionExperiment:
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

        self.sigma = float(SIGMA)

        self.lambda_base = (
            (self.base_mode * math.pi) ** 2
            + self.sigma
        )

        self.lambda_target = (
            (self.mode * math.pi) ** 2
            + self.sigma
        )

        self.base_amplitude = (
            math.pi
            / math.sqrt(self.lambda_base)
        )

        self.amplitude = (
            1.05
            * math.pi
            / math.sqrt(self.lambda_target)
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

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * torch.sin(self.base_mode * math.pi * x)
            +
            self.amplitude
            * torch.sin(self.mode * math.pi * x)
        )

    def exact_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * self.base_mode
            * math.pi
            * torch.cos(self.base_mode * math.pi * x)
            +
            self.amplitude
            * self.mode
            * math.pi
            * torch.cos(self.mode * math.pi * x)
        )

    def forcing(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * self.lambda_base
            * torch.sin(self.base_mode * math.pi * x)
            +
            self.amplitude
            * self.lambda_target
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

        lam = (
            (math.pi * k) ** 2
            + self.sigma
        )

        normalization = (
            math.sqrt(2.0)
            / torch.sqrt(lam)
        )

        with torch.no_grad():
            phase = math.pi * (
                self.x_quad.detach() @ k
            )

            self.test_values = (
                normalization
                * torch.sin(phase)
            )

            self.test_derivatives = (
                normalization
                * (math.pi * k)
                * torch.cos(phase)
            )

            self.forcing_values = self.forcing(
                self.x_quad.detach()
            )

            # Energy Gram.
            gram = (
                self.test_derivatives.T
                @ (self.w_quad * self.test_derivatives)
                +
                self.sigma
                * self.test_values.T
                @ (self.w_quad * self.test_values)
            )

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

            # Manufactured exact weak residual.
            u_exact = self.exact_solution(
                self.x_quad.detach()
            )

            du_exact = self.exact_derivative(
                self.x_quad.detach()
            )

            exact_integrand = (
                du_exact * self.test_derivatives
                +
                self.sigma
                * u_exact * self.test_values
                -
                self.forcing_values * self.test_values
            )

            exact_residual = torch.sum(
                self.w_quad * exact_integrand,
                dim=0,
            )

            self.exact_weak_residual_error = float(
                torch.max(
                    torch.abs(exact_residual)
                ).item()
            )

        if self.gram_error > PREFLIGHT_TOL:
            raise RuntimeError(
                f"Energy Gram failed b={self.base_mode}: "
                f"{self.gram_error:.3e}"
            )

        if self.exact_weak_residual_error > PREFLIGHT_TOL:
            raise RuntimeError(
                f"Manufactured weak residual failed b={self.base_mode}: "
                f"{self.exact_weak_residual_error:.3e}"
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
            +
            self.sigma * u * self.test_values
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
        r = self.weak_residuals().detach()
        e = r.square()
        total = e.sum().clamp_min(1.0e-300)

        t = self.mode - 1
        dominant = int(torch.argmax(e).item())

        return {
            "vpinn_loss":
                float(torch.mean(e).item()),

            "residual_l2_norm":
                float(torch.linalg.vector_norm(r).item()),

            "dominant_residual_mode":
                dominant + 1,

            "dominant_residual_energy_share":
                float((e[dominant] / total).item()),

            "target_mode_residual_energy_share":
                float((e[t] / total).item()),

            "target_mode_abs_residual":
                float(torch.abs(r[t]).item()),
        }

    def train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        r = self.weak_residuals()
        loss = r.square().mean()

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
        modes=(TARGET_MODE,),
        reference_mode=7,
        reference_amplitude=0.15,
        track_interval=TRACK_INTERVAL,
        diagnostic_epochs=(0,),
        convergence_error_threshold=CONVERGENCE_REL_L2,
        localization_share_threshold=LOCALIZE_SHARE,
        resolved_share_threshold=CONVERGENCE_TARGET_SHARE,
        conflict_gamma_threshold=0.20,
        conflict_weighted_negative_threshold=0.50,
        active_relative_tol=1.0e-8,
        grad_absolute_eps=1.0e-300,
        output_dir=str(out_dir),
        dpi=220,
    )


# =============================================================================
# Geometry
# =============================================================================

def mobility(K: np.ndarray, r: np.ndarray) -> float:
    K = 0.5 * (K + K.T)

    rr = float(np.dot(r, r))
    tr = float(np.trace(K))
    num = float(np.dot(r, K @ r))

    if rr <= 0.0 or tr <= 0.0:
        raise RuntimeError("Invalid mobility denominator.")

    if num < -1.0e-12 * max(1.0, rr * abs(tr)):
        raise RuntimeError(
            f"Materially negative r^T K r: {num:.6e}"
        )

    return max(num, 0.0) / (rr * tr)


def geometry_audit(stage18, exp, seed: int, epoch: int, kind: str):
    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    kernel = stage18.residual_jacobian(exp)

    r = kernel["r"].cpu().numpy()
    K = kernel["K"].cpu().numpy()

    mu = mobility(K, r)

    return {
        "seed": seed,
        "base_mode": exp.base_mode,
        "target_mode": TARGET_MODE,
        "epoch": epoch,
        "audit_kind": kind,

        "relative_l2_error": rel,
        **rm,

        "mu_raw": mu,

        "_r": r,
        "_K": K,
    }


def log_shapley(
    K0: np.ndarray,
    r0: np.ndarray,
    Kt: np.ndarray,
    rt: np.ndarray,
):
    mu00 = mobility(K0, r0)
    mu0t = mobility(K0, rt)
    mut0 = mobility(Kt, r0)
    mutt = mobility(Kt, rt)

    floor = 1.0e-300

    f00 = math.log(max(mu00, floor))
    f0t = math.log(max(mu0t, floor))
    ft0 = math.log(max(mut0, floor))
    ftt = math.log(max(mutt, floor))

    phi_K = 0.5 * (
        (ft0 - f00)
        +
        (ftt - f0t)
    )

    phi_r = 0.5 * (
        (f0t - f00)
        +
        (ftt - ft0)
    )

    return {
        "mu_00":
            mu00,

        "mu_0t":
            mu0t,

        "mu_t0":
            mut0,

        "mu_tt":
            mutt,

        "log_shapley_kernel":
            phi_K,

        "log_shapley_residual":
            phi_r,

        "positive_drop_kernel":
            -phi_K,

        "positive_drop_residual":
            -phi_r,

        "residual_drop_dominates":
            bool((-phi_r) > (-phi_K)),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_cell_outcomes(cell_summary: List[dict], path: Path):
    labels = [
        f"b={int(r['base_mode'])}"
        for r in cell_summary
    ]

    deep = [
        int(r["persistent_deep_lock_count"])
        for r in cell_summary
    ]

    escaped = [
        int(r["certified_escape_count"])
        for r in cell_summary
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    ax.bar(
        x - width/2,
        deep,
        width,
        label="Persistent deep lock",
    )

    ax.bar(
        x + width/2,
        escaped,
        width,
        label="Certified escape",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Seeds out of 5")
    ax.set_title("Reaction-diffusion robustness of paired deep-lock phenotype")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_mobility(audit_rows: List[dict], path: Path):
    fig, ax = plt.subplots(figsize=(10.0, 5.7))

    for base_mode in BASE_MODES:
        for seed in SEEDS:
            rr = [
                r for r in audit_rows
                if (
                    int(r["base_mode"]) == base_mode
                    and int(r["seed"]) == seed
                )
            ]

            rr.sort(key=lambda x: int(x["epoch"]))

            if not rr:
                continue

            ax.plot(
                [int(r["epoch"]) for r in rr],
                [float(r["mu_raw"]) for r in rr],
                linewidth=1.0,
                alpha=0.7,
                label=(
                    f"b={base_mode}"
                    if seed == SEEDS[0]
                    else None
                ),
            )

    ax.axhline(
        MOBILITY_COLLAPSE_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label="deep-collapse threshold",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Basis-invariant residual mobility μ")
    ax.set_title("Reaction-diffusion mobility trajectories")
    ax.legend()

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
    stage22_script = resolve(args.stage22_script)
    stage27_script = resolve(args.stage27_script)
    stage27_dir = resolve(args.stage27_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight_provenance(
        stage3_script=stage3_script,
        stage18_script=stage18_script,
        stage19_script=stage19_script,
        stage20_script=stage20_script,
        stage22_script=stage22_script,
        stage27_script=stage27_script,
        stage27_dir=stage27_dir,
    )

    stage3 = load_module(stage3_script, "vpinn_stage3_stage28R")
    stage18 = load_module(stage18_script, "vpinn_stage18_stage28R")
    stage19 = load_module(stage19_script, "vpinn_stage19_stage28R")

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),

        "stage3_solver_sha256": pf["s3"],
        "stage18_script_sha256": pf["s18"],
        "stage19_script_sha256": pf["s19"],
        "stage20_script_sha256": pf["s20"],
        "stage22_script_sha256": pf["s22"],
        "stage27_script_sha256": pf["s27"],
        "stage28r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "reaction_diffusion_independent_pde_robustness",

            "sigma":
                SIGMA,

            "reaction_wavenumber":
                REACTION_WAVENUMBER,

            "seeds":
                list(SEEDS),

            "base_modes":
                list(BASE_MODES),

            "target_mode":
                TARGET_MODE,

            "base_weak_scale":
                "pi/sqrt(2)",

            "target_weak_scale":
                "1.05*pi/sqrt(2)",

            "max_epoch":
                MAX_EPOCH,

            "stop":
                "persistent deep lock OR certified escape OR horizon",

            "R1":
                "all analytic preflights + all paired initializations",

            "R2":
                "b1 localized baseline mobile >=4/5",

            "R3":
                "b1 deep lock >=4/5",

            "R4":
                "b2 deep lock <=1/5 AND b2 escape >=4/5",

            "R5":
                "at b1 deep-lock onset residual drop contribution > kernel in >=80%",

            "no_optimizer_intervention":
                True,
        },
    }

    write_json(out_dir / "manifest.json", manifest)

    print("=" * 178)
    print(
        "VPINN — STAGE 28R REACTION-DIFFUSION INDEPENDENT-PDE ROBUSTNESS"
    )
    print("=" * 178)
    print(f"device                    : {device}")
    print(f"sigma                     : {SIGMA:.12e}")
    print(f"reaction wavenumber       : {REACTION_WAVENUMBER}")
    print(f"new seeds                 : {list(SEEDS)}")
    print(f"base modes                : {list(BASE_MODES)}")
    print(f"target mode               : {TARGET_MODE}")
    print("optimizer intervention    : NONE")
    print("=" * 178)

    preflight_rows = []
    init_rows = []
    tracking_rows = []
    audit_rows = []
    run_rows = []

    for seed in SEEDS:

        initial_vectors = {}

        for base_mode in BASE_MODES:

            run_dir = (
                out_dir
                / f"seed_{seed:03d}"
                / f"base_{base_mode:02d}_target_09"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = make_config(
                stage3=stage3,
                seed=seed,
                device=device,
                out_dir=run_dir,
            )

            exp = ReactionDiffusionExperiment(
                stage3=stage3,
                cfg=cfg,
                device=device,
                base_mode=base_mode,
                target_mode=TARGET_MODE,
                out_dir=run_dir,
            )

            initial_vectors[base_mode] = (
                flatten_params(exp.model).cpu()
            )

            preflight_rows.append(
                {
                    "seed": seed,
                    "base_mode": base_mode,

                    "gram_error":
                        exp.gram_error,

                    "exact_weak_residual_error":
                        exp.exact_weak_residual_error,

                    "gram_pass":
                        bool(exp.gram_error <= PREFLIGHT_TOL),

                    "exact_weak_residual_pass":
                        bool(
                            exp.exact_weak_residual_error
                            <= PREFLIGHT_TOL
                        ),

                    "base_amplitude":
                        exp.base_amplitude,

                    "target_amplitude":
                        exp.amplitude,

                    "base_weak_coefficient":
                        exp.base_amplitude
                        * math.sqrt(exp.lambda_base / 2.0),

                    "target_weak_coefficient":
                        exp.amplitude
                        * math.sqrt(exp.lambda_target / 2.0),
                }
            )

            localized = False
            localization_epoch = -1
            baseline = None

            collapse_streak = 0
            collapse_candidate_epoch = None
            deep_lock_onset = -1
            deep_lock_confirmation = -1
            deep_lock_geometry = None

            escape_streak = 0
            escape_candidate_epoch = None
            escape_onset = -1
            escape_confirmation = -1

            last_epoch = 0

            print()
            print("-" * 178)
            print(
                f"seed={seed} b={base_mode}: "
                f"c_b={exp.base_amplitude:.8f}, "
                f"a9={exp.amplitude:.8f}, "
                f"Gram={exp.gram_error:.3e}, "
                f"exact weak={exp.exact_weak_residual_error:.3e}"
            )

            for epoch in range(MAX_EPOCH + 1):

                last_epoch = epoch

                if epoch % TRACK_INTERVAL == 0:

                    rm = exp.residual_metrics()
                    rel = exp.relative_l2_error()

                    probe = stage19.cheap_probe(
                        exp=exp,
                        mode=TARGET_MODE,
                    )

                    active = bool(
                        rm["target_mode_residual_energy_share"]
                        >= LOCALIZE_SHARE
                        and
                        rel > CONVERGENCE_REL_L2
                    )

                    if active and not localized:
                        localized = True
                        localization_epoch = epoch

                    tracking_rows.append(
                        {
                            "seed": seed,
                            "base_mode": base_mode,
                            "epoch": epoch,
                            "relative_l2_error": rel,
                            **rm,

                            "localized":
                                localized,

                            "mechanism_active":
                                active,

                            "adam_target_uphill_cosine":
                                probe["adam_target_uphill_cosine"],

                            "adam_candidate_target_uphill":
                                probe["adam_candidate_target_uphill"],
                        }
                    )

                    qualifies_escape = bool(
                        rel <= CONVERGENCE_REL_L2
                        and
                        rm["target_mode_residual_energy_share"]
                        <= CONVERGENCE_TARGET_SHARE
                    )

                    if escape_onset < 0:
                        if qualifies_escape:
                            if escape_streak == 0:
                                escape_candidate_epoch = epoch

                            escape_streak += 1
                        else:
                            escape_streak = 0
                            escape_candidate_epoch = None

                        if escape_streak >= ESCAPE_CERTIFY_POINTS:
                            escape_onset = int(escape_candidate_epoch)
                            escape_confirmation = epoch

                do_full = bool(
                    localized
                    and
                    (
                        epoch == localization_epoch
                        or epoch % FULL_AUDIT_INTERVAL == 0
                    )
                )

                if do_full:

                    audit = geometry_audit(
                        stage18=stage18,
                        exp=exp,
                        seed=seed,
                        epoch=epoch,
                        kind=(
                            "LOCALIZED_BASELINE"
                            if epoch == localization_epoch
                            else
                            "SURVEILLANCE_250"
                        ),
                    )

                    if baseline is None:
                        baseline = {
                            "K": audit["_K"],
                            "r": audit["_r"],
                            "mu": audit["mu_raw"],
                        }

                    decomposition = log_shapley(
                        K0=baseline["K"],
                        r0=baseline["r"],
                        Kt=audit["_K"],
                        rt=audit["_r"],
                    )

                    row = {
                        key: value
                        for key, value in audit.items()
                        if not key.startswith("_")
                    }

                    row.update(decomposition)

                    audit_rows.append(row)

                    deep_eligible = bool(
                        row["relative_l2_error"]
                        > CONVERGENCE_REL_L2
                        and
                        row[
                            "target_mode_residual_energy_share"
                        ]
                        >= LOCALIZE_SHARE
                    )

                    collapsed = bool(
                        deep_eligible
                        and
                        float(row["mu_raw"])
                        <= MOBILITY_COLLAPSE_THRESHOLD
                    )

                    if deep_lock_onset < 0:

                        if collapsed:
                            if collapse_streak == 0:
                                collapse_candidate_epoch = epoch

                            collapse_streak += 1
                        else:
                            collapse_streak = 0
                            collapse_candidate_epoch = None

                        if collapse_streak >= DEEP_LOCK_CERTIFY_POINTS:
                            deep_lock_onset = int(
                                collapse_candidate_epoch
                            )

                            deep_lock_confirmation = epoch

                            # Retrieve the onset row already saved.
                            onset_matches = [
                                r for r in audit_rows
                                if (
                                    int(r["seed"]) == seed
                                    and
                                    int(r["base_mode"]) == base_mode
                                    and
                                    int(r["epoch"]) == deep_lock_onset
                                )
                            ]

                            if len(onset_matches) != 1:
                                raise RuntimeError(
                                    "Could not identify unique deep-lock onset row."
                                )

                            deep_lock_geometry = copy.deepcopy(
                                onset_matches[0]
                            )

                if deep_lock_onset >= 0:
                    break

                if escape_onset >= 0 and epoch >= escape_confirmation:
                    break

                if epoch < MAX_EPOCH:
                    exp.train_step()

            run_rows.append(
                {
                    "seed": seed,
                    "base_mode": base_mode,

                    "localized":
                        localized,

                    "localization_epoch":
                        localization_epoch,

                    "localized_mu": (
                        float(baseline["mu"])
                        if baseline is not None
                        else None
                    ),

                    "persistent_deep_lock":
                        deep_lock_onset >= 0,

                    "deep_lock_onset_epoch":
                        deep_lock_onset,

                    "deep_lock_confirmation_epoch":
                        deep_lock_confirmation,

                    "deep_lock_positive_drop_kernel": (
                        float(
                            deep_lock_geometry[
                                "positive_drop_kernel"
                            ]
                        )
                        if deep_lock_geometry is not None
                        else None
                    ),

                    "deep_lock_positive_drop_residual": (
                        float(
                            deep_lock_geometry[
                                "positive_drop_residual"
                            ]
                        )
                        if deep_lock_geometry is not None
                        else None
                    ),

                    "deep_lock_residual_drop_dominates": (
                        bool(
                            deep_lock_geometry[
                                "residual_drop_dominates"
                            ]
                        )
                        if deep_lock_geometry is not None
                        else None
                    ),

                    "certified_escape":
                        escape_onset >= 0,

                    "escape_onset_epoch":
                        escape_onset,

                    "escape_confirmation_epoch":
                        escape_confirmation,

                    "stop_epoch":
                        last_epoch,

                    "stop_reason": (
                        "PERSISTENT_DEEP_LOCK"
                        if deep_lock_onset >= 0
                        else
                        "CERTIFIED_ESCAPE"
                        if escape_onset >= 0
                        else
                        "HORIZON"
                    ),
                }
            )

            print(
                f"  result={run_rows[-1]['stop_reason']} "
                f"loc={localization_epoch} "
                f"mu0={run_rows[-1]['localized_mu']} "
                f"deep={deep_lock_onset} "
                f"escape={escape_onset}"
            )

        # Exact paired initial-network check.
        gap = float(
            torch.max(
                torch.abs(
                    initial_vectors[1]
                    - initial_vectors[2]
                )
            ).item()
        )

        init_rows.append(
            {
                "seed": seed,
                "max_abs_initial_parameter_gap":
                    gap,

                "pass":
                    bool(gap <= INITIAL_CLONE_TOL),
            }
        )

        if gap > INITIAL_CLONE_TOL:
            raise RuntimeError(
                f"Paired initialization failed seed={seed}: {gap:.3e}"
            )

    # =========================================================================
    # Persist
    # =========================================================================
    write_csv(
        out_dir / "analytic_preflight.csv",
        preflight_rows,
    )

    write_csv(
        out_dir / "paired_initialization_checks.csv",
        init_rows,
    )

    write_csv(
        out_dir / "tracking_metrics.csv",
        tracking_rows,
    )

    write_csv(
        out_dir / "mobility_geometry_audits.csv",
        audit_rows,
    )

    write_csv(
        out_dir / "run_summary.csv",
        run_rows,
    )

    # =========================================================================
    # Gates
    # =========================================================================
    analytic_pass = all(
        bool(r["gram_pass"])
        and bool(r["exact_weak_residual_pass"])
        for r in preflight_rows
    )

    init_pass = all(
        bool(r["pass"])
        for r in init_rows
    )

    R1 = bool(
        len(preflight_rows) == 10
        and analytic_pass
        and len(init_rows) == 5
        and init_pass
    )

    b1 = [
        r for r in run_rows
        if int(r["base_mode"]) == 1
    ]

    b2 = [
        r for r in run_rows
        if int(r["base_mode"]) == 2
    ]

    b1_mobile_count = sum(
        int(
            r["localized_mu"] is not None
            and
            float(r["localized_mu"])
            > MOBILITY_COLLAPSE_THRESHOLD
        )
        for r in b1
    )

    R2 = bool(
        b1_mobile_count >= 4
    )

    b1_deep = [
        r for r in b1
        if bool(r["persistent_deep_lock"])
    ]

    b2_deep = [
        r for r in b2
        if bool(r["persistent_deep_lock"])
    ]

    b2_escape = [
        r for r in b2
        if bool(r["certified_escape"])
    ]

    R3 = bool(
        len(b1_deep) >= 4
    )

    R4 = bool(
        len(b2_deep) <= 1
        and
        len(b2_escape) >= 4
    )

    residual_dom_count = sum(
        int(
            bool(r["deep_lock_residual_drop_dominates"])
        )
        for r in b1_deep
    )

    R5 = bool(
        len(b1_deep) > 0
        and
        residual_dom_count
        >= math.ceil(0.80 * len(b1_deep))
    )

    independent_pde = bool(
        R1 and R2 and R3 and R4
    )

    two_stage = bool(
        independent_pde and R5
    )

    if two_stage:

        route_class = (
            "reaction_diffusion_independent_pde_deeplock_and_early_residual_drop_transfer"
        )

        next_route = (
            "stage29R_independent_testspace_robustness"
        )

    elif independent_pde:

        route_class = (
            "reaction_diffusion_deeplock_transfer_without_clean_two_stage_factor_transfer"
        )

        next_route = (
            "stage29R_independent_testspace_robustness_with_joint_geometry_only"
        )

    elif R3 and not R4:

        route_class = (
            "reaction_diffusion_lock_transfers_but_base_mode_specificity_lost"
        )

        next_route = (
            "stage29R_reaction_strength_localization"
        )

    else:

        route_class = (
            "deep_lock_not_robust_to_reaction_diffusion_operator"
        )

        next_route = (
            "stage29R_stop_broadening_operator_claim"
        )

    decision = {
        "sigma":
            SIGMA,

        "reaction_wavenumber":
            REACTION_WAVENUMBER,

        "R1_analytic_and_paired_preflight":
            R1,

        "b1_mobile_localized_count":
            b1_mobile_count,

        "R2_mobile_b1_localized_baseline":
            R2,

        "b1_persistent_deep_lock_count":
            len(b1_deep),

        "R3_b1_deep_lock_transfer":
            R3,

        "b2_persistent_deep_lock_count":
            len(b2_deep),

        "b2_certified_escape_count":
            len(b2_escape),

        "R4_paired_b2_nonlock_control":
            R4,

        "b1_deep_lock_residual_drop_dominance_count":
            residual_dom_count,

        "R5_early_collapse_residual_drop_dominance":
            R5,

        "independent_pde_deep_lock_robustness":
            independent_pde,

        "two_stage_geometry_robustness":
            two_stage,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A PASS transfers the paired deep-lock phenotype from Poisson to "
            "one reaction-diffusion operator with energy-orthonormal Fourier "
            "tests. It does not yet establish robustness to a different test "
            "space, architecture, dimension, or nonlinear PDE."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    # Cell summary.
    cell_summary = []

    for base_mode in BASE_MODES:
        rr = [
            r for r in run_rows
            if int(r["base_mode"]) == base_mode
        ]

        cell_summary.append(
            {
                "base_mode":
                    base_mode,

                "persistent_deep_lock_count":
                    sum(
                        int(bool(r["persistent_deep_lock"]))
                        for r in rr
                    ),

                "certified_escape_count":
                    sum(
                        int(bool(r["certified_escape"]))
                        for r in rr
                    ),

                "horizon_count":
                    sum(
                        int(r["stop_reason"] == "HORIZON")
                        for r in rr
                    ),

                "median_localized_mu": (
                    float(
                        np.median([
                            float(r["localized_mu"])
                            for r in rr
                            if r["localized_mu"] is not None
                        ])
                    )
                    if any(
                        r["localized_mu"] is not None
                        for r in rr
                    )
                    else None
                ),

                "median_deep_lock_onset": (
                    float(
                        np.median([
                            int(r["deep_lock_onset_epoch"])
                            for r in rr
                            if bool(r["persistent_deep_lock"])
                        ])
                    )
                    if any(
                        bool(r["persistent_deep_lock"])
                        for r in rr
                    )
                    else None
                ),
            }
        )

    write_csv(
        out_dir / "cell_summary.csv",
        cell_summary,
    )

    plot_cell_outcomes(
        cell_summary,
        out_dir / "reaction_diffusion_cell_outcomes.png",
    )

    plot_mobility(
        audit_rows,
        out_dir / "reaction_diffusion_mobility.png",
    )

    # =========================================================================
    # Console
    # =========================================================================
    lines = []

    lines.append("=" * 180)
    lines.append(
        "VPINN — STAGE 28R REACTION-DIFFUSION ROBUSTNESS SUMMARY"
    )
    lines.append("=" * 180)

    lines.append(
        "base | deep lock | escape | horizon | median localized mu | median deep onset"
    )

    lines.append("-" * 180)

    for r in cell_summary:
        lines.append(
            f"{int(r['base_mode']):4d} | "
            f"{int(r['persistent_deep_lock_count']):4d}/5   | "
            f"{int(r['certified_escape_count']):4d}/5 | "
            f"{int(r['horizon_count']):4d}/5  | "
            f"{str(r['median_localized_mu']):19s} | "
            f"{str(r['median_deep_lock_onset'])}"
        )

    lines.append("-" * 180)

    lines.append(
        f"R1 analytic + paired preflight         : {R1}"
    )

    lines.append(
        f"R2 mobile b1 localized baseline        : "
        f"{b1_mobile_count}/5 -> {R2}"
    )

    lines.append(
        f"R3 b1 deep-lock transfer               : "
        f"{len(b1_deep)}/5 -> {R3}"
    )

    lines.append(
        f"R4 b2 non-lock control                 : "
        f"deep={len(b2_deep)}/5, escape={len(b2_escape)}/5 -> {R4}"
    )

    lines.append(
        f"R5 residual-drop dominance at collapse : "
        f"{residual_dom_count}/{len(b1_deep)} -> {R5}"
    )

    lines.append(
        f"INDEPENDENT-PDE DEEP-LOCK ROBUSTNESS   : "
        f"{independent_pde}"
    )

    lines.append(
        f"TWO-STAGE GEOMETRY ROBUSTNESS          : "
        f"{two_stage}"
    )

    lines.append(
        f"route class                             : "
        f"{route_class}"
    )

    lines.append(
        f"next route                              : "
        f"{next_route}"
    )

    lines.append("=" * 180)

    lines.append(
        "Guardrail: this is an independent operator test, not yet independent test-space validation."
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
