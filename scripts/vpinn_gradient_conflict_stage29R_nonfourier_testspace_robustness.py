#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 29R
Independent Non-Fourier Test-Space Robustness
==============================================

Scientific status
-----------------
Stage 28R transferred the paired deep-lock phenotype from Poisson to the
reaction-diffusion operator

    -u'' + sigma u = f,
    sigma = (5*pi)^2,

using energy-orthonormal Fourier sine tests.

Stage 28 passed all precommitted gates:

    b=1 deep lock = 5/5
    b=2 deep lock = 1/5
    b=2 escape    = 4/5

and at every b=1 deep-lock onset the early mobility drop was
residual-side dominated.

The next question is whether this survives a genuinely different TEST SPACE,
not merely an orthogonal rotation of the Fourier basis.

Stage 29R replaces the Fourier test span by the 24-dimensional continuous
piecewise-linear H_0^1 finite-element space on a uniform 25-element mesh.

Raw test functions:
    standard interior P1 hat functions.

They are energy-orthonormalized using the same reaction-diffusion bilinear
form

    a(v,w) = int v' w' + sigma v w.

This is a different finite-dimensional subspace, not a basis rotation of the
Fourier sine space.

Piecewise quadrature
--------------------
Use 10-point Gauss-Legendre quadrature on each of 25 elements:

    250 total quadrature points,

close to the prior 256-point budget while exactly respecting P1 breakpoints.

Basis-invariant target localization
-----------------------------------
A non-Fourier test space has no meaningful "coordinate 9".

For unit sine mode e_k(x)=sin(k*pi*x), define its weak-response vector in the
orthonormalized P1 test space:

    q_k[j] = a(e_k, v_j).

Define the unit target template

    t_9 = q_9 / ||q_9||.

For any residual vector r, define target residual share by

    share_9 = (r^T t_9)^2 / ||r||^2.

This is invariant under every orthogonal change of the orthonormal P1 basis.

Weak-scale matching
-------------------
Let

    S_base   = pi/sqrt(2),
    S_target = 1.05*pi/sqrt(2).

Choose manufactured amplitudes

    c_b = S_base / ||q_b||
    a_9 = S_target / ||q_9||.

Therefore the projected weak-response norms are exactly matched across
b=1 and b=2 and to Stage 28:

    ||c_b q_b|| = pi/sqrt(2)
    ||a_9 q_9|| = 1.05*pi/sqrt(2).

Analytic representation checks
------------------------------
Before training require:

    energy Gram error <= 1e-10
    manufactured exact weak residual <= 1e-10
    target-template/base-template |cosine| <= 1e-10
    base projection capture ratio >= 0.99
    target m=9 projection capture ratio >= 0.90
    weak-scale matching errors <= 1e-12.

The target capture ratio is

    ||q_9|| / sqrt(lambda_9/2),

where sqrt(lambda_9/2) is the full energy norm of sin(9*pi*x).

New paired seeds
----------------
    {25,26,27,28,29}.

Cells:
    b=1,m=9
    b=2,m=9.

Initialization must match exactly within each seed.

Ordinary Adam only.

Tracking / stopping
-------------------
Track residual metrics every 25 epochs.

After target localization

    target-template share >= 0.80
    AND relL2 > 1e-2,

compute full residual Jacobian geometry at:
    * localization onset,
    * every global 250 epochs.

Persistent deep lock:
    TWO consecutive eligible full audits with

        mu = r^T K r / (||r||^2 tr K) <= 1e-6.

Certified escape:
    relL2 <= 1e-2
    AND target-template share <= 0.20

for THREE consecutive 25-epoch observations.

Stop at earliest:
    persistent deep-lock certification,
    certified escape,
    epoch 3000.

Two-stage geometry
------------------
At every full audit relative to the localized baseline (K0,r0), compute the
same exact log-Shapley mobility decomposition used in Stage 28.

At b=1 persistent deep-lock onset, test whether

    D_r > D_K,

where D_r and D_K are positive contributions to the log mobility DROP.

Primary gates
-------------

S1 — NON-FOURIER ANALYTIC / PAIRED PREFLIGHT
    all 10 cells pass all analytic checks,
    and 5/5 paired initializations are exact.

S2 — MOBILE B=1 LOCALIZED BASELINE
    mu_localized > 1e-6 in >=4/5 b=1 runs.

S3 — B=1 DEEP-LOCK TEST-SPACE TRANSFER
    persistent deep lock in >=4/5 b=1 runs.

S4 — PAIRED B=2 NON-LOCK CONTROL
    persistent deep lock in <=1/5 b=2 runs
    AND certified escape in >=4/5 b=2 runs.

S5 — EARLY RESIDUAL-DROP DOMINANCE TRANSFER
    among b=1 deep-lock runs,
    D_r > D_K at deep-lock onset in >=80%.

INDEPENDENT TEST-SPACE DEEP-LOCK ROBUSTNESS:
    S1 & S2 & S3 & S4.

CROSS-OPERATOR + CROSS-TESTSPACE TWO-STAGE ROBUSTNESS:
    independent-testspace robustness & S5.

Decision
--------
A) S1-S5 PASS:
       Stage 30R = literature/claim audit + one minimal architecture control
                   chosen by read-only sensitivity bounds.

B) S3 passes but S4 fails:
       deep lock transfers, but base-mode specificity is test-space sensitive.
       Do not claim base-mode robustness.

C) S3 fails:
       deep lock is Fourier-test-space dependent in the current evidence.
       Stop broadening the mechanism claim.

Guardrail
---------
A PASS supports the phenotype across two PDE operators and two materially
different weak-test spaces, within one 1D manufactured family and one neural
architecture. It is not yet a universal VPINN theorem or architecture-robust
claim.
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


SEEDS = (25, 26, 27, 28, 29)
BASE_MODES = (1, 2)
TARGET_MODE = 9

REACTION_WAVENUMBER = 5.0
SIGMA = (REACTION_WAVENUMBER * math.pi) ** 2

N_INTERIOR_TESTS = 24
N_ELEMENTS = N_INTERIOR_TESTS + 1
GL_PER_ELEMENT = 10
N_QUAD = N_ELEMENTS * GL_PER_ELEMENT

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
SCALE_TOL = 1.0e-12
INITIAL_CLONE_TOL = 1.0e-15

BASE_CAPTURE_MIN = 0.99
TARGET_CAPTURE_MIN = 0.90

BASE_WEAK_SCALE = math.pi / math.sqrt(2.0)
TARGET_WEAK_SCALE = 1.05 * math.pi / math.sqrt(2.0)


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-29R independent non-Fourier P1 test-space robustness."
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
        "--stage20-script",
        default="vpinn_gradient_conflict_stage20R_heldout_mobility_unlock.py",
    )
    p.add_argument(
        "--stage28-script",
        default="vpinn_gradient_conflict_stage28R_reaction_diffusion_robustness.py",
    )
    p.add_argument(
        "--stage28-dir",
        default="vpinn_gradient_conflict_stage28R_reaction_diffusion_robustness",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness",
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

def provenance_preflight(
    stage3_script: Path,
    stage18_script: Path,
    stage20_script: Path,
    stage28_script: Path,
    stage28_dir: Path,
) -> dict:

    manifest_path = stage28_dir / "manifest.json"
    decision_path = stage28_dir / "decision.json"

    if not manifest_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError("Stage-28 manifest/decision missing.")

    manifest = read_json(manifest_path)
    decision = read_json(decision_path)

    shas = {
        "s3": sha256_file(stage3_script),
        "s18": sha256_file(stage18_script),
        "s20": sha256_file(stage20_script),
        "s28": sha256_file(stage28_script),
    }

    checks = (
        ("stage3_solver_sha256", "s3"),
        ("stage18_script_sha256", "s18"),
        ("stage20_script_sha256", "s20"),
        ("stage28r_script_sha256", "s28"),
    )

    for key, skey in checks:
        if manifest.get(key) != shas[skey]:
            raise RuntimeError(f"Stage-28 provenance mismatch: {key}")

    if not bool(
        decision.get("independent_pde_deep_lock_robustness", False)
    ):
        raise RuntimeError(
            "Stage 28 did not establish independent-PDE robustness."
        )

    if not bool(
        decision.get("two_stage_geometry_robustness", False)
    ):
        raise RuntimeError(
            "Stage 28 did not establish the two-stage geometry transfer."
        )

    if decision.get("next_route") != (
        "stage29R_independent_testspace_robustness"
    ):
        raise RuntimeError("Unexpected Stage-28 next route.")

    return {
        **shas,
        "stage28_decision": decision,
    }


# =============================================================================
# Non-Fourier P1 test-space experiment
# =============================================================================

class P1ReactionDiffusionExperiment:
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

        self._build_piecewise_quadrature_and_test_space()
        self._build_mode_templates_and_manufactured_problem()

    def _build_piecewise_quadrature_and_test_space(self):
        xi, wi = leggauss(GL_PER_ELEMENT)

        h = 1.0 / N_ELEMENTS

        x_all = []
        w_all = []
        raw_v = []
        raw_d = []

        for e in range(N_ELEMENTS):
            a = e * h
            b = (e + 1) * h

            x_e = (
                (xi + 1.0) * (b - a) / 2.0
                + a
            )

            w_e = wi * (b - a) / 2.0

            for q in range(GL_PER_ELEMENT):
                xq = float(x_e[q])

                vals = np.zeros(
                    N_INTERIOR_TESTS,
                    dtype=np.float64,
                )

                ders = np.zeros_like(vals)

                # Local left global node = e.
                # Local right global node = e+1.
                N_left = (b - xq) / h
                N_right = (xq - a) / h

                left_node = e
                right_node = e + 1

                if 1 <= left_node <= N_INTERIOR_TESTS:
                    vals[left_node - 1] = N_left
                    ders[left_node - 1] = -1.0 / h

                if 1 <= right_node <= N_INTERIOR_TESTS:
                    vals[right_node - 1] = N_right
                    ders[right_node - 1] = 1.0 / h

                x_all.append(xq)
                w_all.append(float(w_e[q]))
                raw_v.append(vals)
                raw_d.append(ders)

        x_np = np.asarray(
            x_all,
            dtype=np.float64,
        ).reshape(-1, 1)

        w_np = np.asarray(
            w_all,
            dtype=np.float64,
        ).reshape(-1, 1)

        V_np = np.asarray(
            raw_v,
            dtype=np.float64,
        )

        D_np = np.asarray(
            raw_d,
            dtype=np.float64,
        )

        x = torch.as_tensor(
            x_np,
            dtype=self.dtype,
            device=self.device,
        )

        w = torch.as_tensor(
            w_np,
            dtype=self.dtype,
            device=self.device,
        )

        V = torch.as_tensor(
            V_np,
            dtype=self.dtype,
            device=self.device,
        )

        D = torch.as_tensor(
            D_np,
            dtype=self.dtype,
            device=self.device,
        )

        G = (
            D.T @ (w * D)
            +
            self.sigma
            * V.T @ (w * V)
        )

        eigvals, eigvecs = torch.linalg.eigh(G)

        if float(torch.min(eigvals).item()) <= 0.0:
            raise RuntimeError(
                "Raw P1 energy Gram is not positive definite."
            )

        inv_sqrt = (
            eigvecs
            @ torch.diag(torch.rsqrt(eigvals))
            @ eigvecs.T
        )

        V_ortho = V @ inv_sqrt
        D_ortho = D @ inv_sqrt

        gram = (
            D_ortho.T @ (w * D_ortho)
            +
            self.sigma
            * V_ortho.T @ (w * V_ortho)
        )

        identity = torch.eye(
            N_INTERIOR_TESTS,
            dtype=self.dtype,
            device=self.device,
        )

        self.gram_error = float(
            torch.max(
                torch.abs(gram - identity)
            ).item()
        )

        self.x_quad = (
            x.detach().clone().requires_grad_(True)
        )

        self.w_quad = w
        self.test_values = V_ortho
        self.test_derivatives = D_ortho

        if self.gram_error > PREFLIGHT_TOL:
            raise RuntimeError(
                f"P1 energy Gram failed: {self.gram_error:.3e}"
            )

    def unit_mode_response(self, k: int) -> torch.Tensor:
        x = self.x_quad.detach()

        s = torch.sin(
            k * math.pi * x
        )

        ds = (
            k
            * math.pi
            * torch.cos(k * math.pi * x)
        )

        q = torch.sum(
            self.w_quad
            * (
                ds * self.test_derivatives
                +
                self.sigma
                * s * self.test_values
            ),
            dim=0,
        )

        return q

    def _build_mode_templates_and_manufactured_problem(self):
        self.lambda_base = (
            (self.base_mode * math.pi) ** 2
            + self.sigma
        )

        self.lambda_target = (
            (self.mode * math.pi) ** 2
            + self.sigma
        )

        q_base = self.unit_mode_response(
            self.base_mode
        )

        q_target = self.unit_mode_response(
            self.mode
        )

        q_base_norm = float(
            torch.linalg.vector_norm(q_base).item()
        )

        q_target_norm = float(
            torch.linalg.vector_norm(q_target).item()
        )

        if q_base_norm <= 0.0 or q_target_norm <= 0.0:
            raise RuntimeError("Degenerate weak-response template.")

        self.base_template = (
            q_base / q_base_norm
        ).detach()

        self.target_template = (
            q_target / q_target_norm
        ).detach()

        self.base_capture_ratio = (
            q_base_norm
            /
            math.sqrt(self.lambda_base / 2.0)
        )

        self.target_capture_ratio = (
            q_target_norm
            /
            math.sqrt(self.lambda_target / 2.0)
        )

        self.template_abs_cosine = abs(
            float(
                torch.dot(
                    self.base_template,
                    self.target_template,
                ).item()
            )
        )

        self.base_amplitude = (
            BASE_WEAK_SCALE
            / q_base_norm
        )

        self.amplitude = (
            TARGET_WEAK_SCALE
            / q_target_norm
        )

        self.base_scale_error = abs(
            self.base_amplitude
            * q_base_norm
            - BASE_WEAK_SCALE
        )

        self.target_scale_error = abs(
            self.amplitude
            * q_target_norm
            - TARGET_WEAK_SCALE
        )

        with torch.no_grad():
            self.forcing_values = self.forcing(
                self.x_quad.detach()
            )

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

            exact_r = torch.sum(
                self.w_quad * exact_integrand,
                dim=0,
            )

            self.exact_weak_residual_error = float(
                torch.max(
                    torch.abs(exact_r)
                ).item()
            )

        if self.base_capture_ratio < BASE_CAPTURE_MIN:
            raise RuntimeError(
                f"Base projection capture too small: "
                f"{self.base_capture_ratio:.6f}"
            )

        if self.target_capture_ratio < TARGET_CAPTURE_MIN:
            raise RuntimeError(
                f"Target projection capture too small: "
                f"{self.target_capture_ratio:.6f}"
            )

        if self.template_abs_cosine > PREFLIGHT_TOL:
            raise RuntimeError(
                f"Base/target weak templates not orthogonal enough: "
                f"{self.template_abs_cosine:.3e}"
            )

        if self.base_scale_error > SCALE_TOL:
            raise RuntimeError(
                f"Base weak-scale matching failed: "
                f"{self.base_scale_error:.3e}"
            )

        if self.target_scale_error > SCALE_TOL:
            raise RuntimeError(
                f"Target weak-scale matching failed: "
                f"{self.target_scale_error:.3e}"
            )

        if self.exact_weak_residual_error > PREFLIGHT_TOL:
            raise RuntimeError(
                f"Manufactured weak residual failed: "
                f"{self.exact_weak_residual_error:.3e}"
            )

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

    def weak_residuals(self) -> torch.Tensor:
        u = self.model(
            self.x_quad
        )

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
            self.sigma
            * u * self.test_values
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

        target_proj = torch.dot(
            r,
            self.target_template,
        )

        base_proj = torch.dot(
            r,
            self.base_template,
        )

        target_share = (
            target_proj.square()
            / total
        )

        base_share = (
            base_proj.square()
            / total
        )

        max_coordinate = int(
            torch.argmax(e).item()
        )

        return {
            "vpinn_loss":
                float(torch.mean(e).item()),

            "residual_l2_norm":
                float(torch.linalg.vector_norm(r).item()),

            "target_template_residual":
                float(target_proj.item()),

            "target_template_abs_residual":
                float(torch.abs(target_proj).item()),

            "target_mode_residual_energy_share":
                float(target_share.item()),

            "base_template_residual_energy_share":
                float(base_share.item()),

            "max_coordinate_index":
                max_coordinate + 1,

            "max_coordinate_energy_share":
                float(
                    (e[max_coordinate] / total).item()
                ),
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
        n_test=N_INTERIOR_TESTS,
        n_quad=N_QUAD,
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

    return {
        "seed": seed,
        "base_mode": exp.base_mode,
        "target_mode": TARGET_MODE,
        "epoch": epoch,
        "audit_kind": kind,

        "relative_l2_error": rel,
        **rm,

        "mu_raw":
            mobility(K, r),

        "_r":
            r,

        "_K":
            K,
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

def plot_outcomes(cell_summary: List[dict], path: Path):
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
    ax.set_title("Non-Fourier P1 test-space robustness")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_mobility(rows: List[dict], path: Path):
    fig, ax = plt.subplots(figsize=(10.0, 5.7))

    for base_mode in BASE_MODES:
        for seed in SEEDS:
            rr = [
                r for r in rows
                if (
                    int(r["base_mode"]) == base_mode
                    and
                    int(r["seed"]) == seed
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
    ax.set_title("P1 test-space residual mobility")
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
    stage20_script = resolve(args.stage20_script)
    stage28_script = resolve(args.stage28_script)
    stage28_dir = resolve(args.stage28_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = provenance_preflight(
        stage3_script=stage3_script,
        stage18_script=stage18_script,
        stage20_script=stage20_script,
        stage28_script=stage28_script,
        stage28_dir=stage28_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage29R",
    )

    stage18 = load_module(
        stage18_script,
        "vpinn_stage18_stage29R",
    )

    manifest = {
        "python":
            sys.version,

        "platform":
            platform.platform(),

        "torch_version":
            torch.__version__,

        "numpy_version":
            np.__version__,

        "device_resolved":
            str(device),

        "stage3_solver_sha256":
            pf["s3"],

        "stage18_script_sha256":
            pf["s18"],

        "stage20_script_sha256":
            pf["s20"],

        "stage28_script_sha256":
            pf["s28"],

        "stage29r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "independent_nonfourier_P1_testspace_robustness",

            "operator":
                "-u'' + sigma u, sigma=(5*pi)^2",

            "raw_test_space":
                "continuous P1 H0^1 hats on 25 uniform elements",

            "orthonormalization":
                "discrete energy Gram inverse square root",

            "n_test":
                N_INTERIOR_TESTS,

            "quadrature":
                f"{GL_PER_ELEMENT}-point GL per {N_ELEMENTS} elements",

            "n_quad":
                N_QUAD,

            "seeds":
                list(SEEDS),

            "base_modes":
                list(BASE_MODES),

            "target_mode":
                TARGET_MODE,

            "target_localization":
                "basis-invariant projection onto normalized weak-response template q9",

            "S1":
                "all analytic representation checks + paired initialization",

            "S2":
                "b1 localized mu>1e-6 >=4/5",

            "S3":
                "b1 deep lock >=4/5",

            "S4":
                "b2 deep lock <=1/5 and escape >=4/5",

            "S5":
                "residual drop dominates at b1 deep-lock onset >=80%",

            "optimizer_intervention":
                False,
        },
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 182)
    print(
        "VPINN — STAGE 29R INDEPENDENT NON-FOURIER P1 TEST-SPACE ROBUSTNESS"
    )
    print("=" * 182)
    print(f"device                    : {device}")
    print(f"test space                : 24 interior P1 hats / 25 elements")
    print(f"quadrature                : {GL_PER_ELEMENT} GL per element = {N_QUAD}")
    print(f"sigma                     : {SIGMA:.12e}")
    print(f"new seeds                 : {list(SEEDS)}")
    print(f"base modes                : {list(BASE_MODES)}")
    print(f"target mode               : {TARGET_MODE}")
    print("optimizer intervention    : NONE")
    print("=" * 182)

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

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            cfg = make_config(
                stage3=stage3,
                seed=seed,
                device=device,
                out_dir=run_dir,
            )

            exp = P1ReactionDiffusionExperiment(
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

            analytic_pass = bool(
                exp.gram_error <= PREFLIGHT_TOL
                and
                exp.exact_weak_residual_error <= PREFLIGHT_TOL
                and
                exp.template_abs_cosine <= PREFLIGHT_TOL
                and
                exp.base_capture_ratio >= BASE_CAPTURE_MIN
                and
                exp.target_capture_ratio >= TARGET_CAPTURE_MIN
                and
                exp.base_scale_error <= SCALE_TOL
                and
                exp.target_scale_error <= SCALE_TOL
            )

            preflight_rows.append(
                {
                    "seed":
                        seed,

                    "base_mode":
                        base_mode,

                    "gram_error":
                        exp.gram_error,

                    "exact_weak_residual_error":
                        exp.exact_weak_residual_error,

                    "base_capture_ratio":
                        exp.base_capture_ratio,

                    "target_capture_ratio":
                        exp.target_capture_ratio,

                    "template_abs_cosine":
                        exp.template_abs_cosine,

                    "base_weak_scale_error":
                        exp.base_scale_error,

                    "target_weak_scale_error":
                        exp.target_scale_error,

                    "base_amplitude":
                        exp.base_amplitude,

                    "target_amplitude":
                        exp.amplitude,

                    "analytic_pass":
                        analytic_pass,
                }
            )

            if not analytic_pass:
                raise RuntimeError(
                    f"Analytic P1 preflight failed seed={seed}, b={base_mode}."
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
            print("-" * 182)
            print(
                f"seed={seed} b={base_mode}: "
                f"capture(base)={exp.base_capture_ratio:.6f}, "
                f"capture(target)={exp.target_capture_ratio:.6f}, "
                f"template cos={exp.template_abs_cosine:.3e}, "
                f"Gram={exp.gram_error:.3e}, "
                f"exact weak={exp.exact_weak_residual_error:.3e}"
            )

            for epoch in range(MAX_EPOCH + 1):

                last_epoch = epoch

                if epoch % TRACK_INTERVAL == 0:

                    rm = exp.residual_metrics()
                    rel = exp.relative_l2_error()

                    active = bool(
                        rm[
                            "target_mode_residual_energy_share"
                        ]
                        >= LOCALIZE_SHARE
                        and
                        rel > CONVERGENCE_REL_L2
                    )

                    if active and not localized:
                        localized = True
                        localization_epoch = epoch

                    tracking_rows.append(
                        {
                            "seed":
                                seed,

                            "base_mode":
                                base_mode,

                            "epoch":
                                epoch,

                            "relative_l2_error":
                                rel,

                            **rm,

                            "localized":
                                localized,

                            "mechanism_active":
                                active,
                        }
                    )

                    qualifies_escape = bool(
                        rel <= CONVERGENCE_REL_L2
                        and
                        rm[
                            "target_mode_residual_energy_share"
                        ]
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
                            escape_onset = int(
                                escape_candidate_epoch
                            )
                            escape_confirmation = epoch

                do_full = bool(
                    localized
                    and
                    (
                        epoch == localization_epoch
                        or
                        epoch % FULL_AUDIT_INTERVAL == 0
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
                            "K":
                                audit["_K"],

                            "r":
                                audit["_r"],

                            "mu":
                                audit["mu_raw"],
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

                    row.update(
                        decomposition
                    )

                    audit_rows.append(
                        row
                    )

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

                            matches = [
                                r for r in audit_rows
                                if (
                                    int(r["seed"]) == seed
                                    and
                                    int(r["base_mode"]) == base_mode
                                    and
                                    int(r["epoch"]) == deep_lock_onset
                                )
                            ]

                            if len(matches) != 1:
                                raise RuntimeError(
                                    "Could not identify unique P1 deep-lock onset row."
                                )

                            deep_lock_geometry = copy.deepcopy(
                                matches[0]
                            )

                if deep_lock_onset >= 0:
                    break

                if escape_onset >= 0 and epoch >= escape_confirmation:
                    break

                if epoch < MAX_EPOCH:
                    exp.train_step()

            run_rows.append(
                {
                    "seed":
                        seed,

                    "base_mode":
                        base_mode,

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
                "seed":
                    seed,

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
    S1 = bool(
        len(preflight_rows) == 10
        and
        all(
            bool(r["analytic_pass"])
            for r in preflight_rows
        )
        and
        len(init_rows) == 5
        and
        all(
            bool(r["pass"])
            for r in init_rows
        )
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

    S2 = bool(
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

    S3 = bool(
        len(b1_deep) >= 4
    )

    S4 = bool(
        len(b2_deep) <= 1
        and
        len(b2_escape) >= 4
    )

    residual_dom_count = sum(
        int(
            bool(
                r[
                    "deep_lock_residual_drop_dominates"
                ]
            )
        )
        for r in b1_deep
    )

    S5 = bool(
        len(b1_deep) > 0
        and
        residual_dom_count
        >= math.ceil(
            0.80 * len(b1_deep)
        )
    )

    independent_testspace = bool(
        S1 and S2 and S3 and S4
    )

    cross_robustness = bool(
        independent_testspace and S5
    )

    if cross_robustness:

        route_class = (
            "reaction_diffusion_deeplock_robust_across_fourier_and_nonfourier_P1_testspaces"
        )

        next_route = (
            "stage30R_claim_literature_audit_and_minimal_architecture_control"
        )

    elif independent_testspace:

        route_class = (
            "deeplock_testspace_transfer_without_clean_early_factor_transfer"
        )

        next_route = (
            "stage30R_claim_literature_audit_joint_geometry_only"
        )

    elif S3 and not S4:

        route_class = (
            "deeplock_transfers_to_P1_but_base_mode_specificity_is_testspace_sensitive"
        )

        next_route = (
            "stage30R_testspace_specific_control_localization"
        )

    else:

        route_class = (
            "deeplock_not_robust_to_nonfourier_P1_testspace"
        )

        next_route = (
            "stage30R_stop_broadening_testspace_claim"
        )

    decision = {
        "test_space":
            "24D continuous P1 H0^1 on 25 uniform elements, energy-orthonormalized",

        "quadrature_points":
            N_QUAD,

        "S1_nonfourier_analytic_and_paired_preflight":
            S1,

        "b1_mobile_localized_count":
            b1_mobile_count,

        "S2_mobile_b1_localized_baseline":
            S2,

        "b1_persistent_deep_lock_count":
            len(b1_deep),

        "S3_b1_deep_lock_testspace_transfer":
            S3,

        "b2_persistent_deep_lock_count":
            len(b2_deep),

        "b2_certified_escape_count":
            len(b2_escape),

        "S4_paired_b2_nonlock_control":
            S4,

        "b1_deep_lock_residual_drop_dominance_count":
            residual_dom_count,

        "S5_early_residual_drop_dominance_transfer":
            S5,

        "independent_testspace_deep_lock_robustness":
            independent_testspace,

        "cross_operator_cross_testspace_two_stage_robustness":
            cross_robustness,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A PASS would establish the paired deep-lock phenotype across "
            "Poisson/Fourier, reaction-diffusion/Fourier, and "
            "reaction-diffusion/non-Fourier P1 test spaces. It remains one "
            "network architecture, one dimension, and a manufactured family."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

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

    plot_outcomes(
        cell_summary,
        out_dir / "nonfourier_P1_cell_outcomes.png",
    )

    plot_mobility(
        audit_rows,
        out_dir / "nonfourier_P1_mobility.png",
    )

    # =========================================================================
    # Console
    # =========================================================================
    capture_target = float(
        np.median([
            float(r["target_capture_ratio"])
            for r in preflight_rows
        ])
    )

    capture_base1 = float(
        np.median([
            float(r["base_capture_ratio"])
            for r in preflight_rows
            if int(r["base_mode"]) == 1
        ])
    )

    capture_base2 = float(
        np.median([
            float(r["base_capture_ratio"])
            for r in preflight_rows
            if int(r["base_mode"]) == 2
        ])
    )

    lines = []

    lines.append("=" * 184)
    lines.append(
        "VPINN — STAGE 29R NON-FOURIER P1 TEST-SPACE ROBUSTNESS SUMMARY"
    )
    lines.append("=" * 184)

    lines.append(
        f"P1 test-space capture ratios: "
        f"b1={capture_base1:.6f}, "
        f"b2={capture_base2:.6f}, "
        f"target9={capture_target:.6f}"
    )

    lines.append(
        "base | deep lock | escape | horizon | median localized mu | median deep onset"
    )

    lines.append("-" * 184)

    for r in cell_summary:
        lines.append(
            f"{int(r['base_mode']):4d} | "
            f"{int(r['persistent_deep_lock_count']):4d}/5   | "
            f"{int(r['certified_escape_count']):4d}/5 | "
            f"{int(r['horizon_count']):4d}/5  | "
            f"{str(r['median_localized_mu']):19s} | "
            f"{str(r['median_deep_lock_onset'])}"
        )

    lines.append("-" * 184)

    lines.append(
        f"S1 non-Fourier analytic + paired preflight: {S1}"
    )

    lines.append(
        f"S2 mobile b1 localized baseline         : "
        f"{b1_mobile_count}/5 -> {S2}"
    )

    lines.append(
        f"S3 b1 deep-lock test-space transfer     : "
        f"{len(b1_deep)}/5 -> {S3}"
    )

    lines.append(
        f"S4 b2 non-lock control                  : "
        f"deep={len(b2_deep)}/5, "
        f"escape={len(b2_escape)}/5 -> {S4}"
    )

    lines.append(
        f"S5 residual-drop dominance at collapse : "
        f"{residual_dom_count}/{len(b1_deep)} -> {S5}"
    )

    lines.append(
        f"INDEPENDENT TEST-SPACE ROBUSTNESS      : "
        f"{independent_testspace}"
    )

    lines.append(
        f"CROSS-OPERATOR + CROSS-TESTSPACE TWO-STAGE ROBUSTNESS: "
        f"{cross_robustness}"
    )

    lines.append(
        f"route class                             : "
        f"{route_class}"
    )

    lines.append(
        f"next route                              : "
        f"{next_route}"
    )

    lines.append("=" * 184)

    lines.append(
        "Guardrail: target localization is a weak-response-template projection, "
        "not a coordinate label in the non-Fourier basis."
    )

    lines.append("=" * 184)

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
