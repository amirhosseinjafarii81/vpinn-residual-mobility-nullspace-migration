#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 33R
Matched-State Persistent Union Test-Space Rescue
=================================================

Scientific status
-----------------
Stage 32R established for the only unresolved Stage-31 refined branch
(seed 27, b=2, m=9):

    E1 exact Stage-31 endpoint replay             PASS
    E2 refined seed27 is a new weak-test floor   PASS
    E3 V26 visibility was consumed by training   PASS
    E4 a fixed union recovers visibility         PASS

Key numbers at epoch 4000:

    Stage-31 branch-state visibility in V26 : 6.591442e-1
    final visibility in V26                 : 5.685215e-11
    final visibility in old V25             : 4.463604e-1
    final visibility in V26 + V25           : 5.718221e-1

Thus replacing V25 by V26 exposed the old error, training reduced that
visible component, and the remaining error became almost completely invisible
to V26 while becoming visible again to the discarded V25 space.

Stage 32 selected the smallest fixed adjunct:

    V26 + V25.

Stage 33 performs the matched-state causal test of the corresponding design
principle:

    KEEP the current test space and ADD the discarded one,
    rather than replacing one finite space by another.

No benchmark expansion
----------------------
One seed only:

    seed 27, b=2, m=9.

No new PDE.
No new architecture.
No new optimizer.
No learning-rate tuning.
No threshold changes.

Exact branch state
------------------
Deterministically reconstruct the Stage-31 REFINED26 endpoint at epoch 4000,
including the exact Adam state.

From that identical state create:

    CONTROL26:
        V26 only.

    UNION26_25:
        span(V26 union V25).

Both branches use the same physical PDE, exact solution, network parameters,
Adam moments, Adam step counter, common quadrature, and loss denominator.

Common quadrature / basis
-------------------------
To remove quadrature as a confound, both Stage-33 branches are represented on
the SAME piecewise Gauss-Legendre quadrature obtained from the merged node set
of the 25- and 26-element meshes.

CONTROL26 uses only the V26 raw hats on that quadrature.

UNION26_25 uses the concatenated raw hats from V26 and V25.

Each raw span is energy-orthonormalized by the symmetric inverse square root
of its exact discrete energy Gram.

The union is expected to have numerical rank 49.

Loss-scale control
------------------
Stage 31 used denominator 24 for the refined V26 branch.

Stage 33 preserves exactly the same denominator in BOTH branches:

    L = (1/24) * sum_j R_j^2.

Thus existing V26 residual-gradient scale is not reduced merely because the
union has more test functions.

Control-equivalence preflight
-----------------------------
Before continuing training, the common-quadrature CONTROL26 representation
must reproduce the original Stage-31 REFINED26 endpoint, in basis-invariant
quantities:

    relL2
    scaled weak loss
    residual norm
    target-template share
    target-template abs residual

and the raw parameter gradient of the loss.

Endpoint union visibility
-------------------------
Before the first Stage-33 optimizer step, require:

    chi_union >= 0.05
    chi_union / chi_control >= 100

using the common physical error energy and orthonormal residual norms.

This independently reproduces the Stage-32 union-visibility conclusion at
the actual branch state used for training.

Continuation
------------
Continue BOTH branches unconditionally from epoch 4000 through epoch 4500.

No early stop.
No backtracking.
No adaptive branch selection.
No optimizer reset.

Track every 25 epochs.

Certified escape remains the inherited physical criterion:

    relL2 <= 1e-2
    AND target-template residual share <= 0.20

for THREE consecutive 25-grid observations.

Matched trajectory comparison
-----------------------------
For each branch compute trapezoidal AUC of relL2 on [4000,4500].

Primary gates
-------------

U1 — EXACT STAGE-31 REFINED ENDPOINT REPLAY
    reconstructed REFINED26 endpoint matches Stage 31 to <=1e-10.

U2 — COMMON-QUADRATURE CONTROL EQUIVALENCE
    CONTROL26 common-quadrature endpoint invariants match original REFINED26
    to <=1e-8
    AND relative loss-gradient gap <=1e-8.

U3 — PERSISTENT UNION VISIBILITY RESCUE
    union chi >=0.05
    AND union/control visibility gain >=100.

U4 — UNION CERTIFIED ESCAPE
    UNION26_25 certifies escape by epoch 4500.

U5 — MATCHED TRAJECTORY BENEFIT
    AUC_union < AUC_control
    AND final relL2_union < final relL2_control.
    If CONTROL26 also certifies escape, UNION escape onset must be earlier.

PERSISTENT-UNION CAUSAL RESCUE:
    U1 & U2 & U3 & U4 & U5.

Interpretation if PASS
----------------------
For this previously unresolved endpoint, replacing finite test spaces caused
a visibility floor to migrate. Persistently retaining the previous test
information via V26+V25 restores visibility from the exact same network/Adam
state and causally improves the subsequent trajectory to certified escape.

This supports a design principle:

    finite adaptive test-space replacement can chase hidden residual error;
    persistent enrichment can be safer than replacement.

It is still a one-seed causal demonstration, not a universal theorem.

Next if PASS
------------
STOP benchmark expansion.

Stage 34R:
    current literature / novelty / claim audit.

Only after that audit may ONE architecture robustness control be authorized
if it is genuinely necessary for the paper-level claim.

If U3 passes but U4/U5 fail:
    do not add more meshes.
    inspect the optimizer response to the union-visible residual.

Guardrail
---------
Stage 31 M3 remains historically FAIL. Stage 33 is a separate matched-state
causal intervention.
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
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss


SEED = 27
BASE_MODE = 2
TARGET_MODE = 9

BRANCH_EPOCH = 4000
FINAL_EPOCH = 4500
TRACK_INTERVAL = 25
CERTIFY_POINTS = 3

CONTROL_ELEMENTS = 26
ADJUNCT_ELEMENTS = 25
GL_PER_INTERVAL = 10

LOSS_DENOMINATOR = 24.0

REPLAY_TOL = 1.0e-10
CONTROL_EQ_TOL = 1.0e-8
GRAD_REL_TOL = 1.0e-8

VISIBILITY_SEEN_MIN = 0.05
VISIBILITY_GAIN_MIN = 100.0

CONVERGENCE_REL_L2 = 1.0e-2
CONVERGENCE_TARGET_SHARE = 0.20

GRAM_TOL = 1.0e-9
RANK_TOL = 1.0e-11


# =============================================================================
# CLI / helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-33R matched-state persistent union test-space rescue."
    )

    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")

    p.add_argument(
        "--stage3-script",
        default="vpinn_gradient_conflict_stage3_frequency_transfer.py",
    )

    p.add_argument(
        "--stage29-script",
        default="vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness.py",
    )

    p.add_argument(
        "--stage29-dir",
        default="vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness",
    )

    p.add_argument(
        "--stage31-script",
        default="vpinn_gradient_conflict_stage31R_minimal_P1_refinement_rescue.py",
    )

    p.add_argument(
        "--stage31-dir",
        default="vpinn_gradient_conflict_stage31R_minimal_P1_refinement_rescue",
    )

    p.add_argument(
        "--stage32-script",
        default="vpinn_gradient_conflict_stage32R_nullspace_migration_audit.py",
    )

    p.add_argument(
        "--stage32-dir",
        default="vpinn_gradient_conflict_stage32R_nullspace_migration_audit",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage33R_persistent_union_rescue",
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


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
        raise RuntimeError(f"Could not import {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def capture_state(exp):
    return {
        "model": copy.deepcopy(exp.model.state_dict()),
        "optimizer": copy.deepcopy(exp.optimizer.state_dict()),
    }


def restore_state(exp, state):
    exp.model.load_state_dict(copy.deepcopy(state["model"]))
    exp.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))


def flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    chunks = []

    for p in model.parameters():
        if p.grad is None:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1))

    return torch.cat(chunks, dim=0)


# =============================================================================
# Provenance
# =============================================================================

def preflight(
    stage3_script: Path,
    stage29_script: Path,
    stage29_dir: Path,
    stage31_script: Path,
    stage31_dir: Path,
    stage32_script: Path,
    stage32_dir: Path,
) -> dict:

    s29m_path = stage29_dir / "manifest.json"
    s31m_path = stage31_dir / "manifest.json"
    s31track_path = stage31_dir / "matched_branch_tracking.csv"
    s32m_path = stage32_dir / "manifest.json"
    s32d_path = stage32_dir / "decision.json"
    s32union_path = stage32_dir / "seed27_union_candidate_summary.csv"

    for p in (
        s29m_path,
        s31m_path,
        s31track_path,
        s32m_path,
        s32d_path,
        s32union_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    s29m = read_json(s29m_path)
    s31m = read_json(s31m_path)
    s32m = read_json(s32m_path)
    s32d = read_json(s32d_path)

    s3 = sha256_file(stage3_script)
    s29 = sha256_file(stage29_script)
    s31 = sha256_file(stage31_script)
    s32 = sha256_file(stage32_script)

    if s29m.get("stage3_solver_sha256") != s3:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 29.")

    if s29m.get("stage29r_script_sha256") != s29:
        raise RuntimeError("Stage-29 SHA mismatch.")

    if s31m.get("stage29_script_sha256") != s29:
        raise RuntimeError("Stage-29 SHA mismatch against Stage 31.")

    if s31m.get("stage31r_script_sha256") != s31:
        raise RuntimeError("Stage-31 SHA mismatch.")

    if s32m.get("stage31_script_sha256") != s31:
        raise RuntimeError("Stage-31 SHA mismatch against Stage 32.")

    if s32m.get("stage32r_script_sha256") != s32:
        raise RuntimeError("Stage-32 SHA mismatch.")

    expected = {
        "E1_exact_stage31_endpoint_replay": True,
        "E2_refined_seed27_is_new_weak_test_floor": True,
        "E3_visibility_consumed_during_refinement_response": True,
        "E4_fixed_union_recovers_migrated_error_visibility": True,
        "nullspace_migration_supported": True,
        "selected_union_adjunct_mesh": 25,
        "next_route": "stage33R_matched_state_persistent_union_testspace_rescue",
    }

    for key, value in expected.items():
        if s32d.get(key) != value:
            raise RuntimeError(
                f"Unexpected Stage-32 decision {key}: "
                f"{s32d.get(key)!r} != {value!r}"
            )

    tracking = read_csv(s31track_path)

    final_rows = [
        row for row in tracking
        if (
            int(row["seed"]) == SEED
            and row["branch"] == "REFINED26"
            and int(row["epoch"]) == BRANCH_EPOCH
        )
    ]

    if len(final_rows) != 1:
        raise RuntimeError(
            "Could not identify unique Stage-31 seed27 REFINED26 epoch-4000 row."
        )

    union_rows = read_csv(s32union_path)

    selected_union = [
        row for row in union_rows
        if int(row["adjunct_mesh"]) == ADJUNCT_ELEMENTS
    ]

    if len(selected_union) != 1:
        raise RuntimeError("Missing Stage-32 V26+V25 union summary.")

    return {
        "stage3_sha256": s3,
        "stage29_sha256": s29,
        "stage31_sha256": s31,
        "stage32_sha256": s32,
        "expected_final": final_rows[0],
        "stage32_decision": s32d,
        "stage32_union": selected_union[0],
    }


# =============================================================================
# Common merged quadrature / raw hats
# =============================================================================

def merged_nodes(meshes: Sequence[int]) -> np.ndarray:
    nodes = {0.0, 1.0}

    for n in meshes:
        for i in range(n + 1):
            nodes.add(i / n)

    return np.asarray(sorted(nodes), dtype=np.float64)


def merged_quadrature(meshes: Sequence[int]):
    nodes = merged_nodes(meshes)
    xi, wi = leggauss(GL_PER_INTERVAL)

    x_all = []
    w_all = []

    for a, b in zip(nodes[:-1], nodes[1:]):
        xe = (xi + 1.0) * (b - a) / 2.0 + a
        we = wi * (b - a) / 2.0

        x_all.extend(float(v) for v in xe)
        w_all.extend(float(v) for v in we)

    return (
        np.asarray(x_all, dtype=np.float64),
        np.asarray(w_all, dtype=np.float64),
    )


def raw_hat_matrix(n_elements: int, x: np.ndarray):
    ntest = n_elements - 1
    h = 1.0 / n_elements

    V = np.zeros((len(x), ntest), dtype=np.float64)
    D = np.zeros_like(V)

    elem = np.floor(x * n_elements).astype(int)
    elem = np.clip(elem, 0, n_elements - 1)

    a = elem / n_elements
    b = (elem + 1) / n_elements

    N_left = (b - x) / h
    N_right = (x - a) / h

    for q in range(len(x)):
        left_node = int(elem[q])
        right_node = left_node + 1

        if 1 <= left_node <= ntest:
            V[q, left_node - 1] = N_left[q]
            D[q, left_node - 1] = -1.0 / h

        if 1 <= right_node <= ntest:
            V[q, right_node - 1] = N_right[q]
            D[q, right_node - 1] = 1.0 / h

    return V, D


# =============================================================================
# Common-span experiment
# =============================================================================

class CommonSpanExperiment:
    """
    Same PDE/network, but a user-specified P1 span represented on the common
    merged quadrature of V25 and V26.

    meshes=(26,)       -> CONTROL26
    meshes=(26,25)     -> UNION26_25
    """

    def __init__(
        self,
        stage3,
        cfg,
        device: torch.device,
        meshes: Sequence[int],
        base_mode: int,
        target_mode: int,
        base_amplitude: float,
        target_amplitude: float,
        sigma: float,
        out_dir: Path,
        loss_denominator: float,
    ):
        self.stage3 = stage3
        self.cfg = cfg
        self.device = device
        self.dtype = torch.float64

        self.meshes = tuple(int(n) for n in meshes)
        self.base_mode = int(base_mode)
        self.mode = int(target_mode)

        self.sigma = float(sigma)
        self.lambda_base = (
            (self.base_mode * math.pi) ** 2
            + self.sigma
        )
        self.lambda_target = (
            (self.mode * math.pi) ** 2
            + self.sigma
        )

        self.base_amplitude = float(base_amplitude)
        self.amplitude = float(target_amplitude)
        self.loss_denominator = float(loss_denominator)

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

        self._build_span()
        self._build_problem()

    def _build_span(self):
        x_np, w_np = merged_quadrature(
            (CONTROL_ELEMENTS, ADJUNCT_ELEMENTS)
        )

        V_parts = []
        D_parts = []

        for n in self.meshes:
            V, D = raw_hat_matrix(n, x_np)
            V_parts.append(V)
            D_parts.append(D)

        Vraw = np.concatenate(V_parts, axis=1)
        Draw = np.concatenate(D_parts, axis=1)

        x = torch.as_tensor(
            x_np.reshape(-1, 1),
            dtype=self.dtype,
            device=self.device,
        )

        w = torch.as_tensor(
            w_np.reshape(-1, 1),
            dtype=self.dtype,
            device=self.device,
        )

        V = torch.as_tensor(
            Vraw,
            dtype=self.dtype,
            device=self.device,
        )

        D = torch.as_tensor(
            Draw,
            dtype=self.dtype,
            device=self.device,
        )

        G = (
            D.T @ (w * D)
            + self.sigma * V.T @ (w * V)
        )

        vals, vecs = torch.linalg.eigh(
            0.5 * (G + G.T)
        )

        vmax = float(torch.max(vals).item())

        rank = int(
            torch.sum(
                vals > RANK_TOL * max(vmax, 1.0)
            ).item()
        )

        self.raw_dimension = int(G.shape[0])
        self.numerical_rank = rank

        if rank != self.raw_dimension:
            raise RuntimeError(
                f"Selected span is rank-deficient: "
                f"{rank}/{self.raw_dimension}"
            )

        vmin = float(torch.min(vals).item())

        if vmin <= 0.0:
            raise RuntimeError("Selected span Gram is not SPD.")

        self.gram_condition = vmax / vmin

        inv_sqrt = (
            vecs
            @ torch.diag(torch.rsqrt(vals))
            @ vecs.T
        )

        self.test_values = V @ inv_sqrt
        self.test_derivatives = D @ inv_sqrt

        gram = (
            self.test_derivatives.T
            @ (w * self.test_derivatives)
            + self.sigma
            * self.test_values.T
            @ (w * self.test_values)
        )

        eye = torch.eye(
            self.raw_dimension,
            dtype=self.dtype,
            device=self.device,
        )

        self.gram_error = float(
            torch.max(
                torch.abs(gram - eye)
            ).item()
        )

        if self.gram_error > GRAM_TOL:
            raise RuntimeError(
                f"Orthonormalized span Gram failed: "
                f"{self.gram_error:.3e}"
            )

        self.x_quad = (
            x.detach()
            .clone()
            .requires_grad_(True)
        )

        self.w_quad = w

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * torch.sin(self.base_mode * math.pi * x)
            + self.amplitude
            * torch.sin(self.mode * math.pi * x)
        )

    def exact_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * self.base_mode
            * math.pi
            * torch.cos(self.base_mode * math.pi * x)
            + self.amplitude
            * self.mode
            * math.pi
            * torch.cos(self.mode * math.pi * x)
        )

    def forcing(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * self.lambda_base
            * torch.sin(self.base_mode * math.pi * x)
            + self.amplitude
            * self.lambda_target
            * torch.sin(self.mode * math.pi * x)
        )

    def unit_mode_response(self, k: int) -> torch.Tensor:
        x = self.x_quad.detach()

        s = torch.sin(k * math.pi * x)
        ds = (
            k * math.pi
            * torch.cos(k * math.pi * x)
        )

        return torch.sum(
            self.w_quad
            * (
                ds * self.test_derivatives
                + self.sigma * s * self.test_values
            ),
            dim=0,
        )

    def _build_problem(self):
        q_target = self.unit_mode_response(self.mode)

        qnorm = torch.linalg.vector_norm(q_target)

        if float(qnorm.item()) <= 0.0:
            raise RuntimeError("Degenerate target template.")

        self.target_template = (
            q_target / qnorm
        ).detach()

        with torch.no_grad():
            self.forcing_values = self.forcing(
                self.x_quad.detach()
            )

            uex = self.exact_solution(
                self.x_quad.detach()
            )

            duex = self.exact_derivative(
                self.x_quad.detach()
            )

            exact_r = torch.sum(
                self.w_quad
                * (
                    duex * self.test_derivatives
                    + self.sigma * uex * self.test_values
                    - self.forcing_values * self.test_values
                ),
                dim=0,
            )

            self.exact_weak_residual_error = float(
                torch.max(
                    torch.abs(exact_r)
                ).item()
            )

        if self.exact_weak_residual_error > 1.0e-9:
            raise RuntimeError(
                f"Exact weak residual failed: "
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

        return torch.sum(
            self.w_quad
            * (
                du * self.test_derivatives
                + self.sigma * u * self.test_values
                - self.forcing_values * self.test_values
            ),
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

        return {
            "scaled_vpinn_loss":
                float(
                    (
                        torch.sum(e)
                        / self.loss_denominator
                    ).item()
                ),

            "residual_l2_norm":
                float(
                    torch.linalg.vector_norm(r).item()
                ),

            "target_mode_residual_energy_share":
                float(
                    (
                        target_proj.square()
                        / total
                    ).item()
                ),

            "target_template_abs_residual":
                float(
                    torch.abs(target_proj).item()
                ),
        }

    def loss_tensor(self) -> torch.Tensor:
        r = self.weak_residuals()

        return (
            torch.sum(r.square())
            / self.loss_denominator
        )

    def train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss = self.loss_tensor()

        loss.backward()
        self.optimizer.step()

        return float(loss.detach().item())


# =============================================================================
# Reconstruction of Stage-31 refined endpoint
# =============================================================================

def reconstruct_stage31_refined_endpoint(
    stage3,
    stage29,
    stage31,
    device,
    out_dir: Path,
):
    cfg = stage29.make_config(
        stage3=stage3,
        seed=SEED,
        device=device,
        out_dir=out_dir / "stage29_replay25",
    )

    replay = stage29.P1ReactionDiffusionExperiment(
        stage3=stage3,
        cfg=cfg,
        device=device,
        base_mode=BASE_MODE,
        target_mode=TARGET_MODE,
        out_dir=out_dir / "stage29_replay25",
    )

    for _ in range(3000):
        replay.train_step()

    branch_state = capture_state(replay)

    refined = stage31.GeneralP1Experiment(
        stage3=stage3,
        cfg=cfg,
        device=device,
        n_elements=26,
        base_mode=BASE_MODE,
        target_mode=TARGET_MODE,
        base_amplitude=float(replay.base_amplitude),
        target_amplitude=float(replay.amplitude),
        out_dir=out_dir / "stage31_refined26",
        loss_denominator=stage31.OLD_LOSS_DENOMINATOR,
    )

    restore_state(refined, branch_state)

    for _ in range(1000):
        refined.train_step()

    return cfg, replay, refined


# =============================================================================
# Common physical error energy
# =============================================================================

def physical_error_energy(exp) -> float:
    # Use the same merged quadrature as Stage-33 spans.
    x_np, w_np = merged_quadrature(
        (CONTROL_ELEMENTS, ADJUNCT_ELEMENTS)
    )

    x = torch.as_tensor(
        x_np.reshape(-1, 1),
        dtype=exp.dtype,
        device=exp.device,
    ).requires_grad_(True)

    w = torch.as_tensor(
        w_np.reshape(-1, 1),
        dtype=exp.dtype,
        device=exp.device,
    )

    u = exp.model(x)

    du = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=False,
        retain_graph=False,
    )[0]

    with torch.no_grad():
        e = (
            u.detach()
            - exp.exact_solution(x.detach())
        )

        de = (
            du.detach()
            - exp.exact_derivative(x.detach())
        )

        return float(
            torch.sum(
                w * (
                    de.square()
                    + exp.sigma * e.square()
                )
            ).item()
        )


# =============================================================================
# Endpoint / gradient diagnostics
# =============================================================================

def endpoint_metrics(exp):
    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    loss = (
        float(rm["vpinn_loss"])
        if "vpinn_loss" in rm
        else
        float(
            rm.get(
                "vpinn_loss_scaled",
                rm.get("scaled_vpinn_loss"),
            )
        )
    )

    return {
        "relative_l2_error":
            rel,

        "scaled_vpinn_loss":
            loss,

        "residual_l2_norm":
            float(rm["residual_l2_norm"]),

        "target_mode_residual_energy_share":
            float(
                rm["target_mode_residual_energy_share"]
            ),

        "target_template_abs_residual":
            float(
                rm["target_template_abs_residual"]
            ),
    }


def raw_loss_gradient(exp) -> torch.Tensor:
    exp.optimizer.zero_grad(set_to_none=True)

    if hasattr(exp, "loss_tensor"):
        loss = exp.loss_tensor()
    else:
        r = exp.weak_residuals()
        loss = (
            torch.sum(r.square())
            / LOSS_DENOMINATOR
        )

    loss.backward()

    g = flatten_gradients(exp.model).clone()

    exp.optimizer.zero_grad(set_to_none=True)

    return g


def trapezoid_auc(rows: List[dict], key: str) -> float:
    rr = sorted(rows, key=lambda r: int(r["epoch"]))

    x = np.asarray(
        [float(r["epoch"]) for r in rr],
        dtype=np.float64,
    )

    y = np.asarray(
        [float(r[key]) for r in rr],
        dtype=np.float64,
    )

    return float(np.trapezoid(y, x))


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
    stage29_script = resolve(args.stage29_script)
    stage29_dir = resolve(args.stage29_dir)
    stage31_script = resolve(args.stage31_script)
    stage31_dir = resolve(args.stage31_dir)
    stage32_script = resolve(args.stage32_script)
    stage32_dir = resolve(args.stage32_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage29_script=stage29_script,
        stage29_dir=stage29_dir,
        stage31_script=stage31_script,
        stage31_dir=stage31_dir,
        stage32_script=stage32_script,
        stage32_dir=stage32_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage33R",
    )

    stage29 = load_module(
        stage29_script,
        "vpinn_stage29_stage33R",
    )

    stage31 = load_module(
        stage31_script,
        "vpinn_stage31_stage33R",
    )

    cfg, replay25, refined31 = reconstruct_stage31_refined_endpoint(
        stage3=stage3,
        stage29=stage29,
        stage31=stage31,
        device=device,
        out_dir=out_dir / "reconstruction",
    )

    # -------------------------------------------------------------------------
    # U1 exact Stage-31 refined endpoint replay.
    # -------------------------------------------------------------------------
    actual_stage31 = endpoint_metrics(refined31)
    expected = pf["expected_final"]

    replay_diffs = {
        "relative_l2_error":
            abs(
                actual_stage31["relative_l2_error"]
                - float(expected["relative_l2_error"])
            ),

        "scaled_vpinn_loss":
            abs(
                actual_stage31["scaled_vpinn_loss"]
                - float(expected["scaled_vpinn_loss"])
            ),

        "residual_l2_norm":
            abs(
                actual_stage31["residual_l2_norm"]
                - float(expected["residual_l2_norm"])
            ),

        "target_share":
            abs(
                actual_stage31[
                    "target_mode_residual_energy_share"
                ]
                - float(
                    expected[
                        "target_mode_residual_energy_share"
                    ]
                )
            ),

        "target_abs_residual":
            abs(
                actual_stage31[
                    "target_template_abs_residual"
                ]
                - float(
                    expected[
                        "target_template_abs_residual"
                    ]
                )
            ),
    }

    replay_gap = max(replay_diffs.values())

    if replay_gap > REPLAY_TOL:
        raise RuntimeError(
            f"Stage-31 refined endpoint replay failed: "
            f"gap={replay_gap:.3e}, diffs={replay_diffs}"
        )

    U1 = True

    endpoint_state = capture_state(refined31)

    # -------------------------------------------------------------------------
    # Common-quadrature CONTROL26 and UNION26_25.
    # -------------------------------------------------------------------------
    control = CommonSpanExperiment(
        stage3=stage3,
        cfg=cfg,
        device=device,
        meshes=(26,),
        base_mode=BASE_MODE,
        target_mode=TARGET_MODE,
        base_amplitude=float(replay25.base_amplitude),
        target_amplitude=float(replay25.amplitude),
        sigma=float(replay25.sigma),
        out_dir=out_dir / "control26_common",
        loss_denominator=LOSS_DENOMINATOR,
    )

    union = CommonSpanExperiment(
        stage3=stage3,
        cfg=cfg,
        device=device,
        meshes=(26,25),
        base_mode=BASE_MODE,
        target_mode=TARGET_MODE,
        base_amplitude=float(replay25.base_amplitude),
        target_amplitude=float(replay25.amplitude),
        sigma=float(replay25.sigma),
        out_dir=out_dir / "union26_25",
        loss_denominator=LOSS_DENOMINATOR,
    )

    restore_state(control, endpoint_state)
    restore_state(union, endpoint_state)

    # -------------------------------------------------------------------------
    # U2 common-quadrature control equivalence.
    # -------------------------------------------------------------------------
    control0 = endpoint_metrics(control)

    control_diffs = {
        key: abs(
            float(control0[key])
            - float(actual_stage31[key])
        )
        for key in actual_stage31
    }

    control_gap = max(control_diffs.values())

    g_original = raw_loss_gradient(refined31)
    g_control = raw_loss_gradient(control)

    grad_gap = float(
        torch.linalg.vector_norm(
            g_control - g_original
        ).item()
        /
        max(
            float(
                torch.linalg.vector_norm(
                    g_original
                ).item()
            ),
            1.0e-300,
        )
    )

    U2 = bool(
        control_gap <= CONTROL_EQ_TOL
        and
        grad_gap <= GRAD_REL_TOL
    )

    if not U2:
        raise RuntimeError(
            f"Common control failed equivalence: "
            f"metric_gap={control_gap:.3e}, "
            f"gradient_gap={grad_gap:.3e}"
        )

    # -------------------------------------------------------------------------
    # U3 exact branch visibility.
    # -------------------------------------------------------------------------
    error_energy = physical_error_energy(control)

    r_control = control.weak_residuals().detach()
    r_union = union.weak_residuals().detach()

    seen_control = float(
        torch.sum(r_control.square()).item()
        / error_energy
    )

    seen_union = float(
        torch.sum(r_union.square()).item()
        / error_energy
    )

    visibility_gain = (
        seen_union
        / max(seen_control, 1.0e-300)
    )

    U3 = bool(
        seen_union >= VISIBILITY_SEEN_MIN
        and
        visibility_gain >= VISIBILITY_GAIN_MIN
    )

    if not U3:
        raise RuntimeError(
            f"Union visibility preflight failed: "
            f"control={seen_control:.6e}, "
            f"union={seen_union:.6e}, "
            f"gain={visibility_gain:.3e}"
        )

    # -------------------------------------------------------------------------
    # Manifest after all no-training preflights are known.
    # -------------------------------------------------------------------------
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
            pf["stage3_sha256"],

        "stage29_script_sha256":
            pf["stage29_sha256"],

        "stage31_script_sha256":
            pf["stage31_sha256"],

        "stage32_script_sha256":
            pf["stage32_sha256"],

        "stage33r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "matched_state_persistent_union_testspace_rescue",

            "seed":
                SEED,

            "branch_epoch":
                BRANCH_EPOCH,

            "final_epoch":
                FINAL_EPOCH,

            "control_span":
                "V26",

            "union_span":
                "V26+V25",

            "common_quadrature":
                "merged 25/26 nodes, 10-point GL per merged interval",

            "loss_denominator_both":
                LOSS_DENOMINATOR,

            "U1":
                "exact Stage31 refined endpoint replay <=1e-10",

            "U2":
                "common V26 metric and gradient equivalence <=1e-8",

            "U3":
                "union seen>=0.05 and visibility gain>=100",

            "U4":
                "union certified escape by 4500",

            "U5":
                "union AUC and final relL2 lower; if control escapes, union earlier",

            "optimizer_reset":
                False,

            "early_stop":
                False,
        },

        "preflight_results": {
            "replay_gap":
                replay_gap,

            "control_metric_gap":
                control_gap,

            "control_gradient_relative_gap":
                grad_gap,

            "control_seen_fraction":
                seen_control,

            "union_seen_fraction":
                seen_union,

            "union_visibility_gain":
                visibility_gain,

            "control_gram_error":
                control.gram_error,

            "union_gram_error":
                union.gram_error,

            "union_raw_dimension":
                union.raw_dimension,

            "union_numerical_rank":
                union.numerical_rank,

            "union_gram_condition":
                union.gram_condition,
        },
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 184)
    print(
        "VPINN — STAGE 33R MATCHED-STATE PERSISTENT UNION TEST-SPACE RESCUE"
    )
    print("=" * 184)
    print(f"device                    : {device}")
    print(f"seed                      : {SEED}")
    print(f"branch epoch              : {BRANCH_EPOCH}")
    print(f"final epoch               : {FINAL_EPOCH}")
    print(f"control                   : V26")
    print(f"union                     : V26 + V25")
    print(f"union rank                : {union.numerical_rank}/{union.raw_dimension}")
    print(f"union Gram condition      : {union.gram_condition:.6e}")
    print(f"control seen              : {seen_control:.6e}")
    print(f"union seen                : {seen_union:.6e}")
    print(f"visibility gain           : {visibility_gain:.6e}")
    print(f"control gradient gap      : {grad_gap:.6e}")
    print("optimizer reset           : NONE")
    print("=" * 184)

    # -------------------------------------------------------------------------
    # Matched unconditional continuation.
    # -------------------------------------------------------------------------
    branches = {
        "CONTROL26": control,
        "UNION26_25": union,
    }

    tracking_rows = []

    escape_streak = {
        name: 0 for name in branches
    }

    escape_candidate = {
        name: None for name in branches
    }

    escape_onset = {
        name: -1 for name in branches
    }

    escape_confirmation = {
        name: -1 for name in branches
    }

    for epoch in range(BRANCH_EPOCH, FINAL_EPOCH + 1):

        if epoch % TRACK_INTERVAL == 0:

            for name, exp in branches.items():
                rm = exp.residual_metrics()
                rel = exp.relative_l2_error()

                share = float(
                    rm[
                        "target_mode_residual_energy_share"
                    ]
                )

                qualifies = bool(
                    rel <= CONVERGENCE_REL_L2
                    and
                    share <= CONVERGENCE_TARGET_SHARE
                )

                if escape_onset[name] < 0:

                    if qualifies:
                        if escape_streak[name] == 0:
                            escape_candidate[name] = epoch

                        escape_streak[name] += 1

                    else:
                        escape_streak[name] = 0
                        escape_candidate[name] = None

                    if escape_streak[name] >= CERTIFY_POINTS:
                        escape_onset[name] = int(
                            escape_candidate[name]
                        )
                        escape_confirmation[name] = epoch

                tracking_rows.append(
                    {
                        "seed":
                            SEED,

                        "branch":
                            name,

                        "epoch":
                            epoch,

                        "relative_l2_error":
                            rel,

                        **rm,

                        "escape_streak":
                            escape_streak[name],

                        "certified_escape_onset_so_far":
                            escape_onset[name],
                    }
                )

        if epoch < FINAL_EPOCH:
            control.train_step()
            union.train_step()

    write_csv(
        out_dir / "matched_branch_tracking.csv",
        tracking_rows,
    )

    # -------------------------------------------------------------------------
    # Branch summary / gates U4,U5.
    # -------------------------------------------------------------------------
    summaries = []

    for name, exp in branches.items():
        rows = [
            r for r in tracking_rows
            if r["branch"] == name
        ]

        auc = trapezoid_auc(
            rows,
            "relative_l2_error",
        )

        final = rows[-1]

        final_error_energy = physical_error_energy(exp)
        r_final = exp.weak_residuals().detach()

        final_seen = float(
            torch.sum(r_final.square()).item()
            / final_error_energy
        )

        summaries.append(
            {
                "branch":
                    name,

                "certified_escape":
                    bool(escape_onset[name] >= 0),

                "escape_onset":
                    escape_onset[name],

                "escape_confirmation":
                    escape_confirmation[name],

                "auc_relL2":
                    auc,

                "final_relL2":
                    float(final["relative_l2_error"]),

                "final_scaled_loss":
                    float(final["scaled_vpinn_loss"]),

                "final_target_share":
                    float(
                        final[
                            "target_mode_residual_energy_share"
                        ]
                    ),

                "final_self_seen_fraction":
                    final_seen,
            }
        )

    write_csv(
        out_dir / "matched_branch_summary.csv",
        summaries,
    )

    smap = {
        r["branch"]: r for r in summaries
    }

    U4 = bool(
        smap["UNION26_25"]["certified_escape"]
    )

    control_escaped = bool(
        smap["CONTROL26"]["certified_escape"]
    )

    escape_order_ok = bool(
        not control_escaped
        or
        int(
            smap["UNION26_25"]["escape_onset"]
        )
        <
        int(
            smap["CONTROL26"]["escape_onset"]
        )
    )

    U5 = bool(
        float(
            smap["UNION26_25"]["auc_relL2"]
        )
        <
        float(
            smap["CONTROL26"]["auc_relL2"]
        )
        and
        float(
            smap["UNION26_25"]["final_relL2"]
        )
        <
        float(
            smap["CONTROL26"]["final_relL2"]
        )
        and
        escape_order_ok
    )

    rescue = bool(
        U1 and U2 and U3 and U4 and U5
    )

    if rescue:
        route_class = (
            "persistent_union_testspace_causally_rescues_refinement_induced_nullspace_migration"
        )

        next_route = (
            "stage34R_current_literature_novelty_claim_audit"
        )

    elif U1 and U2 and U3 and not (U4 and U5):
        route_class = (
            "union_restores_visibility_but_does_not_cleanly_rescue_trajectory"
        )

        next_route = (
            "stage34R_union_visible_residual_optimizer_response_audit"
        )

    else:
        route_class = (
            "persistent_union_rescue_hypothesis_not_supported"
        )

        next_route = (
            "stage34R_stop_union_intervention_and_reassess"
        )

    decision = {
        "U1_exact_stage31_refined_endpoint_replay":
            U1,

        "U2_common_quadrature_control_equivalence":
            U2,

        "U3_persistent_union_visibility_rescue":
            U3,

        "control_seen_fraction":
            seen_control,

        "union_seen_fraction":
            seen_union,

        "visibility_gain":
            visibility_gain,

        "control_certified_escape":
            control_escaped,

        "union_certified_escape":
            bool(
                smap["UNION26_25"]["certified_escape"]
            ),

        "U4_union_certified_escape":
            U4,

        "U5_matched_trajectory_benefit":
            U5,

        "persistent_union_causal_rescue":
            rescue,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A PASS is a one-seed matched-state causal demonstration that "
            "retaining V26 while adding discarded V25 information repairs "
            "the observed nullspace-migration floor. It is not a universal "
            "claim that union enrichment always prevents hidden-error "
            "migration in VPINNs."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    # -------------------------------------------------------------------------
    # Plot.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    for name in ("CONTROL26", "UNION26_25"):
        rr = [
            r for r in tracking_rows
            if r["branch"] == name
        ]

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r["relative_l2_error"]) for r in rr],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=name,
        )

    ax.axhline(
        CONVERGENCE_REL_L2,
        linestyle="--",
        linewidth=1.0,
        label="relL2 escape threshold",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Matched-state persistent union vs V26 replacement floor")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        out_dir / "persistent_union_rescue_relL2.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Console.
    # -------------------------------------------------------------------------
    lines = []

    lines.append("=" * 184)
    lines.append(
        "VPINN — STAGE 33R PERSISTENT UNION TEST-SPACE RESCUE SUMMARY"
    )
    lines.append("=" * 184)

    lines.append(
        f"U1 Stage31 endpoint replay            : {U1} "
        f"(gap={replay_gap:.3e})"
    )

    lines.append(
        f"U2 common V26 equivalence             : {U2} "
        f"(metric={control_gap:.3e}, grad={grad_gap:.3e})"
    )

    lines.append(
        f"U3 union visibility rescue            : {U3} "
        f"(control={seen_control:.6e}, union={seen_union:.6e}, "
        f"gain={visibility_gain:.3e})"
    )

    lines.append("-" * 184)

    for r in summaries:
        lines.append(
            f"{r['branch']:12s} | "
            f"escape={int(r['escape_onset']):4d} | "
            f"AUC={float(r['auc_relL2']):.6e} | "
            f"final rel={float(r['final_relL2']):.6e} | "
            f"final loss={float(r['final_scaled_loss']):.6e} | "
            f"final seen={float(r['final_self_seen_fraction']):.6e}"
        )

    lines.append("-" * 184)

    lines.append(
        f"U4 union certified escape             : {U4}"
    )

    lines.append(
        f"U5 matched trajectory benefit         : {U5}"
    )

    lines.append(
        f"PERSISTENT-UNION CAUSAL RESCUE        : {rescue}"
    )

    lines.append(
        f"route class                            : {route_class}"
    )

    lines.append(
        f"next route                             : {next_route}"
    )

    lines.append("=" * 184)

    lines.append(
        "Guardrail: Stage31 M3 remains FAIL. Stage33 is a separate one-seed matched-state intervention."
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
