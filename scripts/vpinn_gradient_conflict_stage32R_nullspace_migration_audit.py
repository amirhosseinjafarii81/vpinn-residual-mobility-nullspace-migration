#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 32R
Read-Only Refinement-Induced Nullspace Migration Audit
=======================================================

Scientific status
-----------------
Stage 31R was a matched-state intervention at epoch 3000 for the two
Stage-29 b=2 nonescaping controls (seeds 25,27).

At the exact same network + Adam state:

    CONTROL25 : keep the 25-element / 24-test P1 space
    REFINED26 : replace it by a 26-element / 25-test P1 space

Results:

    M1 exact replay                    PASS 2/2
    M2 endpoint visibility rescue      PASS 2/2
    M3 refined certified escape        FAIL 1/2
    M4 matched trajectory improvement  PASS 2/2

The visibility intervention was enormous:

    seed25: seen 1.12e-4 -> 5.77e-1  (~5.14e3 x)
    seed27: seen 2.32e-10 -> 6.59e-1 (~2.84e9 x)

The refined branch then reacted immediately:

    seed25 relL2: 0.010697 -> 0.004056, certified escape
    seed27 relL2: 0.016716 -> 0.010634, no strict escape

For seed27, however, the REFINED26 weak loss at epoch 4000 is already
approximately 4.43e-14 while relL2 remains ~1.063e-2.

That is not the signature of an optimizer that cannot reduce the newly
visible residual. It is the signature expected if training corrected the
visible component and the remaining trial error migrated into (or near) the
complement of the NEW finite test space.

Stage 32 tests that hypothesis before any new optimizer analysis.

No new training experiment
--------------------------
Stage 32 deterministically reconstructs the Stage-31 branches to epoch 4000.
That replay is provenance verification, not a new experimental condition.

No continuation beyond 4000.
No optimizer reset.
No learning-rate change.
No new seed.

Exact endpoint replay
---------------------
For each seed {25,27} and branch {CONTROL25, REFINED26}, reproduce the
Stage-31 epoch-4000 values

    relL2
    scaled VPINN loss
    residual norm
    target-template share
    target-template absolute residual

to <=1e-10.

Basis-invariant endpoint visibility
-----------------------------------
For a physical endpoint error

    e = u_theta - u_exact,

and ANY finite weak-test space V with raw basis phi_i, define

    b_i = a(e,phi_i)
    G_ij = a(phi_i,phi_j)

and the exact energy projection

    ||P_V e||_a^2 = b^T G^{-1} b.

Then

    chi_V = ||P_V e||_a^2 / ||e||_a^2

is the fraction of endpoint error energy visible to V.

This definition does NOT depend on the chosen basis inside V.

Stage 32 evaluates every reconstructed endpoint under the uniform P1 spaces

    V_25, V_26, V_27, V_28, V_29, V_30.

Union visibility
----------------
Replacement can move the nullspace rather than remove it.

Therefore also evaluate, for the REFINED26 endpoint,

    V_26 + V_n,  n in {25,27,28,29,30},

using a single energy Gram over the concatenated raw hat functions.

The union projection is computed exactly by

    b_union^T G_union^{-1} b_union.

No assumption of orthogonality between the two P1 spaces is made.

Primary seed
------------
Seed 27 is the only REFINED26 branch that failed the strict Stage-31 escape
gate. It is therefore the primary endpoint for the migration test.

Precommitted thresholds
-----------------------
Inherited from Stage 30:

    weakly solved loss <= 1e-6
    strict unresolved relL2 > 1e-2
    weak-test invisible chi <= 1e-2

Visibility rescue threshold inherited from Stage 31:

    candidate visibility >= 0.05
    AND gain over current self-space >= 100x.

Primary gates
-------------

E1 — EXACT STAGE-31 ENDPOINT REPLAY
    all 4 branch endpoints reproduce to <=1e-10.

E2 — REFINED26 SEED27 IS A NEW WEAK-TEST FLOOR
    at epoch 4000:
        relL2 > 1e-2
        scaled weak loss <= 1e-6
        chi_V26 <= 1e-2.

E3 — VISIBILITY WAS CONSUMED DURING THE REFINEMENT RESPONSE
    Stage-31 branch-state visibility of seed27 under V26 was >=0.05
    but endpoint chi_V26 <=1e-2.

    This is a state-transition statement:
    the refinement made the old error visible, training reduced that
    component, and the remaining endpoint is again largely outside V26.

E4 — A FIXED UNION SPACE CAN SEE THE MIGRATED ERROR
    among n in {25,27,28,29,30}, at least one union V26+Vn satisfies
        chi_union >=0.05
        AND chi_union / chi_V26 >=100
    for seed27.

NULLSPACE-MIGRATION SUPPORTED:
    E1 & E2 & E3 & E4.

Secondary diagnostics
---------------------
Record:

    * whether the OLD space V25 alone sees the final REFINED26 error;
    * minimal adjunct n satisfying E4;
    * condition number and numerical rank of each union Gram;
    * endpoint visibility matrix for both seeds and both branches.

Decision
--------
A) E1-E4 PASS:
       Stage 33R = matched-state persistent-union test-space rescue,
                   seed27 only.
       Branch from the exact REFINED26 epoch-4000 model/Adam state:
           CONTROL26 vs ORTHONORMALIZED UNION(V26,Vn_selected).
       No further mesh replacement.

B) E1-E3 PASS but E4 FAIL:
       The error has migrated into a broader complement not repaired by any
       one nearby P1 union. Do not keep refining blindly.
       Route to low-cost singular-vector test enrichment.

C) E2 FAIL:
       The Stage-31 residual floor is not a test-space invisibility floor.
       Only then return to optimizer-state analysis.

Guardrail
---------
This stage explains the failed M3 branch of Stage 31. It does not retroactively
turn Stage-31 M3 into PASS and does not claim that all finite VPINNs exhibit
nullspace migration.
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
from typing import List, Sequence

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss


SEEDS = (25, 27)
BASE_MODE = 2
TARGET_MODE = 9

BRANCH_EPOCH = 3000
FINAL_EPOCH = 4000

OLD_ELEMENTS = 25
REFINED_ELEMENTS = 26
CANDIDATE_ELEMENTS = (25, 27, 28, 29, 30)

GL_PER_INTERVAL = 12

REPLAY_TOL = 1.0e-10

REL_L2_UNRESOLVED = 1.0e-2
WEAK_LOSS_TOL = 1.0e-6
INVISIBLE_SEEN_MAX = 1.0e-2

RESCUE_SEEN_MIN = 0.05
RESCUE_GAIN_MIN = 100.0

GRAM_RANK_TOL = 1.0e-11


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-32R read-only refinement-induced nullspace migration audit."
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
        "--output-dir",
        default="vpinn_gradient_conflict_stage32R_nullspace_migration_audit",
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


# =============================================================================
# Provenance
# =============================================================================

def preflight(
    stage3_script: Path,
    stage29_script: Path,
    stage29_dir: Path,
    stage31_script: Path,
    stage31_dir: Path,
) -> dict:

    s29_manifest_path = stage29_dir / "manifest.json"

    s31_manifest_path = stage31_dir / "manifest.json"
    s31_decision_path = stage31_dir / "decision.json"
    s31_tracking_path = stage31_dir / "matched_branch_tracking.csv"
    s31_visibility_path = stage31_dir / "endpoint_visibility_rescue.csv"

    for path in (
        s29_manifest_path,
        s31_manifest_path,
        s31_decision_path,
        s31_tracking_path,
        s31_visibility_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    s29m = read_json(s29_manifest_path)
    s31m = read_json(s31_manifest_path)
    s31d = read_json(s31_decision_path)

    s3 = sha256_file(stage3_script)
    s29 = sha256_file(stage29_script)
    s31 = sha256_file(stage31_script)

    if s29m.get("stage3_solver_sha256") != s3:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 29.")

    if s29m.get("stage29r_script_sha256") != s29:
        raise RuntimeError("Stage-29 SHA mismatch.")

    if s31m.get("stage29_script_sha256") != s29:
        raise RuntimeError("Stage-29 SHA mismatch against Stage 31.")

    if s31m.get("stage31r_script_sha256") != s31:
        raise RuntimeError("Stage-31 SHA mismatch.")

    expected = {
        "M1_exact_stage29_endpoint_replay": True,
        "M2_exact_visibility_rescue": True,
        "M3_refined_branch_escape": False,
        "M4_matched_trajectory_improvement": True,
        "refined_escape_count": 1,
        "route_class":
            "refinement_restores_visibility_but_does_not_cleanly_rescue_training",
    }

    for key, value in expected.items():
        if s31d.get(key) != value:
            raise RuntimeError(
                f"Unexpected Stage-31 decision field {key}: "
                f"{s31d.get(key)!r} != {value!r}"
            )

    tracking = read_csv(s31_tracking_path)
    visibility = read_csv(s31_visibility_path)

    expected_final = {}

    for row in tracking:
        seed = int(row["seed"])
        epoch = int(row["epoch"])

        if seed not in SEEDS or epoch != FINAL_EPOCH:
            continue

        expected_final[(seed, row["branch"])] = row

    needed = {
        (seed, branch)
        for seed in SEEDS
        for branch in ("CONTROL25", "REFINED26")
    }

    if set(expected_final) != needed:
        raise RuntimeError("Incomplete Stage-31 final endpoint rows.")

    branch_seen26 = {
        int(row["seed"]): float(row["seen26_fraction"])
        for row in visibility
    }

    if set(branch_seen26) != set(SEEDS):
        raise RuntimeError("Incomplete Stage-31 branch visibility rows.")

    return {
        "stage3_sha256": s3,
        "stage29_sha256": s29,
        "stage31_sha256": s31,
        "expected_final": expected_final,
        "branch_seen26": branch_seen26,
    }


# =============================================================================
# State helpers
# =============================================================================

def capture_state(exp):
    return {
        "model": copy.deepcopy(exp.model.state_dict()),
        "optimizer": copy.deepcopy(exp.optimizer.state_dict()),
    }


def restore_state(exp, state):
    exp.model.load_state_dict(copy.deepcopy(state["model"]))
    exp.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))


# =============================================================================
# Raw P1 basis on arbitrary common quadrature
# =============================================================================

def merged_nodes(meshes: Sequence[int]) -> np.ndarray:
    nodes = {0.0, 1.0}

    for n in meshes:
        for i in range(n + 1):
            nodes.add(i / n)

    return np.asarray(sorted(nodes), dtype=np.float64)


def common_quadrature(meshes: Sequence[int]):
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

    # Search element with a small right-end correction.
    elem = np.floor(x * n_elements).astype(int)
    elem = np.clip(elem, 0, n_elements - 1)

    a = elem / n_elements
    b = (elem + 1) / n_elements

    N_left = (b - x) / h
    N_right = (x - a) / h

    left_node = elem
    right_node = elem + 1

    for q in range(len(x)):
        ln = int(left_node[q])
        rn = int(right_node[q])

        if 1 <= ln <= ntest:
            V[q, ln - 1] = N_left[q]
            D[q, ln - 1] = -1.0 / h

        if 1 <= rn <= ntest:
            V[q, rn - 1] = N_right[q]
            D[q, rn - 1] = 1.0 / h

    return V, D


# =============================================================================
# Exact projection visibility
# =============================================================================

def endpoint_error_values(exp, x_np: np.ndarray):
    x = torch.as_tensor(
        x_np.reshape(-1, 1),
        dtype=exp.dtype,
        device=exp.device,
    ).requires_grad_(True)

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
        ).reshape(-1).cpu().numpy()

        de = (
            du.detach()
            - exp.exact_derivative(x.detach())
        ).reshape(-1).cpu().numpy()

    return e, de


def projection_visibility(
    exp,
    meshes: Sequence[int],
):
    """
    Visibility in the span of raw P1 spaces listed in meshes.

    For one mesh this is exactly that P1 space.
    For multiple meshes this is their algebraic sum.
    """

    x, w = common_quadrature(meshes)

    e, de = endpoint_error_values(exp, x)

    error_energy = float(
        np.sum(
            w * (
                de * de
                + exp.sigma * e * e
            )
        )
    )

    V_parts = []
    D_parts = []

    for n in meshes:
        V, D = raw_hat_matrix(n, x)
        V_parts.append(V)
        D_parts.append(D)

    V = np.concatenate(V_parts, axis=1)
    D = np.concatenate(D_parts, axis=1)

    W = w[:, None]

    G = (
        D.T @ (W * D)
        + exp.sigma * V.T @ (W * V)
    )

    b = np.sum(
        W * (
            de[:, None] * D
            + exp.sigma * e[:, None] * V
        ),
        axis=0,
    )

    eig = np.linalg.eigvalsh(0.5 * (G + G.T))
    scale = max(float(np.max(eig)), 1.0)

    rank = int(
        np.sum(
            eig > GRAM_RANK_TOL * scale
        )
    )

    if rank != G.shape[0]:
        # Use pseudoinverse only if a mathematically redundant union occurs.
        Ginv = np.linalg.pinv(
            G,
            rcond=GRAM_RANK_TOL,
        )
        coeff = Ginv @ b
    else:
        coeff = np.linalg.solve(G, b)

    projected_energy = float(
        b @ coeff
    )

    seen = (
        projected_energy / error_energy
        if error_energy > 0.0
        else float("nan")
    )

    positive = eig[eig > GRAM_RANK_TOL * scale]

    cond = (
        float(np.max(positive) / np.min(positive))
        if len(positive) > 0
        else float("inf")
    )

    return {
        "meshes":
            "+".join(str(n) for n in meshes),

        "n_raw_functions":
            int(G.shape[0]),

        "numerical_rank":
            rank,

        "gram_condition_positive":
            cond,

        "error_energy_sq":
            error_energy,

        "projected_error_energy_sq":
            projected_energy,

        "seen_fraction":
            seen,

        "invisible_fraction":
            1.0 - seen,
    }


# =============================================================================
# Reconstruction
# =============================================================================

def build_stage31_branches(
    stage3,
    stage29,
    stage31,
    device,
    seed: int,
    out_dir: Path,
):
    cfg = stage29.make_config(
        stage3=stage3,
        seed=seed,
        device=device,
        out_dir=out_dir / "replay25",
    )

    replay = stage29.P1ReactionDiffusionExperiment(
        stage3=stage3,
        cfg=cfg,
        device=device,
        base_mode=BASE_MODE,
        target_mode=TARGET_MODE,
        out_dir=out_dir / "replay25",
    )

    for _ in range(BRANCH_EPOCH):
        replay.train_step()

    branch_state = capture_state(replay)

    control = stage29.P1ReactionDiffusionExperiment(
        stage3=stage3,
        cfg=cfg,
        device=device,
        base_mode=BASE_MODE,
        target_mode=TARGET_MODE,
        out_dir=out_dir / "control25",
    )

    restore_state(control, branch_state)

    refined = stage31.GeneralP1Experiment(
        stage3=stage3,
        cfg=cfg,
        device=device,
        n_elements=REFINED_ELEMENTS,
        base_mode=BASE_MODE,
        target_mode=TARGET_MODE,
        base_amplitude=float(replay.base_amplitude),
        target_amplitude=float(replay.amplitude),
        out_dir=out_dir / "refined26",
        loss_denominator=stage31.OLD_LOSS_DENOMINATOR,
    )

    restore_state(refined, branch_state)

    for _ in range(FINAL_EPOCH - BRANCH_EPOCH):
        control.train_step()
        refined.train_step()

    return control, refined


def endpoint_metrics(exp, branch: str):
    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    loss = (
        float(rm["vpinn_loss"])
        if "vpinn_loss" in rm
        else float(rm["vpinn_loss_scaled"])
    )

    return {
        "branch":
            branch,

        "relative_l2_error":
            rel,

        "scaled_vpinn_loss":
            loss,

        "residual_l2_norm":
            float(rm["residual_l2_norm"]),

        "target_mode_residual_energy_share":
            float(rm["target_mode_residual_energy_share"]),

        "target_template_abs_residual":
            float(rm["target_template_abs_residual"]),
    }


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
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage29_script=stage29_script,
        stage29_dir=stage29_dir,
        stage31_script=stage31_script,
        stage31_dir=stage31_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage32R",
    )

    stage29 = load_module(
        stage29_script,
        "vpinn_stage29_stage32R",
    )

    stage31 = load_module(
        stage31_script,
        "vpinn_stage31_stage32R",
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
            pf["stage3_sha256"],

        "stage29_script_sha256":
            pf["stage29_sha256"],

        "stage31_script_sha256":
            pf["stage31_sha256"],

        "stage32r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "readonly_refinement_induced_nullspace_migration_audit",

            "seeds":
                list(SEEDS),

            "primary_seed":
                27,

            "endpoint_epoch":
                FINAL_EPOCH,

            "single_meshes":
                [25, 26, 27, 28, 29, 30],

            "union_adjunct_candidates":
                list(CANDIDATE_ELEMENTS),

            "invisible_seen_max":
                INVISIBLE_SEEN_MAX,

            "union_rescue_seen_min":
                RESCUE_SEEN_MIN,

            "union_rescue_gain_min":
                RESCUE_GAIN_MIN,

            "no_new_training_condition":
                True,
        },
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 184)
    print(
        "VPINN — STAGE 32R READ-ONLY REFINEMENT-INDUCED NULLSPACE MIGRATION AUDIT"
    )
    print("=" * 184)
    print(f"device                    : {device}")
    print(f"seeds                     : {list(SEEDS)}")
    print(f"endpoint                  : epoch {FINAL_EPOCH}")
    print(f"single P1 meshes          : 25..30")
    print(f"union base               : V26 + candidate adjunct")
    print("new training condition    : NONE")
    print("=" * 184)

    replay_rows = []
    visibility_rows = []
    endpoint_rows = []

    for seed in SEEDS:

        seed_dir = out_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        control, refined = build_stage31_branches(
            stage3=stage3,
            stage29=stage29,
            stage31=stage31,
            device=device,
            seed=seed,
            out_dir=seed_dir,
        )

        for branch_name, exp in (
            ("CONTROL25", control),
            ("REFINED26", refined),
        ):
            actual = endpoint_metrics(
                exp,
                branch_name,
            )

            expected = pf["expected_final"][
                (seed, branch_name)
            ]

            diffs = {
                "relative_l2_error":
                    abs(
                        actual["relative_l2_error"]
                        - float(expected["relative_l2_error"])
                    ),

                "scaled_vpinn_loss":
                    abs(
                        actual["scaled_vpinn_loss"]
                        - float(expected["scaled_vpinn_loss"])
                    ),

                "residual_l2_norm":
                    abs(
                        actual["residual_l2_norm"]
                        - float(expected["residual_l2_norm"])
                    ),

                "target_share":
                    abs(
                        actual[
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
                        actual["target_template_abs_residual"]
                        - float(
                            expected[
                                "target_template_abs_residual"
                            ]
                        )
                    ),
            }

            gap = max(diffs.values())

            if gap > REPLAY_TOL:
                raise RuntimeError(
                    f"Stage-31 final replay failed "
                    f"seed={seed}, branch={branch_name}: "
                    f"gap={gap:.3e}, diffs={diffs}"
                )

            replay_rows.append(
                {
                    "seed":
                        seed,

                    "branch":
                        branch_name,

                    "max_abs_difference":
                        gap,

                    "pass":
                        True,

                    **{
                        f"gap_{k}": v
                        for k, v in diffs.items()
                    },
                }
            )

            endpoint_rows.append(
                {
                    "seed":
                        seed,

                    **actual,
                }
            )

            # Single-mesh visibility matrix.
            for n in (25, 26, 27, 28, 29, 30):
                vis = projection_visibility(
                    exp,
                    (n,),
                )

                visibility_rows.append(
                    {
                        "seed":
                            seed,

                        "branch":
                            branch_name,

                        "visibility_kind":
                            "SINGLE",

                        "adjunct_mesh":
                            n,

                        **vis,
                    }
                )

            # Union visibility only makes scientific sense for the refined
            # endpoint because Stage 32 asks where its new residual floor went.
            if branch_name == "REFINED26":
                for n in CANDIDATE_ELEMENTS:
                    vis = projection_visibility(
                        exp,
                        (26, n),
                    )

                    visibility_rows.append(
                        {
                            "seed":
                                seed,

                            "branch":
                                branch_name,

                            "visibility_kind":
                                "UNION26_PLUS",

                            "adjunct_mesh":
                                n,

                            **vis,
                        }
                    )

        print()
        print(
            f"seed={seed}: "
            f"control rel={endpoint_rows[-2]['relative_l2_error']:.6e}, "
            f"refined rel={endpoint_rows[-1]['relative_l2_error']:.6e}"
        )

    # =========================================================================
    # Persist raw tables
    # =========================================================================
    write_csv(
        out_dir / "stage31_endpoint_replay_checks.csv",
        replay_rows,
    )

    write_csv(
        out_dir / "reconstructed_endpoint_metrics.csv",
        endpoint_rows,
    )

    write_csv(
        out_dir / "endpoint_visibility_matrix.csv",
        visibility_rows,
    )

    # =========================================================================
    # Extract primary seed27 refined endpoint.
    # =========================================================================
    primary_endpoint = next(
        r for r in endpoint_rows
        if int(r["seed"]) == 27
        and r["branch"] == "REFINED26"
    )

    def single_seen(seed: int, branch: str, n: int) -> float:
        return float(
            next(
                r["seen_fraction"]
                for r in visibility_rows
                if (
                    int(r["seed"]) == seed
                    and r["branch"] == branch
                    and r["visibility_kind"] == "SINGLE"
                    and int(r["adjunct_mesh"]) == n
                )
            )
        )

    self26 = single_seen(
        27,
        "REFINED26",
        26,
    )

    old25_on_refined = single_seen(
        27,
        "REFINED26",
        25,
    )

    union_candidates = []

    for n in CANDIDATE_ELEMENTS:
        row = next(
            r for r in visibility_rows
            if (
                int(r["seed"]) == 27
                and r["branch"] == "REFINED26"
                and r["visibility_kind"] == "UNION26_PLUS"
                and int(r["adjunct_mesh"]) == n
            )
        )

        seen = float(row["seen_fraction"])
        gain = seen / max(self26, 1.0e-300)

        qualifies = bool(
            seen >= RESCUE_SEEN_MIN
            and
            gain >= RESCUE_GAIN_MIN
        )

        union_candidates.append(
            {
                "adjunct_mesh":
                    n,

                "union_seen_fraction":
                    seen,

                "gain_over_self26":
                    gain,

                "n_raw_functions":
                    int(row["n_raw_functions"]),

                "numerical_rank":
                    int(row["numerical_rank"]),

                "gram_condition_positive":
                    float(row["gram_condition_positive"]),

                "qualifies_union_rescue":
                    qualifies,
            }
        )

    write_csv(
        out_dir / "seed27_union_candidate_summary.csv",
        union_candidates,
    )

    qualifying = [
        r for r in union_candidates
        if bool(r["qualifies_union_rescue"])
    ]

    selected_adjunct = (
        min(
            int(r["adjunct_mesh"])
            for r in qualifying
        )
        if qualifying
        else -1
    )

    # =========================================================================
    # Gates
    # =========================================================================
    E1 = bool(
        len(replay_rows) == 4
        and
        all(bool(r["pass"]) for r in replay_rows)
    )

    E2 = bool(
        float(primary_endpoint["relative_l2_error"])
        > REL_L2_UNRESOLVED
        and
        float(primary_endpoint["scaled_vpinn_loss"])
        <= WEAK_LOSS_TOL
        and
        self26 <= INVISIBLE_SEEN_MAX
    )

    branch_seen26_seed27 = float(
        pf["branch_seen26"][27]
    )

    E3 = bool(
        branch_seen26_seed27 >= RESCUE_SEEN_MIN
        and
        self26 <= INVISIBLE_SEEN_MAX
    )

    E4 = bool(
        selected_adjunct >= 0
    )

    migration = bool(
        E1 and E2 and E3 and E4
    )

    old_space_sees_migrated_error = bool(
        old25_on_refined >= RESCUE_SEEN_MIN
    )

    if migration:

        route_class = (
            "refinement_corrects_old_visible_error_then_residual_floor_migrates_into_new_testspace_complement"
        )

        next_route = (
            "stage33R_matched_state_persistent_union_testspace_rescue"
        )

    elif E1 and E2 and E3 and not E4:

        route_class = (
            "refinement_induces_broader_nullspace_migration_not_repaired_by_nearby_P1_union"
        )

        next_route = (
            "stage33R_lowcost_singular_vector_test_enrichment"
        )

    else:

        route_class = (
            "stage31_seed27_floor_not_confirmed_as_refined_testspace_invisibility"
        )

        next_route = (
            "stage33R_optimizer_state_response_audit"
        )

    decision = {
        "E1_exact_stage31_endpoint_replay":
            E1,

        "seed27_refined_relL2":
            float(primary_endpoint["relative_l2_error"]),

        "seed27_refined_scaled_loss":
            float(primary_endpoint["scaled_vpinn_loss"]),

        "seed27_self_V26_seen_fraction":
            self26,

        "E2_refined_seed27_is_new_weak_test_floor":
            E2,

        "seed27_branch_state_V26_seen_fraction":
            branch_seen26_seed27,

        "E3_visibility_consumed_during_refinement_response":
            E3,

        "seed27_old_V25_seen_fraction_at_refined_endpoint":
            old25_on_refined,

        "old_space_sees_migrated_error":
            old_space_sees_migrated_error,

        "selected_union_adjunct_mesh":
            selected_adjunct,

        "E4_fixed_union_recovers_migrated_error_visibility":
            E4,

        "nullspace_migration_supported":
            migration,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A PASS shows that replacing V25 by V26 made the old endpoint "
            "error visible and improved training, after which the remaining "
            "error again became largely invisible to V26. This is evidence "
            "of finite-test-space nullspace migration, not a universal "
            "statement about every VPINN or P1 space."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    # =========================================================================
    # Console
    # =========================================================================
    lines = []

    lines.append("=" * 184)
    lines.append(
        "VPINN — STAGE 32R REFINEMENT-INDUCED NULLSPACE MIGRATION SUMMARY"
    )
    lines.append("=" * 184)

    lines.append(
        f"seed27 Stage31 branch V26 visibility : "
        f"{branch_seen26_seed27:.6e}"
    )

    lines.append(
        f"seed27 final self V26 visibility     : "
        f"{self26:.6e}"
    )

    lines.append(
        f"seed27 final old V25 visibility      : "
        f"{old25_on_refined:.6e}"
    )

    lines.append(
        "adjunct | union seen | gain/self26 | rank/raw | condition | qualifies"
    )

    lines.append("-" * 184)

    for r in union_candidates:
        lines.append(
            f"{int(r['adjunct_mesh']):7d} | "
            f"{float(r['union_seen_fraction']):.6e} | "
            f"{float(r['gain_over_self26']):.3e} | "
            f"{int(r['numerical_rank'])}/{int(r['n_raw_functions'])} | "
            f"{float(r['gram_condition_positive']):.3e} | "
            f"{str(r['qualifies_union_rescue'])}"
        )

    lines.append("-" * 184)

    lines.append(
        f"E1 exact Stage31 endpoint replay     : "
        f"{sum(int(r['pass']) for r in replay_rows)}/4 -> {E1}"
    )

    lines.append(
        f"E2 refined seed27 is new weak floor  : {E2}"
    )

    lines.append(
        f"E3 branch visibility consumed        : {E3}"
    )

    lines.append(
        f"E4 fixed union recovers visibility   : "
        f"selected adjunct={selected_adjunct} -> {E4}"
    )

    lines.append(
        f"NULLSPACE MIGRATION SUPPORTED        : {migration}"
    )

    lines.append(
        f"route class                           : {route_class}"
    )

    lines.append(
        f"next route                            : {next_route}"
    )

    lines.append("=" * 184)

    lines.append(
        "Guardrail: Stage31 M3 remains FAIL. Stage32 is a read-only mechanism localization."
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
