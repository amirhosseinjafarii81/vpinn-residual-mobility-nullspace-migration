#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 11
Earliest-Failure Joint Total/Target Descent-Cone Projection Audit
=================================================================

Scientific motivation
---------------------
Stage 10 identified the dominant failure mechanism of repeated REFLECT:

For 4/5 seeds, the EARLIEST active failure is DIRECTION_CONFLICT:

    <grad T, Delta_R> < 0
    <grad L, Delta_R> > 0

where

    T(theta) = R_9(theta)^2 / M
    L(theta) = (1/M) sum_k R_k(theta)^2.

Moreover, at those earliest failures, even alpha=1/64 along Delta_R increases
the total VPINN loss. Therefore simple damping of the SAME reflected direction
is not the fundamental fix.

Stage 11 asks the next minimal question:

    Can the failed REFLECT displacement be minimally modified so that it lies
    in the INTERSECTION of the total-loss and target-loss descent half-spaces?

Define the closed first-order joint descent cone

    C(theta) = {
        d :
        <grad L, d> <= 0,
        <grad T, d> <= 0
    }.

At each seed's EARLIEST Stage-10 DIRECTION_CONFLICT state, compute

    d_C = argmin_d  0.5 ||d - Delta_R||^2
          subject to
              <grad L, d> <= 0,
              <grad T, d> <= 0.

This is a convex two-constraint Euclidean projection with an exact
active-set/KKT solution. No optimizer hyperparameter is tuned.

Why earliest failure only?
--------------------------
Later states on the Stage-9 REFLECT trajectory may already be consequences of
earlier divergence. The earliest failure is the least-confounded state at
which the mechanism becomes invalid.

Active seeds
------------
Stage 10 reported:
    seed 0 : earliest failure 2505
    seed 1 : earliest failure 2505
    seed 2 : earliest failure 2505
    seed 3 : earliest failure 2510
    seed 4 : no active failure in 2500..2600

Seed 4 is recorded as NO_TRIGGER and is not counted in the active-mechanism
group denominator.

Read-only exact line scan
-------------------------
After projecting Delta_R to d_C, evaluate

    theta + alpha d_C

for the precommitted grid

    alpha in {1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 3/4, 1}.

The optimizer state is never changed by this line scan.

Important diagnostics
---------------------
For each active seed:
  * KKT feasibility/stationarity/complementarity;
  * projection active set;
  * ||d_C|| / ||Delta_R||;
  * ||d_C - Delta_R|| / ||Delta_R||;
  * retained target-descent fraction

        q_T =
            (-<grad T,d_C>)
            / (-<grad T,Delta_R>),

    when Delta_R is target descent;
  * exact total and target losses along the line scan.

Classification
--------------
FULL_SAFE_TARGET_DESCENT:
    d_C retains strict first-order target descent and alpha=1 is jointly safe.

DAMPED_SAFE_TARGET_DESCENT:
    d_C retains strict first-order target descent, alpha=1 is unsafe, but some
    alpha<1 is jointly safe.

TARGET_STALLED:
    d_C is joint-cone feasible but target directional derivative is numerically
    zero. The zero-margin cone projection has sacrificed active correction of
    R_9 and is not enough for an escape mechanism.

STRICT_DESCENT_BUT_NO_SAFE_ALPHA:
    d_C is strict target descent and first-order total non-ascent, but no
    scanned alpha is jointly safe.

DEGENERATE_ZERO_STEP:
    projected direction is numerically zero.

KKT_FAILURE:
    projection verification fails.

Primary route
-------------
Let N_active be the number of seeds with a Stage-10 earliest failure.
Here N_active is expected to be 4.

If >=3 active seeds are FULL_SAFE_TARGET_DESCENT:
    authorize a bounded full-step joint-cone continuation pilot.

Else if >=3 active seeds are either FULL_SAFE_TARGET_DESCENT or
DAMPED_SAFE_TARGET_DESCENT:
    authorize a backtracking joint-cone continuation pilot.

Else if >=3 active seeds are TARGET_STALLED:
    authorize a STRICT-MARGIN joint total/target descent QP audit.

Otherwise:
    route to mixed Pareto-geometry audit.

This stage is an AUDIT only. It does not train a proposed method.
"""

from __future__ import annotations

import argparse
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


ALPHAS = (
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    3.0 / 4.0,
    1.0,
)

TARGET_MODE = 9


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-11 earliest-failure joint descent-cone audit."
    )
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")

    p.add_argument(
        "--stage3-script",
        default="vpinn_gradient_conflict_stage3_frequency_transfer.py",
    )
    p.add_argument(
        "--stage5-dir",
        default="vpinn_gradient_conflict_stage5_escape_time",
    )
    p.add_argument(
        "--stage9-script",
        default="vpinn_gradient_conflict_stage9_reflected_adam_continuation.py",
    )
    p.add_argument(
        "--stage9-dir",
        default="vpinn_gradient_conflict_stage9_reflected_adam_continuation",
    )
    p.add_argument(
        "--stage10-script",
        default="vpinn_gradient_conflict_stage10_local_feasibility_audit.py",
    )
    p.add_argument(
        "--stage10-dir",
        default="vpinn_gradient_conflict_stage10_local_feasibility_audit",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage11_joint_descent_cone_audit",
    )

    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

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


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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

    normalized = [
        {key: row.get(key, None) for key in fields}
        for row in rows
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Python source not found: {path}")

    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import source: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def flatten(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in parts], dim=0)


# =============================================================================
# Preflight
# =============================================================================

def preflight(
    stage3_script: Path,
    stage5_dir: Path,
    stage9_script: Path,
    stage9_dir: Path,
    stage10_script: Path,
    stage10_dir: Path,
) -> dict:
    paths = {
        "s5_manifest": stage5_dir / "manifest.json",
        "s9_manifest": stage9_dir / "manifest.json",
        "s10_manifest": stage10_dir / "manifest.json",
        "s10_decision": stage10_dir / "decision.json",
        "s10_seed_summary": stage10_dir / "seed_summary.csv",
        "s10_probe": stage10_dir / "probe_metrics.csv",
        "s10_replay": stage10_dir / "stage9_replay_checks.csv",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    s5_manifest = read_json(paths["s5_manifest"])
    s9_manifest = read_json(paths["s9_manifest"])
    s10_manifest = read_json(paths["s10_manifest"])
    s10_decision = read_json(paths["s10_decision"])

    if s10_decision.get("next_route") != (
        "joint_total_target_descent_cone_projection_audit"
    ):
        raise RuntimeError(
            "Stage 10 did not authorize joint descent-cone projection."
        )

    if s10_decision.get("route_class") != (
        "multiobjective_direction_conflict_dominant"
    ):
        raise RuntimeError(
            "Stage-10 route class is not direction-conflict dominant."
        )

    if not bool(
        s10_decision.get("all_stage9_replay_checks_pass", False)
    ):
        raise RuntimeError("Stage-10 Stage-9 replay checks are not all PASS.")

    actual_s3_sha = sha256_file(stage3_script)
    actual_s9_sha = sha256_file(stage9_script)
    actual_s10_sha = sha256_file(stage10_script)

    if s5_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 5.")

    if s9_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 9.")

    if s10_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 10.")

    if s10_manifest.get("stage9_script_sha256") != actual_s9_sha:
        raise RuntimeError("Stage-9 source SHA mismatch against Stage 10.")

    if s10_manifest.get("stage10_script_sha256") != actual_s10_sha:
        raise RuntimeError("Stage-10 source SHA mismatch against its manifest.")

    replay_rows = read_csv(paths["s10_replay"])

    if not replay_rows or not all(
        str(r["pass"]).lower() == "true"
        for r in replay_rows
    ):
        raise RuntimeError("Stage-10 replay-check CSV is not all PASS.")

    seed_summary = read_csv(paths["s10_seed_summary"])
    probe_rows = read_csv(paths["s10_probe"])

    active_targets = {}

    for row in seed_summary:
        seed = int(row["seed"])
        epoch = int(row["earliest_active_non_SAFE_FULL_epoch"])
        cls = row["earliest_failure_class"]

        if epoch >= 0:
            if cls != "DIRECTION_CONFLICT":
                raise RuntimeError(
                    f"Seed {seed} earliest failure is {cls}, not "
                    "DIRECTION_CONFLICT."
                )
            active_targets[seed] = epoch

    if len(active_targets) < 3:
        raise RuntimeError(
            "Fewer than three active direction-conflict seeds. "
            "Stage-11 group route is not justified."
        )

    for seed in range(5):
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing locked checkpoint: {checkpoint}")

    return {
        "seeds": [0, 1, 2, 3, 4],
        "active_targets": active_targets,
        "probe_rows": probe_rows,
        "stage3_sha256": actual_s3_sha,
        "stage9_sha256": actual_s9_sha,
        "stage10_sha256": actual_s10_sha,
    }


# =============================================================================
# Exact current geometry
# =============================================================================

def current_geometry(exp, target_mode: int = TARGET_MODE) -> dict:
    residuals = exp.weak_residuals()
    params = tuple(p for p in exp.model.parameters() if p.requires_grad)

    t = target_mode - 1
    M = residuals.numel()

    target_loss = residuals[t].square() / M
    total_loss = residuals.square().mean()

    gt_parts = torch.autograd.grad(
        target_loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )

    gL_parts = torch.autograd.grad(
        total_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    gt = flatten(gt_parts).detach()
    gL = flatten(gL_parts).detach()

    group = exp.optimizer.param_groups[0]

    if float(group.get("weight_decay", 0.0)) != 0.0:
        raise RuntimeError("Expected Adam weight_decay=0.")
    if bool(group.get("amsgrad", False)):
        raise RuntimeError("Expected Adam amsgrad=False.")
    if bool(group.get("maximize", False)):
        raise RuntimeError("Expected Adam maximize=False.")

    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    lr = float(group["lr"])

    candidate_parts = []
    offset = 0

    for p in params:
        n = p.numel()
        gp = gL[offset:offset+n].reshape_as(p)
        offset += n

        state = exp.optimizer.state[p]

        m_old = state["exp_avg"]
        v_old = state["exp_avg_sq"]

        step_raw = state["step"]
        step_old = (
            int(step_raw.item())
            if torch.is_tensor(step_raw)
            else int(step_raw)
        )

        m_new = beta1 * m_old + (1.0 - beta1) * gp
        v_new = beta2 * v_old + (1.0 - beta2) * gp.square()

        step_new = step_old + 1
        bc1 = 1.0 - beta1 ** step_new
        bc2 = 1.0 - beta2 ** step_new

        denom = (v_new.sqrt() / math.sqrt(bc2)).add(eps)
        delta = -lr / bc1 * m_new / denom

        candidate_parts.append(delta.reshape(-1))

    candidate = flatten(candidate_parts)

    gt2 = torch.dot(gt, gt)
    candidate_target_dot = torch.dot(gt, candidate)

    if float(gt2.item()) <= 0.0:
        raise RuntimeError("Target gradient norm is zero.")

    reflect_active = bool(candidate_target_dot.item() > 0.0)

    if reflect_active:
        component = (candidate_target_dot / gt2) * gt
        reflected = candidate - 2.0 * component
    else:
        reflected = candidate.detach().clone()

    return {
        "params": params,
        "g_total": gL,
        "g_target": gt,
        "adam_candidate": candidate,
        "reflected": reflected,
        "reflect_active": reflect_active,
        "pre_total_loss": float(total_loss.detach().item()),
        "pre_target_loss": float(target_loss.detach().item()),
        "candidate_norm": float(torch.linalg.vector_norm(candidate).item()),
        "reflected_norm": float(torch.linalg.vector_norm(reflected).item()),
        "reflect_total_dot": float(torch.dot(gL, reflected).item()),
        "reflect_target_dot": float(torch.dot(gt, reflected).item()),
    }


# =============================================================================
# Projection onto intersection of two homogeneous half-spaces
# =============================================================================

def project_two_descent_halfspaces(
    source: torch.Tensor,
    g_total: torch.Tensor,
    g_target: torch.Tensor,
) -> dict:
    """
    Exact Euclidean projection:

        min_d 0.5 ||d-source||^2
        s.t. g_total^T d <= 0
             g_target^T d <= 0

    Solve by enumerating the complete two-constraint KKT active set.
    """
    dtype = source.dtype
    device = source.device

    normals = [g_total, g_target]
    names = ["TOTAL", "TARGET"]

    norm_sq = [torch.dot(g, g) for g in normals]
    violations = [torch.dot(g, source) for g in normals]

    source_norm = torch.linalg.vector_norm(source)

    feas_tol = [
        1.0e-11
        * max(
            1.0e-300,
            float(torch.sqrt(ns).item() * source_norm.item()),
        )
        for ns in norm_sq
    ]

    candidates = []

    def add_candidate(
        d: torch.Tensor,
        lambda_total: float,
        lambda_target: float,
        active_set: str,
    ):
        vals = [
            float(torch.dot(g_total, d).item()),
            float(torch.dot(g_target, d).item()),
        ]

        feasible = (
            vals[0] <= feas_tol[0]
            and vals[1] <= feas_tol[1]
        )

        dual_ok = (
            lambda_total >= -1.0e-12
            and lambda_target >= -1.0e-12
        )

        if not (feasible and dual_ok):
            return

        distance_sq = float(
            torch.dot(d - source, d - source).item()
        )

        candidates.append(
            {
                "d": d.detach().clone(),
                "lambda_total": float(max(0.0, lambda_total)),
                "lambda_target": float(max(0.0, lambda_target)),
                "active_set": active_set,
                "distance_sq": distance_sq,
            }
        )

    # No active constraints.
    if (
        float(violations[0].item()) <= feas_tol[0]
        and float(violations[1].item()) <= feas_tol[1]
    ):
        add_candidate(
            source,
            0.0,
            0.0,
            "NONE",
        )

    # TOTAL only.
    if float(norm_sq[0].item()) > 0.0:
        lam = float(
            (violations[0] / norm_sq[0]).item()
        )

        if lam >= -1.0e-12:
            d = source - lam * g_total
            add_candidate(
                d,
                lam,
                0.0,
                "TOTAL",
            )

    # TARGET only.
    if float(norm_sq[1].item()) > 0.0:
        lam = float(
            (violations[1] / norm_sq[1]).item()
        )

        if lam >= -1.0e-12:
            d = source - lam * g_target
            add_candidate(
                d,
                0.0,
                lam,
                "TARGET",
            )

    # BOTH active.
    a = norm_sq[0]
    b = torch.dot(g_total, g_target)
    c = norm_sq[1]

    G = torch.stack(
        [
            torch.stack([a, b]),
            torch.stack([b, c]),
        ]
    )

    rhs = torch.stack([violations[0], violations[1]])

    scale = max(
        1.0e-300,
        float(torch.max(torch.abs(G)).item()),
    )

    det = float(torch.det(G).item())

    if abs(det) > 1.0e-14 * scale * scale:
        lambdas = torch.linalg.solve(G, rhs)

        lt = float(lambdas[0].item())
        lr = float(lambdas[1].item())

        if lt >= -1.0e-12 and lr >= -1.0e-12:
            d = source - lt * g_total - lr * g_target

            add_candidate(
                d,
                lt,
                lr,
                "TOTAL+TARGET",
            )

    if not candidates:
        raise RuntimeError(
            "No valid KKT candidate found for two-halfspace projection."
        )

    best = min(candidates, key=lambda x: x["distance_sq"])

    d = best["d"]
    lt = best["lambda_total"]
    lr = best["lambda_target"]

    total_dot = float(torch.dot(g_total, d).item())
    target_dot = float(torch.dot(g_target, d).item())

    stationarity = (
        d
        - source
        + lt * g_total
        + lr * g_target
    )

    stationarity_norm = float(
        torch.linalg.vector_norm(stationarity).item()
    )

    source_scale = max(
        1.0e-300,
        float(torch.linalg.vector_norm(source).item()),
    )

    stationarity_rel = stationarity_norm / source_scale

    complementarity_total = abs(lt * total_dot)
    complementarity_target = abs(lr * target_dot)

    dual_feasible = bool(lt >= -1.0e-12 and lr >= -1.0e-12)

    primal_feasible = bool(
        total_dot <= feas_tol[0]
        and target_dot <= feas_tol[1]
    )

    kkt_pass = bool(
        primal_feasible
        and dual_feasible
        and stationarity_rel <= 1.0e-9
        and complementarity_total
            <= 1.0e-9 * max(1.0e-300, source_scale)
        and complementarity_target
            <= 1.0e-9 * max(1.0e-300, source_scale)
    )

    return {
        "direction": d,
        "active_set": best["active_set"],
        "lambda_total": lt,
        "lambda_target": lr,
        "source_total_dot": float(violations[0].item()),
        "source_target_dot": float(violations[1].item()),
        "projected_total_dot": total_dot,
        "projected_target_dot": target_dot,
        "projection_distance": math.sqrt(max(0.0, best["distance_sq"])),
        "stationarity_relative_residual": stationarity_rel,
        "complementarity_total": complementarity_total,
        "complementarity_target": complementarity_target,
        "primal_feasible": primal_feasible,
        "dual_feasible": dual_feasible,
        "kkt_pass": kkt_pass,
    }


# =============================================================================
# Exact read-only line scan
# =============================================================================

def loss_metrics(exp, target_mode: int = TARGET_MODE) -> dict:
    residuals = exp.weak_residuals().detach()
    energy = residuals.square()

    return {
        "total_loss": float(torch.mean(energy).item()),
        "target_loss": float(
            (energy[target_mode - 1] / energy.numel()).item()
        ),
    }


def set_displaced_params(
    params,
    base_parts,
    d: torch.Tensor,
    alpha: float,
) -> None:
    offset = 0

    with torch.no_grad():
        for p, p0 in zip(params, base_parts):
            n = p.numel()

            p.copy_(
                p0
                + alpha
                * d[offset:offset+n].reshape_as(p)
            )

            offset += n

    if offset != d.numel():
        raise RuntimeError("Displacement size mismatch.")


def restore_params(params, base_parts) -> None:
    with torch.no_grad():
        for p, p0 in zip(params, base_parts):
            p.copy_(p0)


def line_scan(exp, params, d: torch.Tensor) -> tuple[list[dict], dict]:
    base_parts = [p.detach().clone() for p in params]
    pre = loss_metrics(exp)

    tol_total = 1.0e-12 * max(1.0, abs(pre["total_loss"]))
    tol_target = 1.0e-12 * max(1.0, abs(pre["target_loss"]))

    rows = []

    try:
        for alpha in ALPHAS:
            set_displaced_params(
                params=params,
                base_parts=base_parts,
                d=d,
                alpha=alpha,
            )

            post = loss_metrics(exp)

            dL = post["total_loss"] - pre["total_loss"]
            dT = post["target_loss"] - pre["target_loss"]

            joint_safe = bool(
                dL <= tol_total
                and dT <= tol_target
            )

            strict_target_improvement = bool(
                dT < -tol_target
            )

            rows.append(
                {
                    "alpha": float(alpha),
                    "pre_total_loss": pre["total_loss"],
                    "post_total_loss": post["total_loss"],
                    "total_loss_change": dL,
                    "pre_target_loss": pre["target_loss"],
                    "post_target_loss": post["target_loss"],
                    "target_loss_change": dT,
                    "joint_safe": joint_safe,
                    "strict_target_improvement":
                        strict_target_improvement,
                }
            )

    finally:
        restore_params(params, base_parts)

    safe = [r for r in rows if r["joint_safe"]]

    strict_safe = [
        r for r in rows
        if r["joint_safe"] and r["strict_target_improvement"]
    ]

    full = next(r for r in rows if r["alpha"] == 1.0)

    return rows, {
        "full_joint_safe": bool(full["joint_safe"]),
        "full_strict_target_improvement":
            bool(full["strict_target_improvement"]),
        "max_joint_safe_alpha": (
            max(r["alpha"] for r in safe)
            if safe else 0.0
        ),
        "max_strict_target_safe_alpha": (
            max(r["alpha"] for r in strict_safe)
            if strict_safe else 0.0
        ),
        "any_joint_safe": bool(safe),
        "any_strict_target_safe": bool(strict_safe),
    }


# =============================================================================
# Stage-10 reproduction at earliest failure
# =============================================================================

def probe_reference_map(rows: List[dict]) -> dict:
    return {
        (int(r["seed"]), int(r["epoch"])): r
        for r in rows
    }


def verify_against_stage10(
    seed: int,
    epoch: int,
    geo: dict,
    ref_map: dict,
    tolerance: float = 1.0e-10,
) -> dict:
    key = (seed, epoch)

    if key not in ref_map:
        raise RuntimeError(
            f"Stage-10 probe row missing: seed={seed}, epoch={epoch}."
        )

    old = ref_map[key]

    comparisons = {
        "candidate_update_norm": (
            float(old["candidate_update_norm"]),
            geo["candidate_norm"],
        ),
        "reflected_update_norm": (
            float(old["reflected_update_norm"]),
            geo["reflected_norm"],
        ),
        "target_directional_derivative": (
            float(old["target_directional_derivative"]),
            geo["reflect_target_dot"],
        ),
        "total_directional_derivative": (
            float(old["total_directional_derivative"]),
            geo["reflect_total_dot"],
        ),
        "pre_total_loss": (
            float(old["pre_total_loss"]),
            geo["pre_total_loss"],
        ),
        "pre_target_loss": (
            float(old["pre_target_loss"]),
            geo["pre_target_loss"],
        ),
    }

    diffs = {
        name: abs(a - b)
        for name, (a, b) in comparisons.items()
    }

    max_diff = max(diffs.values())

    return {
        "seed": seed,
        "epoch": epoch,
        "max_abs_difference": max_diff,
        "field_abs_differences": diffs,
        "pass": bool(max_diff <= tolerance),
    }


# =============================================================================
# Classification
# =============================================================================

def classify_projection(
    projection: dict,
    reflected: torch.Tensor,
    g_target: torch.Tensor,
    scan_summary: dict,
) -> tuple[str, dict]:
    d = projection["direction"]

    source_norm = float(torch.linalg.vector_norm(reflected).item())
    proj_norm = float(torch.linalg.vector_norm(d).item())

    norm_ratio = proj_norm / max(source_norm, 1.0e-300)

    modification_ratio = (
        float(torch.linalg.vector_norm(d - reflected).item())
        / max(source_norm, 1.0e-300)
    )

    source_target_dot = float(torch.dot(g_target, reflected).item())
    projected_target_dot = float(torch.dot(g_target, d).item())

    if source_target_dot < 0.0:
        target_retention = (
            -projected_target_dot
            / max(-source_target_dot, 1.0e-300)
        )
    else:
        target_retention = float("nan")

    target_scale = max(
        1.0e-300,
        float(
            torch.linalg.vector_norm(g_target).item()
            * torch.linalg.vector_norm(d).item()
        ),
    )

    strict_target_descent = bool(
        projected_target_dot
        < -1.0e-11 * target_scale
    )

    numerically_zero = bool(
        proj_norm <= 1.0e-12 * max(1.0, source_norm)
    )

    if not projection["kkt_pass"]:
        cls = "KKT_FAILURE"

    elif numerically_zero:
        cls = "DEGENERATE_ZERO_STEP"

    elif not strict_target_descent:
        cls = "TARGET_STALLED"

    elif (
        scan_summary["full_joint_safe"]
        and scan_summary["full_strict_target_improvement"]
    ):
        cls = "FULL_SAFE_TARGET_DESCENT"

    elif (
        scan_summary["any_joint_safe"]
        and scan_summary["any_strict_target_safe"]
    ):
        cls = "DAMPED_SAFE_TARGET_DESCENT"

    else:
        cls = "STRICT_DESCENT_BUT_NO_SAFE_ALPHA"

    diagnostics = {
        "projected_norm": proj_norm,
        "projected_over_reflected_norm": norm_ratio,
        "projection_modification_over_reflected_norm":
            modification_ratio,
        "target_descent_retention_fraction":
            target_retention,
        "strict_target_descent": strict_target_descent,
        "numerically_zero_step": numerically_zero,
    }

    return cls, diagnostics


# =============================================================================
# Plotting
# =============================================================================

def plot_retention(rows: List[dict], path: Path) -> None:
    active = [
        r for r in rows
        if r["status"] == "ACTIVE_AUDIT"
    ]

    if not active:
        return

    seeds = [int(r["seed"]) for r in active]
    vals = [
        float(r["target_descent_retention_fraction"])
        for r in active
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar([str(s) for s in seeds], vals)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Retained target-descent fraction")
    ax.set_title(
        "How much REFLECT target descent survives joint-cone projection?"
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_norm_ratio(rows: List[dict], path: Path) -> None:
    active = [
        r for r in rows
        if r["status"] == "ACTIVE_AUDIT"
    ]

    if not active:
        return

    seeds = [int(r["seed"]) for r in active]
    vals = [
        float(r["projected_over_reflected_norm"])
        for r in active
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar([str(s) for s in seeds], vals)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Seed")
    ax.set_ylabel("||d_cone|| / ||Delta_REFLECT||")
    ax.set_title("Magnitude retained after joint-cone projection")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    root = Path(__file__).resolve().parent

    stage3_script = Path(args.stage3_script)
    if not stage3_script.is_absolute():
        stage3_script = root / stage3_script

    stage5_dir = Path(args.stage5_dir)
    if not stage5_dir.is_absolute():
        stage5_dir = root / stage5_dir

    stage9_script = Path(args.stage9_script)
    if not stage9_script.is_absolute():
        stage9_script = root / stage9_script

    stage9_dir = Path(args.stage9_dir)
    if not stage9_dir.is_absolute():
        stage9_dir = root / stage9_dir

    stage10_script = Path(args.stage10_script)
    if not stage10_script.is_absolute():
        stage10_script = root / stage10_script

    stage10_dir = Path(args.stage10_dir)
    if not stage10_dir.is_absolute():
        stage10_dir = root / stage10_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage5_dir=stage5_dir,
        stage9_script=stage9_script,
        stage9_dir=stage9_dir,
        stage10_script=stage10_script,
        stage10_dir=stage10_dir,
    )

    stage9 = load_module(
        stage9_script,
        "vpinn_stage9_replay_stage11",
    )

    stage3 = stage9.load_stage3_module(stage3_script)

    probe_ref = probe_reference_map(pf["probe_rows"])

    precommitment = {
        "stage":
            "earliest_failure_joint_total_target_descent_cone_projection",
        "source_direction":
            "Stage-9 REFLECT displacement at each seed's earliest "
            "Stage-10 DIRECTION_CONFLICT state",
        "constraints": [
            "<grad total VPINN loss, d> <= 0",
            "<grad target R9^2/M loss, d> <= 0",
        ],
        "projection":
            "exact minimum-Euclidean-distance two-halfspace KKT projection",
        "line_scan_alphas": list(ALPHAS),
        "active_seed_denominator":
            "only seeds with Stage-10 earliest DIRECTION_CONFLICT",
        "classification": {
            "FULL_SAFE_TARGET_DESCENT":
                "strict target descent retained and alpha=1 jointly safe",
            "DAMPED_SAFE_TARGET_DESCENT":
                "strict target descent retained; some alpha<1 jointly safe",
            "TARGET_STALLED":
                "joint cone feasible but target first-order progress lost",
            "STRICT_DESCENT_BUT_NO_SAFE_ALPHA":
                "strict target descent retained but no scanned alpha safe",
            "DEGENERATE_ZERO_STEP":
                "projection collapses numerically to zero",
            "KKT_FAILURE":
                "projection verification fails",
        },
        "decision_routes": {
            "full_safe_in_at_least_3_active_seeds":
                "bounded_full_step_joint_cone_continuation_pilot",
            "any_safe_target_descent_in_at_least_3_active_seeds":
                "backtracking_joint_cone_continuation_pilot",
            "target_stalled_in_at_least_3_active_seeds":
                "strict_margin_joint_descent_qp_audit",
            "otherwise":
                "mixed_pareto_geometry_audit",
        },
        "audit_only": True,
    }

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),
        "stage3_solver_sha256": pf["stage3_sha256"],
        "stage9_script_sha256": pf["stage9_sha256"],
        "stage10_script_sha256": pf["stage10_sha256"],
        "stage10_dir": str(stage10_dir),
        "stage11_script_sha256": sha256_file(Path(__file__).resolve()),
        "precommitment": precommitment,
    }
    write_json(out_dir / "manifest.json", manifest)

    print("=" * 138)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 11 EARLIEST-FAILURE JOINT DESCENT-CONE AUDIT"
    )
    print("=" * 138)
    print(f"device                    : {device}")
    print(f"active failure seeds      : {pf['active_targets']}")
    print(
        "projection source         : failed Stage-9 REFLECT displacement"
    )
    print(
        "cone                      : <grad L,d><=0 AND <grad T,d><=0"
    )
    print(f"line-scan alphas          : {list(ALPHAS)}")
    print(f"Stage-3 SHA256            : {pf['stage3_sha256']}")
    print(f"Stage-9 SHA256            : {pf['stage9_sha256']}")
    print(f"Stage-10 SHA256           : {pf['stage10_sha256']}")
    print("=" * 138)

    summary_rows: List[dict] = []
    scan_rows: List[dict] = []
    replay_checks: List[dict] = []

    start = time.perf_counter()

    for seed in pf["seeds"]:
        if seed not in pf["active_targets"]:
            summary_rows.append(
                {
                    "seed": seed,
                    "status": "NO_TRIGGER",
                    "earliest_failure_epoch": -1,
                    "classification": "NO_TRIGGER",
                }
            )

            print()
            print(
                f"SEED {seed}: NO_TRIGGER "
                "(Stage 10 found no active early REFLECT failure)"
            )
            continue

        failure_epoch = pf["active_targets"][seed]

        seed_dir = out_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        exp = stage9.make_experiment(
            stage3=stage3,
            device=device,
            seed=seed,
            out_dir=seed_dir,
        )

        stage9.load_locked_checkpoint(
            exp=exp,
            checkpoint_path=checkpoint,
            expected_seed=seed,
        )

        # Replay the exact failed REFLECT trajectory to the earliest failure.
        for _epoch in range(2500, failure_epoch):
            stage9.intervention_step(
                exp=exp,
                branch="REFLECT",
                target_mode=TARGET_MODE,
            )

        geo = current_geometry(exp)

        if not geo["reflect_active"]:
            raise RuntimeError(
                f"Seed {seed} earliest failure state is no longer REFLECT-active."
            )

        reproduction = verify_against_stage10(
            seed=seed,
            epoch=failure_epoch,
            geo=geo,
            ref_map=probe_ref,
            tolerance=1.0e-10,
        )

        replay_checks.append(reproduction)

        if not reproduction["pass"]:
            raise RuntimeError(
                f"Stage-10 earliest-failure reproduction failed: "
                f"seed={seed}, epoch={failure_epoch}, "
                f"max diff={reproduction['max_abs_difference']:.3e}"
            )

        projection = project_two_descent_halfspaces(
            source=geo["reflected"],
            g_total=geo["g_total"],
            g_target=geo["g_target"],
        )

        line_rows, scan_summary = line_scan(
            exp=exp,
            params=geo["params"],
            d=projection["direction"],
        )

        cls, diag = classify_projection(
            projection=projection,
            reflected=geo["reflected"],
            g_target=geo["g_target"],
            scan_summary=scan_summary,
        )

        summary = {
            "seed": seed,
            "status": "ACTIVE_AUDIT",
            "earliest_failure_epoch": failure_epoch,
            "classification": cls,

            "reflect_total_directional_derivative":
                geo["reflect_total_dot"],
            "reflect_target_directional_derivative":
                geo["reflect_target_dot"],

            "cone_active_set":
                projection["active_set"],
            "cone_lambda_total":
                projection["lambda_total"],
            "cone_lambda_target":
                projection["lambda_target"],
            "cone_total_directional_derivative":
                projection["projected_total_dot"],
            "cone_target_directional_derivative":
                projection["projected_target_dot"],

            "cone_kkt_pass":
                projection["kkt_pass"],
            "cone_stationarity_relative_residual":
                projection["stationarity_relative_residual"],
            "cone_complementarity_total":
                projection["complementarity_total"],
            "cone_complementarity_target":
                projection["complementarity_target"],

            **diag,
            **scan_summary,
        }

        summary_rows.append(summary)

        for row in line_rows:
            scan_rows.append(
                {
                    "seed": seed,
                    "failure_epoch": failure_epoch,
                    "classification": cls,
                    "cone_active_set":
                        projection["active_set"],
                    **row,
                }
            )

        print()
        print("-" * 138)
        print(
            f"SEED {seed} @ {failure_epoch} | "
            f"class={cls}"
        )
        print(
            f"REFLECT: dL={geo['reflect_total_dot']:+.6e}, "
            f"dT={geo['reflect_target_dot']:+.6e}"
        )
        print(
            f"CONE   : dL={projection['projected_total_dot']:+.6e}, "
            f"dT={projection['projected_target_dot']:+.6e}, "
            f"active={projection['active_set']}"
        )
        print(
            f"norm retained={diag['projected_over_reflected_norm']:.6f}, "
            f"target descent retained="
            f"{diag['target_descent_retention_fraction']:.6f}"
        )
        print(
            f"max jointly safe alpha={scan_summary['max_joint_safe_alpha']:.6f}, "
            f"max strict-target safe alpha="
            f"{scan_summary['max_strict_target_safe_alpha']:.6f}"
        )
        print(
            f"KKT={'PASS' if projection['kkt_pass'] else 'FAIL'} | "
            f"replay gap={reproduction['max_abs_difference']:.3e}"
        )

    write_csv(out_dir / "seed_summary.csv", summary_rows)
    write_csv(out_dir / "line_scan_metrics.csv", scan_rows)
    write_csv(out_dir / "stage10_reproduction_checks.csv", replay_checks)

    active = [
        r for r in summary_rows
        if r["status"] == "ACTIVE_AUDIT"
    ]

    n_active = len(active)

    full_safe = sum(
        int(r["classification"] == "FULL_SAFE_TARGET_DESCENT")
        for r in active
    )

    damped_safe = sum(
        int(r["classification"] == "DAMPED_SAFE_TARGET_DESCENT")
        for r in active
    )

    target_stalled = sum(
        int(r["classification"] == "TARGET_STALLED")
        for r in active
    )

    strict_no_safe = sum(
        int(
            r["classification"]
            == "STRICT_DESCENT_BUT_NO_SAFE_ALPHA"
        )
        for r in active
    )

    degenerate = sum(
        int(r["classification"] == "DEGENERATE_ZERO_STEP")
        for r in active
    )

    kkt_fail = sum(
        int(r["classification"] == "KKT_FAILURE")
        for r in active
    )

    safe_target_descent = full_safe + damped_safe

    group_need = max(1, math.ceil(0.75 * n_active))

    if full_safe >= group_need:
        route_class = "full_step_joint_cone_locally_viable"
        next_route = (
            "bounded_full_step_joint_cone_continuation_pilot"
        )
    elif safe_target_descent >= group_need:
        route_class = "damped_joint_cone_locally_viable"
        next_route = (
            "backtracking_joint_cone_continuation_pilot"
        )
    elif target_stalled >= group_need:
        route_class = "zero_margin_cone_sacrifices_target_progress"
        next_route = "strict_margin_joint_descent_qp_audit"
    else:
        route_class = "mixed_or_unresolved_joint_cone_geometry"
        next_route = "mixed_pareto_geometry_audit"

    retention_vals = [
        float(r["target_descent_retention_fraction"])
        for r in active
        if np.isfinite(
            float(r["target_descent_retention_fraction"])
        )
    ]

    norm_vals = [
        float(r["projected_over_reflected_norm"])
        for r in active
    ]

    decision = {
        "n_active_seeds": n_active,
        "active_seeds": [
            int(r["seed"]) for r in active
        ],
        "group_required_count": group_need,

        "full_safe_target_descent_count": full_safe,
        "damped_safe_target_descent_count": damped_safe,
        "safe_target_descent_total_count":
            safe_target_descent,
        "target_stalled_count": target_stalled,
        "strict_descent_but_no_safe_alpha_count":
            strict_no_safe,
        "degenerate_zero_step_count": degenerate,
        "kkt_failure_count": kkt_fail,

        "all_stage10_reproductions_pass":
            all(bool(r["pass"]) for r in replay_checks),
        "all_cone_kkt_checks_pass":
            all(bool(r["cone_kkt_pass"]) for r in active),

        "median_target_descent_retention_fraction": (
            float(np.median(retention_vals))
            if retention_vals else None
        ),
        "median_projected_over_reflected_norm": (
            float(np.median(norm_vals))
            if norm_vals else None
        ),

        "route_class": route_class,
        "next_route": next_route,

        "interpretation_guardrail": (
            "Stage 11 tests only local geometric feasibility at the earliest "
            "Stage-10 direction-conflict states. A locally safe cone-projected "
            "direction does not establish accelerated VPINN escape; only the "
            "authorized bounded continuation stage may test that."
        ),
    }

    write_json(out_dir / "decision.json", decision)

    plot_retention(
        summary_rows,
        out_dir / "target_descent_retention.png",
    )

    plot_norm_ratio(
        summary_rows,
        out_dir / "projected_norm_ratio.png",
    )

    elapsed = time.perf_counter() - start

    lines = []
    lines.append("=" * 150)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 11 JOINT DESCENT-CONE SUMMARY"
    )
    lines.append("=" * 150)
    lines.append(
        "seed | epoch | class                            | active set     | "
        "norm keep | target keep | max safe a | max strict a | KKT"
    )
    lines.append("-" * 150)

    for r in summary_rows:
        if r["status"] != "ACTIVE_AUDIT":
            lines.append(
                f"{int(r['seed']):4d} | "
                f"{-1:5d} | "
                f"{'NO_TRIGGER':32s} | "
                f"{'-':14s} | "
                f"{'-':9s} | "
                f"{'-':11s} | "
                f"{'-':10s} | "
                f"{'-':12s} | "
                f"{'-'}"
            )
            continue

        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['earliest_failure_epoch']):5d} | "
            f"{str(r['classification']):32s} | "
            f"{str(r['cone_active_set']):14s} | "
            f"{float(r['projected_over_reflected_norm']):9.5f} | "
            f"{float(r['target_descent_retention_fraction']):11.5f} | "
            f"{float(r['max_joint_safe_alpha']):10.5f} | "
            f"{float(r['max_strict_target_safe_alpha']):12.5f} | "
            f"{'PASS' if r['cone_kkt_pass'] else 'FAIL'}"
        )

    lines.append("-" * 150)
    lines.append(
        f"active direction-conflict seeds      : {n_active}"
    )
    lines.append(
        f"group requirement                    : {group_need}/{n_active}"
    )
    lines.append(
        f"FULL_SAFE_TARGET_DESCENT             : {full_safe}/{n_active}"
    )
    lines.append(
        f"DAMPED_SAFE_TARGET_DESCENT           : {damped_safe}/{n_active}"
    )
    lines.append(
        f"TARGET_STALLED                       : {target_stalled}/{n_active}"
    )
    lines.append(
        f"STRICT_DESCENT_BUT_NO_SAFE_ALPHA     : {strict_no_safe}/{n_active}"
    )
    lines.append(
        f"all Stage-10 reproductions           : "
        f"{'PASS' if decision['all_stage10_reproductions_pass'] else 'FAIL'}"
    )
    lines.append(
        f"all cone KKT checks                  : "
        f"{'PASS' if decision['all_cone_kkt_checks_pass'] else 'FAIL'}"
    )
    lines.append(
        f"median target-descent retention      : "
        f"{decision['median_target_descent_retention_fraction']}"
    )
    lines.append(
        f"median projected/reflected norm      : "
        f"{decision['median_projected_over_reflected_norm']}"
    )
    lines.append(
        f"route class                          : {route_class}"
    )
    lines.append(
        f"next route                           : {next_route}"
    )
    lines.append(
        f"elapsed seconds                      : {elapsed:.2f}"
    )
    lines.append("=" * 150)
    lines.append(
        "Guardrail: do not run a continuation unless the earliest-failure "
        "projection retains target descent and is exactly jointly safe."
    )
    lines.append("=" * 150)

    summary_text = "\n".join(lines)

    print()
    print(summary_text)

    (out_dir / "console_summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
