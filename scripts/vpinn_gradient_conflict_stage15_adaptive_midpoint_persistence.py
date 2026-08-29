#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 15
Bounded State-Adaptive Pareto-Midpoint Persistence Pilot
========================================================

Scientific purpose
------------------
Stage 14 established a strong local result at the three Stage-13 failure states:

    3/3 ADAPTIVE_FULL_SAFE

when lambda was updated from the CURRENT Pareto geometry,

    lambda_mid(theta)
        = 0.5 * [lambda_low(theta) + lambda_high(theta)],

where the interval is the strict first-order joint-descent set for

    d(lambda)
        = (1-lambda) Delta_Adam
          + lambda Delta_REFLECT.

That result is still LOCAL. Stage 15 asks whether the SAME deterministic
state-adaptive rule remains valid along a bounded trajectory.

Active seeds and branch points
------------------------------
Use all four seeds that entered the Stage-10 direction-conflict mechanism:

    seed 0 : branch at epoch 2505
    seed 1 : branch at epoch 2505
    seed 2 : branch at epoch 2505
    seed 3 : branch at epoch 2510

Seed 4 is NO_TRIGGER and is excluded from the active denominator.

Each branch state is reconstructed by exact Stage-9 REFLECT replay from the
saved Stage-5 epoch-2500 checkpoint.

State-consistent adaptive step
------------------------------
At every optimization step:

  1) compute the current target gradient
         g_T = grad(R_9^2 / M);

  2) compute the current total VPINN gradient
         g_L = grad(mean_k R_k^2);

  3) using the CURRENT Adam state and g_L, predict the exact next inherited
     Adam parameter displacement Delta_A;

  4) if
         <g_T, Delta_A> <= 0,
     Adam is already target-nonuphill and NO intervention is applied;

  5) otherwise construct the exact target reflection
         Delta_R
           = Delta_A
             - 2 <g_T,Delta_A>/||g_T||^2 g_T;

  6) analytically compute the CURRENT strict joint-descent interval

         I(theta)
           = {lambda in [0,1]:
                <g_L,d(lambda)> < 0
                and
                <g_T,d(lambda)> < 0};

  7) if I(theta) has nonzero width, choose ONLY

         lambda_mid(theta)
           = midpoint(I(theta));

     exact post-step losses are NEVER used to select lambda;

  8) let real PyTorch Adam perform its ordinary step so that exp_avg,
     exp_avg_sq, and step counter evolve normally;

  9) verify the actual Adam displacement agrees with the predicted one;

 10) overwrite ONLY the model parameter displacement with d(lambda_mid),
     preserving the newly updated Adam optimizer state.

No reset.
No fixed lambda.
No backtracking.
No line-search rescue.
No post-hoc lambda selection.

No-rescue failure gates
-----------------------
NO_STRICT_INTERVAL:
    the current strict joint-descent lambda interval is empty.

DEGENERATE_STEP:
    ||d_mid|| / ||Delta_Adam|| < 0.5.

NONLINEAR_SAFETY_FAILURE:
    at an ACTIVE intervention step, first-order joint descent is valid but the
    exact nonlinear full step increases total or target loss beyond

        1e-12 * max(1, |pre_loss|).

The run stops immediately on any of these events.

Bounded horizon
---------------
Continue only through epoch 2700.

This is intentionally before the earliest validated Stage-5 certified escape
onset (2800). The objective is persistence, not yet escape-time acceleration.

Primary promotion gate
----------------------
Active denominator = 4.

PERSISTENT_SAFE:
    reaches epoch 2700 without any no-rescue failure.

NET_JOINT_PROGRESS:
    at epoch 2700,

        total_loss(2700) < total_loss(branch_start)
        target_loss(2700) < target_loss(branch_start).

Authorize Stage 16 escape-time comparison only if

    >= 3/4 PERSISTENT_SAFE
    AND
    >= 3/4 NET_JOINT_PROGRESS.

Secondary diagnostics
---------------------
At epoch 2700 compare against:
  * validated Stage-5 ordinary Adam;
  * validated Stage-9 full REFLECT.

Also record:
  * lambda trajectory;
  * current interval width/margin;
  * active/inactive intervention fraction;
  * applied step norm relative to inherited Adam;
  * exact per-step total/target loss changes;
  * fraction of active steps with STRICT exact decrease of both objectives.

Interpretation guardrail
------------------------
A Stage-15 PASS establishes bounded persistence of the adaptive Pareto rule.
It does NOT establish faster escape. Only Stage 16 may make an escape-time
comparison.
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
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


TARGET_MODE = 9
END_EPOCH = 2700
TRACK_INTERVAL = 25
DEGENERACY_FLOOR = 0.5
NONLINEAR_TOL_FACTOR = 1.0e-12
ADAM_FORMULA_TOL = 5.0e-12


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-15 bounded state-adaptive Pareto midpoint pilot."
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
        "--stage10-dir",
        default="vpinn_gradient_conflict_stage10_local_feasibility_audit",
    )

    p.add_argument(
        "--stage12-dir",
        default="vpinn_gradient_conflict_stage12_common_pareto_blend_audit",
    )

    p.add_argument(
        "--stage13-dir",
        default="vpinn_gradient_conflict_stage13_fixed_common_blend_pilot",
    )

    p.add_argument(
        "--stage14-script",
        default="vpinn_gradient_conflict_stage14_adaptive_midpoint_failure_audit.py",
    )

    p.add_argument(
        "--stage14-dir",
        default="vpinn_gradient_conflict_stage14_adaptive_midpoint_failure_audit",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage15_adaptive_midpoint_persistence",
    )

    return p.parse_args()


# =============================================================================
# Generic utilities
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
        for key in row.keys():
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
    return torch.cat([x.reshape(-1) for x in parts], dim=0)


# =============================================================================
# Preflight
# =============================================================================

def preflight(
    stage3_script: Path,
    stage5_dir: Path,
    stage9_script: Path,
    stage9_dir: Path,
    stage10_dir: Path,
    stage12_dir: Path,
    stage13_dir: Path,
    stage14_script: Path,
    stage14_dir: Path,
) -> dict:

    paths = {
        "s5_manifest": stage5_dir / "manifest.json",
        "s5_aggregate": stage5_dir / "aggregate_postlock_metrics.csv",

        "s9_manifest": stage9_dir / "manifest.json",
        "s9_aggregate": stage9_dir / "aggregate_trajectories.csv",

        "s10_seed_summary": stage10_dir / "seed_summary.csv",

        "s12_manifest": stage12_dir / "manifest.json",
        "s12_seed_summary": stage12_dir / "seed_summary.csv",

        "s13_manifest": stage13_dir / "manifest.json",
        "s13_decision": stage13_dir / "decision.json",

        "s14_manifest": stage14_dir / "manifest.json",
        "s14_decision": stage14_dir / "decision.json",
        "s14_reproduction":
            stage14_dir / "stage13_failure_state_reproduction.json",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    s5_manifest = read_json(paths["s5_manifest"])
    s9_manifest = read_json(paths["s9_manifest"])
    s12_manifest = read_json(paths["s12_manifest"])
    s13_manifest = read_json(paths["s13_manifest"])
    s13_decision = read_json(paths["s13_decision"])
    s14_manifest = read_json(paths["s14_manifest"])
    s14_decision = read_json(paths["s14_decision"])
    s14_repro = read_json(paths["s14_reproduction"])

    if s14_decision.get("next_route") != (
        "bounded_state_adaptive_midpoint_persistence_pilot"
    ):
        raise RuntimeError(
            "Stage 14 did not authorize the bounded adaptive-midpoint pilot."
        )

    if s14_decision.get("route_class") != (
        "adaptive_midpoint_full_step_rescues_all_failures"
    ):
        raise RuntimeError(
            "Stage-14 route class is not full-step adaptive rescue."
        )

    if int(s14_decision.get("adaptive_full_safe_count", -1)) != 3:
        raise RuntimeError("Stage 14 did not report 3/3 adaptive full-safe.")

    if not bool(
        s14_decision.get("all_failure_state_reproductions_pass", False)
    ):
        raise RuntimeError(
            "Stage-14 failure-state reproductions are not all PASS."
        )

    if not bool(s14_repro.get("all_pass", False)):
        raise RuntimeError(
            "Stage-14 reproduction file is not all PASS."
        )

    # Stage-13 failure pattern is a prerequisite for interpreting Stage 14.
    if bool(s13_decision.get("promotion_gate_pass", True)):
        raise RuntimeError("Stage 13 unexpectedly passed.")

    if int(s13_decision.get("geometry_failure_count", -1)) != 1:
        raise RuntimeError("Unexpected Stage-13 geometry failure count.")

    if int(s13_decision.get("nonlinear_safety_failure_count", -1)) != 2:
        raise RuntimeError("Unexpected Stage-13 nonlinear failure count.")

    actual_s3_sha = sha256_file(stage3_script)
    actual_s9_sha = sha256_file(stage9_script)
    actual_s14_sha = sha256_file(stage14_script)

    if s5_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 5.")

    if s9_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 9.")

    if s12_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 12.")

    if s13_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 13.")

    if s14_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 14.")

    if s14_manifest.get("stage9_script_sha256") != actual_s9_sha:
        raise RuntimeError("Stage-9 SHA mismatch against Stage 14.")

    if s14_manifest.get("stage14_script_sha256") != actual_s14_sha:
        raise RuntimeError(
            "Stage-14 source SHA mismatch against its manifest."
        )

    # Active branch points come from Stage 10 and must remain frozen.
    s10_summary = read_csv(paths["s10_seed_summary"])

    branch_points = {}

    for row in s10_summary:
        seed = int(row["seed"])
        epoch = int(row["earliest_active_non_SAFE_FULL_epoch"])

        if epoch >= 0:
            if row["earliest_failure_class"] != "DIRECTION_CONFLICT":
                raise RuntimeError(
                    f"Seed {seed} Stage-10 earliest failure is not "
                    "DIRECTION_CONFLICT."
                )

            branch_points[seed] = epoch

    expected = {
        0: 2505,
        1: 2505,
        2: 2505,
        3: 2510,
    }

    if branch_points != expected:
        raise RuntimeError(
            f"Unexpected active branch points: {branch_points}"
        )

    # Stage-12 local interval references at the branch states.
    s12_summary = read_csv(paths["s12_seed_summary"])

    interval_reference = {}

    for row in s12_summary:
        if row["status"] == "ACTIVE_AUDIT":
            seed = int(row["seed"])

            interval_reference[seed] = {
                "lower": float(row["lambda_interval_lower"]),
                "upper": float(row["lambda_interval_upper"]),
            }

    if set(interval_reference) != set(branch_points):
        raise RuntimeError(
            "Stage-12 local interval references are incomplete."
        )

    for seed in range(5):
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing locked checkpoint: {checkpoint}"
            )

    return {
        "branch_points": branch_points,
        "interval_reference": interval_reference,

        "stage3_sha256": actual_s3_sha,
        "stage9_sha256": actual_s9_sha,
        "stage14_sha256": actual_s14_sha,

        "stage5_aggregate":
            read_csv(paths["s5_aggregate"]),

        "stage9_aggregate":
            read_csv(paths["s9_aggregate"]),
    }


# =============================================================================
# Historical lookup maps
# =============================================================================

def baseline_map(rows: List[dict]) -> dict:
    return {
        (int(r["seed"]), int(r["epoch"])): r
        for r in rows
    }


def reflect_map(rows: List[dict]) -> dict:
    return {
        (int(r["seed"]), str(r["branch"]), int(r["epoch"])): r
        for r in rows
    }


# =============================================================================
# State metrics
# =============================================================================

def state_metrics(exp, target_mode: int = TARGET_MODE) -> dict:
    residuals = exp.weak_residuals().detach()
    energy = residuals.square()

    total_energy = energy.sum().clamp_min(1.0e-300)
    t = target_mode - 1

    dominant = int(torch.argmax(energy).item())

    return {
        "relative_l2_error":
            exp.relative_l2_error(),

        "vpinn_loss":
            float(torch.mean(energy).item()),

        "target_loss":
            float((energy[t] / energy.numel()).item()),

        "target_mode_abs_residual":
            float(torch.abs(residuals[t]).item()),

        "target_mode_residual_energy_share":
            float((energy[t] / total_energy).item()),

        "dominant_residual_mode":
            dominant + 1,
    }


# =============================================================================
# Strict interval algebra
# =============================================================================

def strict_negative_interval(
    f0: float,
    f1: float,
    tol: float = 1.0e-15,
) -> Tuple[float, float]:

    slope = f1 - f0

    if abs(slope) <= tol * max(1.0, abs(f0), abs(f1)):
        if f0 < 0.0:
            return 0.0, 1.0

        return math.inf, -math.inf

    root = -f0 / slope

    if slope > 0.0:
        return 0.0, min(1.0, root)

    return max(0.0, root), 1.0


def intersect_open_intervals(
    intervals: List[Tuple[float, float]],
) -> Tuple[float, float]:

    return (
        max(x[0] for x in intervals),
        min(x[1] for x in intervals),
    )


# =============================================================================
# Exact Adam candidate prediction
# =============================================================================

def predict_adam_candidate(
    exp,
    params,
    g_total: torch.Tensor,
) -> torch.Tensor:

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

    parts = []
    offset = 0

    for p in params:
        n = p.numel()

        gp = g_total[offset:offset+n].reshape_as(p)
        offset += n

        state = exp.optimizer.state[p]

        if "exp_avg" not in state or "exp_avg_sq" not in state:
            raise RuntimeError("Adam moment state is missing.")

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

        denom = (
            v_new.sqrt() / math.sqrt(bc2)
        ).add(eps)

        delta = -lr / bc1 * m_new / denom

        parts.append(delta.reshape(-1))

    return flatten(parts)


# =============================================================================
# One state-adaptive step
# =============================================================================

def adaptive_midpoint_step(
    exp,
    target_mode: int = TARGET_MODE,
) -> dict:

    exp.optimizer.zero_grad(set_to_none=True)

    residuals = exp.weak_residuals()

    params = tuple(
        p for p in exp.model.parameters()
        if p.requires_grad
    )

    t = target_mode - 1
    M = residuals.numel()

    target_loss = residuals[t].square() / M
    total_loss = residuals.square().mean()

    pre_total = float(total_loss.detach().item())
    pre_target = float(target_loss.detach().item())

    # Target gradient for Pareto geometry.
    gt_parts = torch.autograd.grad(
        target_loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )

    gt = flatten(gt_parts).detach()

    # Total raw gradient is exactly what Adam consumes.
    total_loss.backward()

    gL = flatten([
        p.grad.detach().clone()
        for p in params
    ])

    # Predict the exact inherited Adam candidate BEFORE mutating optimizer state.
    predicted_candidate = predict_adam_candidate(
        exp=exp,
        params=params,
        g_total=gL,
    )

    candidate_total_dot = float(
        torch.dot(gL, predicted_candidate).item()
    )

    candidate_target_dot = float(
        torch.dot(gt, predicted_candidate).item()
    )

    candidate_norm = float(
        torch.linalg.vector_norm(predicted_candidate).item()
    )

    active = bool(candidate_target_dot > 0.0)

    interval_lo = None
    interval_hi = None
    interval_width = None
    interval_margin = None
    lambda_mid = None

    reflect_total_dot = None
    reflect_target_dot = None

    if active:
        gt2 = torch.dot(gt, gt)

        if float(gt2.item()) <= 0.0:
            raise RuntimeError("Target gradient norm is zero.")

        component = (
            torch.dot(gt, predicted_candidate) / gt2
        ) * gt

        reflected = predicted_candidate - 2.0 * component

        reflect_total_dot = float(
            torch.dot(gL, reflected).item()
        )

        reflect_target_dot = float(
            torch.dot(gt, reflected).item()
        )

        interval_L = strict_negative_interval(
            candidate_total_dot,
            reflect_total_dot,
        )

        interval_T = strict_negative_interval(
            candidate_target_dot,
            reflect_target_dot,
        )

        interval_lo, interval_hi = intersect_open_intervals(
            [interval_L, interval_T]
        )

        if not (
            math.isfinite(interval_lo)
            and math.isfinite(interval_hi)
            and interval_lo < interval_hi
        ):
            return {
                "status": "NO_STRICT_INTERVAL",

                "intervention_active": True,

                "candidate_total_dot":
                    candidate_total_dot,
                "candidate_target_dot":
                    candidate_target_dot,

                "reflect_total_dot":
                    reflect_total_dot,
                "reflect_target_dot":
                    reflect_target_dot,

                "interval_lower":
                    interval_lo,
                "interval_upper":
                    interval_hi,

                "candidate_norm":
                    candidate_norm,

                "pre_total_loss":
                    pre_total,
                "pre_target_loss":
                    pre_target,
            }

        lambda_mid = 0.5 * (interval_lo + interval_hi)

        interval_width = interval_hi - interval_lo

        interval_margin = min(
            lambda_mid - interval_lo,
            interval_hi - lambda_mid,
        )

        applied = (
            (1.0 - lambda_mid) * predicted_candidate
            + lambda_mid * reflected
        )

        applied_total_dot = float(
            torch.dot(gL, applied).item()
        )

        applied_target_dot = float(
            torch.dot(gt, applied).item()
        )

        if not (
            applied_total_dot < 0.0
            and applied_target_dot < 0.0
        ):
            raise RuntimeError(
                "Adaptive midpoint is not strict first-order joint descent."
            )

    else:
        reflected = predicted_candidate
        applied = predicted_candidate

        reflect_total_dot = candidate_total_dot
        reflect_target_dot = candidate_target_dot

        applied_total_dot = candidate_total_dot
        applied_target_dot = candidate_target_dot

    applied_norm = float(
        torch.linalg.vector_norm(applied).item()
    )

    applied_over_candidate = (
        applied_norm / max(candidate_norm, 1.0e-300)
    )

    if active and applied_over_candidate < DEGENERACY_FLOOR:
        return {
            "status": "DEGENERATE_STEP",

            "intervention_active": True,

            "lambda_mid":
                lambda_mid,

            "interval_lower":
                interval_lo,
            "interval_upper":
                interval_hi,
            "interval_width":
                interval_width,
            "interval_margin":
                interval_margin,

            "candidate_total_dot":
                candidate_total_dot,
            "candidate_target_dot":
                candidate_target_dot,

            "reflect_total_dot":
                reflect_total_dot,
            "reflect_target_dot":
                reflect_target_dot,

            "applied_total_dot":
                applied_total_dot,
            "applied_target_dot":
                applied_target_dot,

            "candidate_norm":
                candidate_norm,
            "applied_norm":
                applied_norm,
            "applied_over_candidate_norm":
                applied_over_candidate,

            "pre_total_loss":
                pre_total,
            "pre_target_loss":
                pre_target,
        }

    # Snapshot the pre-step parameters.
    before_parts = [
        p.detach().clone()
        for p in params
    ]

    before = flatten(before_parts)

    # Real Adam step updates both parameters and optimizer state.
    exp.optimizer.step()

    actual_candidate = flatten([
        p.detach().clone()
        for p in params
    ]) - before

    formula_max_abs_diff = float(
        torch.max(
            torch.abs(
                actual_candidate - predicted_candidate
            )
        ).item()
    )

    formula_rel_diff = float(
        (
            torch.linalg.vector_norm(
                actual_candidate - predicted_candidate
            )
            /
            torch.clamp(
                torch.linalg.vector_norm(actual_candidate),
                min=1.0e-300,
            )
        ).item()
    )

    if formula_max_abs_diff > ADAM_FORMULA_TOL:
        raise RuntimeError(
            "Predicted-vs-runtime Adam candidate mismatch: "
            f"{formula_max_abs_diff:.3e}"
        )

    # Overwrite ONLY model displacement. Adam state remains the real state
    # created by optimizer.step().
    offset = 0

    with torch.no_grad():
        for p, p0 in zip(params, before_parts):
            n = p.numel()

            p.copy_(
                p0
                + applied[offset:offset+n].reshape_as(p)
            )

            offset += n

    if offset != applied.numel():
        raise RuntimeError("Applied displacement size mismatch.")

    # Exact nonlinear post-step losses.
    post_residuals = exp.weak_residuals().detach()
    post_energy = post_residuals.square()

    post_total = float(
        torch.mean(post_energy).item()
    )

    post_target = float(
        (post_energy[t] / post_energy.numel()).item()
    )

    total_change = post_total - pre_total
    target_change = post_target - pre_target

    tol_total = (
        NONLINEAR_TOL_FACTOR
        * max(1.0, abs(pre_total))
    )

    tol_target = (
        NONLINEAR_TOL_FACTOR
        * max(1.0, abs(pre_target))
    )

    nonlinear_failure = bool(
        active
        and (
            total_change > tol_total
            or target_change > tol_target
        )
    )

    strict_joint_exact = bool(
        total_change < 0.0
        and target_change < 0.0
    )

    return {
        "status": (
            "NONLINEAR_SAFETY_FAILURE"
            if nonlinear_failure
            else "OK"
        ),

        "intervention_active":
            active,

        "lambda_mid":
            lambda_mid,

        "interval_lower":
            interval_lo,
        "interval_upper":
            interval_hi,
        "interval_width":
            interval_width,
        "interval_margin":
            interval_margin,

        "candidate_total_dot":
            candidate_total_dot,
        "candidate_target_dot":
            candidate_target_dot,

        "reflect_total_dot":
            reflect_total_dot,
        "reflect_target_dot":
            reflect_target_dot,

        "applied_total_dot":
            applied_total_dot,
        "applied_target_dot":
            applied_target_dot,

        "candidate_norm":
            candidate_norm,
        "applied_norm":
            applied_norm,
        "applied_over_candidate_norm":
            applied_over_candidate,

        "adam_formula_max_abs_diff":
            formula_max_abs_diff,
        "adam_formula_relative_diff":
            formula_rel_diff,

        "pre_total_loss":
            pre_total,
        "post_total_loss":
            post_total,
        "total_loss_change":
            total_change,

        "pre_target_loss":
            pre_target,
        "post_target_loss":
            post_target,
        "target_loss_change":
            target_change,

        "strict_joint_exact_decrease":
            strict_joint_exact,

        "nonlinear_safety_failure":
            nonlinear_failure,
    }


# =============================================================================
# First-state Stage-12 interval reproduction
# =============================================================================

def verify_branch_interval(
    seed: int,
    result: dict,
    reference: dict,
    tolerance: float = 1.0e-10,
) -> dict:

    if not bool(result["intervention_active"]):
        raise RuntimeError(
            f"Seed {seed} branch state unexpectedly intervention-inactive."
        )

    diffs = {
        "lower":
            abs(
                float(result["interval_lower"])
                - float(reference["lower"])
            ),

        "upper":
            abs(
                float(result["interval_upper"])
                - float(reference["upper"])
            ),
    }

    max_diff = max(diffs.values())

    return {
        "seed":
            seed,

        "max_abs_difference":
            max_diff,

        "field_abs_differences":
            diffs,

        "tolerance":
            tolerance,

        "pass":
            bool(max_diff <= tolerance),
    }


# =============================================================================
# Plots
# =============================================================================

def plot_lambda_trajectories(
    step_rows: List[dict],
    path: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(10, 6))

    for seed in sorted(
        set(int(r["seed"]) for r in step_rows)
    ):
        rr = [
            r for r in step_rows
            if int(r["seed"]) == seed
            and str(r["intervention_active"]).lower() == "true"
            and r.get("lambda_mid") not in (None, "")
        ]

        rr.sort(key=lambda x: int(x["epoch_after"]))

        if not rr:
            continue

        ax.plot(
            [int(r["epoch_after"]) for r in rr],
            [float(r["lambda_mid"]) for r in rr],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"seed {seed}",
        )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.0,
        label="target-neutral boundary",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Adaptive lambda midpoint")
    ax.set_title("Moving Pareto compromise during Stage 15")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_interval_widths(
    step_rows: List[dict],
    path: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(10, 6))

    for seed in sorted(
        set(int(r["seed"]) for r in step_rows)
    ):
        rr = [
            r for r in step_rows
            if int(r["seed"]) == seed
            and str(r["intervention_active"]).lower() == "true"
            and r.get("interval_width") not in (None, "")
        ]

        rr.sort(key=lambda x: int(x["epoch_after"]))

        if not rr:
            continue

        ax.plot(
            [int(r["epoch_after"]) for r in rr],
            [float(r["interval_width"]) for r in rr],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"seed {seed}",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Strict Pareto interval width")
    ax.set_title("How the feasible Pareto corridor moves and contracts")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tracking_error(
    tracking_rows: List[dict],
    path: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(10, 6))

    for seed in sorted(
        set(int(r["seed"]) for r in tracking_rows)
    ):
        rr = [
            r for r in tracking_rows
            if int(r["seed"]) == seed
        ]

        rr.sort(key=lambda x: int(x["epoch"]))

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r["relative_l2_error"]) for r in rr],
            marker="o",
            markersize=3,
            label=f"seed {seed}",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Adaptive midpoint bounded trajectories")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_comparison(
    summaries: List[dict],
    path: Path,
) -> None:

    reached = [
        r for r in summaries
        if r["status"] == "REACHED_2700_SAFE"
    ]

    if not reached:
        return

    x = np.arange(len(reached))
    width = 0.36

    seeds = [
        str(int(r["seed"]))
        for r in reached
    ]

    vs_adam = [
        float(r["relL2_ratio_vs_stage5_adam_2700"])
        for r in reached
    ]

    vs_reflect = [
        float(r["relL2_ratio_vs_stage9_reflect_2700"])
        for r in reached
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))

    ax.bar(
        x - width / 2,
        vs_adam,
        width,
        label="vs Stage-5 Adam",
    )

    ax.bar(
        x + width / 2,
        vs_reflect,
        width,
        label="vs Stage-9 REFLECT",
    )

    ax.axhline(1.0, linestyle="--", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(seeds)

    ax.set_xlabel("Seed")
    ax.set_ylabel("Relative L2 ratio at epoch 2700")
    ax.set_title("Stage-15 endpoint comparison")
    ax.legend()

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

    stage10_dir = Path(args.stage10_dir)
    if not stage10_dir.is_absolute():
        stage10_dir = root / stage10_dir

    stage12_dir = Path(args.stage12_dir)
    if not stage12_dir.is_absolute():
        stage12_dir = root / stage12_dir

    stage13_dir = Path(args.stage13_dir)
    if not stage13_dir.is_absolute():
        stage13_dir = root / stage13_dir

    stage14_script = Path(args.stage14_script)
    if not stage14_script.is_absolute():
        stage14_script = root / stage14_script

    stage14_dir = Path(args.stage14_dir)
    if not stage14_dir.is_absolute():
        stage14_dir = root / stage14_dir

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
        stage10_dir=stage10_dir,
        stage12_dir=stage12_dir,
        stage13_dir=stage13_dir,
        stage14_script=stage14_script,
        stage14_dir=stage14_dir,
    )

    stage9 = load_module(
        stage9_script,
        "vpinn_stage9_replay_stage15",
    )

    stage3 = stage9.load_stage3_module(stage3_script)

    s5_map = baseline_map(pf["stage5_aggregate"])
    s9_map = reflect_map(pf["stage9_aggregate"])

    precommitment = {
        "stage":
            "bounded_state_adaptive_pareto_midpoint_persistence",

        "active_branch_points":
            pf["branch_points"],

        "end_epoch":
            END_EPOCH,

        "adaptive_lambda_rule":
            "midpoint of CURRENT strict total/target first-order descent interval",

        "lambda_selection_uses_post_step_losses":
            False,

        "inactive_rule":
            "use inherited Adam unchanged when target-nonuphill",

        "no_rescue_failure_gates": {
            "NO_STRICT_INTERVAL":
                "current strict Pareto interval is empty",

            "DEGENERATE_STEP":
                "adaptive step norm / Adam candidate norm < 0.5",

            "NONLINEAR_SAFETY_FAILURE":
                "active full step increases total or target exact loss "
                "beyond 1e-12 relative-scale numerical tolerance",
        },

        "promotion_gate":
            ">=3/4 REACHED_2700_SAFE AND >=3/4 NET_JOINT_PROGRESS",

        "next_route_if_pass":
            "stage16_adaptive_midpoint_escape_time_comparison",

        "next_route_if_geometry_failure_dominant":
            "strict_margin_or_broader_pareto_family_audit",

        "next_route_if_nonlinear_failure_dominant":
            "adaptive_midpoint_backtracking_audit",
    }

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

        "stage9_script_sha256":
            pf["stage9_sha256"],

        "stage14_script_sha256":
            pf["stage14_sha256"],

        "stage15_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "stage14_dir":
            str(stage14_dir),

        "precommitment":
            precommitment,
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 154)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 15 STATE-ADAPTIVE MIDPOINT PERSISTENCE PILOT"
    )
    print("=" * 154)
    print(f"device                    : {device}")
    print(f"active branch points      : {pf['branch_points']}")
    print(f"bounded horizon           : through epoch {END_EPOCH}")
    print(
        "lambda rule               : midpoint of CURRENT strict Pareto interval"
    )
    print(
        "rescue policy              : NONE"
    )
    print(
        "promotion gate             : >=3/4 safe + >=3/4 net joint progress"
    )
    print("=" * 154)

    summaries: List[dict] = []
    step_rows: List[dict] = []
    tracking_rows: List[dict] = []
    branch_checks: List[dict] = []

    global_start = time.perf_counter()

    for seed in range(5):

        if seed not in pf["branch_points"]:

            summaries.append(
                {
                    "seed":
                        seed,

                    "status":
                        "NO_TRIGGER",

                    "branch_epoch":
                        -1,
                }
            )

            print()
            print(f"SEED {seed}: NO_TRIGGER")
            continue

        branch_epoch = pf["branch_points"][seed]

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

        # Exact Stage-9 REFLECT replay to the frozen branch point.
        for _epoch in range(2500, branch_epoch):

            stage9.intervention_step(
                exp=exp,
                branch="REFLECT",
                target_mode=TARGET_MODE,
            )

        branch_state = state_metrics(exp)

        tracking_rows.append(
            {
                "seed":
                    seed,

                "epoch":
                    branch_epoch,

                "event":
                    "BRANCH_START",

                **branch_state,
            }
        )

        print()
        print("-" * 154)
        print(
            f"SEED {seed} | branch={branch_epoch} | "
            f"relL2={branch_state['relative_l2_error']:.6e} | "
            f"L={branch_state['vpinn_loss']:.6e} | "
            f"T={branch_state['target_loss']:.6e}"
        )

        status = "RUNNING"
        failure_epoch = -1
        failure_type = None

        active_steps = 0
        inactive_steps = 0

        strict_exact_active_steps = 0
        tolerance_safe_active_steps = 0

        lambda_values = []
        interval_widths = []
        interval_margins = []
        norm_ratios = []

        max_adam_formula_gap = 0.0

        branch_interval_verified = False

        epoch = branch_epoch

        while epoch < END_EPOCH:

            result = adaptive_midpoint_step(
                exp=exp,
                target_mode=TARGET_MODE,
            )

            # Geometry/degeneracy failures occur BEFORE the optimizer state is
            # mutated. This makes the failure checkpoint clean.
            if result["status"] in (
                "NO_STRICT_INTERVAL",
                "DEGENERATE_STEP",
            ):
                status = result["status"]
                failure_epoch = epoch
                failure_type = result["status"]

                step_rows.append(
                    {
                        "seed":
                            seed,

                        "epoch_before":
                            epoch,

                        "epoch_after":
                            epoch,

                        **result,
                    }
                )

                print(
                    f"  STOP {status} before step at epoch {epoch}"
                )

                break

            # Verify branch-state interval against Stage 12 BEFORE accepting
            # the first adaptive trajectory step.
            if not branch_interval_verified:

                check = verify_branch_interval(
                    seed=seed,
                    result=result,
                    reference=pf["interval_reference"][seed],
                    tolerance=1.0e-10,
                )

                branch_checks.append(check)

                if not check["pass"]:
                    raise RuntimeError(
                        f"Stage-12 branch interval reproduction failed "
                        f"for seed {seed}: "
                        f"{check['max_abs_difference']:.3e}"
                    )

                branch_interval_verified = True

            epoch_before = epoch
            epoch += 1

            if result["intervention_active"]:
                active_steps += 1

                lambda_values.append(
                    float(result["lambda_mid"])
                )

                interval_widths.append(
                    float(result["interval_width"])
                )

                interval_margins.append(
                    float(result["interval_margin"])
                )

                norm_ratios.append(
                    float(
                        result[
                            "applied_over_candidate_norm"
                        ]
                    )
                )

                if result["strict_joint_exact_decrease"]:
                    strict_exact_active_steps += 1

                if not result["nonlinear_safety_failure"]:
                    tolerance_safe_active_steps += 1

            else:
                inactive_steps += 1

            max_adam_formula_gap = max(
                max_adam_formula_gap,
                float(result["adam_formula_max_abs_diff"]),
            )

            step_rows.append(
                {
                    "seed":
                        seed,

                    "epoch_before":
                        epoch_before,

                    "epoch_after":
                        epoch,

                    **result,
                }
            )

            if result["status"] == "NONLINEAR_SAFETY_FAILURE":

                status = "NONLINEAR_SAFETY_FAILURE"
                failure_epoch = epoch
                failure_type = status

                current = state_metrics(exp)

                tracking_rows.append(
                    {
                        "seed":
                            seed,

                        "epoch":
                            epoch,

                        "event":
                            "NONLINEAR_SAFETY_FAILURE",

                        **current,
                    }
                )

                print(
                    f"  STOP nonlinear failure at epoch {epoch}: "
                    f"dL={result['total_loss_change']:+.3e}, "
                    f"dT={result['target_loss_change']:+.3e}, "
                    f"lambda={result['lambda_mid']}"
                )

                break

            if (
                epoch % TRACK_INTERVAL == 0
                or epoch == END_EPOCH
            ):

                current = state_metrics(exp)

                tracking_rows.append(
                    {
                        "seed":
                            seed,

                        "epoch":
                            epoch,

                        "event":
                            "TRACK",

                        **current,
                    }
                )

        if not branch_interval_verified:
            raise RuntimeError(
                f"Seed {seed} terminated before branch interval verification."
            )

        if status == "RUNNING":
            status = "REACHED_2700_SAFE"

        if status == "REACHED_2700_SAFE":

            endpoint = state_metrics(exp)

            net_total_progress = bool(
                endpoint["vpinn_loss"]
                < branch_state["vpinn_loss"]
            )

            net_target_progress = bool(
                endpoint["target_loss"]
                < branch_state["target_loss"]
            )

            net_joint_progress = bool(
                net_total_progress
                and net_target_progress
            )

            key5 = (seed, END_EPOCH)
            key9 = (seed, "REFLECT", END_EPOCH)

            if key5 not in s5_map:
                raise RuntimeError(
                    f"Stage-5 comparator missing seed={seed}, "
                    f"epoch={END_EPOCH}."
                )

            if key9 not in s9_map:
                raise RuntimeError(
                    f"Stage-9 REFLECT comparator missing seed={seed}, "
                    f"epoch={END_EPOCH}."
                )

            hist5 = s5_map[key5]
            hist9 = s9_map[key9]

            summary = {
                "seed":
                    seed,

                "status":
                    status,

                "branch_epoch":
                    branch_epoch,

                "active_steps":
                    active_steps,

                "inactive_steps":
                    inactive_steps,

                "active_fraction":
                    active_steps
                    / max(active_steps + inactive_steps, 1),

                "strict_exact_active_steps":
                    strict_exact_active_steps,

                "strict_exact_active_fraction":
                    strict_exact_active_steps
                    / max(active_steps, 1),

                "tolerance_safe_active_steps":
                    tolerance_safe_active_steps,

                "minimum_lambda":
                    min(lambda_values)
                    if lambda_values
                    else None,

                "maximum_lambda":
                    max(lambda_values)
                    if lambda_values
                    else None,

                "median_lambda":
                    float(np.median(lambda_values))
                    if lambda_values
                    else None,

                "minimum_interval_width":
                    min(interval_widths)
                    if interval_widths
                    else None,

                "minimum_midpoint_margin":
                    min(interval_margins)
                    if interval_margins
                    else None,

                "minimum_applied_over_adam_norm":
                    min(norm_ratios)
                    if norm_ratios
                    else None,

                "median_applied_over_adam_norm":
                    float(np.median(norm_ratios))
                    if norm_ratios
                    else None,

                "max_adam_formula_gap":
                    max_adam_formula_gap,

                "branch_start_relative_l2":
                    branch_state["relative_l2_error"],

                "branch_start_total_loss":
                    branch_state["vpinn_loss"],

                "branch_start_target_loss":
                    branch_state["target_loss"],

                "endpoint_relative_l2":
                    endpoint["relative_l2_error"],

                "endpoint_total_loss":
                    endpoint["vpinn_loss"],

                "endpoint_target_loss":
                    endpoint["target_loss"],

                "endpoint_target_share":
                    endpoint[
                        "target_mode_residual_energy_share"
                    ],

                "net_total_progress":
                    net_total_progress,

                "net_target_progress":
                    net_target_progress,

                "net_joint_progress":
                    net_joint_progress,

                "relL2_ratio_vs_stage5_adam_2700":
                    endpoint["relative_l2_error"]
                    / float(hist5["relative_l2_error"]),

                "total_loss_ratio_vs_stage5_adam_2700":
                    endpoint["vpinn_loss"]
                    / float(hist5["vpinn_loss"]),

                "target_share_difference_vs_stage5_adam_2700":
                    endpoint[
                        "target_mode_residual_energy_share"
                    ]
                    - float(
                        hist5[
                            "target_mode_residual_energy_share"
                        ]
                    ),

                "relL2_ratio_vs_stage9_reflect_2700":
                    endpoint["relative_l2_error"]
                    / float(hist9["relative_l2_error"]),

                "total_loss_ratio_vs_stage9_reflect_2700":
                    endpoint["vpinn_loss"]
                    / float(hist9["vpinn_loss"]),
            }

        else:

            summary = {
                "seed":
                    seed,

                "status":
                    status,

                "branch_epoch":
                    branch_epoch,

                "failure_epoch":
                    failure_epoch,

                "failure_type":
                    failure_type,

                "active_steps":
                    active_steps,

                "inactive_steps":
                    inactive_steps,

                "active_fraction":
                    active_steps
                    / max(active_steps + inactive_steps, 1),

                "strict_exact_active_steps":
                    strict_exact_active_steps,

                "strict_exact_active_fraction":
                    strict_exact_active_steps
                    / max(active_steps, 1),

                "minimum_lambda":
                    min(lambda_values)
                    if lambda_values
                    else None,

                "maximum_lambda":
                    max(lambda_values)
                    if lambda_values
                    else None,

                "median_lambda":
                    float(np.median(lambda_values))
                    if lambda_values
                    else None,

                "minimum_interval_width":
                    min(interval_widths)
                    if interval_widths
                    else None,

                "minimum_midpoint_margin":
                    min(interval_margins)
                    if interval_margins
                    else None,

                "minimum_applied_over_adam_norm":
                    min(norm_ratios)
                    if norm_ratios
                    else None,

                "median_applied_over_adam_norm":
                    float(np.median(norm_ratios))
                    if norm_ratios
                    else None,

                "max_adam_formula_gap":
                    max_adam_formula_gap,

                "branch_start_relative_l2":
                    branch_state["relative_l2_error"],

                "branch_start_total_loss":
                    branch_state["vpinn_loss"],

                "branch_start_target_loss":
                    branch_state["target_loss"],

                "net_joint_progress":
                    False,
            }

        summaries.append(summary)

        write_json(
            seed_dir / "summary.json",
            summary,
        )

        print(
            f"  result={status} | "
            f"active={active_steps} | "
            f"inactive={inactive_steps} | "
            f"strict active={strict_exact_active_steps}/{max(active_steps,1)}"
        )

        if status == "REACHED_2700_SAFE":

            print(
                f"  endpoint relL2={summary['endpoint_relative_l2']:.6e} | "
                f"joint progress={summary['net_joint_progress']} | "
                f"relL2/Adam={summary['relL2_ratio_vs_stage5_adam_2700']:.6f} | "
                f"relL2/REFLECT={summary['relL2_ratio_vs_stage9_reflect_2700']:.6f}"
            )

            print(
                f"  lambda range=[{summary['minimum_lambda']:.9f}, "
                f"{summary['maximum_lambda']:.9f}] | "
                f"min interval width={summary['minimum_interval_width']:.6e}"
            )

    # -------------------------------------------------------------------------
    # Persist results.
    # -------------------------------------------------------------------------
    write_csv(
        out_dir / "seed_summary.csv",
        summaries,
    )

    write_csv(
        out_dir / "step_metrics.csv",
        step_rows,
    )

    write_csv(
        out_dir / "tracking_metrics.csv",
        tracking_rows,
    )

    write_json(
        out_dir / "stage12_branch_interval_reproduction.json",
        {
            "all_pass":
                all(r["pass"] for r in branch_checks),

            "results":
                branch_checks,
        },
    )

    active_summaries = [
        r for r in summaries
        if r["status"] != "NO_TRIGGER"
    ]

    persistent_safe_count = sum(
        int(r["status"] == "REACHED_2700_SAFE")
        for r in active_summaries
    )

    net_joint_progress_count = sum(
        int(bool(r.get("net_joint_progress", False)))
        for r in active_summaries
    )

    no_interval_count = sum(
        int(r["status"] == "NO_STRICT_INTERVAL")
        for r in active_summaries
    )

    degenerate_count = sum(
        int(r["status"] == "DEGENERATE_STEP")
        for r in active_summaries
    )

    nonlinear_count = sum(
        int(r["status"] == "NONLINEAR_SAFETY_FAILURE")
        for r in active_summaries
    )

    promotion = bool(
        persistent_safe_count >= 3
        and net_joint_progress_count >= 3
    )

    reached = [
        r for r in active_summaries
        if r["status"] == "REACHED_2700_SAFE"
    ]

    if promotion:

        route_class = (
            "adaptive_midpoint_persists_through_bounded_horizon"
        )

        next_route = (
            "stage16_adaptive_midpoint_escape_time_comparison"
        )

    elif no_interval_count + degenerate_count >= 2:

        route_class = (
            "adaptive_pareto_geometry_breaks_or_collapses"
        )

        next_route = (
            "strict_margin_or_broader_pareto_family_audit"
        )

    elif nonlinear_count >= 2:

        route_class = (
            "adaptive_midpoint_first_order_valid_but_curvature_limited"
        )

        next_route = (
            "adaptive_midpoint_backtracking_audit"
        )

    else:

        route_class = "mixed_adaptive_midpoint_failure"

        next_route = (
            "mixed_adaptive_midpoint_failure_audit"
        )

    decision = {
        "n_active_seeds":
            len(active_summaries),

        "active_seeds":
            sorted(int(r["seed"]) for r in active_summaries),

        "persistent_safe_count":
            persistent_safe_count,

        "net_joint_progress_count":
            net_joint_progress_count,

        "no_strict_interval_count":
            no_interval_count,

        "degenerate_step_count":
            degenerate_count,

        "nonlinear_safety_failure_count":
            nonlinear_count,

        "promotion_gate_pass":
            promotion,

        "all_stage12_branch_interval_reproductions_pass":
            all(r["pass"] for r in branch_checks),

        "median_relL2_ratio_vs_stage5_adam_2700": (
            float(
                np.median([
                    r["relL2_ratio_vs_stage5_adam_2700"]
                    for r in reached
                ])
            )
            if reached
            else None
        ),

        "median_relL2_ratio_vs_stage9_reflect_2700": (
            float(
                np.median([
                    r["relL2_ratio_vs_stage9_reflect_2700"]
                    for r in reached
                ])
            )
            if reached
            else None
        ),

        "median_active_fraction": (
            float(
                np.median([
                    r["active_fraction"]
                    for r in reached
                ])
            )
            if reached
            else None
        ),

        "median_strict_exact_active_fraction": (
            float(
                np.median([
                    r["strict_exact_active_fraction"]
                    for r in reached
                ])
            )
            if reached
            else None
        ),

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "Stage 15 establishes only bounded persistence through epoch 2700. "
            "Even a promotion PASS does not establish faster convergence or "
            "earlier escape. Stage 16 must compare certified escape times "
            "against the validated Stage-5 Adam baseline."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    # -------------------------------------------------------------------------
    # Figures.
    # -------------------------------------------------------------------------
    plot_lambda_trajectories(
        step_rows,
        out_dir / "adaptive_lambda_trajectories.png",
    )

    plot_interval_widths(
        step_rows,
        out_dir / "pareto_interval_widths.png",
    )

    plot_tracking_error(
        tracking_rows,
        out_dir / "relative_l2_trajectories.png",
    )

    plot_endpoint_comparison(
        summaries,
        out_dir / "endpoint_relL2_comparison.png",
    )

    elapsed = time.perf_counter() - global_start

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------
    lines = []

    lines.append("=" * 164)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 15 STATE-ADAPTIVE MIDPOINT PERSISTENCE SUMMARY"
    )
    lines.append("=" * 164)

    lines.append(
        "seed | branch | status                     | active | inactive | "
        "lambda range               | min width    | strict active | "
        "joint progress | relL2/Adam@2700"
    )

    lines.append("-" * 164)

    for r in summaries:

        if r["status"] == "NO_TRIGGER":

            lines.append(
                f"{int(r['seed']):4d} | "
                f"{-1:6d} | "
                f"{'NO_TRIGGER':26s} | "
                f"{'-':6s} | "
                f"{'-':8s} | "
                f"{'-':26s} | "
                f"{'-':12s} | "
                f"{'-':13s} | "
                f"{'-':14s} | "
                f"{'-'}"
            )

            continue

        if r["minimum_lambda"] is not None:
            lambda_text = (
                f"[{r['minimum_lambda']:.8f},"
                f"{r['maximum_lambda']:.8f}]"
            )
        else:
            lambda_text = "-"

        if r["status"] == "REACHED_2700_SAFE":
            rel_adam = (
                f"{r['relL2_ratio_vs_stage5_adam_2700']:.6f}"
            )
        else:
            rel_adam = "-"

        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['branch_epoch']):6d} | "
            f"{r['status']:26s} | "
            f"{int(r['active_steps']):6d} | "
            f"{int(r['inactive_steps']):8d} | "
            f"{lambda_text:26s} | "
            f"{str(r.get('minimum_interval_width')):12s} | "
            f"{int(r['strict_exact_active_steps']):5d}/"
            f"{max(int(r['active_steps']),1):<7d} | "
            f"{str(bool(r.get('net_joint_progress',False))):14s} | "
            f"{rel_adam}"
        )

    lines.append("-" * 164)

    lines.append(
        f"Stage-12 branch interval reproductions : "
        f"{sum(int(r['pass']) for r in branch_checks)}/"
        f"{len(branch_checks)} PASS"
    )

    lines.append(
        f"PERSISTENT_SAFE                        : "
        f"{persistent_safe_count}/4"
    )

    lines.append(
        f"NET_JOINT_PROGRESS                     : "
        f"{net_joint_progress_count}/4"
    )

    lines.append(
        f"NO_STRICT_INTERVAL                     : "
        f"{no_interval_count}/4"
    )

    lines.append(
        f"DEGENERATE_STEP                        : "
        f"{degenerate_count}/4"
    )

    lines.append(
        f"NONLINEAR_SAFETY_FAILURE               : "
        f"{nonlinear_count}/4"
    )

    lines.append(
        f"promotion gate                         : "
        f"{'PASS' if promotion else 'FAIL'}"
    )

    lines.append(
        f"route class                            : "
        f"{route_class}"
    )

    lines.append(
        f"next route                             : "
        f"{next_route}"
    )

    lines.append(
        f"elapsed seconds                        : "
        f"{elapsed:.2f}"
    )

    lines.append("=" * 164)

    lines.append(
        "Guardrail: a Stage-15 PASS authorizes only the Stage-16 certified "
        "escape-time comparison. It is not itself an acceleration claim."
    )

    lines.append("=" * 164)

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
