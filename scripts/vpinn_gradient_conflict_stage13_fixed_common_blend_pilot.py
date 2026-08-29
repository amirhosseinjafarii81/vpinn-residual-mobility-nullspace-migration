#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 13
Bounded Fixed Common-Pareto-Blend Persistence Pilot
====================================================

Purpose
-------
Stage 12 established, at the FOUR earliest Stage-10 direction-conflict states,
a common strict first-order joint-descent interval for

    d(lambda) = (1-lambda) Delta_Adam + lambda Delta_REFLECT

and selected, without post-step tuning,

    lambda_star = midpoint(common interval).

At those four earliest failure states:
  * lambda_star was strict first-order descent for both total and target loss;
  * the exact FULL step decreased both losses in 4/4 seeds;
  * the step retained ~95-98% of Adam's norm.

But Stage 12 is LOCAL. It does not show that one FIXED common lambda remains
Pareto-compatible as the intervention trajectory evolves.

Stage 13 asks exactly that persistence question before spending compute on a
full escape-time experiment.

Branching point
---------------
For each active seed, replay the exact Stage-9 REFLECT trajectory from the
saved epoch-2500 checkpoint to that seed's EARLIEST Stage-10
DIRECTION_CONFLICT state:

    seed 0 -> epoch 2505
    seed 1 -> epoch 2505
    seed 2 -> epoch 2505
    seed 3 -> epoch 2510

Then branch to the FIXED Stage-12 lambda_star rule.

Seed 4 is NO_TRIGGER and is not part of the active group denominator.

State-consistent fixed-lambda step
----------------------------------
At each step:

  1) compute current raw VPINN gradient and target gradient;
  2) let the real PyTorch Adam optimizer take its ordinary candidate step,
     thereby updating exp_avg, exp_avg_sq and step exactly;
  3) if the candidate Adam displacement is already target-nonuphill,
     keep the Adam displacement unchanged;
  4) if Adam is target-uphill, construct the current REFLECT endpoint and
     the current strict joint-descent lambda interval;
  5) the PRECOMMITTED fixed lambda_star must lie strictly inside that interval;
  6) apply

         d_star = (1-lambda_star) Delta_Adam
                  + lambda_star Delta_REFLECT

     while retaining Adam's newly updated optimizer state.

No state reset.
No lambda adaptation.
No line-search rescue.
No backtracking.

No-rescue early-stop gates
--------------------------
An active seed stops immediately if either:

GEOMETRY_DRIFT:
    lambda_star no longer lies strictly inside the current exact first-order
    joint-descent interval.

NONLINEAR_SAFETY_FAILURE:
    lambda_star is first-order jointly feasible, but the ACTUAL applied step
    increases total or target loss by more than the numerical safety tolerance.

The numerical tolerance is used ONLY to distinguish floating-point noise from
meaningful positive change:

    tol = 1e-12 * max(1, |pre_loss|).

This is not used to turn a clearly positive change into scientific "descent".

Bounded horizon
---------------
Only continue through epoch 2700.

Why 2700?
  * it is before the earliest Stage-5 certified escape onset (2800);
  * Stage 9 already exhibited severe REFLECT pathology by this horizon;
  * it is long enough to test whether the fixed common lambda remains valid
    beyond its local Stage-12 construction;
  * it avoids paying for a full escape study before persistence is established.

Primary promotion gate
----------------------
Active denominator = 4.

A seed is PERSISTENT_SAFE if it reaches epoch 2700 without GEOMETRY_DRIFT or
NONLINEAR_SAFETY_FAILURE.

A seed has NET_JOINT_PROGRESS if, at epoch 2700,

    total_loss(2700) < total_loss(branch_start)
    target_loss(2700) < target_loss(branch_start).

Authorize the full fixed-lambda escape-time continuation only if:

    >= 3/4 seeds are PERSISTENT_SAFE
    AND
    >= 3/4 seeds have NET_JOINT_PROGRESS.

Secondary historical comparisons
--------------------------------
At epoch 2700, compare the pilot trajectory with:
  * validated Stage-5 ordinary-Adam baseline;
  * Stage-9 failed REFLECT trajectory.

These historical comparisons are descriptive and are NOT used to choose
lambda_star or rescue a branch.

Decision routes
---------------
If promotion gate passes:
    bounded_fixed_common_blend_escape_time_continuation

Else if GEOMETRY_DRIFT occurs in >=2 active seeds:
    state_adaptive_pareto_midpoint_audit

Else if NONLINEAR_SAFETY_FAILURE occurs in >=2 active seeds:
    backtracking_fixed_common_blend_audit

Otherwise:
    mixed_fixed_blend_failure_audit

This is still a bounded pilot, not the final escape-time claim.
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


TARGET_MODE = 9
END_EPOCH = 2700


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-13 bounded fixed common Pareto-blend persistence pilot."
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
        "--stage12-script",
        default="vpinn_gradient_conflict_stage12_common_pareto_blend_audit.py",
    )
    p.add_argument(
        "--stage12-dir",
        default="vpinn_gradient_conflict_stage12_common_pareto_blend_audit",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage13_fixed_common_blend_pilot",
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
    stage12_script: Path,
    stage12_dir: Path,
) -> dict:

    paths = {
        "s5_manifest": stage5_dir / "manifest.json",
        "s5_aggregate": stage5_dir / "aggregate_postlock_metrics.csv",

        "s9_manifest": stage9_dir / "manifest.json",
        "s9_aggregate": stage9_dir / "aggregate_trajectories.csv",

        "s10_seed_summary": stage10_dir / "seed_summary.csv",

        "s12_manifest": stage12_dir / "manifest.json",
        "s12_decision": stage12_dir / "decision.json",
        "s12_seed_summary": stage12_dir / "seed_summary.csv",
        "s12_line_scan": stage12_dir / "line_scan_metrics.csv",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  "
            + "\n  ".join(missing)
        )

    s5_manifest = read_json(paths["s5_manifest"])
    s9_manifest = read_json(paths["s9_manifest"])
    s12_manifest = read_json(paths["s12_manifest"])
    s12_decision = read_json(paths["s12_decision"])

    if s12_decision.get("next_route") != (
        "bounded_common_pareto_blend_continuation_pilot"
    ):
        raise RuntimeError(
            "Stage 12 did not authorize the bounded common-blend pilot."
        )

    if s12_decision.get("route_class") != (
        "common_interior_pareto_full_step_viable"
    ):
        raise RuntimeError(
            "Stage-12 route class is not full-step common Pareto viability."
        )

    if int(s12_decision.get("full_safe_count", -1)) != 4:
        raise RuntimeError("Stage 12 did not report 4/4 full-safe active seeds.")

    if not bool(s12_decision.get("common_interval_exists", False)):
        raise RuntimeError("Stage 12 common strict interval does not exist.")

    lambda_star = float(s12_decision["lambda_star"])
    common_lo = float(s12_decision["common_interval_lower"])
    common_hi = float(s12_decision["common_interval_upper"])

    if not (common_lo < lambda_star < common_hi):
        raise RuntimeError("Stage-12 lambda_star is not strictly interior.")

    actual_s3_sha = sha256_file(stage3_script)
    actual_s9_sha = sha256_file(stage9_script)
    actual_s12_sha = sha256_file(stage12_script)

    if s5_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 5.")

    if s9_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 9.")

    if s12_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 12.")

    if s12_manifest.get("stage9_script_sha256") != actual_s9_sha:
        raise RuntimeError("Stage-9 SHA mismatch against Stage 12.")

    if s12_manifest.get("stage12_script_sha256") != actual_s12_sha:
        raise RuntimeError(
            "Stage-12 source SHA mismatch against its result manifest."
        )

    s10_summary = read_csv(paths["s10_seed_summary"])
    s12_summary = read_csv(paths["s12_seed_summary"])
    s12_scan = read_csv(paths["s12_line_scan"])

    active_targets = {}

    for row in s10_summary:
        seed = int(row["seed"])
        epoch = int(row["earliest_active_non_SAFE_FULL_epoch"])

        if epoch >= 0:
            if row["earliest_failure_class"] != "DIRECTION_CONFLICT":
                raise RuntimeError(
                    f"Seed {seed} Stage-10 earliest failure is not "
                    "DIRECTION_CONFLICT."
                )

            active_targets[seed] = epoch

    if active_targets != {0: 2505, 1: 2505, 2: 2505, 3: 2510}:
        raise RuntimeError(
            f"Unexpected Stage-10 active targets: {active_targets}"
        )

    s12_active = {
        int(r["seed"])
        for r in s12_summary
        if r["status"] == "ACTIVE_AUDIT"
    }

    if s12_active != set(active_targets):
        raise RuntimeError(
            "Stage-10 and Stage-12 active seed sets do not match."
        )

    # Stage-12 exact full-step reference at alpha=1.
    first_step_reference = {}

    for row in s12_scan:
        if abs(float(row["alpha"]) - 1.0) <= 1.0e-15:
            first_step_reference[int(row["seed"])] = row

    if set(first_step_reference) != set(active_targets):
        raise RuntimeError("Stage-12 alpha=1 reference is incomplete.")

    for seed in range(5):
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing locked checkpoint: {checkpoint}")

    return {
        "active_targets": active_targets,

        "lambda_star": lambda_star,
        "common_lo": common_lo,
        "common_hi": common_hi,

        "stage3_sha256": actual_s3_sha,
        "stage9_sha256": actual_s9_sha,
        "stage12_sha256": actual_s12_sha,

        "stage5_aggregate": read_csv(paths["s5_aggregate"]),
        "stage9_aggregate": read_csv(paths["s9_aggregate"]),

        "first_step_reference": first_step_reference,
    }


# =============================================================================
# Historical lookup maps
# =============================================================================

def stage5_map(rows: List[dict]) -> dict:
    return {
        (int(r["seed"]), int(r["epoch"])): r
        for r in rows
    }


def stage9_map(rows: List[dict]) -> dict:
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
        "relative_l2_error": exp.relative_l2_error(),

        "vpinn_loss": float(torch.mean(energy).item()),
        "target_loss": float((energy[t] / energy.numel()).item()),

        "target_mode_abs_residual":
            float(torch.abs(residuals[t]).item()),

        "target_mode_residual_energy_share":
            float((energy[t] / total_energy).item()),

        "dominant_residual_mode": dominant + 1,
    }


# =============================================================================
# Strict interval helpers
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
        max(i[0] for i in intervals),
        min(i[1] for i in intervals),
    )


# =============================================================================
# Fixed common-lambda state-consistent step
# =============================================================================

def fixed_common_blend_step(
    exp,
    lambda_star: float,
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

    # Raw total loss gradient is exactly what Adam consumes.
    total_loss.backward()

    gL = flatten([
        p.grad.detach().clone()
        for p in params
    ])

    before_parts = [
        p.detach().clone()
        for p in params
    ]

    before = flatten(before_parts)

    # Real PyTorch Adam step. Optimizer state is updated exactly.
    exp.optimizer.step()

    after_candidate = flatten([
        p.detach().clone()
        for p in params
    ])

    candidate = after_candidate - before

    candidate_total_dot = float(
        torch.dot(gL, candidate).item()
    )

    candidate_target_dot = float(
        torch.dot(gt, candidate).item()
    )

    candidate_norm = float(
        torch.linalg.vector_norm(candidate).item()
    )

    active = bool(candidate_target_dot > 0.0)

    geometry_failure = False
    interval_lo = None
    interval_hi = None
    interval_margin = None

    if active:
        gt2 = torch.dot(gt, gt)

        if float(gt2.item()) <= 0.0:
            raise RuntimeError("Target gradient norm is zero.")

        component = (
            torch.dot(gt, candidate) / gt2
        ) * gt

        reflected = candidate - 2.0 * component

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

        interval_margin = min(
            lambda_star - interval_lo,
            interval_hi - lambda_star,
        )

        # Strict interior check. The 1e-12 tolerance only protects against
        # binary representation of an exact boundary.
        geometry_failure = not (
            interval_lo + 1.0e-12
            < lambda_star
            < interval_hi - 1.0e-12
        )

        if geometry_failure:
            # Restore parameters to the pre-step state. Optimizer state has
            # already advanced, but the branch stops immediately and is never
            # continued after a geometry failure.
            with torch.no_grad():
                for p, p0 in zip(params, before_parts):
                    p.copy_(p0)

            return {
                "intervention_active": True,
                "geometry_failure": True,
                "nonlinear_safety_failure": False,

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
                "interval_margin":
                    interval_margin,

                "candidate_norm":
                    candidate_norm,

                "applied_norm":
                    None,
                "applied_over_candidate_norm":
                    None,

                "applied_total_dot":
                    None,
                "applied_target_dot":
                    None,

                "pre_total_loss":
                    pre_total,
                "pre_target_loss":
                    pre_target,

                "post_total_loss":
                    None,
                "post_target_loss":
                    None,

                "total_loss_change":
                    None,
                "target_loss_change":
                    None,
            }

        applied = (
            (1.0 - lambda_star) * candidate
            + lambda_star * reflected
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
                "lambda_star passed interval check but applied direction "
                "is not strict joint first-order descent."
            )

    else:
        reflected = candidate
        reflect_total_dot = candidate_total_dot
        reflect_target_dot = candidate_target_dot

        applied = candidate
        applied_total_dot = candidate_total_dot
        applied_target_dot = candidate_target_dot

    applied_norm = float(
        torch.linalg.vector_norm(applied).item()
    )

    # Overwrite only model displacement while preserving the Adam state that
    # was produced by the ordinary raw-gradient optimizer step.
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

    # Exact post-step losses.
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

    tol_total = 1.0e-12 * max(1.0, abs(pre_total))
    tol_target = 1.0e-12 * max(1.0, abs(pre_target))

    nonlinear_failure = bool(
        active
        and (
            total_change > tol_total
            or target_change > tol_target
        )
    )

    return {
        "intervention_active": active,
        "geometry_failure": False,
        "nonlinear_safety_failure":
            nonlinear_failure,

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
        "interval_margin":
            interval_margin,

        "candidate_norm":
            candidate_norm,

        "applied_norm":
            applied_norm,
        "applied_over_candidate_norm":
            applied_norm / max(candidate_norm, 1.0e-300),

        "applied_total_dot":
            applied_total_dot,
        "applied_target_dot":
            applied_target_dot,

        "pre_total_loss":
            pre_total,
        "pre_target_loss":
            pre_target,

        "post_total_loss":
            post_total,
        "post_target_loss":
            post_target,

        "total_loss_change":
            total_change,
        "target_loss_change":
            target_change,
    }


# =============================================================================
# Stage-12 first-step reproduction
# =============================================================================

def verify_first_step(
    seed: int,
    result: dict,
    reference: dict,
    tolerance: float = 1.0e-10,
) -> dict:

    comparisons = {
        "post_total_loss": (
            float(reference["post_total_loss"]),
            float(result["post_total_loss"]),
        ),

        "post_target_loss": (
            float(reference["post_target_loss"]),
            float(result["post_target_loss"]),
        ),

        "total_loss_change": (
            float(reference["total_loss_change"]),
            float(result["total_loss_change"]),
        ),

        "target_loss_change": (
            float(reference["target_loss_change"]),
            float(result["target_loss_change"]),
        ),
    }

    diffs = {
        name: abs(a - b)
        for name, (a, b) in comparisons.items()
    }

    max_diff = max(diffs.values())

    return {
        "seed": seed,
        "max_abs_difference": max_diff,
        "field_abs_differences": diffs,
        "tolerance": tolerance,
        "pass": bool(max_diff <= tolerance),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_endpoint_ratios(
    summaries: List[dict],
    path: Path,
) -> None:

    active = [
        r for r in summaries
        if r["status"] == "REACHED_2700_SAFE"
    ]

    if not active:
        return

    seeds = [str(int(r["seed"])) for r in active]

    baseline = [
        float(r["relL2_ratio_vs_stage5_baseline_2700"])
        for r in active
    ]

    reflect = [
        float(r["relL2_ratio_vs_stage9_reflect_2700"])
        for r in active
    ]

    x = np.arange(len(seeds))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.bar(
        x - width / 2,
        baseline,
        width,
        label="vs Stage-5 Adam",
    )

    ax.bar(
        x + width / 2,
        reflect,
        width,
        label="vs Stage-9 REFLECT",
    )

    ax.axhline(1.0, linestyle="--", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(seeds)

    ax.set_xlabel("Seed")
    ax.set_ylabel("Relative L2 ratio at epoch 2700")
    ax.set_title("Fixed common-blend endpoint comparison")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_min_margin(
    summaries: List[dict],
    path: Path,
) -> None:

    reached = [
        r for r in summaries
        if r["status"] == "REACHED_2700_SAFE"
    ]

    if not reached:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.bar(
        [str(int(r["seed"])) for r in reached],
        [float(r["minimum_active_lambda_margin"]) for r in reached],
    )

    ax.set_xlabel("Seed")
    ax.set_ylabel("Minimum strict lambda margin")
    ax.set_title(
        "How far the fixed lambda stayed from the moving Pareto boundary"
    )

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

    stage12_script = Path(args.stage12_script)
    if not stage12_script.is_absolute():
        stage12_script = root / stage12_script

    stage12_dir = Path(args.stage12_dir)
    if not stage12_dir.is_absolute():
        stage12_dir = root / stage12_dir

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
        stage12_script=stage12_script,
        stage12_dir=stage12_dir,
    )

    stage9 = load_module(
        stage9_script,
        "vpinn_stage9_replay_stage13",
    )

    stage3 = stage9.load_stage3_module(stage3_script)

    s5_map = stage5_map(pf["stage5_aggregate"])
    s9_map = stage9_map(pf["stage9_aggregate"])

    lambda_star = pf["lambda_star"]

    precommitment = {
        "stage":
            "bounded_fixed_common_pareto_blend_persistence_pilot",

        "active_branch_points":
            pf["active_targets"],

        "lambda_star":
            lambda_star,

        "lambda_rule":
            "fixed Stage-12 common midpoint; never adapted",

        "end_epoch":
            END_EPOCH,

        "no_rescue_stops": {
            "GEOMETRY_DRIFT":
                "lambda_star leaves current strict joint-descent interval",
            "NONLINEAR_SAFETY_FAILURE":
                "active first-order-feasible step increases total or target "
                "loss beyond 1e-12 relative-scale numerical tolerance",
        },

        "promotion_gate":
            ">=3/4 PERSISTENT_SAFE through 2700 AND "
            ">=3/4 NET_JOINT_PROGRESS at 2700",

        "comparators":
            [
                "Stage-5 validated Adam at 2700",
                "Stage-9 validated REFLECT at 2700",
            ],
    }

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),

        "stage3_solver_sha256":
            pf["stage3_sha256"],
        "stage9_script_sha256":
            pf["stage9_sha256"],
        "stage12_script_sha256":
            pf["stage12_sha256"],
        "stage13_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "stage12_dir": str(stage12_dir),

        "precommitment": precommitment,
    }

    write_json(out_dir / "manifest.json", manifest)

    print("=" * 146)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 13 FIXED COMMON-BLEND PERSISTENCE PILOT"
    )
    print("=" * 146)
    print(f"device                    : {device}")
    print(f"active branch points      : {pf['active_targets']}")
    print(f"lambda_star               : {lambda_star:.15f}")
    print(
        f"Stage-12 common interval  : "
        f"({pf['common_lo']:.15f}, {pf['common_hi']:.15f})"
    )
    print(f"bounded horizon           : through epoch {END_EPOCH}")
    print(
        "no rescue                : stop on geometry drift or nonlinear safety failure"
    )
    print("=" * 146)

    summaries: List[dict] = []
    trajectory_rows: List[dict] = []
    first_step_checks: List[dict] = []

    global_start = time.perf_counter()

    for seed in range(5):

        if seed not in pf["active_targets"]:
            summaries.append(
                {
                    "seed": seed,
                    "status": "NO_TRIGGER",
                    "branch_epoch": -1,
                }
            )

            print()
            print(f"SEED {seed}: NO_TRIGGER")
            continue

        branch_epoch = pf["active_targets"][seed]

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

        # Exact Stage-9 REFLECT replay to the identified first failure state.
        for _epoch in range(2500, branch_epoch):
            stage9.intervention_step(
                exp=exp,
                branch="REFLECT",
                target_mode=TARGET_MODE,
            )

        branch_state = state_metrics(exp)

        print()
        print("-" * 146)
        print(
            f"SEED {seed} | branch epoch={branch_epoch} | "
            f"relL2={branch_state['relative_l2_error']:.6e} | "
            f"L={branch_state['vpinn_loss']:.6e} | "
            f"T={branch_state['target_loss']:.6e}"
        )

        status = "RUNNING"

        failure_epoch = -1
        failure_type = None

        active_steps = 0
        inactive_steps = 0

        min_margin = math.inf
        exact_safe_active_steps = 0

        first_step_done = False

        # Record branch state.
        trajectory_rows.append(
            {
                "seed": seed,
                "epoch": branch_epoch,
                "event": "BRANCH_START",
                **branch_state,
            }
        )

        epoch = branch_epoch

        while epoch < END_EPOCH:

            step = fixed_common_blend_step(
                exp=exp,
                lambda_star=lambda_star,
                target_mode=TARGET_MODE,
            )

            if step["geometry_failure"]:
                status = "GEOMETRY_DRIFT"
                failure_epoch = epoch
                failure_type = "GEOMETRY_DRIFT"

                trajectory_rows.append(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "event": "GEOMETRY_DRIFT",
                        **step,
                    }
                )

                print(
                    f"  STOP geometry drift at step from epoch {epoch}: "
                    f"interval=({step['interval_lower']},"
                    f"{step['interval_upper']})"
                )

                break

            if step["intervention_active"]:
                active_steps += 1

                min_margin = min(
                    min_margin,
                    float(step["interval_margin"]),
                )

                if not step["nonlinear_safety_failure"]:
                    exact_safe_active_steps += 1
            else:
                inactive_steps += 1

            epoch += 1

            # First-step exact reproduction against Stage 12.
            if not first_step_done:
                check = verify_first_step(
                    seed=seed,
                    result=step,
                    reference=pf["first_step_reference"][seed],
                    tolerance=1.0e-10,
                )

                first_step_checks.append(check)

                if not check["pass"]:
                    raise RuntimeError(
                        f"Stage-12 first-step reproduction failed for seed "
                        f"{seed}: {check['max_abs_difference']:.3e}"
                    )

                first_step_done = True

            if step["nonlinear_safety_failure"]:
                status = "NONLINEAR_SAFETY_FAILURE"
                failure_epoch = epoch
                failure_type = "NONLINEAR_SAFETY_FAILURE"

                current = state_metrics(exp)

                trajectory_rows.append(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "event": "NONLINEAR_SAFETY_FAILURE",
                        **current,
                        **step,
                    }
                )

                print(
                    f"  STOP nonlinear safety failure at epoch {epoch}: "
                    f"dL={step['total_loss_change']:+.3e}, "
                    f"dT={step['target_loss_change']:+.3e}"
                )

                break

            if (
                epoch % 25 == 0
                or epoch == END_EPOCH
            ):
                current = state_metrics(exp)

                trajectory_rows.append(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "event": "TRACK",
                        **current,
                        **step,
                    }
                )

        if status == "RUNNING":
            status = "REACHED_2700_SAFE"

        if status == "REACHED_2700_SAFE":
            endpoint = state_metrics(exp)

            net_joint_progress = bool(
                endpoint["vpinn_loss"]
                    < branch_state["vpinn_loss"]
                and endpoint["target_loss"]
                    < branch_state["target_loss"]
            )

            s5_key = (seed, END_EPOCH)
            s9_key = (seed, "REFLECT", END_EPOCH)

            if s5_key not in s5_map:
                raise RuntimeError(
                    f"Stage-5 comparator missing seed={seed}, epoch={END_EPOCH}."
                )

            if s9_key not in s9_map:
                raise RuntimeError(
                    f"Stage-9 REFLECT comparator missing seed={seed}, "
                    f"epoch={END_EPOCH}."
                )

            s5 = s5_map[s5_key]
            s9 = s9_map[s9_key]

            rel_ratio_s5 = (
                endpoint["relative_l2_error"]
                / float(s5["relative_l2_error"])
            )

            rel_ratio_s9 = (
                endpoint["relative_l2_error"]
                / float(s9["relative_l2_error"])
            )

            loss_ratio_s5 = (
                endpoint["vpinn_loss"]
                / float(s5["vpinn_loss"])
            )

            loss_ratio_s9 = (
                endpoint["vpinn_loss"]
                / float(s9["vpinn_loss"])
            )

            summary = {
                "seed": seed,
                "status": status,
                "branch_epoch": branch_epoch,

                "active_steps":
                    active_steps,
                "inactive_steps":
                    inactive_steps,

                "minimum_active_lambda_margin":
                    min_margin if active_steps > 0 else None,

                "exact_safe_active_steps":
                    exact_safe_active_steps,

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

                "net_total_progress":
                    endpoint["vpinn_loss"]
                    < branch_state["vpinn_loss"],

                "net_target_progress":
                    endpoint["target_loss"]
                    < branch_state["target_loss"],

                "net_joint_progress":
                    net_joint_progress,

                "relL2_ratio_vs_stage5_baseline_2700":
                    rel_ratio_s5,
                "relL2_ratio_vs_stage9_reflect_2700":
                    rel_ratio_s9,

                "total_loss_ratio_vs_stage5_baseline_2700":
                    loss_ratio_s5,
                "total_loss_ratio_vs_stage9_reflect_2700":
                    loss_ratio_s9,
            }

        else:
            summary = {
                "seed": seed,
                "status": status,
                "branch_epoch": branch_epoch,

                "failure_epoch":
                    failure_epoch,
                "failure_type":
                    failure_type,

                "active_steps":
                    active_steps,
                "inactive_steps":
                    inactive_steps,

                "minimum_active_lambda_margin":
                    (
                        min_margin
                        if active_steps > 0
                        else None
                    ),

                "exact_safe_active_steps":
                    exact_safe_active_steps,

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
        write_json(seed_dir / "summary.json", summary)

        print(
            f"  result={status} | active={active_steps} | "
            f"inactive={inactive_steps} | "
            f"min margin="
            f"{summary.get('minimum_active_lambda_margin')}"
        )

        if status == "REACHED_2700_SAFE":
            print(
                f"  endpoint relL2={summary['endpoint_relative_l2']:.6e} | "
                f"joint progress={summary['net_joint_progress']} | "
                f"relL2/baseline={summary['relL2_ratio_vs_stage5_baseline_2700']:.6f} | "
                f"relL2/REFLECT={summary['relL2_ratio_vs_stage9_reflect_2700']:.6f}"
            )

    write_csv(out_dir / "seed_summary.csv", summaries)
    write_csv(out_dir / "trajectory_metrics.csv", trajectory_rows)

    write_json(
        out_dir / "stage12_first_step_reproduction.json",
        {
            "all_pass":
                all(r["pass"] for r in first_step_checks),

            "results":
                first_step_checks,
        },
    )

    active_summaries = [
        r for r in summaries
        if r["status"] != "NO_TRIGGER"
    ]

    safe_count = sum(
        int(r["status"] == "REACHED_2700_SAFE")
        for r in active_summaries
    )

    joint_progress_count = sum(
        int(bool(r.get("net_joint_progress", False)))
        for r in active_summaries
    )

    geometry_failure_count = sum(
        int(r["status"] == "GEOMETRY_DRIFT")
        for r in active_summaries
    )

    nonlinear_failure_count = sum(
        int(r["status"] == "NONLINEAR_SAFETY_FAILURE")
        for r in active_summaries
    )

    promotion = bool(
        safe_count >= 3
        and joint_progress_count >= 3
    )

    if promotion:
        route_class = "fixed_common_blend_persists_locally"
        next_route = (
            "bounded_fixed_common_blend_escape_time_continuation"
        )

    elif geometry_failure_count >= 2:
        route_class = "fixed_lambda_geometry_drift"
        next_route = "state_adaptive_pareto_midpoint_audit"

    elif nonlinear_failure_count >= 2:
        route_class = "fixed_lambda_curvature_failure"
        next_route = "backtracking_fixed_common_blend_audit"

    else:
        route_class = "mixed_fixed_blend_failure"
        next_route = "mixed_fixed_blend_failure_audit"

    reached = [
        r for r in active_summaries
        if r["status"] == "REACHED_2700_SAFE"
    ]

    decision = {
        "n_active_seeds": len(active_summaries),
        "active_seeds": sorted(
            int(r["seed"]) for r in active_summaries
        ),

        "lambda_star": lambda_star,

        "persistent_safe_count":
            safe_count,

        "net_joint_progress_count":
            joint_progress_count,

        "geometry_failure_count":
            geometry_failure_count,

        "nonlinear_safety_failure_count":
            nonlinear_failure_count,

        "promotion_gate_pass":
            promotion,

        "all_stage12_first_step_reproductions_pass":
            all(r["pass"] for r in first_step_checks),

        "median_relL2_ratio_vs_stage5_baseline_2700": (
            float(
                np.median([
                    r["relL2_ratio_vs_stage5_baseline_2700"]
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

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "Stage 13 tests persistence of the FIXED Stage-12 common lambda "
            "from the exact earliest REFLECT failure states only through "
            "epoch 2700. A promotion PASS authorizes, but does not replace, "
            "the subsequent escape-time comparison."
        ),
    }

    write_json(out_dir / "decision.json", decision)

    plot_endpoint_ratios(
        summaries,
        out_dir / "endpoint_relL2_ratios.png",
    )

    plot_min_margin(
        summaries,
        out_dir / "minimum_lambda_margin.png",
    )

    elapsed = time.perf_counter() - global_start

    lines = []

    lines.append("=" * 150)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 13 FIXED COMMON-BLEND PERSISTENCE SUMMARY"
    )
    lines.append("=" * 150)
    lines.append(
        "seed | branch | status                     | active | inactive | "
        "min margin | joint progress | relL2/base@2700 | relL2/refl@2700"
    )
    lines.append("-" * 150)

    for r in summaries:
        if r["status"] == "NO_TRIGGER":
            lines.append(
                f"{int(r['seed']):4d} | "
                f"{-1:6d} | "
                f"{'NO_TRIGGER':26s} | "
                f"{'-':6s} | {'-':8s} | {'-':10s} | "
                f"{'-':14s} | {'-':16s} | {'-'}"
            )
            continue

        if r["status"] == "REACHED_2700_SAFE":
            base_ratio = (
                f"{r['relL2_ratio_vs_stage5_baseline_2700']:.6f}"
            )
            refl_ratio = (
                f"{r['relL2_ratio_vs_stage9_reflect_2700']:.6f}"
            )
        else:
            base_ratio = "-"
            refl_ratio = "-"

        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['branch_epoch']):6d} | "
            f"{r['status']:26s} | "
            f"{int(r['active_steps']):6d} | "
            f"{int(r['inactive_steps']):8d} | "
            f"{str(r.get('minimum_active_lambda_margin')):10s} | "
            f"{str(bool(r.get('net_joint_progress', False))):14s} | "
            f"{base_ratio:16s} | "
            f"{refl_ratio}"
        )

    lines.append("-" * 150)
    lines.append(
        f"Stage-12 first-step reproductions : "
        f"{sum(int(r['pass']) for r in first_step_checks)}/"
        f"{len(first_step_checks)} PASS"
    )
    lines.append(
        f"PERSISTENT_SAFE                   : "
        f"{safe_count}/4"
    )
    lines.append(
        f"NET_JOINT_PROGRESS                : "
        f"{joint_progress_count}/4"
    )
    lines.append(
        f"GEOMETRY_DRIFT                    : "
        f"{geometry_failure_count}/4"
    )
    lines.append(
        f"NONLINEAR_SAFETY_FAILURE          : "
        f"{nonlinear_failure_count}/4"
    )
    lines.append(
        f"promotion gate                    : "
        f"{'PASS' if promotion else 'FAIL'}"
    )
    lines.append(
        f"route class                       : "
        f"{route_class}"
    )
    lines.append(
        f"next route                        : "
        f"{next_route}"
    )
    lines.append(
        f"elapsed seconds                   : "
        f"{elapsed:.2f}"
    )
    lines.append("=" * 150)
    lines.append(
        "Guardrail: fixed-lambda persistence must be established before "
        "paying for a full escape-time continuation."
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
