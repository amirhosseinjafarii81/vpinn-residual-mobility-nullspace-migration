#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 14
Failure-State Adaptive Pareto-Midpoint Rescue Audit
===================================================

Scientific motivation
---------------------
Stage 13 tested a FIXED common Pareto blend

    d(lambda*) = (1-lambda*) Delta_Adam + lambda* Delta_REFLECT

with
    lambda* = 0.5285577788190368.

The bounded persistence pilot failed:
  seed 0 : NONLINEAR_SAFETY_FAILURE at 2514
  seed 1 : GEOMETRY_DRIFT at 2510
  seed 2 : REACHED_2700_SAFE
  seed 3 : NONLINEAR_SAFETY_FAILURE at 2514
  seed 4 : NO_TRIGGER.

The Stage-13 automatic route suggested fixed-lambda backtracking because two
seeds failed nonlinearly. That route is incomplete: seed 1 failed because the
fixed lambda LEFT the strict first-order joint-descent interval. No positive
step-size damping of the same direction can fix a first-order sign violation.

Stage 14 therefore performs the cheapest discriminating audit:

At the exact PRE-FAILURE state of each failed seed, compare

  A) the original FIXED lambda* direction;
  B) a STATE-ADAPTIVE midpoint direction

         lambda_mid(theta)
             = 0.5 * [lambda_low(theta) + lambda_high(theta)]

     where (lambda_low,lambda_high) is the CURRENT strict first-order
     joint-descent interval for

         d(lambda)
             = (1-lambda) Delta_Adam + lambda Delta_REFLECT.

lambda_mid is selected from first-order geometry ONLY. Exact nonlinear losses
are used only for validation.

Failed-state reconstruction
---------------------------
Reconstruct from exact saved epoch-2500 states:

  seed 0:
      replay Stage-9 REFLECT to 2505,
      then 8 successful Stage-13 fixed-blend steps,
      audit the pre-failure state at epoch 2513.

  seed 1:
      replay Stage-9 REFLECT to 2505,
      then 5 successful Stage-13 fixed-blend steps,
      audit the geometry-drift state at epoch 2510.

  seed 3:
      replay Stage-9 REFLECT to 2510,
      then 3 successful Stage-13 fixed-blend steps,
      audit the pre-failure state at epoch 2513.

The reconstructed state/geometry must reproduce the Stage-13 failure row
within 1e-10.

Exact line scan
---------------
For FIXED and ADAPTIVE directions, evaluate

    alpha in {1/64,1/32,1/16,1/8,1/4,1/2,3/4,1}

with exact read-only total and target losses.

No optimizer state is changed by the line scan.

Classification
--------------
ADAPTIVE_FULL_SAFE:
    adaptive midpoint is strict first-order joint descent and alpha=1
    strictly decreases BOTH exact losses.

ADAPTIVE_DAMPED_SAFE:
    full step is unsafe, but some alpha<1 strictly decreases both.

ADAPTIVE_FIRST_ORDER_ONLY:
    first-order joint descent holds, but no scanned alpha is strictly safe.

ADAPTIVE_DEGENERATE:
    ||d_mid|| < 0.5 ||Delta_Adam||.

NO_STRICT_INTERVAL:
    no nonzero-width current strict joint-descent interval exists.

Primary route
-------------
Target denominator = the THREE Stage-13 failed active seeds {0,1,3}.

If 3/3 are ADAPTIVE_FULL_SAFE:
    authorize bounded state-adaptive midpoint persistence pilot.

Else if >=2/3 are ADAPTIVE_FULL_SAFE or ADAPTIVE_DAMPED_SAFE:
    authorize backtracking state-adaptive midpoint audit.

Else:
    route to strict-margin Pareto/QP audit.

This stage is an AUDIT only. It does not continue training past the failure
state.
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


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-14 adaptive Pareto-midpoint failure-state audit."
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
        "--stage12-script",
        default="vpinn_gradient_conflict_stage12_common_pareto_blend_audit.py",
    )
    p.add_argument(
        "--stage13-script",
        default="vpinn_gradient_conflict_stage13_fixed_common_blend_pilot.py",
    )
    p.add_argument(
        "--stage13-dir",
        default="vpinn_gradient_conflict_stage13_fixed_common_blend_pilot",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage14_adaptive_midpoint_failure_audit",
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
    stage12_script: Path,
    stage13_script: Path,
    stage13_dir: Path,
) -> dict:

    paths = {
        "s5_manifest": stage5_dir / "manifest.json",

        "s13_manifest": stage13_dir / "manifest.json",
        "s13_decision": stage13_dir / "decision.json",
        "s13_seed_summary": stage13_dir / "seed_summary.csv",
        "s13_trajectory": stage13_dir / "trajectory_metrics.csv",
        "s13_reproduction": stage13_dir / "stage12_first_step_reproduction.json",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    s5_manifest = read_json(paths["s5_manifest"])
    s13_manifest = read_json(paths["s13_manifest"])
    s13_decision = read_json(paths["s13_decision"])
    s13_reproduction = read_json(paths["s13_reproduction"])

    if bool(s13_decision.get("promotion_gate_pass", True)):
        raise RuntimeError("Stage 13 unexpectedly passed its promotion gate.")

    if s13_decision.get("route_class") != "fixed_lambda_curvature_failure":
        raise RuntimeError(
            "Stage-13 route class is not fixed_lambda_curvature_failure."
        )

    if s13_decision.get("next_route") != "backtracking_fixed_common_blend_audit":
        raise RuntimeError(
            "Unexpected Stage-13 automatic next route."
        )

    if int(s13_decision.get("persistent_safe_count", -1)) != 1:
        raise RuntimeError("Unexpected Stage-13 persistent-safe count.")

    if int(s13_decision.get("geometry_failure_count", -1)) != 1:
        raise RuntimeError("Unexpected Stage-13 geometry-failure count.")

    if int(s13_decision.get("nonlinear_safety_failure_count", -1)) != 2:
        raise RuntimeError("Unexpected Stage-13 nonlinear-failure count.")

    if not bool(s13_reproduction.get("all_pass", False)):
        raise RuntimeError("Stage-13 Stage-12 first-step reproduction failed.")

    actual_s3_sha = sha256_file(stage3_script)
    actual_s9_sha = sha256_file(stage9_script)
    actual_s12_sha = sha256_file(stage12_script)
    actual_s13_sha = sha256_file(stage13_script)

    if s5_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 5.")

    if s13_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 13.")

    if s13_manifest.get("stage9_script_sha256") != actual_s9_sha:
        raise RuntimeError("Stage-9 SHA mismatch against Stage 13.")

    if s13_manifest.get("stage12_script_sha256") != actual_s12_sha:
        raise RuntimeError("Stage-12 SHA mismatch against Stage 13.")

    if s13_manifest.get("stage13_script_sha256") != actual_s13_sha:
        raise RuntimeError("Stage-13 source SHA mismatch against its manifest.")

    seed_summary = read_csv(paths["s13_seed_summary"])
    trajectory = read_csv(paths["s13_trajectory"])

    expected_status = {
        0: "NONLINEAR_SAFETY_FAILURE",
        1: "GEOMETRY_DRIFT",
        2: "REACHED_2700_SAFE",
        3: "NONLINEAR_SAFETY_FAILURE",
        4: "NO_TRIGGER",
    }

    failed = {}

    for row in seed_summary:
        seed = int(row["seed"])
        status = row["status"]

        if status != expected_status[seed]:
            raise RuntimeError(
                f"Unexpected Stage-13 seed {seed} status: {status}"
            )

        if seed in (0, 1, 3):
            failed[seed] = {
                "status": status,
                "branch_epoch": int(row["branch_epoch"]),
                "failure_epoch": int(float(row["failure_epoch"])),
            }

    if failed != {
        0: {
            "status": "NONLINEAR_SAFETY_FAILURE",
            "branch_epoch": 2505,
            "failure_epoch": 2514,
        },
        1: {
            "status": "GEOMETRY_DRIFT",
            "branch_epoch": 2505,
            "failure_epoch": 2510,
        },
        3: {
            "status": "NONLINEAR_SAFETY_FAILURE",
            "branch_epoch": 2510,
            "failure_epoch": 2514,
        },
    }:
        raise RuntimeError(f"Unexpected Stage-13 failed-state map: {failed}")

    failure_rows = {}

    for row in trajectory:
        seed = int(row["seed"])
        event = row["event"]

        if seed in failed and event in (
            "GEOMETRY_DRIFT",
            "NONLINEAR_SAFETY_FAILURE",
        ):
            failure_rows[seed] = row

    if set(failure_rows) != set(failed):
        raise RuntimeError("Stage-13 failure-event rows are incomplete.")

    for seed in range(5):
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing locked checkpoint: {checkpoint}")

    return {
        "failed": failed,
        "failure_rows": failure_rows,

        "lambda_fixed": float(s13_decision["lambda_star"]),

        "stage3_sha256": actual_s3_sha,
        "stage9_sha256": actual_s9_sha,
        "stage12_sha256": actual_s12_sha,
        "stage13_sha256": actual_s13_sha,
    }


# =============================================================================
# Read-only current Adam / REFLECT geometry
# =============================================================================

def current_geometry(exp, target_mode: int = TARGET_MODE) -> dict:

    residuals = exp.weak_residuals()

    params = tuple(
        p for p in exp.model.parameters()
        if p.requires_grad
    )

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

        denom = (
            v_new.sqrt() / math.sqrt(bc2)
        ).add(eps)

        delta = -lr / bc1 * m_new / denom

        candidate_parts.append(delta.reshape(-1))

    candidate = flatten(candidate_parts)

    gt2 = torch.dot(gt, gt)

    if float(gt2.item()) <= 0.0:
        raise RuntimeError("Target gradient norm is zero.")

    candidate_target_dot = float(
        torch.dot(gt, candidate).item()
    )

    candidate_total_dot = float(
        torch.dot(gL, candidate).item()
    )

    active = bool(candidate_target_dot > 0.0)

    if not active:
        raise RuntimeError(
            "Targeted Stage-13 failure state is unexpectedly intervention-inactive."
        )

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

    return {
        "params": params,

        "g_total": gL,
        "g_target": gt,

        "candidate": candidate,
        "reflected": reflected,

        "candidate_total_dot":
            candidate_total_dot,
        "candidate_target_dot":
            candidate_target_dot,

        "reflect_total_dot":
            reflect_total_dot,
        "reflect_target_dot":
            reflect_target_dot,

        "candidate_norm":
            float(torch.linalg.vector_norm(candidate).item()),

        "reflect_norm":
            float(torch.linalg.vector_norm(reflected).item()),

        "pre_total_loss":
            float(total_loss.detach().item()),
        "pre_target_loss":
            float(target_loss.detach().item()),
    }


# =============================================================================
# Interval algebra
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
# Exact strict line scan
# =============================================================================

def loss_metrics(exp, target_mode: int = TARGET_MODE) -> dict:

    residuals = exp.weak_residuals().detach()
    energy = residuals.square()

    return {
        "total_loss":
            float(torch.mean(energy).item()),

        "target_loss":
            float(
                (energy[target_mode - 1] / energy.numel()).item()
            ),
    }


def set_temp_params(
    params,
    base_parts,
    direction: torch.Tensor,
    alpha: float,
) -> None:

    offset = 0

    with torch.no_grad():
        for p, p0 in zip(params, base_parts):
            n = p.numel()

            p.copy_(
                p0
                + alpha
                * direction[offset:offset+n].reshape_as(p)
            )

            offset += n

    if offset != direction.numel():
        raise RuntimeError("Direction size mismatch.")


def restore_params(params, base_parts) -> None:

    with torch.no_grad():
        for p, p0 in zip(params, base_parts):
            p.copy_(p0)


def strict_line_scan(
    exp,
    params,
    direction: torch.Tensor,
) -> Tuple[List[dict], dict]:

    base_parts = [
        p.detach().clone()
        for p in params
    ]

    pre = loss_metrics(exp)

    rows = []

    try:
        for alpha in ALPHAS:
            set_temp_params(
                params=params,
                base_parts=base_parts,
                direction=direction,
                alpha=alpha,
            )

            post = loss_metrics(exp)

            dL = post["total_loss"] - pre["total_loss"]
            dT = post["target_loss"] - pre["target_loss"]

            strict_safe = bool(
                dL < 0.0
                and dT < 0.0
            )

            rows.append(
                {
                    "alpha": float(alpha),

                    "pre_total_loss":
                        pre["total_loss"],
                    "post_total_loss":
                        post["total_loss"],
                    "total_loss_change":
                        dL,

                    "pre_target_loss":
                        pre["target_loss"],
                    "post_target_loss":
                        post["target_loss"],
                    "target_loss_change":
                        dT,

                    "joint_strict_improvement":
                        strict_safe,
                }
            )

    finally:
        restore_params(params, base_parts)

    safe = [
        row for row in rows
        if row["joint_strict_improvement"]
    ]

    full = next(
        row for row in rows
        if row["alpha"] == 1.0
    )

    return rows, {
        "full_step_strictly_safe":
            bool(full["joint_strict_improvement"]),

        "any_strictly_safe_alpha":
            bool(safe),

        "max_strictly_safe_alpha":
            (
                max(row["alpha"] for row in safe)
                if safe
                else 0.0
            ),

        "full_total_loss_change":
            float(full["total_loss_change"]),

        "full_target_loss_change":
            float(full["target_loss_change"]),
    }


# =============================================================================
# Reproduction against Stage 13
# =============================================================================

def compare_float(old_row: dict, key: str, current: float) -> float:

    raw = old_row.get(key)

    if raw in (None, "", "nan", "NaN"):
        raise RuntimeError(f"Missing Stage-13 failure field: {key}")

    return abs(float(raw) - float(current))


def verify_failure_state(
    seed: int,
    geo: dict,
    failure_row: dict,
    tolerance: float = 1.0e-10,
) -> dict:

    comparisons = {
        "pre_total_loss":
            compare_float(
                failure_row,
                "pre_total_loss",
                geo["pre_total_loss"],
            ),

        "pre_target_loss":
            compare_float(
                failure_row,
                "pre_target_loss",
                geo["pre_target_loss"],
            ),

        "candidate_total_dot":
            compare_float(
                failure_row,
                "candidate_total_dot",
                geo["candidate_total_dot"],
            ),

        "candidate_target_dot":
            compare_float(
                failure_row,
                "candidate_target_dot",
                geo["candidate_target_dot"],
            ),

        "reflect_total_dot":
            compare_float(
                failure_row,
                "reflect_total_dot",
                geo["reflect_total_dot"],
            ),

        "reflect_target_dot":
            compare_float(
                failure_row,
                "reflect_target_dot",
                geo["reflect_target_dot"],
            ),
    }

    max_diff = max(comparisons.values())

    return {
        "seed": seed,
        "tolerance": tolerance,
        "max_abs_difference": max_diff,
        "field_abs_differences": comparisons,
        "pass": bool(max_diff <= tolerance),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_lambdas(rows: List[dict], path: Path) -> None:

    x = np.arange(len(rows))

    fixed = [
        float(r["lambda_fixed"])
        for r in rows
    ]

    midpoint = [
        float(r["lambda_adaptive_midpoint"])
        for r in rows
    ]

    upper = [
        float(r["current_interval_upper"])
        for r in rows
    ]

    lower = [
        float(r["current_interval_lower"])
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.plot(x, fixed, marker="o", label="fixed lambda*")
    ax.plot(x, midpoint, marker="o", label="adaptive midpoint")
    ax.plot(x, upper, marker="o", label="current upper bound")
    ax.plot(x, lower, marker="o", label="current lower bound")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"seed {int(r['seed'])}" for r in rows]
    )

    ax.set_ylabel("lambda")
    ax.set_title("Fixed lambda vs failure-state adaptive Pareto midpoint")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_full_step_changes(rows: List[dict], path: Path) -> None:

    x = np.arange(len(rows))
    width = 0.36

    fixed = [
        float(r["fixed_full_total_loss_change"])
        for r in rows
    ]

    adaptive = [
        float(r["adaptive_full_total_loss_change"])
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.bar(
        x - width / 2,
        fixed,
        width,
        label="fixed lambda*",
    )

    ax.bar(
        x + width / 2,
        adaptive,
        width,
        label="adaptive midpoint",
    )

    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"seed {int(r['seed'])}" for r in rows]
    )

    ax.set_ylabel("Exact full-step total-loss change")
    ax.set_title("Failure-state rescue of total VPINN descent")
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

    stage12_script = Path(args.stage12_script)
    if not stage12_script.is_absolute():
        stage12_script = root / stage12_script

    stage13_script = Path(args.stage13_script)
    if not stage13_script.is_absolute():
        stage13_script = root / stage13_script

    stage13_dir = Path(args.stage13_dir)
    if not stage13_dir.is_absolute():
        stage13_dir = root / stage13_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage5_dir=stage5_dir,
        stage9_script=stage9_script,
        stage12_script=stage12_script,
        stage13_script=stage13_script,
        stage13_dir=stage13_dir,
    )

    stage9 = load_module(
        stage9_script,
        "vpinn_stage9_replay_stage14",
    )

    stage13 = load_module(
        stage13_script,
        "vpinn_stage13_replay_stage14",
    )

    stage3 = stage9.load_stage3_module(stage3_script)

    lambda_fixed = pf["lambda_fixed"]

    print("=" * 148)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 14 FAILURE-STATE ADAPTIVE MIDPOINT AUDIT"
    )
    print("=" * 148)
    print(f"device                    : {device}")
    print(f"failed seeds              : {sorted(pf['failed'])}")
    print(f"fixed lambda*             : {lambda_fixed:.15f}")
    print(
        "audit question            : fixed backtracking vs state-adaptive midpoint"
    )
    print(
        "selection                  : midpoint of CURRENT strict Pareto interval"
    )
    print("=" * 148)

    summary_rows: List[dict] = []
    line_rows: List[dict] = []
    reproduction_checks: List[dict] = []

    start = time.perf_counter()

    for seed in sorted(pf["failed"]):

        info = pf["failed"][seed]
        branch_epoch = info["branch_epoch"]
        failure_epoch = info["failure_epoch"]
        failure_type = info["status"]

        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        seed_dir = out_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

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

        # Exact Stage-9 REFLECT replay to Stage-13 branch state.
        for _epoch in range(2500, branch_epoch):
            stage9.intervention_step(
                exp=exp,
                branch="REFLECT",
                target_mode=TARGET_MODE,
            )

        if failure_type == "NONLINEAR_SAFETY_FAILURE":
            successful_fixed_steps = (
                failure_epoch - branch_epoch - 1
            )

            audit_epoch = failure_epoch - 1

        elif failure_type == "GEOMETRY_DRIFT":
            successful_fixed_steps = (
                failure_epoch - branch_epoch
            )

            audit_epoch = failure_epoch

        else:
            raise RuntimeError(
                f"Unsupported failure type: {failure_type}"
            )

        for _ in range(successful_fixed_steps):
            step = stage13.fixed_common_blend_step(
                exp=exp,
                lambda_star=lambda_fixed,
                target_mode=TARGET_MODE,
            )

            if step["geometry_failure"]:
                raise RuntimeError(
                    f"Unexpected early geometry failure while reconstructing "
                    f"seed {seed}."
                )

            if step["nonlinear_safety_failure"]:
                raise RuntimeError(
                    f"Unexpected early nonlinear failure while reconstructing "
                    f"seed {seed}."
                )

        geo = current_geometry(
            exp=exp,
            target_mode=TARGET_MODE,
        )

        reproduction = verify_failure_state(
            seed=seed,
            geo=geo,
            failure_row=pf["failure_rows"][seed],
            tolerance=1.0e-10,
        )

        reproduction_checks.append(reproduction)

        if not reproduction["pass"]:
            raise RuntimeError(
                f"Stage-13 failure-state reproduction failed for seed {seed}: "
                f"{reproduction['max_abs_difference']:.3e}"
            )

        interval_L = strict_negative_interval(
            geo["candidate_total_dot"],
            geo["reflect_total_dot"],
        )

        interval_T = strict_negative_interval(
            geo["candidate_target_dot"],
            geo["reflect_target_dot"],
        )

        lo, hi = intersect_open_intervals(
            [interval_L, interval_T]
        )

        strict_interval_exists = bool(lo < hi)

        if not strict_interval_exists:
            adaptive_class = "NO_STRICT_INTERVAL"

            summary_rows.append(
                {
                    "seed": seed,
                    "failure_type": failure_type,
                    "audit_epoch": audit_epoch,
                    "classification": adaptive_class,
                }
            )

            continue

        lambda_mid = 0.5 * (lo + hi)

        candidate = geo["candidate"]
        reflected = geo["reflected"]

        fixed_direction = (
            (1.0 - lambda_fixed) * candidate
            + lambda_fixed * reflected
        )

        adaptive_direction = (
            (1.0 - lambda_mid) * candidate
            + lambda_mid * reflected
        )

        gL = geo["g_total"]
        gT = geo["g_target"]

        fixed_dL = float(
            torch.dot(gL, fixed_direction).item()
        )

        fixed_dT = float(
            torch.dot(gT, fixed_direction).item()
        )

        adaptive_dL = float(
            torch.dot(gL, adaptive_direction).item()
        )

        adaptive_dT = float(
            torch.dot(gT, adaptive_direction).item()
        )

        if not (
            adaptive_dL < 0.0
            and adaptive_dT < 0.0
        ):
            raise RuntimeError(
                f"Adaptive midpoint is not strict first-order joint descent "
                f"for seed {seed}."
            )

        fixed_scan_rows, fixed_scan = strict_line_scan(
            exp=exp,
            params=geo["params"],
            direction=fixed_direction,
        )

        adaptive_scan_rows, adaptive_scan = strict_line_scan(
            exp=exp,
            params=geo["params"],
            direction=adaptive_direction,
        )

        for row in fixed_scan_rows:
            line_rows.append(
                {
                    "seed": seed,
                    "audit_epoch": audit_epoch,
                    "direction": "FIXED",
                    **row,
                }
            )

        for row in adaptive_scan_rows:
            line_rows.append(
                {
                    "seed": seed,
                    "audit_epoch": audit_epoch,
                    "direction": "ADAPTIVE_MIDPOINT",
                    **row,
                }
            )

        adam_norm = geo["candidate_norm"]

        adaptive_norm = float(
            torch.linalg.vector_norm(
                adaptive_direction
            ).item()
        )

        norm_ratio = (
            adaptive_norm
            / max(adam_norm, 1.0e-300)
        )

        target_retention = (
            -adaptive_dT
            / max(-geo["reflect_target_dot"], 1.0e-300)
        )

        if norm_ratio < 0.5:
            adaptive_class = "ADAPTIVE_DEGENERATE"

        elif adaptive_scan["full_step_strictly_safe"]:
            adaptive_class = "ADAPTIVE_FULL_SAFE"

        elif adaptive_scan["any_strictly_safe_alpha"]:
            adaptive_class = "ADAPTIVE_DAMPED_SAFE"

        else:
            adaptive_class = "ADAPTIVE_FIRST_ORDER_ONLY"

        summary = {
            "seed": seed,

            "failure_type":
                failure_type,

            "audit_epoch":
                audit_epoch,

            "lambda_fixed":
                lambda_fixed,

            "current_interval_lower":
                lo,
            "current_interval_upper":
                hi,
            "current_interval_width":
                hi - lo,

            "lambda_fixed_inside_current_interval":
                bool(lo < lambda_fixed < hi),

            "lambda_adaptive_midpoint":
                lambda_mid,

            "fixed_distance_to_upper":
                hi - lambda_fixed,

            "adaptive_midpoint_margin":
                min(
                    lambda_mid - lo,
                    hi - lambda_mid,
                ),

            "fixed_first_order_total_dot":
                fixed_dL,
            "fixed_first_order_target_dot":
                fixed_dT,

            "adaptive_first_order_total_dot":
                adaptive_dL,
            "adaptive_first_order_target_dot":
                adaptive_dT,

            "adaptive_over_adam_norm":
                norm_ratio,

            "adaptive_target_descent_retention_vs_reflect":
                target_retention,

            "fixed_full_step_strictly_safe":
                fixed_scan["full_step_strictly_safe"],
            "fixed_max_strictly_safe_alpha":
                fixed_scan["max_strictly_safe_alpha"],
            "fixed_full_total_loss_change":
                fixed_scan["full_total_loss_change"],
            "fixed_full_target_loss_change":
                fixed_scan["full_target_loss_change"],

            "adaptive_full_step_strictly_safe":
                adaptive_scan["full_step_strictly_safe"],
            "adaptive_max_strictly_safe_alpha":
                adaptive_scan["max_strictly_safe_alpha"],
            "adaptive_full_total_loss_change":
                adaptive_scan["full_total_loss_change"],
            "adaptive_full_target_loss_change":
                adaptive_scan["full_target_loss_change"],

            "classification":
                adaptive_class,

            "failure_state_reproduction_gap":
                reproduction["max_abs_difference"],
        }

        summary_rows.append(summary)

        write_json(
            seed_dir / "summary.json",
            summary,
        )

        print()
        print("-" * 148)
        print(
            f"SEED {seed} @ {audit_epoch} | "
            f"Stage-13 failure={failure_type}"
        )
        print(
            f"current interval = "
            f"({lo:.12f}, {hi:.12f})"
        )
        print(
            f"fixed lambda     = {lambda_fixed:.12f} | "
            f"inside={lo < lambda_fixed < hi} | "
            f"max safe alpha={fixed_scan['max_strictly_safe_alpha']:.5f}"
        )
        print(
            f"adaptive midpoint= {lambda_mid:.12f} | "
            f"class={adaptive_class} | "
            f"max safe alpha={adaptive_scan['max_strictly_safe_alpha']:.5f}"
        )
        print(
            f"adaptive exact full: "
            f"dL={adaptive_scan['full_total_loss_change']:+.6e}, "
            f"dT={adaptive_scan['full_target_loss_change']:+.6e}"
        )
        print(
            f"adaptive norm/Adam={norm_ratio:.6f} | "
            f"target retention={target_retention:.6f}"
        )

    write_csv(
        out_dir / "failure_state_summary.csv",
        summary_rows,
    )

    write_csv(
        out_dir / "line_scan_metrics.csv",
        line_rows,
    )

    write_json(
        out_dir / "stage13_failure_state_reproduction.json",
        {
            "all_pass":
                all(r["pass"] for r in reproduction_checks),

            "results":
                reproduction_checks,
        },
    )

    n = len(summary_rows)

    full_safe = sum(
        int(
            r.get("classification")
            == "ADAPTIVE_FULL_SAFE"
        )
        for r in summary_rows
    )

    damped_safe = sum(
        int(
            r.get("classification")
            == "ADAPTIVE_DAMPED_SAFE"
        )
        for r in summary_rows
    )

    degenerate = sum(
        int(
            r.get("classification")
            == "ADAPTIVE_DEGENERATE"
        )
        for r in summary_rows
    )

    first_order_only = sum(
        int(
            r.get("classification")
            == "ADAPTIVE_FIRST_ORDER_ONLY"
        )
        for r in summary_rows
    )

    no_interval = sum(
        int(
            r.get("classification")
            == "NO_STRICT_INTERVAL"
        )
        for r in summary_rows
    )

    fixed_any_safe = sum(
        int(
            float(r.get("fixed_max_strictly_safe_alpha", 0.0))
            > 0.0
        )
        for r in summary_rows
    )

    if (
        n == 3
        and full_safe == 3
        and degenerate == 0
    ):
        route_class = "adaptive_midpoint_full_step_rescues_all_failures"

        next_route = (
            "bounded_state_adaptive_midpoint_persistence_pilot"
        )

    elif (
        full_safe + damped_safe >= 2
        and degenerate == 0
    ):
        route_class = "adaptive_midpoint_requires_damping"

        next_route = (
            "backtracking_state_adaptive_midpoint_audit"
        )

    else:
        route_class = "adaptive_midpoint_local_rescue_insufficient"

        next_route = "strict_margin_pareto_qp_audit"

    decision = {
        "n_failed_states":
            n,

        "failed_seeds":
            sorted(int(r["seed"]) for r in summary_rows),

        "fixed_lambda":
            lambda_fixed,

        "fixed_direction_has_some_safe_alpha_count":
            fixed_any_safe,

        "adaptive_full_safe_count":
            full_safe,

        "adaptive_damped_safe_count":
            damped_safe,

        "adaptive_degenerate_count":
            degenerate,

        "adaptive_first_order_only_count":
            first_order_only,

        "no_strict_interval_count":
            no_interval,

        "all_failure_state_reproductions_pass":
            all(r["pass"] for r in reproduction_checks),

        "median_adaptive_midpoint": (
            float(
                np.median([
                    r["lambda_adaptive_midpoint"]
                    for r in summary_rows
                    if "lambda_adaptive_midpoint" in r
                ])
            )
            if summary_rows
            else None
        ),

        "median_adaptive_over_adam_norm": (
            float(
                np.median([
                    r["adaptive_over_adam_norm"]
                    for r in summary_rows
                    if "adaptive_over_adam_norm" in r
                ])
            )
            if summary_rows
            else None
        ),

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "Stage 14 is a local failure-state audit. A 3/3 adaptive midpoint "
            "rescue establishes that the Stage-13 failures are locally "
            "repairable by updating the Pareto blend from current geometry. "
            "It does not establish persistence or faster escape. The next "
            "bounded persistence pilot must test that separately."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    precommitment = {
        "stage":
            "failure_state_adaptive_pareto_midpoint_rescue_audit",

        "targeted_failed_seeds":
            [0, 1, 3],

        "adaptive_lambda_rule":
            "midpoint of current strict total/target first-order descent interval",

        "lambda_selection_uses_post_step_loss":
            False,

        "line_scan_alphas":
            list(ALPHAS),

        "strict_scientific_safety":
            "total_loss_change < 0 AND target_loss_change < 0",

        "degeneracy_guard":
            "adaptive step norm / Adam candidate norm >= 0.5",

        "primary_gate":
            "3/3 ADAPTIVE_FULL_SAFE",

        "audit_only":
            True,
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

        "stage12_script_sha256":
            pf["stage12_sha256"],

        "stage13_script_sha256":
            pf["stage13_sha256"],

        "stage14_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "stage13_dir":
            str(stage13_dir),

        "precommitment":
            precommitment,
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    plot_lambdas(
        summary_rows,
        out_dir / "failure_state_lambda_geometry.png",
    )

    plot_full_step_changes(
        summary_rows,
        out_dir / "fixed_vs_adaptive_total_loss_change.png",
    )

    elapsed = time.perf_counter() - start

    lines = []

    lines.append("=" * 154)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 14 ADAPTIVE MIDPOINT FAILURE-STATE SUMMARY"
    )
    lines.append("=" * 154)
    lines.append(
        "seed | epoch | Stage13 failure            | current interval          | "
        "fixed max a | midpoint     | adaptive class       | adaptive dL       | adaptive dT"
    )
    lines.append("-" * 154)

    for r in summary_rows:

        if "lambda_adaptive_midpoint" not in r:
            lines.append(
                f"{int(r['seed']):4d} | "
                f"{int(r['audit_epoch']):5d} | "
                f"{r['failure_type']:26s} | "
                f"{'-':25s} | "
                f"{'-':11s} | "
                f"{'-':12s} | "
                f"{r['classification']:20s}"
            )

            continue

        interval_text = (
            f"({r['current_interval_lower']:.9f},"
            f"{r['current_interval_upper']:.9f})"
        )

        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['audit_epoch']):5d} | "
            f"{r['failure_type']:26s} | "
            f"{interval_text:25s} | "
            f"{r['fixed_max_strictly_safe_alpha']:11.5f} | "
            f"{r['lambda_adaptive_midpoint']:12.9f} | "
            f"{r['classification']:20s} | "
            f"{r['adaptive_full_total_loss_change']:+.6e} | "
            f"{r['adaptive_full_target_loss_change']:+.6e}"
        )

    lines.append("-" * 154)
    lines.append(
        f"Stage-13 failure reproductions       : "
        f"{sum(int(r['pass']) for r in reproduction_checks)}/"
        f"{len(reproduction_checks)} PASS"
    )

    lines.append(
        f"fixed direction has some safe alpha  : "
        f"{fixed_any_safe}/{n}"
    )

    lines.append(
        f"ADAPTIVE_FULL_SAFE                   : "
        f"{full_safe}/{n}"
    )

    lines.append(
        f"ADAPTIVE_DAMPED_SAFE                 : "
        f"{damped_safe}/{n}"
    )

    lines.append(
        f"median adaptive midpoint             : "
        f"{decision['median_adaptive_midpoint']}"
    )

    lines.append(
        f"median adaptive/Adam norm            : "
        f"{decision['median_adaptive_over_adam_norm']}"
    )

    lines.append(
        f"route class                          : "
        f"{route_class}"
    )

    lines.append(
        f"next route                           : "
        f"{next_route}"
    )

    lines.append(
        f"elapsed seconds                      : "
        f"{elapsed:.2f}"
    )

    lines.append("=" * 154)
    lines.append(
        "Guardrail: local rescue is not persistence. Do not run an escape-time "
        "experiment until the adaptive midpoint survives a bounded trajectory pilot."
    )
    lines.append("=" * 154)

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
