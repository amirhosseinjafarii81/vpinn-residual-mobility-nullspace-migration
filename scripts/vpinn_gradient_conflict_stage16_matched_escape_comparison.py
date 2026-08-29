#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 16
Matched-State Certified Escape-Time Causal Comparison
=====================================================

Question
--------
Does the state-adaptive Pareto-midpoint rule actually escape the VPINN plateau
earlier than ordinary Adam when BOTH methods start from the EXACT SAME
parameter AND optimizer state?

Why a matched-state control?
----------------------------
Stages 12-15 branched after a short, identical Stage-9 REFLECT replay to each
seed's earliest direction-conflict state. Comparing the adaptive trajectory
only with the old Stage-5 historical Adam run would therefore mix:
    * the short REFLECT prelude, and
    * the adaptive Pareto intervention.

Stage 16 removes that confound.

For each active seed:
    1) reconstruct the exact earliest direction-conflict state;
    2) clone model + Adam state exactly;
    3) launch two branches:

       CONTROL:
           ordinary Adam only

       ADAPTIVE:
           Stage-15 state-adaptive Pareto midpoint rule

Both branches therefore have identical theta, exp_avg, exp_avg_sq, step
counter, and optimizer hyperparameters at the branch point.

Active seeds
------------
    seed 0 : branch epoch 2505
    seed 1 : branch epoch 2505
    seed 2 : branch epoch 2505
    seed 3 : branch epoch 2510

Seed 4 is NO_TRIGGER and is excluded from the primary active-mechanism
denominator. Its historical Stage-5 behavior remains descriptive context.

Adaptive safety rules
---------------------
The ADAPTIVE branch inherits Stage 15 without modification:

    * use ordinary Adam unchanged when its candidate step is target-nonuphill;
    * otherwise compute the CURRENT strict total/target Pareto interval;
    * choose its midpoint;
    * no lambda tuning from post-step losses;
    * no backtracking;
    * no reset;
    * stop on:
          NO_STRICT_INTERVAL,
          DEGENERATE_STEP,
          NONLINEAR_SAFETY_FAILURE.

An adaptive safety failure counts as NON-ACCELERATION in the primary result.
Seed 2 is not silently excluded.

Certified escape definition
---------------------------
Inherited unchanged from Stage 5:

    relative L2 error <= 1e-2
    AND
    target-mode residual-energy share <= 0.20

for THREE consecutive observations on the global 25-epoch tracking grid.

The certified escape onset is the first observation in that 3-point run.

Fixed maximum horizon
---------------------
    max epoch = 4000

This exceeds every Stage-5 baseline confirmation epoch and prevents
result-dependent horizon extension.

Primary paired endpoint
-----------------------
For seed s:

    acceleration_s
        = t_escape_CONTROL,s - t_escape_ADAPTIVE,s

ADAPTIVE_ACCELERATED iff:
    * adaptive branch has no safety failure,
    * both branches have certified escape,
    * t_escape_ADAPTIVE < t_escape_CONTROL.

If adaptive fails a safety gate or is censored, it is NOT accelerated.

Primary group gate
------------------
PASS only if:

    >= 3/4 active seeds are ADAPTIVE_ACCELERATED
    AND
    >= 3/4 adaptive branches have certified escape
    AND
    adaptive safety failures <= 1/4.

This threshold is inherited from the Stage-15 3/4 promotion policy.

Secondary metrics
-----------------
For seeds where both branches escape:
    * epochs saved;
    * percent reduction in branch-to-escape delay;
    * escape-delay speedup ratio;
    * comparison with historical Stage-5 escape onset (descriptive only).

Reproducibility gates
---------------------
Before Stage-16 continuation is accepted:

    * Stage-3 SHA must match Stage-15 manifest;
    * Stage-9 SHA must match Stage-15 manifest;
    * local Stage-15 source SHA must match the executed Stage-15 manifest;
    * Stage-15 promotion gate must be PASS;
    * Stage-15 branch-interval reproductions must be PASS;
    * CONTROL and ADAPTIVE cloned states must be exactly identical;
    * ADAPTIVE replay through epoch 2700 must reproduce Stage-15 results
      at every stored tracking point for safe seeds;
    * seed-2 Stage-15 nonlinear failure must reproduce at epoch 2560.

No further optimizer rescue is authorized inside this stage.

Final route
-----------
If primary group gate PASS:
    linkedin_final_evidence_package_with_escape_claim

If primary group gate FAIL:
    linkedin_final_mechanism_package_without_acceleration_claim

This is deliberately the last optimizer experiment for the LinkedIn thread.
A FAIL changes the story; it does not trigger another post-hoc rescue sweep.
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
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


TARGET_MODE = 9
TRACK_INTERVAL = 25
MAX_EPOCH = 4000

ESCAPE_REL_L2 = 1.0e-2
ESCAPE_TARGET_SHARE = 0.20
ESCAPE_CONSECUTIVE = 3

REPLAY_TOL = 1.0e-10


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-16 matched-state certified escape-time comparison."
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
        "--stage10-dir",
        default="vpinn_gradient_conflict_stage10_local_feasibility_audit",
    )

    p.add_argument(
        "--stage15-script",
        default="vpinn_gradient_conflict_stage15_adaptive_midpoint_persistence.py",
    )

    p.add_argument(
        "--stage15-dir",
        default="vpinn_gradient_conflict_stage15_adaptive_midpoint_persistence",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage16_matched_escape_comparison",
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


# =============================================================================
# Preflight
# =============================================================================

def preflight(
    stage3_script: Path,
    stage5_dir: Path,
    stage9_script: Path,
    stage10_dir: Path,
    stage15_script: Path,
    stage15_dir: Path,
) -> dict:

    paths = {
        "s5_manifest":
            stage5_dir / "manifest.json",

        "s5_escape":
            stage5_dir / "escape_time_summary.csv",

        "s10_seed_summary":
            stage10_dir / "seed_summary.csv",

        "s15_manifest":
            stage15_dir / "manifest.json",

        "s15_decision":
            stage15_dir / "decision.json",

        "s15_seed_summary":
            stage15_dir / "seed_summary.csv",

        "s15_tracking":
            stage15_dir / "tracking_metrics.csv",

        "s15_steps":
            stage15_dir / "step_metrics.csv",

        "s15_branch_repro":
            stage15_dir / "stage12_branch_interval_reproduction.json",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    s5_manifest = read_json(paths["s5_manifest"])
    s15_manifest = read_json(paths["s15_manifest"])
    s15_decision = read_json(paths["s15_decision"])
    s15_branch_repro = read_json(paths["s15_branch_repro"])

    if not bool(s15_decision.get("promotion_gate_pass", False)):
        raise RuntimeError("Stage 15 promotion gate is not PASS.")

    if s15_decision.get("next_route") != (
        "stage16_adaptive_midpoint_escape_time_comparison"
    ):
        raise RuntimeError(
            "Stage 15 did not authorize the Stage-16 escape comparison."
        )

    if int(s15_decision.get("persistent_safe_count", -1)) != 3:
        raise RuntimeError("Unexpected Stage-15 persistent-safe count.")

    if int(s15_decision.get("net_joint_progress_count", -1)) != 3:
        raise RuntimeError("Unexpected Stage-15 joint-progress count.")

    if int(s15_decision.get("nonlinear_safety_failure_count", -1)) != 1:
        raise RuntimeError(
            "Unexpected Stage-15 nonlinear-safety-failure count."
        )

    if not bool(
        s15_decision.get(
            "all_stage12_branch_interval_reproductions_pass",
            False,
        )
    ):
        raise RuntimeError(
            "Stage-15 branch interval reproductions are not all PASS."
        )

    if not bool(s15_branch_repro.get("all_pass", False)):
        raise RuntimeError(
            "Stage-15 branch reproduction JSON is not all PASS."
        )

    actual_s3_sha = sha256_file(stage3_script)
    actual_s9_sha = sha256_file(stage9_script)
    actual_s15_sha = sha256_file(stage15_script)

    if s5_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 5.")

    if s15_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 15.")

    if s15_manifest.get("stage9_script_sha256") != actual_s9_sha:
        raise RuntimeError("Stage-9 source SHA mismatch against Stage 15.")

    if s15_manifest.get("stage15_script_sha256") != actual_s15_sha:
        raise RuntimeError(
            "Stage-15 source SHA mismatch against its executed manifest. "
            "Use the exact Stage-15 script that produced the result directory."
        )

    # Freeze active branch points from Stage 10.
    s10_summary = read_csv(paths["s10_seed_summary"])

    branch_points = {}

    for row in s10_summary:
        seed = int(row["seed"])
        epoch = int(row["earliest_active_non_SAFE_FULL_epoch"])

        if epoch >= 0:
            if row["earliest_failure_class"] != "DIRECTION_CONFLICT":
                raise RuntimeError(
                    f"Seed {seed} earliest Stage-10 failure is not "
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
            f"Unexpected Stage-10 branch points: {branch_points}"
        )

    # Validate Stage-15 status pattern.
    s15_summary = read_csv(paths["s15_seed_summary"])

    status_map = {
        int(r["seed"]): r["status"]
        for r in s15_summary
    }

    expected_status = {
        0: "REACHED_2700_SAFE",
        1: "REACHED_2700_SAFE",
        2: "NONLINEAR_SAFETY_FAILURE",
        3: "REACHED_2700_SAFE",
        4: "NO_TRIGGER",
    }

    if status_map != expected_status:
        raise RuntimeError(
            f"Unexpected Stage-15 seed status map: {status_map}"
        )

    # Historical Stage-5 escape context.
    stage5_escape_rows = read_csv(paths["s5_escape"])

    historical_escape = {}

    for r in stage5_escape_rows:
        seed = int(r["seed"])

        onset_key = (
            "certified_escape_onset_epoch"
            if "certified_escape_onset_epoch" in r
            else "escape_onset_epoch"
        )

        confirm_key = (
            "certified_escape_confirmation_epoch"
            if "certified_escape_confirmation_epoch" in r
            else "escape_confirmation_epoch"
        )

        historical_escape[seed] = {
            "onset":
                int(float(r[onset_key])),

            "confirmation":
                int(float(r[confirm_key])),
        }

    if not all(seed in historical_escape for seed in range(5)):
        raise RuntimeError("Stage-5 escape summary is incomplete.")

    # Exact epoch-2500 checkpoints.
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
        "branch_points":
            branch_points,

        "historical_escape":
            historical_escape,

        "stage15_tracking":
            read_csv(paths["s15_tracking"]),

        "stage15_steps":
            read_csv(paths["s15_steps"]),

        "stage15_status_map":
            status_map,

        "stage3_sha256":
            actual_s3_sha,

        "stage9_sha256":
            actual_s9_sha,

        "stage15_sha256":
            actual_s15_sha,
    }


# =============================================================================
# State cloning
# =============================================================================

def capture_state(exp) -> dict:
    return {
        "model":
            copy.deepcopy(exp.model.state_dict()),

        "optimizer":
            copy.deepcopy(exp.optimizer.state_dict()),
    }


def load_captured_state(exp, captured: dict) -> None:
    exp.model.load_state_dict(
        copy.deepcopy(captured["model"])
    )

    exp.optimizer.load_state_dict(
        copy.deepcopy(captured["optimizer"])
    )

    for state in exp.optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(exp.device)


def exact_state_equality(exp_a, exp_b) -> dict:
    model_equal = True
    model_max_abs = 0.0

    for pa, pb in zip(
        exp_a.model.parameters(),
        exp_b.model.parameters(),
    ):
        diff = float(
            torch.max(
                torch.abs(
                    pa.detach() - pb.detach()
                )
            ).item()
        )

        model_max_abs = max(model_max_abs, diff)

        if not torch.equal(pa.detach(), pb.detach()):
            model_equal = False

    optimizer_equal = True
    optimizer_max_abs = 0.0

    params_a = list(exp_a.model.parameters())
    params_b = list(exp_b.model.parameters())

    for pa, pb in zip(params_a, params_b):
        sa = exp_a.optimizer.state[pa]
        sb = exp_b.optimizer.state[pb]

        if set(sa.keys()) != set(sb.keys()):
            optimizer_equal = False
            continue

        for key in sa:
            va = sa[key]
            vb = sb[key]

            if torch.is_tensor(va) and torch.is_tensor(vb):
                if va.shape != vb.shape:
                    optimizer_equal = False
                    continue

                diff = float(
                    torch.max(
                        torch.abs(
                            va.detach() - vb.detach()
                        )
                    ).item()
                )

                optimizer_max_abs = max(
                    optimizer_max_abs,
                    diff,
                )

                if not torch.equal(
                    va.detach(),
                    vb.detach(),
                ):
                    optimizer_equal = False

            else:
                if va != vb:
                    optimizer_equal = False

    return {
        "model_exact_equal":
            model_equal,

        "optimizer_exact_equal":
            optimizer_equal,

        "model_max_abs_difference":
            model_max_abs,

        "optimizer_max_abs_difference":
            optimizer_max_abs,

        "pass":
            bool(model_equal and optimizer_equal),
    }


# =============================================================================
# Core metrics
# =============================================================================

def state_metrics(exp, target_mode: int = TARGET_MODE) -> dict:
    residuals = exp.weak_residuals().detach()
    energy = residuals.square()

    t = target_mode - 1

    total_energy = energy.sum().clamp_min(1.0e-300)

    return {
        "relative_l2_error":
            exp.relative_l2_error(),

        "vpinn_loss":
            float(torch.mean(energy).item()),

        "target_loss":
            float((energy[t] / energy.numel()).item()),

        "target_mode_residual_energy_share":
            float((energy[t] / total_energy).item()),

        "target_mode_abs_residual":
            float(torch.abs(residuals[t]).item()),

        "dominant_residual_mode":
            int(torch.argmax(energy).item()) + 1,
    }


def ordinary_adam_step(exp) -> None:
    exp.optimizer.zero_grad(set_to_none=True)

    residuals = exp.weak_residuals()
    loss = residuals.square().mean()

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Non-finite ordinary-Adam loss."
        )

    loss.backward()
    exp.optimizer.step()


# =============================================================================
# Escape tracker
# =============================================================================

def new_escape_tracker() -> dict:
    return {
        "streak":
            0,

        "candidate_onset":
            -1,

        "certified_onset":
            -1,

        "certified_confirmation":
            -1,
    }


def update_escape_tracker(
    tracker: dict,
    epoch: int,
    metrics: dict,
) -> bool:

    qualifies = bool(
        metrics["relative_l2_error"] <= ESCAPE_REL_L2
        and
        metrics[
            "target_mode_residual_energy_share"
        ] <= ESCAPE_TARGET_SHARE
    )

    if qualifies:
        if tracker["streak"] == 0:
            tracker["candidate_onset"] = epoch

        tracker["streak"] += 1

    else:
        tracker["streak"] = 0
        tracker["candidate_onset"] = -1

    if (
        tracker["certified_onset"] < 0
        and
        tracker["streak"] >= ESCAPE_CONSECUTIVE
    ):
        tracker["certified_onset"] = (
            tracker["candidate_onset"]
        )

        tracker["certified_confirmation"] = epoch

        return True

    return False


# =============================================================================
# Stage-15 replay verification
# =============================================================================

def stage15_tracking_map(rows: List[dict]) -> dict:
    return {
        (int(r["seed"]), int(r["epoch"])):
            r
        for r in rows
    }


def stage15_failure_map(rows: List[dict]) -> dict:
    result = {}

    for r in rows:
        if r.get("status") == "NONLINEAR_SAFETY_FAILURE":
            result[int(r["seed"])] = r

    return result


def verify_stage15_tracking_state(
    seed: int,
    epoch: int,
    metrics: dict,
    reference_map: dict,
    tolerance: float = REPLAY_TOL,
) -> dict:

    key = (seed, epoch)

    if key not in reference_map:
        raise RuntimeError(
            f"Stage-15 tracking reference missing: {key}"
        )

    old = reference_map[key]

    comparisons = {
        "relative_l2_error": (
            float(old["relative_l2_error"]),
            float(metrics["relative_l2_error"]),
        ),

        "vpinn_loss": (
            float(old["vpinn_loss"]),
            float(metrics["vpinn_loss"]),
        ),

        "target_loss": (
            float(old["target_loss"]),
            float(metrics["target_loss"]),
        ),

        "target_share": (
            float(
                old[
                    "target_mode_residual_energy_share"
                ]
            ),
            float(
                metrics[
                    "target_mode_residual_energy_share"
                ]
            ),
        ),
    }

    diffs = {
        k: abs(a - b)
        for k, (a, b) in comparisons.items()
    }

    max_diff = max(diffs.values())

    return {
        "seed":
            seed,

        "epoch":
            epoch,

        "type":
            "TRACKING",

        "max_abs_difference":
            max_diff,

        "pass":
            bool(max_diff <= tolerance),
    }


def verify_stage15_failure_step(
    seed: int,
    result: dict,
    reference: dict,
    tolerance: float = REPLAY_TOL,
) -> dict:

    fields = [
        "lambda_mid",
        "interval_lower",
        "interval_upper",
        "pre_total_loss",
        "post_total_loss",
        "total_loss_change",
        "pre_target_loss",
        "post_target_loss",
        "target_loss_change",
    ]

    diffs = {}

    for field in fields:
        diffs[field] = abs(
            float(result[field])
            - float(reference[field])
        )

    max_diff = max(diffs.values())

    return {
        "seed":
            seed,

        "epoch":
            int(reference["epoch_after"]),

        "type":
            "FAILURE_STEP",

        "max_abs_difference":
            max_diff,

        "pass":
            bool(max_diff <= tolerance),
    }


# =============================================================================
# Branch runners
# =============================================================================

def run_control_branch(
    exp,
    seed: int,
    branch_epoch: int,
    out_rows: List[dict],
) -> dict:

    tracker = new_escape_tracker()

    start_metrics = state_metrics(exp)

    out_rows.append(
        {
            "seed":
                seed,

            "branch":
                "CONTROL",

            "epoch":
                branch_epoch,

            "event":
                "BRANCH_START",

            **start_metrics,
        }
    )

    epoch = branch_epoch

    while epoch < MAX_EPOCH:

        ordinary_adam_step(exp)
        epoch += 1

        if epoch % TRACK_INTERVAL != 0:
            continue

        metrics = state_metrics(exp)

        newly_certified = update_escape_tracker(
            tracker=tracker,
            epoch=epoch,
            metrics=metrics,
        )

        out_rows.append(
            {
                "seed":
                    seed,

                "branch":
                    "CONTROL",

                "epoch":
                    epoch,

                "event":
                    (
                        "ESCAPE_CONFIRMATION"
                        if newly_certified
                        else "TRACK"
                    ),

                **metrics,
            }
        )

        if newly_certified:
            break

    return {
        "seed":
            seed,

        "branch":
            "CONTROL",

        "status":
            (
                "CERTIFIED_ESCAPE"
                if tracker["certified_onset"] >= 0
                else "CENSORED"
            ),

        "escape_onset_epoch":
            tracker["certified_onset"],

        "escape_confirmation_epoch":
            tracker["certified_confirmation"],

        "max_epoch_reached":
            epoch,
    }


def run_adaptive_branch(
    exp,
    stage15,
    seed: int,
    branch_epoch: int,
    out_rows: List[dict],
    replay_checks: List[dict],
    s15_track_map: dict,
    s15_failure_map_: dict,
) -> dict:

    tracker = new_escape_tracker()

    start_metrics = state_metrics(exp)

    # Exact branch-state replay check.
    check = verify_stage15_tracking_state(
        seed=seed,
        epoch=branch_epoch,
        metrics=start_metrics,
        reference_map=s15_track_map,
    )

    replay_checks.append(check)

    if not check["pass"]:
        raise RuntimeError(
            f"Stage-15 branch-state replay failed seed={seed}."
        )

    out_rows.append(
        {
            "seed":
                seed,

            "branch":
                "ADAPTIVE",

            "epoch":
                branch_epoch,

            "event":
                "BRANCH_START",

            **start_metrics,
        }
    )

    epoch = branch_epoch

    failure_type = None
    failure_epoch = -1

    adaptive_steps = 0
    active_steps = 0
    inactive_steps = 0

    lambda_values = []

    while epoch < MAX_EPOCH:

        result = stage15.adaptive_midpoint_step(
            exp=exp,
            target_mode=TARGET_MODE,
        )

        if result["status"] in (
            "NO_STRICT_INTERVAL",
            "DEGENERATE_STEP",
        ):
            failure_type = result["status"]
            failure_epoch = epoch

            out_rows.append(
                {
                    "seed":
                        seed,

                    "branch":
                        "ADAPTIVE",

                    "epoch":
                        epoch,

                    "event":
                        failure_type,

                    **state_metrics(exp),
                }
            )

            break

        adaptive_steps += 1

        if result["intervention_active"]:
            active_steps += 1

            if result.get("lambda_mid") is not None:
                lambda_values.append(
                    float(result["lambda_mid"])
                )

        else:
            inactive_steps += 1

        epoch += 1

        if result["status"] == "NONLINEAR_SAFETY_FAILURE":
            failure_type = "NONLINEAR_SAFETY_FAILURE"
            failure_epoch = epoch

            # Seed 2 must reproduce the Stage-15 failure exactly.
            if seed in s15_failure_map_:
                check = verify_stage15_failure_step(
                    seed=seed,
                    result=result,
                    reference=s15_failure_map_[seed],
                )

                replay_checks.append(check)

                if not check["pass"]:
                    raise RuntimeError(
                        f"Stage-15 nonlinear failure replay failed "
                        f"seed={seed}."
                    )

            metrics = state_metrics(exp)

            out_rows.append(
                {
                    "seed":
                        seed,

                    "branch":
                        "ADAPTIVE",

                    "epoch":
                        epoch,

                    "event":
                        "NONLINEAR_SAFETY_FAILURE",

                    **metrics,
                }
            )

            break

        # Reproduce all Stage-15 stored tracking states up through 2700.
        if (
            epoch <= 2700
            and
            (seed, epoch) in s15_track_map
        ):
            metrics_repro = state_metrics(exp)

            check = verify_stage15_tracking_state(
                seed=seed,
                epoch=epoch,
                metrics=metrics_repro,
                reference_map=s15_track_map,
            )

            replay_checks.append(check)

            if not check["pass"]:
                raise RuntimeError(
                    f"Stage-15 tracking replay failed "
                    f"seed={seed}, epoch={epoch}."
                )

        if epoch % TRACK_INTERVAL != 0:
            continue

        metrics = state_metrics(exp)

        newly_certified = update_escape_tracker(
            tracker=tracker,
            epoch=epoch,
            metrics=metrics,
        )

        out_rows.append(
            {
                "seed":
                    seed,

                "branch":
                    "ADAPTIVE",

                "epoch":
                    epoch,

                "event":
                    (
                        "ESCAPE_CONFIRMATION"
                        if newly_certified
                        else "TRACK"
                    ),

                **metrics,

                "lambda_mid":
                    result.get("lambda_mid"),

                "intervention_active":
                    result.get("intervention_active"),
            }
        )

        if newly_certified:
            break

    if failure_type is not None:
        status = "SAFETY_FAILURE"

    elif tracker["certified_onset"] >= 0:
        status = "CERTIFIED_ESCAPE"

    else:
        status = "CENSORED"

    return {
        "seed":
            seed,

        "branch":
            "ADAPTIVE",

        "status":
            status,

        "failure_type":
            failure_type,

        "failure_epoch":
            failure_epoch,

        "escape_onset_epoch":
            tracker["certified_onset"],

        "escape_confirmation_epoch":
            tracker["certified_confirmation"],

        "max_epoch_reached":
            epoch,

        "adaptive_steps":
            adaptive_steps,

        "active_steps":
            active_steps,

        "inactive_steps":
            inactive_steps,

        "active_fraction":
            (
                active_steps / max(adaptive_steps, 1)
            ),

        "lambda_min":
            min(lambda_values)
            if lambda_values
            else None,

        "lambda_max":
            max(lambda_values)
            if lambda_values
            else None,

        "lambda_median":
            float(np.median(lambda_values))
            if lambda_values
            else None,
    }


# =============================================================================
# Plots
# =============================================================================

def plot_escape_times(
    paired: List[dict],
    path: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    seeds = [str(int(r["seed"])) for r in paired]
    x = np.arange(len(seeds))
    width = 0.35

    control = [
        (
            float(r["control_escape_onset_epoch"])
            if int(r["control_escape_onset_epoch"]) >= 0
            else MAX_EPOCH
        )
        for r in paired
    ]

    adaptive = [
        (
            float(r["adaptive_escape_onset_epoch"])
            if int(r["adaptive_escape_onset_epoch"]) >= 0
            else MAX_EPOCH
        )
        for r in paired
    ]

    ax.bar(
        x - width / 2,
        control,
        width,
        label="Matched Adam control",
    )

    ax.bar(
        x + width / 2,
        adaptive,
        width,
        label="Adaptive midpoint",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(seeds)

    ax.set_xlabel("Conflict-active seed")
    ax.set_ylabel("Certified escape onset epoch")
    ax.set_title("Matched-state certified escape-time comparison")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    rows: List[dict],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    log_y: bool,
) -> None:

    fig, ax = plt.subplots(figsize=(10.2, 6.0))

    for seed in sorted(set(int(r["seed"]) for r in rows)):
        for branch in ("CONTROL", "ADAPTIVE"):

            rr = [
                r for r in rows
                if int(r["seed"]) == seed
                and r["branch"] == branch
                and metric in r
                and r[metric] not in (None, "")
            ]

            rr.sort(key=lambda x: int(x["epoch"]))

            if not rr:
                continue

            ax.plot(
                [int(r["epoch"]) for r in rr],
                [float(r[metric]) for r in rr],
                linewidth=1.2,
                label=f"seed {seed} {branch}",
            )

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)

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

    stage10_dir = Path(args.stage10_dir)
    if not stage10_dir.is_absolute():
        stage10_dir = root / stage10_dir

    stage15_script = Path(args.stage15_script)
    if not stage15_script.is_absolute():
        stage15_script = root / stage15_script

    stage15_dir = Path(args.stage15_dir)
    if not stage15_dir.is_absolute():
        stage15_dir = root / stage15_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage5_dir=stage5_dir,
        stage9_script=stage9_script,
        stage10_dir=stage10_dir,
        stage15_script=stage15_script,
        stage15_dir=stage15_dir,
    )

    stage9 = load_module(
        stage9_script,
        "vpinn_stage9_replay_stage16",
    )

    stage15 = load_module(
        stage15_script,
        "vpinn_stage15_rule_stage16",
    )

    stage3 = stage9.load_stage3_module(stage3_script)

    s15_track_map = stage15_tracking_map(
        pf["stage15_tracking"]
    )

    s15_failure_map_ = stage15_failure_map(
        pf["stage15_steps"]
    )

    precommitment = {
        "stage":
            "matched_state_certified_escape_time_causal_comparison",

        "active_branch_points":
            pf["branch_points"],

        "branches": [
            "CONTROL ordinary Adam",
            "ADAPTIVE Stage-15 midpoint rule",
        ],

        "identical_start_state_required":
            True,

        "max_epoch":
            MAX_EPOCH,

        "tracking_interval":
            TRACK_INTERVAL,

        "escape_definition": {
            "relative_l2_le":
                ESCAPE_REL_L2,

            "target_share_le":
                ESCAPE_TARGET_SHARE,

            "consecutive_observations":
                ESCAPE_CONSECUTIVE,
        },

        "adaptive_failure_counts_as_nonacceleration":
            True,

        "primary_group_gate": (
            ">=3/4 adaptive accelerated AND "
            ">=3/4 adaptive certified escape AND "
            "<=1/4 adaptive safety failure"
        ),

        "no_posthoc_rescue_after_stage16":
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

        "stage15_script_sha256":
            pf["stage15_sha256"],

        "stage16_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment":
            precommitment,
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 158)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 16 MATCHED-STATE CERTIFIED ESCAPE COMPARISON"
    )
    print("=" * 158)

    print(f"device                    : {device}")
    print(f"active branch points      : {pf['branch_points']}")
    print(f"maximum epoch             : {MAX_EPOCH}")
    print(
        "primary comparator        : ordinary Adam from IDENTICAL branch state"
    )
    print(
        "adaptive failure          : counts as non-acceleration"
    )
    print("=" * 158)

    all_trajectory_rows: List[dict] = []
    paired_rows: List[dict] = []
    clone_checks: List[dict] = []
    replay_checks: List[dict] = []

    global_start = time.perf_counter()

    for seed, branch_epoch in pf["branch_points"].items():

        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        # -------------------------------------------------------------
        # Reconstruct the exact common branch state once.
        # -------------------------------------------------------------
        base_exp = stage9.make_experiment(
            stage3=stage3,
            device=device,
            seed=seed,
            out_dir=out_dir / f"seed_{seed:03d}" / "_base",
        )

        stage9.load_locked_checkpoint(
            exp=base_exp,
            checkpoint_path=checkpoint,
            expected_seed=seed,
        )

        for _epoch in range(2500, branch_epoch):
            stage9.intervention_step(
                exp=base_exp,
                branch="REFLECT",
                target_mode=TARGET_MODE,
            )

        captured = capture_state(base_exp)

        # -------------------------------------------------------------
        # Construct exact matched branches.
        # -------------------------------------------------------------
        control_exp = stage9.make_experiment(
            stage3=stage3,
            device=device,
            seed=seed,
            out_dir=out_dir / f"seed_{seed:03d}" / "control",
        )

        adaptive_exp = stage9.make_experiment(
            stage3=stage3,
            device=device,
            seed=seed,
            out_dir=out_dir / f"seed_{seed:03d}" / "adaptive",
        )

        load_captured_state(control_exp, captured)
        load_captured_state(adaptive_exp, captured)

        clone_check = exact_state_equality(
            control_exp,
            adaptive_exp,
        )

        clone_check.update(
            {
                "seed":
                    seed,

                "branch_epoch":
                    branch_epoch,
            }
        )

        clone_checks.append(clone_check)

        if not clone_check["pass"]:
            raise RuntimeError(
                f"Matched-state clone equality failed seed={seed}."
            )

        print()
        print("-" * 158)
        print(
            f"SEED {seed} | branch={branch_epoch} | "
            f"exact matched state: PASS"
        )

        # -------------------------------------------------------------
        # Run branches.
        # -------------------------------------------------------------
        control_summary = run_control_branch(
            exp=control_exp,
            seed=seed,
            branch_epoch=branch_epoch,
            out_rows=all_trajectory_rows,
        )

        adaptive_summary = run_adaptive_branch(
            exp=adaptive_exp,
            stage15=stage15,
            seed=seed,
            branch_epoch=branch_epoch,
            out_rows=all_trajectory_rows,
            replay_checks=replay_checks,
            s15_track_map=s15_track_map,
            s15_failure_map_=s15_failure_map_,
        )

        control_onset = int(
            control_summary["escape_onset_epoch"]
        )

        adaptive_onset = int(
            adaptive_summary["escape_onset_epoch"]
        )

        both_escape = bool(
            control_onset >= 0
            and adaptive_onset >= 0
        )

        accelerated = bool(
            adaptive_summary["status"] == "CERTIFIED_ESCAPE"
            and
            both_escape
            and
            adaptive_onset < control_onset
        )

        if both_escape:
            epochs_saved = (
                control_onset - adaptive_onset
            )

            control_delay = (
                control_onset - branch_epoch
            )

            adaptive_delay = (
                adaptive_onset - branch_epoch
            )

            delay_reduction_pct = (
                100.0
                * (
                    1.0
                    - adaptive_delay
                    / max(control_delay, 1)
                )
            )

            delay_speedup = (
                control_delay
                / max(adaptive_delay, 1)
            )

        else:
            epochs_saved = None
            delay_reduction_pct = None
            delay_speedup = None

        historical = pf["historical_escape"][seed]

        paired = {
            "seed":
                seed,

            "branch_epoch":
                branch_epoch,

            "control_status":
                control_summary["status"],

            "control_escape_onset_epoch":
                control_onset,

            "control_escape_confirmation_epoch":
                control_summary[
                    "escape_confirmation_epoch"
                ],

            "adaptive_status":
                adaptive_summary["status"],

            "adaptive_failure_type":
                adaptive_summary.get("failure_type"),

            "adaptive_failure_epoch":
                adaptive_summary.get("failure_epoch"),

            "adaptive_escape_onset_epoch":
                adaptive_onset,

            "adaptive_escape_confirmation_epoch":
                adaptive_summary[
                    "escape_confirmation_epoch"
                ],

            "adaptive_accelerated":
                accelerated,

            "epochs_saved":
                epochs_saved,

            "escape_delay_reduction_percent":
                delay_reduction_pct,

            "escape_delay_speedup":
                delay_speedup,

            "historical_stage5_escape_onset":
                historical["onset"],

            "adaptive_vs_historical_stage5_epochs":
                (
                    historical["onset"] - adaptive_onset
                    if adaptive_onset >= 0
                    else None
                ),

            "adaptive_active_fraction":
                adaptive_summary.get("active_fraction"),

            "adaptive_lambda_min":
                adaptive_summary.get("lambda_min"),

            "adaptive_lambda_max":
                adaptive_summary.get("lambda_max"),

            "adaptive_lambda_median":
                adaptive_summary.get("lambda_median"),
        }

        paired_rows.append(paired)

        print(
            f"  CONTROL  : status={control_summary['status']} | "
            f"onset={control_onset}"
        )

        print(
            f"  ADAPTIVE : status={adaptive_summary['status']} | "
            f"onset={adaptive_onset} | "
            f"failure={adaptive_summary.get('failure_type')}"
        )

        print(
            f"  paired result: "
            f"{'ACCELERATED' if accelerated else 'NOT ACCELERATED'}"
        )

        if epochs_saved is not None:
            print(
                f"  epochs saved={epochs_saved} | "
                f"delay reduction={delay_reduction_pct:.2f}%"
            )

    # -----------------------------------------------------------------
    # Aggregate decision.
    # -----------------------------------------------------------------
    accelerated_count = sum(
        int(bool(r["adaptive_accelerated"]))
        for r in paired_rows
    )

    adaptive_escape_count = sum(
        int(
            r["adaptive_status"]
            == "CERTIFIED_ESCAPE"
        )
        for r in paired_rows
    )

    control_escape_count = sum(
        int(
            r["control_status"]
            == "CERTIFIED_ESCAPE"
        )
        for r in paired_rows
    )

    adaptive_failure_count = sum(
        int(
            r["adaptive_status"]
            == "SAFETY_FAILURE"
        )
        for r in paired_rows
    )

    gate_pass = bool(
        accelerated_count >= 3
        and
        adaptive_escape_count >= 3
        and
        adaptive_failure_count <= 1
    )

    saved_values = [
        float(r["epochs_saved"])
        for r in paired_rows
        if r["epochs_saved"] is not None
    ]

    reduction_values = [
        float(r["escape_delay_reduction_percent"])
        for r in paired_rows
        if r["escape_delay_reduction_percent"] is not None
    ]

    speedup_values = [
        float(r["escape_delay_speedup"])
        for r in paired_rows
        if r["escape_delay_speedup"] is not None
    ]

    if gate_pass:
        route_class = (
            "matched_state_adaptive_escape_acceleration_supported"
        )

        next_route = (
            "linkedin_final_evidence_package_with_escape_claim"
        )

    else:
        route_class = (
            "matched_state_escape_acceleration_not_supported"
        )

        next_route = (
            "linkedin_final_mechanism_package_without_acceleration_claim"
        )

    decision = {
        "n_active_seeds":
            len(paired_rows),

        "control_certified_escape_count":
            control_escape_count,

        "adaptive_certified_escape_count":
            adaptive_escape_count,

        "adaptive_safety_failure_count":
            adaptive_failure_count,

        "adaptive_accelerated_count":
            accelerated_count,

        "primary_group_gate_pass":
            gate_pass,

        "all_matched_state_clone_checks_pass":
            all(r["pass"] for r in clone_checks),

        "all_stage15_replay_checks_pass":
            all(r["pass"] for r in replay_checks),

        "median_epochs_saved_among_both_escape": (
            float(np.median(saved_values))
            if saved_values
            else None
        ),

        "median_escape_delay_reduction_percent": (
            float(np.median(reduction_values))
            if reduction_values
            else None
        ),

        "median_escape_delay_speedup": (
            float(np.median(speedup_values))
            if speedup_values
            else None
        ),

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "The primary claim is a matched-state causal comparison. "
            "Adaptive safety failures count as non-acceleration. "
            "Historical Stage-5 comparisons are secondary because their "
            "starting state differs from the matched branch state. "
            "No additional optimizer rescue sweep is authorized after Stage 16."
        ),
    }

    write_csv(
        out_dir / "paired_escape_summary.csv",
        paired_rows,
    )

    write_csv(
        out_dir / "trajectory_metrics.csv",
        all_trajectory_rows,
    )

    write_csv(
        out_dir / "matched_state_clone_checks.csv",
        clone_checks,
    )

    write_csv(
        out_dir / "stage15_replay_checks.csv",
        replay_checks,
    )

    write_json(
        out_dir / "decision.json",
        decision,
    )

    # A compact JSON deliberately useful for the final LinkedIn evidence card.
    write_json(
        out_dir / "linkedin_candidate_metrics.json",
        {
            "active_seeds":
                4,

            "adaptive_accelerated":
                accelerated_count,

            "adaptive_certified_escape":
                adaptive_escape_count,

            "adaptive_safety_failures":
                adaptive_failure_count,

            "median_epochs_saved":
                decision[
                    "median_epochs_saved_among_both_escape"
                ],

            "median_escape_delay_reduction_percent":
                decision[
                    "median_escape_delay_reduction_percent"
                ],

            "matched_state_gate_pass":
                gate_pass,
        },
    )

    # -----------------------------------------------------------------
    # Figures.
    # -----------------------------------------------------------------
    plot_escape_times(
        paired_rows,
        out_dir / "matched_escape_times.png",
    )

    plot_trajectories(
        rows=all_trajectory_rows,
        metric="relative_l2_error",
        ylabel="Relative L2 error",
        title="Matched-state Adam vs adaptive Pareto midpoint",
        path=out_dir / "matched_relative_l2_trajectories.png",
        log_y=True,
    )

    plot_trajectories(
        rows=all_trajectory_rows,
        metric="target_mode_residual_energy_share",
        ylabel="Target residual-energy share",
        title="Matched-state release of the unresolved weak mode",
        path=out_dir / "matched_target_share_trajectories.png",
        log_y=False,
    )

    elapsed = time.perf_counter() - global_start

    # -----------------------------------------------------------------
    # Console summary.
    # -----------------------------------------------------------------
    lines = []

    lines.append("=" * 170)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 16 MATCHED-STATE CERTIFIED ESCAPE SUMMARY"
    )
    lines.append("=" * 170)

    lines.append(
        "seed | branch epoch | control onset | adaptive onset | adaptive status     | "
        "failure                  | epochs saved | delay reduction | result"
    )

    lines.append("-" * 170)

    for r in paired_rows:

        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['branch_epoch']):12d} | "
            f"{int(r['control_escape_onset_epoch']):13d} | "
            f"{int(r['adaptive_escape_onset_epoch']):14d} | "
            f"{str(r['adaptive_status']):19s} | "
            f"{str(r['adaptive_failure_type']):24s} | "
            f"{str(r['epochs_saved']):12s} | "
            f"{str(r['escape_delay_reduction_percent']):15s} | "
            f"{'ACCEL' if r['adaptive_accelerated'] else 'NO'}"
        )

    lines.append("-" * 170)

    lines.append(
        f"matched-state clone checks          : "
        f"{sum(int(r['pass']) for r in clone_checks)}/"
        f"{len(clone_checks)} PASS"
    )

    lines.append(
        f"Stage-15 replay checks              : "
        f"{sum(int(r['pass']) for r in replay_checks)}/"
        f"{len(replay_checks)} PASS"
    )

    lines.append(
        f"CONTROL certified escape            : "
        f"{control_escape_count}/4"
    )

    lines.append(
        f"ADAPTIVE certified escape           : "
        f"{adaptive_escape_count}/4"
    )

    lines.append(
        f"ADAPTIVE safety failure             : "
        f"{adaptive_failure_count}/4"
    )

    lines.append(
        f"ADAPTIVE accelerated                : "
        f"{accelerated_count}/4"
    )

    lines.append(
        f"median epochs saved                 : "
        f"{decision['median_epochs_saved_among_both_escape']}"
    )

    lines.append(
        f"median escape-delay reduction       : "
        f"{decision['median_escape_delay_reduction_percent']}%"
    )

    lines.append(
        f"primary group gate                  : "
        f"{'PASS' if gate_pass else 'FAIL'}"
    )

    lines.append(
        f"route class                         : "
        f"{route_class}"
    )

    lines.append(
        f"next route                          : "
        f"{next_route}"
    )

    lines.append(
        f"elapsed seconds                     : "
        f"{elapsed:.2f}"
    )

    lines.append("=" * 170)

    lines.append(
        "Guardrail: Stage 16 is the final optimizer experiment for this LinkedIn "
        "thread. A FAIL changes the public claim; it does not authorize another "
        "post-hoc rescue sweep."
    )

    lines.append("=" * 170)

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
