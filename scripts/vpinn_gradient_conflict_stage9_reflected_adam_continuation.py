#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 9
State-Consistent PROJECT vs REFLECT Continuation
================================================

Scientific purpose
------------------
Stage 8 established a clean one-step causal fact:

  * REFLECT preserves the exact inherited Adam step norm,
  * flips only the component that is uphill for the unresolved target
    residual R_9^2,
  * improves R_9^2 relative to CONTROL in all five seeds,
  * does not increase total VPINN loss in the one-step audit.

But the one-step effect is tiny in several seeds. Therefore the next required
question is not "is the direction locally better?" but:

    Does repeated, norm-preserving correction of target-uphill Adam
    displacements actually accelerate escape from the VPINN plateau?

This stage performs the paired continuation.

Historical CONTROL
------------------
The ordinary Adam baseline is NOT rerun. Stage 5 already provides a fully
validated historical continuation from the exact same epoch-2500 checkpoints.

Intervention branches
---------------------
PROJECT:
    If the candidate Adam displacement Delta_A is target-uphill,

        <g_t, Delta_A> > 0,

    remove only that component:

        Delta_P =
            Delta_A
            - (<g_t,Delta_A>/||g_t||^2) g_t.

    Otherwise use Delta_P = Delta_A.

REFLECT:
    If target-uphill,

        Delta_R =
            Delta_A
            - 2(<g_t,Delta_A>/||g_t||^2) g_t.

    Otherwise use Delta_R = Delta_A.

Crucial optimizer-state consistency
-----------------------------------
At EVERY step:

  1) compute the current raw VPINN gradient;
  2) let the real PyTorch Adam optimizer perform its ordinary step;
  3) KEEP Adam's newly updated first/second moments and step counter;
  4) overwrite only the PARAMETER displacement with PROJECT or REFLECT.

Thus the optimizer state evolves exactly according to the raw gradient at the
current intervention trajectory. We do not freeze moments, reset moments, or
manually invent Adam state.

REFLECT remains norm preserving at every active intervention:

    ||Delta_R|| = ||Delta_A||.

PROJECT is diagnostic and uses the minimum-distance target-neutral correction.

Historical escape definition
----------------------------
Inherited unchanged from Stage 5:

    relative L2 <= 1e-2
    AND
    target residual-energy share <= 0.20

for THREE consecutive observations spaced 25 epochs apart.

Primary endpoint
----------------
For each seed:

    acceleration_REFLECT =
        Stage5_baseline_escape_onset
        - REFLECT_escape_onset.

A seed accelerates iff REFLECT has a certified escape onset strictly earlier
than the historical baseline onset.

Primary group gate:
    >= 4/5 REFLECT seeds accelerate.

PROJECT is mechanistic context, not a rescue fallback.

Mechanism interpretation
------------------------
If REFLECT accelerates >=4/5 and PROJECT <=1/5:
    direction-specific target descent is supported.

If REFLECT accelerates >=4/5 and PROJECT >=4/5:
    merely preventing target-uphill motion may be sufficient.

If REFLECT accelerates >=4/5 and PROJECT is 2-3/5:
    mixed suppression-vs-reflection mechanism.

If REFLECT fails:
    do not promote the one-step result to an escape-time mechanism.

Efficiency
----------
Each intervention branch for each seed runs only until the PRE-EXISTING
Stage-5 baseline confirmation epoch. Once a branch has certified escape and
has reached the common endpoint epoch 2700, that branch stops.

No baseline recomputation.
No architecture sweep.
No optimizer reset sweep.

Preflight
---------
* Stage-8 continuation route must be authorized.
* Stage-3 solver identity must match.
* First intervention step must reproduce Stage-8 one-step metrics to strict
  tolerance before continuation is accepted.
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


BRANCHES = ("PROJECT", "REFLECT")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-9 state-consistent PROJECT/REFLECT continuation."
    )
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--track-interval", type=int, default=25)
    p.add_argument("--common-endpoint-epoch", type=int, default=2700)

    p.add_argument(
        "--stage3-script",
        default="vpinn_gradient_conflict_stage3_frequency_transfer.py",
    )
    p.add_argument(
        "--stage5-dir",
        default="vpinn_gradient_conflict_stage5_escape_time",
    )
    p.add_argument(
        "--stage8-dir",
        default="vpinn_gradient_conflict_stage8_adam_target_reflection",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage9_reflected_adam_continuation",
    )

    args = p.parse_args()

    if args.track_interval != 25:
        raise ValueError(
            "Stage 9 is precommitted to the Stage-5 25-epoch tracking grid."
        )

    if args.common_endpoint_epoch <= 2500:
        raise ValueError("--common-endpoint-epoch must be > 2500.")

    if args.common_endpoint_epoch % args.track_interval != 0:
        raise ValueError(
            "--common-endpoint-epoch must be on the tracking grid."
        )

    return args


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

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    normalized = [
        {key: row.get(key, None) for key in fieldnames}
        for row in rows
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)


def load_stage3_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Stage-3 solver not found: {path}")

    spec = importlib.util.spec_from_file_location(
        "vpinn_stage3_solver_stage9", str(path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Stage-3 solver: {path}")

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
    stage8_dir: Path,
) -> dict:
    paths = {
        "s5_manifest": stage5_dir / "manifest.json",
        "s5_decision": stage5_dir / "decision.json",
        "s5_summary": stage5_dir / "escape_time_summary.csv",
        "s5_aggregate": stage5_dir / "aggregate_postlock_metrics.csv",
        "s8_manifest": stage8_dir / "manifest.json",
        "s8_decision": stage8_dir / "decision.json",
        "s8_results": stage8_dir / "branch_results.csv",
        "s8_runtime": stage8_dir / "control_runtime_verification.json",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    s5_manifest = read_json(paths["s5_manifest"])
    s5_decision = read_json(paths["s5_decision"])
    s8_manifest = read_json(paths["s8_manifest"])
    s8_decision = read_json(paths["s8_decision"])
    s8_runtime = read_json(paths["s8_runtime"])

    if s8_decision.get("next_route") != (
        "norm_preserving_reflected_adam_continuation"
    ):
        raise RuntimeError(
            "Stage 8 did not authorize reflected-Adam continuation."
        )

    if not bool(s8_decision.get("continuation_authorized", False)):
        raise RuntimeError("Stage-8 continuation_authorized is false.")

    if not bool(s8_decision.get("group_gate_pass", False)):
        raise RuntimeError("Stage-8 group gate is not PASS.")

    if not bool(s8_runtime.get("all_pass", False)):
        raise RuntimeError("Stage-8 CONTROL runtime verification is not PASS.")

    actual_sha = sha256_file(stage3_script)

    if actual_sha != s5_manifest.get("stage3_solver_sha256"):
        raise RuntimeError("Stage-3 SHA mismatch vs Stage 5.")

    if actual_sha != s8_manifest.get("stage3_solver_sha256"):
        raise RuntimeError("Stage-3 SHA mismatch vs Stage 8.")

    summary = read_csv(paths["s5_summary"])
    aggregate = read_csv(paths["s5_aggregate"])
    stage8_rows = read_csv(paths["s8_results"])

    seeds = [int(r["seed"]) for r in summary]
    if seeds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Unexpected Stage-5 seeds: {seeds}")

    baseline = {
        int(r["seed"]): {
            "release_onset": int(r["release_onset_epoch"]),
            "escape_onset": int(r["certified_escape_onset_epoch"]),
            "escape_confirmation":
                int(r["certified_escape_confirmation_epoch"]),
        }
        for r in summary
    }

    earliest_baseline_escape = min(
        x["escape_onset"] for x in baseline.values()
    )

    return {
        "seeds": seeds,
        "stage3_sha256": actual_sha,
        "baseline": baseline,
        "aggregate": aggregate,
        "stage8_rows": stage8_rows,
        "earliest_baseline_escape": earliest_baseline_escape,
    }


# =============================================================================
# Experiment + checkpoint
# =============================================================================

def make_experiment(stage3, device, seed: int, out_dir: Path):
    cfg = stage3.Config(
        seed=seed,
        device=str(device),
        epochs=4000,
        learning_rate=1.0e-3,
        width=32,
        depth=3,
        n_test=24,
        n_quad=256,
        n_eval=4001,
        modes=(9,),
        reference_mode=7,
        reference_amplitude=0.15,
        track_interval=25,
        diagnostic_epochs=(2500,),
        output_dir=str(out_dir),
    )

    return stage3.ModeExperiment(
        cfg=cfg,
        device=device,
        mode=9,
        out_dir=out_dir,
    )


def load_locked_checkpoint(
    exp,
    checkpoint_path: Path,
    expected_seed: int,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=exp.device,
        weights_only=False,
    )

    if int(checkpoint["seed"]) != expected_seed:
        raise RuntimeError("Checkpoint seed mismatch.")
    if int(checkpoint["epoch"]) != 2500:
        raise RuntimeError("Checkpoint epoch is not 2500.")
    if int(checkpoint["mode"]) != 9:
        raise RuntimeError("Checkpoint mode is not 9.")

    exp.model.load_state_dict(checkpoint["model_state_dict"])
    exp.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for state in exp.optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(exp.device)


# =============================================================================
# State metrics
# =============================================================================

def state_metrics(exp, target_mode: int = 9) -> dict:
    residuals = exp.weak_residuals().detach()
    energy = residuals.square()
    total_energy = energy.sum().clamp_min(1.0e-300)

    t = target_mode - 1
    dominant = int(torch.argmax(energy).item())

    return {
        "relative_l2_error": exp.relative_l2_error(),
        "vpinn_loss": float(torch.mean(energy).item()),
        "target_loss_unscaled": float(energy[t].item()),
        "target_mode_abs_residual": float(torch.abs(residuals[t]).item()),
        "target_mode_residual_energy_share": float(
            (energy[t] / total_energy).item()
        ),
        "dominant_residual_mode": dominant + 1,
    }


# =============================================================================
# State-consistent intervention step
# =============================================================================

def intervention_step(exp, branch: str, target_mode: int = 9) -> dict:
    if branch not in BRANCHES:
        raise ValueError(f"Unknown branch: {branch}")

    exp.optimizer.zero_grad(set_to_none=True)

    residuals = exp.weak_residuals()
    params = tuple(p for p in exp.model.parameters() if p.requires_grad)

    t = target_mode - 1
    M = residuals.numel()

    target_loss = residuals[t].square() / M
    total_loss = residuals.square().mean()

    # Target gradient for geometry only.
    gt_parts = torch.autograd.grad(
        target_loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )
    gt = flatten(gt_parts).detach()

    # Raw total gradient is exactly what Adam consumes.
    total_loss.backward()

    before_parts = [
        p.detach().clone()
        for p in params
    ]
    before = flatten(before_parts)

    # IMPORTANT: real PyTorch Adam step.
    # This updates both optimizer state and parameters.
    exp.optimizer.step()

    after_candidate = flatten(
        [p.detach().clone() for p in params]
    )

    delta_adam = after_candidate - before

    gt2 = torch.dot(gt, gt)
    dot = torch.dot(gt, delta_adam)

    if float(gt2.item()) <= 0.0:
        raise RuntimeError("Target gradient norm is zero.")

    active = bool(dot.item() > 0.0)

    if active:
        component = (dot / gt2) * gt

        if branch == "PROJECT":
            delta = delta_adam - component
        elif branch == "REFLECT":
            delta = delta_adam - 2.0 * component
        else:
            raise AssertionError
    else:
        delta = delta_adam

    # Re-place model parameters while KEEPING the optimizer state produced
    # by the ordinary Adam step above.
    offset = 0

    with torch.no_grad():
        for p, p_before in zip(params, before_parts):
            n = p.numel()
            corrected = (
                p_before
                + delta[offset:offset+n].reshape_as(p)
            )
            p.copy_(corrected)
            offset += n

    if offset != delta.numel():
        raise RuntimeError("Parameter/displacement size mismatch.")

    adam_norm = torch.linalg.vector_norm(delta_adam)
    branch_norm = torch.linalg.vector_norm(delta)

    adam_alignment = float(
        (
            -dot
            / torch.clamp(
                torch.linalg.vector_norm(gt) * adam_norm,
                min=1.0e-300,
            )
        ).item()
    )

    branch_dot = torch.dot(gt, delta)

    branch_alignment = float(
        (
            -branch_dot
            / torch.clamp(
                torch.linalg.vector_norm(gt) * branch_norm,
                min=1.0e-300,
            )
        ).item()
    )

    if active and branch == "PROJECT":
        if abs(float(branch_dot.item())) > (
            5.0e-11
            * max(
                1.0,
                float(
                    torch.linalg.vector_norm(gt).item()
                    * branch_norm.item()
                ),
            )
        ):
            raise RuntimeError("PROJECT orthogonality identity failed.")

    if active and branch == "REFLECT":
        norm_ratio = float(
            (
                branch_norm
                / torch.clamp(adam_norm, min=1.0e-300)
            ).item()
        )

        if abs(norm_ratio - 1.0) > 1.0e-10:
            raise RuntimeError(
                f"REFLECT norm preservation failed: {norm_ratio:.16e}"
            )

        if abs(branch_alignment + adam_alignment) > 1.0e-10:
            raise RuntimeError(
                "REFLECT target-alignment identity failed."
            )

    return {
        "intervention_active": active,
        "candidate_adam_target_descent_alignment":
            adam_alignment,
        "applied_target_descent_alignment":
            branch_alignment,
        "candidate_adam_update_norm":
            float(adam_norm.item()),
        "applied_update_norm":
            float(branch_norm.item()),
        "applied_over_candidate_update_norm":
            float(
                (
                    branch_norm
                    / torch.clamp(adam_norm, min=1.0e-300)
                ).item()
            ),
    }


# =============================================================================
# First-step Stage-8 reproduction
# =============================================================================

def stage8_expected_map(rows: List[dict]) -> dict:
    return {
        (int(r["seed"]), str(r["branch"])): r
        for r in rows
    }


def verify_first_step_against_stage8(
    seed: int,
    branch: str,
    post: dict,
    step_diag: dict,
    expected_map: dict,
    tolerance: float = 1.0e-10,
) -> dict:
    key = (seed, branch)

    if key not in expected_map:
        raise RuntimeError(
            f"Stage-8 expected row missing for seed={seed}, branch={branch}."
        )

    old = expected_map[key]

    fields = {
        "post_relative_l2_error": (
            float(old["post_relative_l2_error"]),
            float(post["relative_l2_error"]),
        ),
        "post_vpinn_loss": (
            float(old["post_vpinn_loss"]),
            float(post["vpinn_loss"]),
        ),
        "post_target_loss_unscaled": (
            float(old["post_target_loss_unscaled"]),
            float(post["target_loss_unscaled"]),
        ),
        "post_target_share": (
            float(old["post_target_share"]),
            float(post["target_mode_residual_energy_share"]),
        ),
        "target_descent_alignment": (
            float(old["target_descent_alignment"]),
            float(step_diag["applied_target_descent_alignment"]),
        ),
        "update_norm_ratio": (
            float(old["update_norm_ratio_vs_control"]),
            float(step_diag["applied_over_candidate_update_norm"]),
        ),
    }

    diffs = {
        name: abs(a - b)
        for name, (a, b) in fields.items()
    }

    max_diff = max(diffs.values())

    return {
        "seed": seed,
        "branch": branch,
        "tolerance": tolerance,
        "max_abs_difference": max_diff,
        "field_abs_differences": diffs,
        "pass": bool(max_diff <= tolerance),
    }


# =============================================================================
# Historical baseline lookup
# =============================================================================

def aggregate_map(rows: List[dict]) -> dict:
    return {
        (int(r["seed"]), int(r["epoch"])): r
        for r in rows
    }


# =============================================================================
# Single branch continuation
# =============================================================================

def run_branch(
    stage3,
    device,
    seed: int,
    branch: str,
    checkpoint_path: Path,
    branch_dir: Path,
    historical_baseline: dict,
    historical_map: dict,
    stage8_map: dict,
    track_interval: int,
    common_endpoint_epoch: int,
) -> tuple[list[dict], dict, dict]:

    exp = make_experiment(stage3, device, seed, branch_dir)
    load_locked_checkpoint(exp, checkpoint_path, seed)

    baseline_onset = int(historical_baseline["escape_onset"])
    baseline_confirmation = int(
        historical_baseline["escape_confirmation"]
    )

    max_epoch = baseline_confirmation

    rows: List[dict] = []

    active_steps = 0
    total_steps = 0
    active_candidate_alignments = []
    applied_alignments = []

    consecutive_escape = 0
    candidate_escape_onset = -1
    certified_escape_onset = -1
    certified_escape_confirmation = -1

    first_step_check = None
    common_endpoint_row = None

    # Epoch-2500 pre-step state.
    pre2500 = state_metrics(exp)

    rows.append(
        {
            "seed": seed,
            "branch": branch,
            "epoch": 2500,
            **pre2500,
            "intervention_active_step": None,
            "candidate_adam_target_descent_alignment": None,
            "applied_target_descent_alignment": None,
            "candidate_adam_update_norm": None,
            "applied_update_norm": None,
            "applied_over_candidate_update_norm": None,
            "qualifies_escape": False,
        }
    )

    # Apply steps 2500 -> 2501, ..., until stop.
    epoch = 2500

    while epoch < max_epoch:
        step_diag = intervention_step(exp, branch)
        total_steps += 1

        if step_diag["intervention_active"]:
            active_steps += 1
            active_candidate_alignments.append(
                step_diag[
                    "candidate_adam_target_descent_alignment"
                ]
            )

        applied_alignments.append(
            step_diag["applied_target_descent_alignment"]
        )

        epoch += 1

        # Strict first-step reproduction against Stage 8.
        if epoch == 2501:
            post = state_metrics(exp)

            first_step_check = verify_first_step_against_stage8(
                seed=seed,
                branch=branch,
                post=post,
                step_diag=step_diag,
                expected_map=stage8_map,
                tolerance=1.0e-10,
            )

            if not first_step_check["pass"]:
                raise RuntimeError(
                    f"Stage-8 first-step reproduction failed: "
                    f"seed={seed}, branch={branch}, "
                    f"max diff={first_step_check['max_abs_difference']:.3e}"
                )

        is_track = (epoch % track_interval == 0)

        if not is_track:
            continue

        state = state_metrics(exp)

        qualifies = bool(
            state["relative_l2_error"] <= 1.0e-2
            and state["target_mode_residual_energy_share"] <= 0.20
        )

        if qualifies:
            if consecutive_escape == 0:
                candidate_escape_onset = epoch
            consecutive_escape += 1
        else:
            consecutive_escape = 0
            candidate_escape_onset = -1

        row = {
            "seed": seed,
            "branch": branch,
            "epoch": epoch,
            **state,
            "intervention_active_step":
                step_diag["intervention_active"],
            "candidate_adam_target_descent_alignment":
                step_diag[
                    "candidate_adam_target_descent_alignment"
                ],
            "applied_target_descent_alignment":
                step_diag["applied_target_descent_alignment"],
            "candidate_adam_update_norm":
                step_diag["candidate_adam_update_norm"],
            "applied_update_norm":
                step_diag["applied_update_norm"],
            "applied_over_candidate_update_norm":
                step_diag["applied_over_candidate_update_norm"],
            "qualifies_escape": qualifies,
        }

        rows.append(row)

        if epoch == common_endpoint_epoch:
            common_endpoint_row = row

        if consecutive_escape >= 3:
            certified_escape_onset = candidate_escape_onset
            certified_escape_confirmation = epoch

            # Do not stop before the common endpoint.
            if epoch >= common_endpoint_epoch:
                break

    if first_step_check is None:
        raise RuntimeError("First-step verification was not performed.")

    if common_endpoint_row is None:
        # If branch reaches max before common endpoint, the design is invalid.
        raise RuntimeError(
            f"Common endpoint {common_endpoint_epoch} not reached for "
            f"seed={seed}, branch={branch}."
        )

    baseline_common_key = (seed, common_endpoint_epoch)

    if baseline_common_key not in historical_map:
        raise RuntimeError(
            f"Historical Stage-5 baseline row missing at "
            f"seed={seed}, epoch={common_endpoint_epoch}."
        )

    baseline_common = historical_map[baseline_common_key]

    accelerated = bool(
        certified_escape_onset >= 0
        and certified_escape_onset < baseline_onset
    )

    if certified_escape_onset >= 0:
        acceleration_epochs = (
            baseline_onset - certified_escape_onset
        )
        baseline_delay = baseline_onset - 2500
        branch_delay = certified_escape_onset - 2500

        speedup = (
            baseline_delay / branch_delay
            if branch_delay > 0
            else float("inf")
        )
    else:
        acceleration_epochs = None
        speedup = None

    summary = {
        "seed": seed,
        "branch": branch,

        "historical_baseline_escape_onset_epoch":
            baseline_onset,
        "historical_baseline_escape_confirmation_epoch":
            baseline_confirmation,

        "branch_escape_onset_epoch":
            certified_escape_onset,
        "branch_escape_confirmation_epoch":
            certified_escape_confirmation,

        "accelerated_vs_historical_baseline":
            accelerated,
        "acceleration_epochs":
            acceleration_epochs,
        "escape_delay_speedup":
            speedup,

        "censored_at_historical_baseline_confirmation":
            bool(certified_escape_onset < 0),

        "total_intervention_steps":
            total_steps,
        "active_intervention_steps":
            active_steps,
        "active_intervention_fraction":
            (
                active_steps / total_steps
                if total_steps > 0
                else 0.0
            ),

        "mean_candidate_alignment_when_active":
            (
                float(np.mean(active_candidate_alignments))
                if active_candidate_alignments
                else None
            ),
        "mean_applied_alignment":
            (
                float(np.mean(applied_alignments))
                if applied_alignments
                else None
            ),

        "baseline_relL2_at_common_endpoint":
            float(baseline_common["relative_l2_error"]),
        "branch_relL2_at_common_endpoint":
            float(common_endpoint_row["relative_l2_error"]),
        "common_endpoint_relL2_ratio_branch_over_baseline":
            (
                float(common_endpoint_row["relative_l2_error"])
                / float(baseline_common["relative_l2_error"])
            ),

        "baseline_target_share_at_common_endpoint":
            float(
                baseline_common[
                    "target_mode_residual_energy_share"
                ]
            ),
        "branch_target_share_at_common_endpoint":
            float(
                common_endpoint_row[
                    "target_mode_residual_energy_share"
                ]
            ),
        "common_endpoint_target_share_difference":
            (
                float(
                    common_endpoint_row[
                        "target_mode_residual_energy_share"
                    ]
                )
                - float(
                    baseline_common[
                        "target_mode_residual_energy_share"
                    ]
                )
            ),
    }

    return rows, summary, first_step_check


# =============================================================================
# Plotting
# =============================================================================

def plot_metric(
    intervention_rows: List[dict],
    historical_rows: List[dict],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    seeds = sorted(
        set(int(r["seed"]) for r in intervention_rows)
    )

    # Historical baseline: thin dashed curves.
    for seed in seeds:
        h = [
            r for r in historical_rows
            if int(r["seed"]) == seed
            and 2500 <= int(r["epoch"]) <= 3700
        ]
        h.sort(key=lambda x: int(x["epoch"]))

        if h:
            ax.plot(
                [int(r["epoch"]) for r in h],
                [float(r[metric]) for r in h],
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
            )

    # Intervention branches: aggregate median across seeds on common epochs.
    for branch in BRANCHES:
        rr = [
            r for r in intervention_rows
            if r["branch"] == branch
        ]

        epochs = sorted(set(int(r["epoch"]) for r in rr))

        x = []
        med = []
        lo = []
        hi = []

        for e in epochs:
            vals = [
                float(r[metric])
                for r in rr
                if int(r["epoch"]) == e
            ]

            if len(vals) < 2:
                continue

            x.append(e)
            med.append(float(np.median(vals)))
            lo.append(float(np.min(vals)))
            hi.append(float(np.max(vals)))

        if x:
            ax.plot(
                x,
                med,
                linewidth=2.2,
                label=f"{branch} median",
            )
            ax.fill_between(
                x,
                lo,
                hi,
                alpha=0.12,
            )

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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

    stage8_dir = Path(args.stage8_dir)
    if not stage8_dir.is_absolute():
        stage8_dir = root / stage8_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage5_dir=stage5_dir,
        stage8_dir=stage8_dir,
    )

    if (
        args.common_endpoint_epoch
        >= pf["earliest_baseline_escape"]
    ):
        raise RuntimeError(
            "Common endpoint must be strictly earlier than the earliest "
            f"historical baseline escape ({pf['earliest_baseline_escape']})."
        )

    stage3 = load_stage3_module(stage3_script)
    s8_map = stage8_expected_map(pf["stage8_rows"])
    hist_map = aggregate_map(pf["aggregate"])

    precommitment = {
        "stage": "state_consistent_project_reflect_continuation",
        "locked_epoch": 2500,
        "seeds": pf["seeds"],
        "historical_control": "Stage-5 validated baseline; not rerun",
        "branches": {
            "PROJECT": (
                "real Adam state update every step; then remove only "
                "target-uphill parameter displacement component"
            ),
            "REFLECT": (
                "real Adam state update every step; then reflect only "
                "target-uphill parameter displacement component; "
                "candidate step norm preserved"
            ),
        },
        "escape_definition": {
            "relative_l2_error_le": 1.0e-2,
            "target_residual_energy_share_le": 0.20,
            "consecutive_tracking_observations": 3,
            "tracking_interval": 25,
        },
        "primary_reflect_success":
            "certified REFLECT escape onset strictly earlier than "
            "historical Stage-5 baseline onset",
        "primary_group_gate": "at least 4/5 REFLECT seeds accelerate",
        "project_is_mechanistic_context_not_fallback": True,
        "common_secondary_endpoint_epoch":
            args.common_endpoint_epoch,
        "per_seed_max_epoch":
            "historical Stage-5 baseline confirmation epoch",
        "decision_routes": {
            "reflect_pass_project_le1":
                "direction_specific_reflection_supported",
            "reflect_pass_project_ge4":
                "uphill_suppression_sufficient",
            "reflect_pass_project_2_or_3":
                "mixed_suppression_reflection_mechanism",
            "reflect_fail":
                "local_curvature_trust_region_audit",
        },
    }

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),
        "stage3_solver_sha256": pf["stage3_sha256"],
        "stage5_dir": str(stage5_dir),
        "stage8_dir": str(stage8_dir),
        "stage9_script_sha256": sha256_file(Path(__file__).resolve()),
        "precommitment": precommitment,
    }
    write_json(out_dir / "manifest.json", manifest)

    print("=" * 132)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 9 STATE-CONSISTENT PROJECT/REFLECT CONTINUATION"
    )
    print("=" * 132)
    print(f"device                    : {device}")
    print(f"seeds                     : {pf['seeds']}")
    print(f"historical CONTROL        : Stage 5 reused, not rerun")
    print(f"branches                  : {list(BRANCHES)}")
    print(
        "Adam state evolution      : ordinary raw-gradient Adam update every step"
    )
    print(
        "parameter intervention    : after Adam candidate step, PROJECT/REFLECT only"
    )
    print(
        f"common endpoint           : {args.common_endpoint_epoch}"
    )
    print(
        "primary gate              : >=4/5 REFLECT seeds escape earlier than baseline"
    )
    print(f"Stage-3 SHA256            : {pf['stage3_sha256']}")
    print("=" * 132)

    all_rows: List[dict] = []
    summaries: List[dict] = []
    first_step_checks: List[dict] = []

    start = time.perf_counter()

    for seed in pf["seeds"]:
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        print()
        print("-" * 132)
        print(
            f"SEED {seed} | historical onset="
            f"{pf['baseline'][seed]['escape_onset']} | "
            f"confirmation="
            f"{pf['baseline'][seed]['escape_confirmation']}"
        )

        for branch in BRANCHES:
            branch_dir = (
                out_dir
                / f"seed_{seed:03d}"
                / branch.lower()
            )
            branch_dir.mkdir(parents=True, exist_ok=True)

            rows, summary, first_check = run_branch(
                stage3=stage3,
                device=device,
                seed=seed,
                branch=branch,
                checkpoint_path=checkpoint,
                branch_dir=branch_dir,
                historical_baseline=pf["baseline"][seed],
                historical_map=hist_map,
                stage8_map=s8_map,
                track_interval=args.track_interval,
                common_endpoint_epoch=args.common_endpoint_epoch,
            )

            all_rows.extend(rows)
            summaries.append(summary)
            first_step_checks.append(first_check)

            write_csv(
                branch_dir / "trajectory.csv",
                rows,
            )
            write_json(
                branch_dir / "summary.json",
                summary,
            )

            print(
                f"{branch:8s} | "
                f"escape onset={summary['branch_escape_onset_epoch']:4d} | "
                f"accelerated="
                f"{summary['accelerated_vs_historical_baseline']} | "
                f"active={summary['active_intervention_steps']}/"
                f"{summary['total_intervention_steps']} | "
                f"relL2 ratio@2700="
                f"{summary['common_endpoint_relL2_ratio_branch_over_baseline']:.6f}"
            )

    write_csv(out_dir / "aggregate_trajectories.csv", all_rows)
    write_csv(out_dir / "paired_summary.csv", summaries)

    write_json(
        out_dir / "first_step_stage8_reproduction.json",
        {
            "all_pass": all(x["pass"] for x in first_step_checks),
            "results": first_step_checks,
        },
    )

    def branch_summaries(branch: str):
        return [x for x in summaries if x["branch"] == branch]

    reflect_s = branch_summaries("REFLECT")
    project_s = branch_summaries("PROJECT")

    reflect_accel_count = sum(
        int(x["accelerated_vs_historical_baseline"])
        for x in reflect_s
    )

    project_accel_count = sum(
        int(x["accelerated_vs_historical_baseline"])
        for x in project_s
    )

    reflect_group_pass = bool(reflect_accel_count >= 4)

    reflect_accelerations = [
        int(x["acceleration_epochs"])
        for x in reflect_s
        if x["acceleration_epochs"] is not None
    ]

    project_accelerations = [
        int(x["acceleration_epochs"])
        for x in project_s
        if x["acceleration_epochs"] is not None
    ]

    reflect_common_ratios = [
        float(x["common_endpoint_relL2_ratio_branch_over_baseline"])
        for x in reflect_s
    ]

    project_common_ratios = [
        float(x["common_endpoint_relL2_ratio_branch_over_baseline"])
        for x in project_s
    ]

    if not reflect_group_pass:
        next_route = "local_curvature_trust_region_audit"
        mechanism_class = "one_step_reflection_did_not_generalize_to_escape"
    elif project_accel_count <= 1:
        next_route = "frequency_transfer_of_reflected_optimizer"
        mechanism_class = "direction_specific_reflection_supported"
    elif project_accel_count >= 4:
        next_route = "frequency_transfer_of_uphill_suppression"
        mechanism_class = "uphill_suppression_sufficient"
    else:
        next_route = "suppression_vs_reflection_disambiguation"
        mechanism_class = "mixed_suppression_reflection_mechanism"

    decision = {
        "n_seeds": len(pf["seeds"]),
        "reflect_accelerated_seed_count":
            reflect_accel_count,
        "project_accelerated_seed_count":
            project_accel_count,
        "reflect_group_gate_pass":
            reflect_group_pass,
        "reflect_accelerated_seeds": [
            int(x["seed"])
            for x in reflect_s
            if x["accelerated_vs_historical_baseline"]
        ],
        "project_accelerated_seeds": [
            int(x["seed"])
            for x in project_s
            if x["accelerated_vs_historical_baseline"]
        ],
        "median_reflect_acceleration_epochs": (
            float(np.median(reflect_accelerations))
            if reflect_accelerations else None
        ),
        "median_project_acceleration_epochs": (
            float(np.median(project_accelerations))
            if project_accelerations else None
        ),
        "median_reflect_relL2_ratio_at_2700":
            float(np.median(reflect_common_ratios)),
        "median_project_relL2_ratio_at_2700":
            float(np.median(project_common_ratios)),
        "all_first_step_stage8_reproductions_pass":
            all(x["pass"] for x in first_step_checks),
        "mechanism_class": mechanism_class,
        "next_route": next_route,
        "interpretation_guardrail": (
            "A REFLECT group PASS establishes that repeated state-consistent, "
            "norm-preserving correction of target-uphill Adam displacements "
            "accelerates certified VPINN escape from identical locked states. "
            "PROJECT behavior determines whether active target descent is "
            "needed or simple uphill suppression is sufficient."
        ),
    }
    write_json(out_dir / "decision.json", decision)

    # Figures.
    plot_metric(
        intervention_rows=all_rows,
        historical_rows=pf["aggregate"],
        metric="relative_l2_error",
        ylabel="Relative L2 error",
        title="Historical Adam vs PROJECT/REFLECT continuation",
        path=out_dir / "relative_l2_trajectories.png",
        log_y=True,
    )

    plot_metric(
        intervention_rows=all_rows,
        historical_rows=pf["aggregate"],
        metric="target_mode_residual_energy_share",
        ylabel="Target-mode residual energy share",
        title="Release of unresolved mode under PROJECT/REFLECT",
        path=out_dir / "target_share_trajectories.png",
        log_y=False,
    )

    elapsed = time.perf_counter() - start

    lines = []
    lines.append("=" * 148)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 9 STATE-CONSISTENT CONTINUATION SUMMARY"
    )
    lines.append("=" * 148)
    lines.append(
        "seed | branch   | baseline onset | branch onset | acceleration | "
        "active steps | relL2 ratio@2700 | accelerated"
    )
    lines.append("-" * 148)

    for x in sorted(
        summaries,
        key=lambda z: (int(z["seed"]), z["branch"]),
    ):
        lines.append(
            f"{int(x['seed']):4d} | "
            f"{x['branch']:8s} | "
            f"{int(x['historical_baseline_escape_onset_epoch']):14d} | "
            f"{int(x['branch_escape_onset_epoch']):12d} | "
            f"{str(x['acceleration_epochs']):12s} | "
            f"{int(x['active_intervention_steps']):12d} | "
            f"{x['common_endpoint_relL2_ratio_branch_over_baseline']:16.6f} | "
            f"{'YES' if x['accelerated_vs_historical_baseline'] else 'NO'}"
        )

    lines.append("-" * 148)
    lines.append(
        f"Stage-8 first-step reproductions    : "
        f"{sum(int(x['pass']) for x in first_step_checks)}/"
        f"{len(first_step_checks)} PASS"
    )
    lines.append(
        f"REFLECT accelerated seeds           : "
        f"{reflect_accel_count}/5"
    )
    lines.append(
        f"PROJECT accelerated seeds           : "
        f"{project_accel_count}/5"
    )

    if reflect_accelerations:
        lines.append(
            f"median REFLECT acceleration         : "
            f"{np.median(reflect_accelerations):.1f} epochs"
        )

    if project_accelerations:
        lines.append(
            f"median PROJECT acceleration         : "
            f"{np.median(project_accelerations):.1f} epochs"
        )

    lines.append(
        f"median REFLECT relL2 ratio @2700    : "
        f"{np.median(reflect_common_ratios):.6f}"
    )
    lines.append(
        f"median PROJECT relL2 ratio @2700    : "
        f"{np.median(project_common_ratios):.6f}"
    )
    lines.append(
        f"REFLECT primary group gate          : "
        f"{'PASS' if reflect_group_pass else 'FAIL'}"
    )
    lines.append(
        f"mechanism class                     : "
        f"{mechanism_class}"
    )
    lines.append(
        f"next route                          : "
        f"{next_route}"
    )
    lines.append(f"elapsed seconds                     : {elapsed:.2f}")
    lines.append("=" * 148)
    lines.append(
        "Guardrail: repeated directional intervention is a modified optimizer. "
        "Its mechanism is interpreted only relative to the paired historical "
        "baseline and PROJECT context defined before this run."
    )
    lines.append("=" * 148)

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
