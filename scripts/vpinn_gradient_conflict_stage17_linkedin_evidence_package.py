#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 17
Final Read-Only Evidence Audit + LinkedIn Asset Builder
=======================================================

THIS STAGE DOES NOT TRAIN ANY MODEL.

Purpose
-------
Stage 16 was precommitted as the final optimizer experiment for the LinkedIn
thread. Its matched-state result was:

    CONTROL certified escape : 4/4
    ADAPTIVE certified escape: 0/4
    ADAPTIVE safety failures : 4/4
    ADAPTIVE accelerated     : 0/4
    primary gate             : FAIL

Therefore no additional optimizer rescue sweep is scientifically authorized.

Stage 17 is a READ-ONLY evidence and communication stage. It:

  1) validates the final Stage-16 matched-state result;
  2) reconstructs the strongest defensible mechanism chain from Stages
     12, 14, 15, and 16;
  3) computes final LinkedIn-safe metrics;
  4) explicitly separates ALLOWED from FORBIDDEN public claims;
  5) creates communication figures that represent safety failures honestly;
  6) writes a concise evidence table and a post-structure draft.

No parameter update is executed.
No optimizer is instantiated.
No result-dependent scientific threshold is introduced.

Final scientific story
----------------------
The evidence chain supports:

  * At the conflict-active branch states, the unresolved target weak mode
    carries ~99% of the residual energy.

  * At those states, inherited Adam is total-loss descent but target-loss
    uphill, while full REFLECT has the opposite trade-off.

  * A local Pareto-compatible interval exists and a state-adaptive midpoint
    can locally rescue failed fixed-direction states.

  * The Pareto-compatible compromise moves during training.

  * Bounded persistence succeeds in 3/4 active seeds through epoch 2700.

  * In the final matched-state causal comparison, ordinary Adam certifies
    escape in 4/4 active seeds, while the adaptive rule hits its precommitted
    safety gate in 4/4 and certifies escape in 0/4.

Therefore the evidence SUPPORTS a moving optimization-geometry mechanism,
but DOES NOT SUPPORT an acceleration or optimizer-improvement claim.

Communication guardrail
-----------------------
Allowed:
    "The weak residual can identify the unresolved mode while the optimizer
     update moves against correcting it."

Allowed:
    "A fixed safe compromise was not dynamically robust; the local
     Pareto-compatible corridor moved during training."

Allowed:
    "Local adaptive Pareto correction repaired the conflict, but did not
     survive the final matched-state escape test."

Forbidden:
    "Our adaptive optimizer converges faster."
    "We solved VPINN gradient conflict."
    "Adam fails on VPINNs in general."
    "This is the first discovery of this phenomenon."
    "The adaptive method escaped earlier."

The negative final result is part of the story, not something to hide.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-17 final read-only LinkedIn evidence package."
    )

    p.add_argument(
        "--stage12-dir",
        default="vpinn_gradient_conflict_stage12_common_pareto_blend_audit",
    )

    p.add_argument(
        "--stage14-dir",
        default="vpinn_gradient_conflict_stage14_adaptive_midpoint_failure_audit",
    )

    p.add_argument(
        "--stage15-dir",
        default="vpinn_gradient_conflict_stage15_adaptive_midpoint_persistence",
    )

    p.add_argument(
        "--stage16-dir",
        default="vpinn_gradient_conflict_stage16_matched_escape_comparison",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage17_linkedin_evidence_package",
    )

    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

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


def resolve_dir(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else root / p


def as_bool(x) -> bool:
    return str(x).strip().lower() == "true"


# =============================================================================
# Preflight
# =============================================================================

def preflight(
    stage12_dir: Path,
    stage14_dir: Path,
    stage15_dir: Path,
    stage16_dir: Path,
) -> dict:

    paths = {
        "s12_decision":
            stage12_dir / "decision.json",

        "s12_summary":
            stage12_dir / "seed_summary.csv",

        "s14_decision":
            stage14_dir / "decision.json",

        "s14_summary":
            stage14_dir / "failure_state_summary.csv",

        "s15_decision":
            stage15_dir / "decision.json",

        "s15_summary":
            stage15_dir / "seed_summary.csv",

        "s15_steps":
            stage15_dir / "step_metrics.csv",

        "s16_decision":
            stage16_dir / "decision.json",

        "s16_paired":
            stage16_dir / "paired_escape_summary.csv",

        "s16_traj":
            stage16_dir / "trajectory_metrics.csv",

        "s16_clones":
            stage16_dir / "matched_state_clone_checks.csv",

        "s16_replay":
            stage16_dir / "stage15_replay_checks.csv",

        "s16_linkedin":
            stage16_dir / "linkedin_candidate_metrics.json",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    d12 = read_json(paths["s12_decision"])
    d14 = read_json(paths["s14_decision"])
    d15 = read_json(paths["s15_decision"])
    d16 = read_json(paths["s16_decision"])

    if not bool(d12.get("common_interval_exists", False)):
        raise RuntimeError("Stage 12 common Pareto interval is missing.")

    if int(d12.get("full_safe_count", -1)) != 4:
        raise RuntimeError("Stage 12 did not have 4/4 full-safe local seeds.")

    if int(d14.get("adaptive_full_safe_count", -1)) != 3:
        raise RuntimeError("Stage 14 did not have 3/3 adaptive local rescue.")

    if not bool(d15.get("promotion_gate_pass", False)):
        raise RuntimeError("Stage 15 bounded persistence gate is not PASS.")

    if bool(d16.get("primary_group_gate_pass", True)):
        raise RuntimeError(
            "Stage 16 unexpectedly PASSed. Stage-17 claim rules must be "
            "updated before use."
        )

    if d16.get("next_route") != (
        "linkedin_final_mechanism_package_without_acceleration_claim"
    ):
        raise RuntimeError("Unexpected Stage-16 final route.")

    if int(d16.get("control_certified_escape_count", -1)) != 4:
        raise RuntimeError("Stage-16 matched controls did not escape 4/4.")

    if int(d16.get("adaptive_certified_escape_count", -1)) != 0:
        raise RuntimeError("Unexpected Stage-16 adaptive escape count.")

    if int(d16.get("adaptive_safety_failure_count", -1)) != 4:
        raise RuntimeError("Unexpected Stage-16 adaptive failure count.")

    if int(d16.get("adaptive_accelerated_count", -1)) != 0:
        raise RuntimeError("Unexpected Stage-16 acceleration count.")

    clones = read_csv(paths["s16_clones"])
    replay = read_csv(paths["s16_replay"])

    if not clones or not all(as_bool(r["pass"]) for r in clones):
        raise RuntimeError("Matched-state clone checks are not all PASS.")

    if not replay or not all(as_bool(r["pass"]) for r in replay):
        raise RuntimeError("Stage-15 replay checks are not all PASS.")

    return {
        "d12": d12,
        "d14": d14,
        "d15": d15,
        "d16": d16,

        "s12_summary":
            read_csv(paths["s12_summary"]),

        "s14_summary":
            read_csv(paths["s14_summary"]),

        "s15_summary":
            read_csv(paths["s15_summary"]),

        "s15_steps":
            read_csv(paths["s15_steps"]),

        "s16_paired":
            read_csv(paths["s16_paired"]),

        "s16_traj":
            read_csv(paths["s16_traj"]),

        "s16_clones":
            clones,

        "s16_replay":
            replay,

        "s16_linkedin":
            read_json(paths["s16_linkedin"]),
    }


# =============================================================================
# Evidence extraction
# =============================================================================

def branch_start_rows(traj: List[dict]) -> List[dict]:
    rows = []

    for r in traj:
        if (
            r["branch"] == "ADAPTIVE"
            and r["event"] == "BRANCH_START"
        ):
            rows.append(r)

    return sorted(rows, key=lambda x: int(x["seed"]))


def final_failure_rows(traj: List[dict]) -> List[dict]:
    rows = []

    for r in traj:
        if (
            r["branch"] == "ADAPTIVE"
            and "FAILURE" in r["event"]
        ):
            rows.append(r)

    return sorted(rows, key=lambda x: int(x["seed"]))


def qualify_escape(r: dict) -> bool:
    return bool(
        float(r["relative_l2_error"]) <= 1.0e-2
        and
        float(r["target_mode_residual_energy_share"]) <= 0.20
    )


# =============================================================================
# Communication figures
# =============================================================================

def plot_matched_outcome(
    paired: List[dict],
    path: Path,
) -> None:
    """
    Honest outcome figure:
      circle = certified CONTROL escape onset
      x      = ADAPTIVE safety-failure epoch

    We intentionally do NOT draw adaptive failures as bars at epoch 4000.
    """
    seeds = [int(r["seed"]) for r in paired]

    control = [
        int(float(r["control_escape_onset_epoch"]))
        for r in paired
    ]

    failure = [
        int(float(r["adaptive_failure_epoch"]))
        for r in paired
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    ax.scatter(
        seeds,
        control,
        marker="o",
        s=90,
        label="Matched Adam: certified escape",
    )

    ax.scatter(
        seeds,
        failure,
        marker="x",
        s=110,
        label="Adaptive midpoint: safety failure",
    )

    for seed, c, f in zip(seeds, control, failure):
        ax.plot([seed, seed], [min(c, f), max(c, f)], linewidth=1.0)

    ax.set_xticks(seeds)
    ax.set_xlabel("Conflict-active seed")
    ax.set_ylabel("Epoch")
    ax.set_title(
        "Same starting state, different outcome: Adam escapes; adaptive rule hits safety gate"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_branch_target_share(
    starts: List[dict],
    path: Path,
) -> None:
    seeds = [int(r["seed"]) for r in starts]

    shares = [
        100.0 * float(r["target_mode_residual_energy_share"])
        for r in starts
    ]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    ax.bar(
        [str(s) for s in seeds],
        shares,
    )

    ax.set_ylim(95.0, 100.0)
    ax.set_xlabel("Conflict-active seed")
    ax.set_ylabel("Target-mode residual energy share (%)")
    ax.set_title(
        "The unresolved weak mode was not hidden: it carried ~99% of residual energy"
    )

    for i, value in enumerate(shares):
        ax.text(
            i,
            value + 0.08,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_motion(
    step_rows: List[dict],
    path: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    seeds = sorted(
        {
            int(r["seed"])
            for r in step_rows
            if str(r.get("intervention_active", "")).lower() == "true"
            and r.get("lambda_mid") not in (None, "")
        }
    )

    for seed in seeds:
        rr = [
            r for r in step_rows
            if int(r["seed"]) == seed
            and str(r.get("intervention_active", "")).lower() == "true"
            and r.get("lambda_mid") not in (None, "")
        ]

        rr.sort(key=lambda x: int(x["epoch_after"]))

        ax.plot(
            [int(r["epoch_after"]) for r in rr],
            [float(r["lambda_mid"]) for r in rr],
            linewidth=1.3,
            label=f"seed {seed}",
        )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.0,
        label="target-neutral boundary",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Adaptive Pareto midpoint λ")
    ax.set_title(
        "The locally safe compromise was not fixed; it moved with the trajectory"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_stage16_relL2(
    traj: List[dict],
    path: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    seeds = sorted(set(int(r["seed"]) for r in traj))

    for seed in seeds:
        for branch in ("CONTROL", "ADAPTIVE"):
            rr = [
                r for r in traj
                if int(r["seed"]) == seed
                and r["branch"] == branch
                and r.get("relative_l2_error") not in (None, "")
            ]

            rr.sort(key=lambda x: int(x["epoch"]))

            if not rr:
                continue

            ax.plot(
                [int(r["epoch"]) for r in rr],
                [float(r["relative_l2_error"]) for r in rr],
                linewidth=1.2,
                label=f"seed {seed} {branch}",
            )

    ax.axhline(
        1.0e-2,
        linestyle="--",
        linewidth=1.0,
        label="escape error threshold",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Matched-state trajectories: local repair did not become a globally safe optimizer")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    root = Path(__file__).resolve().parent

    stage12_dir = resolve_dir(root, args.stage12_dir)
    stage14_dir = resolve_dir(root, args.stage14_dir)
    stage15_dir = resolve_dir(root, args.stage15_dir)
    stage16_dir = resolve_dir(root, args.stage16_dir)
    out_dir = resolve_dir(root, args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    pf = preflight(
        stage12_dir=stage12_dir,
        stage14_dir=stage14_dir,
        stage15_dir=stage15_dir,
        stage16_dir=stage16_dir,
    )

    starts = branch_start_rows(pf["s16_traj"])
    failures = final_failure_rows(pf["s16_traj"])

    if len(starts) != 4:
        raise RuntimeError(
            f"Expected four adaptive branch-start rows, found {len(starts)}."
        )

    if len(failures) != 4:
        raise RuntimeError(
            f"Expected four final adaptive safety failures, found {len(failures)}."
        )

    target_shares = np.array([
        float(r["target_mode_residual_energy_share"])
        for r in starts
    ])

    median_target_share = float(np.median(target_shares))
    min_target_share = float(np.min(target_shares))
    max_target_share = float(np.max(target_shares))

    paired = sorted(
        pf["s16_paired"],
        key=lambda x: int(x["seed"]),
    )

    control_onsets = [
        int(float(r["control_escape_onset_epoch"]))
        for r in paired
    ]

    adaptive_failure_epochs = [
        int(float(r["adaptive_failure_epoch"]))
        for r in paired
    ]

    historical_matches = sum(
        int(
            int(float(r["control_escape_onset_epoch"]))
            ==
            int(float(r["historical_stage5_escape_onset"]))
        )
        for r in paired
    )

    # Special diagnostic: seed 1 reached two consecutive qualifying tracked
    # states but failed before a third confirmation.
    seed1_adaptive = sorted(
        [
            r for r in pf["s16_traj"]
            if int(r["seed"]) == 1
            and r["branch"] == "ADAPTIVE"
        ],
        key=lambda x: int(x["epoch"]),
    )

    seed1_qualifying = [
        int(r["epoch"])
        for r in seed1_adaptive
        if r["event"] in ("TRACK", "ESCAPE_CONFIRMATION")
        and qualify_escape(r)
    ]

    # -------------------------------------------------------------------------
    # Evidence chain.
    # -------------------------------------------------------------------------
    evidence_rows = [
        {
            "claim_id": "E1",
            "status": "SUPPORTED",
            "claim": (
                "At the matched conflict-active branch states, the unresolved "
                "target mode carried approximately 99% of residual energy."
            ),
            "metric": (
                f"median={100*median_target_share:.3f}%, "
                f"range=[{100*min_target_share:.3f}%, "
                f"{100*max_target_share:.3f}%]"
            ),
            "source": "Stage 16 trajectory_metrics.csv branch-start rows",
        },
        {
            "claim_id": "E2",
            "status": "SUPPORTED",
            "claim": (
                "A local common Pareto-compatible blend existed and was "
                "full-step safe at the initial conflict states."
            ),
            "metric": "Stage 12: 4/4 COMMON_FULL_SAFE",
            "source": "Stage 12 decision.json",
        },
        {
            "claim_id": "E3",
            "status": "SUPPORTED",
            "claim": (
                "Updating the Pareto midpoint from current geometry locally "
                "rescued all previously failed fixed-blend states."
            ),
            "metric": "Stage 14: 3/3 ADAPTIVE_FULL_SAFE",
            "source": "Stage 14 decision.json",
        },
        {
            "claim_id": "E4",
            "status": "SUPPORTED",
            "claim": (
                "The state-adaptive rule showed bounded persistence through "
                "epoch 2700 in most active seeds, but not all."
            ),
            "metric": (
                f"Stage 15: {pf['d15']['persistent_safe_count']}/4 safe, "
                f"{pf['d15']['net_joint_progress_count']}/4 net joint progress"
            ),
            "source": "Stage 15 decision.json",
        },
        {
            "claim_id": "E5",
            "status": "SUPPORTED",
            "claim": (
                "The final matched-state causal comparison gave ordinary Adam "
                "certified escape in all active seeds."
            ),
            "metric": (
                f"4/4 controls escaped; onsets={control_onsets}; "
                f"{historical_matches}/4 exactly match Stage-5 historical onsets"
            ),
            "source": "Stage 16 paired_escape_summary.csv",
        },
        {
            "claim_id": "E6",
            "status": "SUPPORTED",
            "claim": (
                "The adaptive Pareto midpoint did not produce a certified "
                "escape in the final matched-state test."
            ),
            "metric": (
                "0/4 certified escape; 4/4 precommitted safety failures; "
                f"failure epochs={adaptive_failure_epochs}"
            ),
            "source": "Stage 16 decision.json + paired_escape_summary.csv",
        },
        {
            "claim_id": "E7",
            "status": "NOT_SUPPORTED",
            "claim": "The adaptive Pareto optimizer converges faster than Adam.",
            "metric": "Stage 16 acceleration gate: 0/4, FAIL",
            "source": "Stage 16 decision.json",
        },
        {
            "claim_id": "E8",
            "status": "NOT_SUPPORTED",
            "claim": "The gradient-conflict problem has been solved.",
            "metric": "Final adaptive branch safety failure: 4/4",
            "source": "Stage 16 decision.json",
        },
    ]

    write_csv(
        out_dir / "final_evidence_table.csv",
        evidence_rows,
    )

    allowed_claims = [
        (
            "In this controlled VPINN experiment, the unresolved weak mode "
            "was clearly visible in the residual while the optimizer update "
            "could move against correcting it."
        ),
        (
            "The locally Pareto-compatible compromise was not a fixed "
            "hyperparameter; its feasible geometry moved along the training "
            "trajectory."
        ),
        (
            "Local adaptive Pareto correction repaired several conflict "
            "states and persisted for a bounded horizon in 3/4 active seeds."
        ),
        (
            "In the final matched-state test, ordinary Adam certified escape "
            "in 4/4 active seeds, while the adaptive rule hit its precommitted "
            "safety gate in 4/4 and certified escape in 0/4."
        ),
        (
            "The final result supports a mechanism diagnosis, not an "
            "optimizer-improvement claim."
        ),
    ]

    forbidden_claims = [
        "The adaptive midpoint converges faster than Adam.",
        "The adaptive midpoint escaped the plateau earlier.",
        "We solved VPINN gradient conflict.",
        "Adam fails for VPINNs in general.",
        "This phenomenon occurs in all VPINNs.",
        "This is the first-ever discovery of VPINN gradient conflict.",
        "The adaptive method is a new superior optimizer.",
    ]

    final_metrics = {
        "branch_target_residual_share": {
            "median":
                median_target_share,

            "median_percent":
                100.0 * median_target_share,

            "minimum_percent":
                100.0 * min_target_share,

            "maximum_percent":
                100.0 * max_target_share,
        },

        "stage12_local_common_blend": {
            "full_safe":
                int(pf["d12"]["full_safe_count"]),

            "active":
                int(pf["d12"]["n_active_seeds"]),

            "lambda_star":
                pf["d12"]["lambda_star"],
        },

        "stage14_local_adaptive_rescue": {
            "full_safe":
                int(pf["d14"]["adaptive_full_safe_count"]),

            "failed_states":
                int(pf["d14"]["n_failed_states"]),
        },

        "stage15_bounded_persistence": {
            "persistent_safe":
                int(pf["d15"]["persistent_safe_count"]),

            "net_joint_progress":
                int(pf["d15"]["net_joint_progress_count"]),

            "active":
                int(pf["d15"]["n_active_seeds"]),

            "nonlinear_safety_failures":
                int(pf["d15"]["nonlinear_safety_failure_count"]),
        },

        "stage16_matched_state_final": {
            "control_certified_escape":
                int(pf["d16"]["control_certified_escape_count"]),

            "adaptive_certified_escape":
                int(pf["d16"]["adaptive_certified_escape_count"]),

            "adaptive_safety_failures":
                int(pf["d16"]["adaptive_safety_failure_count"]),

            "adaptive_accelerated":
                int(pf["d16"]["adaptive_accelerated_count"]),

            "control_escape_onsets":
                control_onsets,

            "adaptive_failure_epochs":
                adaptive_failure_epochs,

            "matched_clone_checks_pass":
                bool(pf["d16"]["all_matched_state_clone_checks_pass"]),

            "stage15_replay_checks_pass":
                bool(pf["d16"]["all_stage15_replay_checks_pass"]),

            "control_onsets_match_historical_stage5":
                historical_matches,
        },

        "seed1_near_certification_context": {
            "qualifying_tracked_epochs_before_failure":
                seed1_qualifying,

            "note": (
                "Descriptive only. Certification still failed because the "
                "third consecutive qualifying observation was not reached "
                "before the safety failure."
            ),
        },

        "final_public_claim_class":
            "MECHANISM_ONLY_NO_ACCELERATION",

        "allowed_claims":
            allowed_claims,

        "forbidden_claims":
            forbidden_claims,
    }

    write_json(
        out_dir / "final_linkedin_metrics_and_claims.json",
        final_metrics,
    )

    # -------------------------------------------------------------------------
    # Figures.
    # -------------------------------------------------------------------------
    plot_matched_outcome(
        paired=paired,
        path=out_dir / "01_matched_state_final_outcome.png",
    )

    plot_branch_target_share(
        starts=starts,
        path=out_dir / "02_target_mode_was_visible.png",
    )

    plot_lambda_motion(
        step_rows=pf["s15_steps"],
        path=out_dir / "03_pareto_midpoint_moves.png",
    )

    plot_stage16_relL2(
        traj=pf["s16_traj"],
        path=out_dir / "04_matched_relative_l2_story.png",
    )

    # -------------------------------------------------------------------------
    # LinkedIn story skeleton.
    # -------------------------------------------------------------------------
    post_outline = f"""# LinkedIn evidence-first story skeleton

## Hook
My VPINN knew which weak mode was wrong.

The optimizer still moved against fixing it.

## Concrete observation
At the conflict-active branch states, the target mode carried a median
{100.0*median_target_share:.2f}% of the residual energy
(range {100.0*min_target_share:.2f}% to {100.0*max_target_share:.2f}%).

So the weak formulation was not blind to the missing mode.

## The contradiction
The optimization geometry was the problem.

A direction that helped the unresolved weak residual could hurt the full
variational objective.

A full reflection repaired the target direction locally, but destabilized
other weak equations.

## The moving geometry
A local Pareto-compatible corridor existed.

A fixed compromise was not dynamically robust.

When the compromise was recomputed from the current state, the midpoint moved
during training rather than behaving like a fixed hyperparameter.

## What worked locally
Stage 12:
4/4 initial conflict states were full-step safe under the common local blend.

Stage 14:
3/3 failed fixed-blend states were locally rescued by the current adaptive
midpoint.

Stage 15:
3/4 active seeds stayed safe through epoch 2700.

## The result I will NOT hide
The final matched-state test changed the conclusion.

Starting CONTROL and ADAPTIVE from exactly identical model and Adam states:

- ordinary Adam certified escape in 4/4 active seeds;
- adaptive midpoint certified escape in 0/4;
- adaptive midpoint hit the precommitted safety gate in 4/4.

So I do NOT have evidence that the adaptive rule is a better optimizer.

## Actual takeaway
The interesting result is the mechanism:

A VPINN residual can clearly identify an unresolved mode while the
parameter-space optimizer dynamics conflict with correcting it.

And the locally safe compromise itself can move during training.

Local geometric repair is not the same thing as a globally stable optimizer.

## Discussion question
Should VPINN test functions be viewed only as weak residual probes, or also
as competing optimization objectives once they are mapped into parameter
space?

## Claims to avoid
- "faster convergence"
- "new superior optimizer"
- "solved gradient conflict"
- "first-ever discovery"
"""

    (out_dir / "linkedin_story_skeleton.md").write_text(
        post_outline,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Human-readable final report.
    # -------------------------------------------------------------------------
    report = f"""
STAGE 17 FINAL READ-ONLY EVIDENCE AUDIT
=======================================

FINAL SCIENTIFIC CLASS:
    MECHANISM_ONLY_NO_ACCELERATION

Conflict-active branch residual concentration:
    median target share = {100.0*median_target_share:.4f}%
    range               = [{100.0*min_target_share:.4f}%,
                           {100.0*max_target_share:.4f}%]

Local Pareto evidence:
    Stage 12 common blend full-safe = {pf['d12']['full_safe_count']}/4
    Stage 14 adaptive rescue        = {pf['d14']['adaptive_full_safe_count']}/3

Bounded persistence:
    Stage 15 persistent safe        = {pf['d15']['persistent_safe_count']}/4
    Stage 15 joint progress         = {pf['d15']['net_joint_progress_count']}/4

Final matched-state causal comparison:
    CONTROL certified escape        = {pf['d16']['control_certified_escape_count']}/4
    ADAPTIVE certified escape       = {pf['d16']['adaptive_certified_escape_count']}/4
    ADAPTIVE safety failure         = {pf['d16']['adaptive_safety_failure_count']}/4
    ADAPTIVE accelerated            = {pf['d16']['adaptive_accelerated_count']}/4

CONTROL escape onsets:
    {control_onsets}

ADAPTIVE safety-failure epochs:
    {adaptive_failure_epochs}

Matched-state clone verification:
    PASS

Stage-15 replay verification:
    PASS

Historical Stage-5 onset reproduction by matched CONTROL:
    {historical_matches}/4 exact onset matches

FINAL DECISION:
    No additional optimizer rescue experiment is authorized for this
    LinkedIn evidence thread.

    The final public claim should center on:
        * visible unresolved weak mode,
        * optimizer-space conflict,
        * moving Pareto-compatible geometry,
        * failure of local geometric repair to become a globally safe
          accelerated optimizer.

    Acceleration claim: NOT SUPPORTED.
"""

    (out_dir / "final_evidence_report.txt").write_text(
        report.strip() + "\n",
        encoding="utf-8",
    )

    print(report)

    print("Created LinkedIn evidence package:")
    print(f"  {out_dir / 'final_evidence_table.csv'}")
    print(f"  {out_dir / 'final_linkedin_metrics_and_claims.json'}")
    print(f"  {out_dir / 'linkedin_story_skeleton.md'}")
    print(f"  {out_dir / 'final_evidence_report.txt'}")
    print(f"  {out_dir / '01_matched_state_final_outcome.png'}")
    print(f"  {out_dir / '02_target_mode_was_visible.png'}")
    print(f"  {out_dir / '03_pareto_midpoint_moves.png'}")
    print(f"  {out_dir / '04_matched_relative_l2_story.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
