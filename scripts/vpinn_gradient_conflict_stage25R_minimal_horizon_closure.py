#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 25R
Minimal Horizon Closure for the Three Censored b=1,m=9 Deep-Lock Runs
=====================================================================

Why this stage is justified
---------------------------
Stage 24R established, for paired seeds {15,...,19} and fixed target m=9:

    P1 exact b=1 Stage-22 replay                PASS 5/5
    P2 persistent deep mobility lock            PASS 5/5
    P3 b=1 conflict-release delay >=2x b=2      PASS 5/5
    P4 b=1 conflict-escape delay > b=2          PASS 5/5

Only P5 failed because the epoch-4000 horizon censored 3/5 b=1 runs.

The two completed b=1 runs escaped at:
    seed 15: release 3575 -> escape 3950  (375 epochs)
    seed 16: release 2700 -> escape 3100  (400 epochs)

The three censored runs had already released by epoch 4000:
    seed 17: release 3750
    seed 18: release 3850
    seed 19: release 3925

and their epoch-4000 mobility had already recovered by huge factors relative
to conflict onset.

Therefore Stage 25R performs the minimum justified continuation:

    ONLY seeds {17,18,19}
    ONLY b=1,m=9
    from exact reconstructed epoch 4000
    through at most epoch 4500
    ordinary Adam only.

No new seed, target mode, base mode, optimizer, threshold, architecture, or PDE.

Why epoch 4500?
---------------
The latest current release is 3925. A 4500 horizon gives 575 post-release
epochs, exceeding the observed 250-400 epoch release-to-escape lags in the
previous completed deep-lock trajectories, while avoiding an arbitrary long
rerun.

Exact epoch-4000 reconstruction
-------------------------------
Stage 24 did not serialize checkpoints, so each of the three runs is
deterministically rebuilt from epoch 0 to epoch 4000.

At epoch 4000 Stage 25 must reproduce Stage-24:

    relative L2 error
    VPINN loss
    target residual-energy share
    target absolute residual
    Adam target-uphill cosine
    mu_raw
    top eigenvalue fraction
    effective rank

to maximum absolute difference <= 1e-10.

Any mismatch aborts.

Inherited definitions
---------------------
Certified escape:
    relL2 <= 1e-2
    AND target share <= 0.20
for THREE consecutive 25-epoch observations.

Persistent deep lock:
    already established 5/5 by Stage 24. Not redefined.

Mobility recovery:
    after persistent deep lock, TWO consecutive 250-grid audits with

        mu / mu_conflict >= 100.

The recovery definition and threshold are unchanged.

Important state carry
---------------------
Stage 25 reconstructs the Stage-24 mobility-recovery streak at epoch 4000
from the existing Stage-24 surveillance rows.

This matters because seeds 18 and 19 already have one qualifying recovery
audit at epoch 4000. The 4250 audit must be allowed to certify a recovery
onset at 4000 rather than incorrectly restarting the streak.

Precommitted closure gates
--------------------------
C1 — EXACT EPOCH-4000 RECONSTRUCTION
    3/3 PASS <=1e-10.

C2 — NEW ESCAPE CLOSURE
    at least 2/3 previously censored runs certify escape by epoch 4500.

C3 — CUMULATIVE STAGE-24 P5 CLOSURE
    Combining Stage-24 completed runs and Stage-25 extensions:
        total b=1 certified escapes >=4/5
    AND among all escaped b=1 runs:
        mobility recovery onset < escape onset
    in >=80%.

If C1&C2&C3 PASS:
    the previously horizon-censored Stage-24 P5 is closed,
    and all five Stage-24 base-mode specificity gates are considered closed.

Next:
    Stage 26R = read-only paired tangent-eigenspace mechanism localization.

If C2 fails:
    do NOT launch a new sweep. Extend only still-censored runs once more using
    the observed 4000-4500 trajectory.

Scientific guardrail
--------------------
This stage is horizon closure only. It cannot create new mechanistic claims.
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
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


BASE_MODE = 1
TARGET_MODE = 9
EXTEND_SEEDS = (17, 18, 19)

RECONSTRUCT_EPOCH = 4000
MAX_EPOCH = 4500

TRACK_INTERVAL = 25
MOBILITY_AUDIT_INTERVAL = 250

CONVERGENCE_REL_L2 = 1.0e-2
CONVERGENCE_TARGET_SHARE = 0.20
CERTIFY_POINTS = 3

MOBILITY_RECOVERY_RATIO = 100.0
MOBILITY_RECOVERY_POINTS = 2

REPLAY_TOL = 1.0e-10


# =============================================================================
# CLI / helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-25R minimal horizon closure for censored b=1,m=9 runs."
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
        "--stage19-script",
        default="vpinn_gradient_conflict_stage19R_temporal_conflict_parity.py",
    )
    p.add_argument(
        "--stage20-script",
        default="vpinn_gradient_conflict_stage20R_heldout_mobility_unlock.py",
    )
    p.add_argument(
        "--stage22-script",
        default="vpinn_gradient_conflict_stage22R_symmetry_sector_swap.py",
    )
    p.add_argument(
        "--stage24-script",
        default="vpinn_gradient_conflict_stage24R_paired_base_mode_deep_lock.py",
    )
    p.add_argument(
        "--stage24-dir",
        default="vpinn_gradient_conflict_stage24R_paired_base_mode_deep_lock",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage25R_minimal_horizon_closure",
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
        raise RuntimeError(f"Could not import: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def bool_from_csv(value) -> bool:
    return str(value).strip().lower() == "true"


# =============================================================================
# Preflight
# =============================================================================

def preflight(
    stage3_script: Path,
    stage18_script: Path,
    stage19_script: Path,
    stage20_script: Path,
    stage22_script: Path,
    stage24_script: Path,
    stage24_dir: Path,
) -> dict:

    manifest_path = stage24_dir / "manifest.json"
    decision_path = stage24_dir / "decision.json"
    paired_path = stage24_dir / "paired_run_summary.csv"
    tracking_path = stage24_dir / "tracking_metrics.csv"
    mobility_path = stage24_dir / "mobility_audits.csv"

    for path in (
        manifest_path,
        decision_path,
        paired_path,
        tracking_path,
        mobility_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    decision = read_json(decision_path)

    shas = {
        "s3": sha256_file(stage3_script),
        "s18": sha256_file(stage18_script),
        "s19": sha256_file(stage19_script),
        "s20": sha256_file(stage20_script),
        "s22": sha256_file(stage22_script),
        "s24": sha256_file(stage24_script),
    }

    checks = (
        ("stage3_solver_sha256", "s3"),
        ("stage18_script_sha256", "s18"),
        ("stage19_script_sha256", "s19"),
        ("stage20_script_sha256", "s20"),
        ("stage22_script_sha256", "s22"),
        ("stage24r_script_sha256", "s24"),
    )

    for manifest_key, sha_key in checks:
        if manifest.get(manifest_key) != shas[sha_key]:
            raise RuntimeError(
                f"SHA mismatch: {manifest_key}"
            )

    # Stage-24 route must be exactly the horizon-censoring branch.
    if decision.get("route_class") != (
        "deep_lock_replication_passes_but_recovery_or_escape_is_horizon_censored"
    ):
        raise RuntimeError(
            "Stage 24 is not in the expected horizon-censored route."
        )

    if decision.get("next_route") != "stage25R_extend_only_censored_b1_runs":
        raise RuntimeError("Unexpected Stage-24 next route.")

    if not bool(decision.get("P1_exact_b1_stage22_replay", False)):
        raise RuntimeError("Stage-24 P1 did not pass.")
    if not bool(decision.get("P2_persistent_deep_lock_replication", False)):
        raise RuntimeError("Stage-24 P2 did not pass.")
    if not bool(decision.get("P3_conflict_persistence_contrast", False)):
        raise RuntimeError("Stage-24 P3 did not pass.")
    if not bool(decision.get("P4_escape_delay_contrast", False)):
        raise RuntimeError("Stage-24 P4 did not pass.")
    if bool(decision.get("P5_mobility_recovery_precedes_escape", True)):
        raise RuntimeError("Stage-24 P5 unexpectedly already passed.")

    paired_rows = read_csv(paired_path)
    tracking_rows = read_csv(tracking_path)
    mobility_rows = read_csv(mobility_path)

    paired = {int(r["seed"]): r for r in paired_rows}

    censored = sorted(
        seed
        for seed, row in paired.items()
        if not bool_from_csv(row["b1_certified_escape"])
    )

    if censored != list(EXTEND_SEEDS):
        raise RuntimeError(
            f"Expected censored seeds {list(EXTEND_SEEDS)}, got {censored}."
        )

    # Exact Stage-24 epoch-4000 reference rows.
    tracking_4000 = {}

    for row in tracking_rows:
        seed = int(row["seed"])
        epoch = int(row["epoch"])

        if seed in EXTEND_SEEDS and epoch == RECONSTRUCT_EPOCH:
            tracking_4000[seed] = row

    mobility_4000 = {}

    for row in mobility_rows:
        seed = int(row["seed"])
        epoch = int(row["epoch"])

        if seed in EXTEND_SEEDS and epoch == RECONSTRUCT_EPOCH:
            mobility_4000[seed] = row

    if set(tracking_4000) != set(EXTEND_SEEDS):
        raise RuntimeError("Missing Stage-24 epoch-4000 tracking references.")

    if set(mobility_4000) != set(EXTEND_SEEDS):
        raise RuntimeError("Missing Stage-24 epoch-4000 mobility references.")

    # Read all Stage-24 mobility rows by seed so the recovery streak can be
    # carried exactly into Stage 25.
    mobility_by_seed = {
        seed: sorted(
            [
                r for r in mobility_rows
                if int(r["seed"]) == seed
            ],
            key=lambda x: int(x["epoch"]),
        )
        for seed in EXTEND_SEEDS
    }

    completed_stage24 = [
        r for r in paired_rows
        if bool_from_csv(r["b1_certified_escape"])
    ]

    if len(completed_stage24) != 2:
        raise RuntimeError(
            "Expected exactly 2 Stage-24 completed b=1 escapes."
        )

    return {
        **shas,
        "decision": decision,
        "paired": paired,
        "tracking_4000": tracking_4000,
        "mobility_4000": mobility_4000,
        "mobility_by_seed": mobility_by_seed,
        "completed_stage24": completed_stage24,
    }


# =============================================================================
# Reconstruct exact epoch-4000 state
# =============================================================================

def reconstruct_epoch_4000(
    stage3,
    stage18,
    stage19,
    stage20,
    stage22,
    stage24,
    device,
    seed: int,
    expected_track: dict,
    expected_mobility: dict,
    out_dir: Path,
):

    cfg = stage22.make_config(
        stage3=stage3,
        seed=seed,
        device=device,
        out_dir=out_dir,
    )

    exp = stage22.SymmetrySwapExperiment(
        stage3=stage3,
        cfg=cfg,
        device=device,
        base_mode=BASE_MODE,
        target_mode=TARGET_MODE,
        out_dir=out_dir,
    )

    for _ in range(RECONSTRUCT_EPOCH):
        exp.train_step()

    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    probe = stage19.cheap_probe(
        exp=exp,
        mode=TARGET_MODE,
    )

    audit = stage24.light_audit(
        stage18=stage18,
        stage20=stage20,
        exp=exp,
        seed=seed,
        epoch=RECONSTRUCT_EPOCH,
        kind="RECONSTRUCTED_EPOCH_4000",
    )

    diffs = {
        "relative_l2_error":
            abs(
                rel
                - float(expected_track["relative_l2_error"])
            ),

        "vpinn_loss":
            abs(
                float(rm["vpinn_loss"])
                - float(expected_track["vpinn_loss"])
            ),

        "target_share":
            abs(
                float(rm["target_mode_residual_energy_share"])
                - float(
                    expected_track[
                        "target_mode_residual_energy_share"
                    ]
                )
            ),

        "target_abs_residual":
            abs(
                float(rm["target_mode_abs_residual"])
                - float(expected_track["target_mode_abs_residual"])
            ),

        "adam_target_uphill_cosine":
            abs(
                float(probe["adam_target_uphill_cosine"])
                - float(expected_track["adam_target_uphill_cosine"])
            ),

        "mu_raw":
            abs(
                float(audit["mu_raw"])
                - float(expected_mobility["mu_raw"])
            ),

        "top_eigenvalue_fraction":
            abs(
                float(audit["top_eigenvalue_fraction"])
                - float(
                    expected_mobility[
                        "top_eigenvalue_fraction"
                    ]
                )
            ),

        "effective_rank":
            abs(
                float(audit["effective_rank"])
                - float(expected_mobility["effective_rank"])
            ),
    }

    max_gap = max(diffs.values())

    if max_gap > REPLAY_TOL:
        raise RuntimeError(
            f"Epoch-4000 replay failed seed={seed}: "
            f"max_gap={max_gap:.3e}, diffs={diffs}"
        )

    return exp, audit, {
        "seed": seed,
        "epoch": RECONSTRUCT_EPOCH,
        "max_abs_difference": max_gap,
        "pass": True,
        **{f"gap_{k}": v for k, v in diffs.items()},
    }


# =============================================================================
# Carry Stage-24 mobility recovery state
# =============================================================================

def stage24_recovery_state(
    mobility_rows: List[dict],
    conflict_mu: float,
    prior_recovery_onset: int,
):
    """
    Reconstruct the recovery certification state at epoch 4000.

    Returns:
        recovery_onset, current_streak, current_candidate_epoch
    """

    if prior_recovery_onset >= 0:
        return prior_recovery_onset, MOBILITY_RECOVERY_POINTS, prior_recovery_onset

    rows = [
        r for r in mobility_rows
        if (
            int(r["epoch"]) <= RECONSTRUCT_EPOCH
            and r["audit_kind"] == "SURVEILLANCE_250"
        )
    ]

    rows.sort(key=lambda x: int(x["epoch"]))

    streak = 0
    candidate = None

    for row in rows:
        recovered = bool(
            float(row["mu_raw"])
            / max(conflict_mu, 1.0e-300)
            >= MOBILITY_RECOVERY_RATIO
        )

        if recovered:
            if streak == 0:
                candidate = int(row["epoch"])

            streak += 1

            if streak >= MOBILITY_RECOVERY_POINTS:
                return candidate, streak, candidate

        else:
            streak = 0
            candidate = None

    return -1, streak, candidate


# =============================================================================
# Plot
# =============================================================================

def plot_extension(rows: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.5))

    for seed in EXTEND_SEEDS:
        rr = [
            r for r in rows
            if int(r["seed"]) == seed
        ]

        rr.sort(key=lambda x: int(x["epoch"]))

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r["relative_l2_error"]) for r in rr],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"seed {seed}",
        )

    ax.axhline(
        CONVERGENCE_REL_L2,
        linestyle="--",
        linewidth=1.0,
        label="escape relL2 threshold",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Minimal extension of the three horizon-censored deep-lock runs")
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
    stage19_script = resolve(args.stage19_script)
    stage20_script = resolve(args.stage20_script)
    stage22_script = resolve(args.stage22_script)
    stage24_script = resolve(args.stage24_script)
    stage24_dir = resolve(args.stage24_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage18_script=stage18_script,
        stage19_script=stage19_script,
        stage20_script=stage20_script,
        stage22_script=stage22_script,
        stage24_script=stage24_script,
        stage24_dir=stage24_dir,
    )

    stage3 = load_module(stage3_script, "vpinn_stage3_stage25R")
    stage18 = load_module(stage18_script, "vpinn_stage18_stage25R")
    stage19 = load_module(stage19_script, "vpinn_stage19_stage25R")
    stage20 = load_module(stage20_script, "vpinn_stage20_stage25R")
    stage22 = load_module(stage22_script, "vpinn_stage22_stage25R")
    stage24 = load_module(stage24_script, "vpinn_stage24_stage25R")

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),

        "stage3_solver_sha256": pf["s3"],
        "stage18_script_sha256": pf["s18"],
        "stage19_script_sha256": pf["s19"],
        "stage20_script_sha256": pf["s20"],
        "stage22_script_sha256": pf["s22"],
        "stage24_script_sha256": pf["s24"],
        "stage25r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "minimal_horizon_closure_of_stage24_p5",

            "continued_seeds":
                list(EXTEND_SEEDS),

            "reconstruct_epoch":
                RECONSTRUCT_EPOCH,

            "new_horizon":
                MAX_EPOCH,

            "horizon_rationale":
                "575 epochs after latest Stage-24 release 3925; exceeds observed 250-400 post-release escape lags",

            "C1":
                "3/3 epoch-4000 reconstruction <=1e-10",

            "C2":
                "new certified escape >=2/3",

            "C3":
                "cumulative b1 escape >=4/5 and recovery-before-escape >=80% of escaped",

            "no_new_factor":
                True,

            "optimizer_intervention":
                False,
        },
    }

    write_json(out_dir / "manifest.json", manifest)

    print("=" * 176)
    print(
        "VPINN — STAGE 25R MINIMAL HORIZON CLOSURE OF STAGE-24 P5"
    )
    print("=" * 176)
    print(f"device                    : {device}")
    print(f"continued seeds           : {list(EXTEND_SEEDS)}")
    print(f"reconstruct epoch         : {RECONSTRUCT_EPOCH}")
    print(f"extended horizon          : {MAX_EPOCH}")
    print("optimizer intervention    : NONE")
    print("=" * 176)

    replay_rows = []
    tracking_rows = []
    mobility_rows = []
    extension_rows = []

    global_start = time.perf_counter()

    for seed in EXTEND_SEEDS:

        run_dir = (
            out_dir
            / f"seed_{seed:03d}"
            / "base_01_target_09"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        stage24_row = pf["paired"][seed]

        conflict_mu = float(stage24_row["b1_conflict_mu"])
        prior_recovery_onset = int(
            stage24_row["b1_mobility_recovery_onset_epoch"]
        )

        exp, audit4000, replay = reconstruct_epoch_4000(
            stage3=stage3,
            stage18=stage18,
            stage19=stage19,
            stage20=stage20,
            stage22=stage22,
            stage24=stage24,
            device=device,
            seed=seed,
            expected_track=pf["tracking_4000"][seed],
            expected_mobility=pf["mobility_4000"][seed],
            out_dir=run_dir,
        )

        replay_rows.append(replay)
        mobility_rows.append(audit4000)

        (
            recovery_onset,
            recovery_streak,
            recovery_candidate_epoch,
        ) = stage24_recovery_state(
            mobility_rows=pf["mobility_by_seed"][seed],
            conflict_mu=conflict_mu,
            prior_recovery_onset=prior_recovery_onset,
        )

        escape_streak = 0
        escape_candidate_epoch = None
        escape_candidate_state = None
        escape_onset = -1
        escape_confirmation = -1

        # Save exact reconstructed horizon state for later read-only work.
        torch.save(
            {
                "seed": seed,
                "epoch": RECONSTRUCT_EPOCH,
                "base_mode": BASE_MODE,
                "target_mode": TARGET_MODE,
                "model_state_dict":
                    copy.deepcopy(exp.model.state_dict()),
                "optimizer_state_dict":
                    copy.deepcopy(exp.optimizer.state_dict()),
            },
            run_dir / "state_epoch_4000.pt",
        )

        print()
        print("-" * 176)
        print(
            f"seed={seed}: replay4000 gap={replay['max_abs_difference']:.3e}, "
            f"relL2={audit4000['relative_l2_error']:.6e}, "
            f"share={audit4000['target_mode_residual_energy_share']:.6f}, "
            f"mu={audit4000['mu_raw']:.6e}, "
            f"recovery_onset_inherited={recovery_onset}, "
            f"recovery_streak={recovery_streak}"
        )

        for epoch in range(RECONSTRUCT_EPOCH, MAX_EPOCH + 1):

            if epoch % TRACK_INTERVAL == 0:

                rm = exp.residual_metrics()
                rel = exp.relative_l2_error()

                probe = stage19.cheap_probe(
                    exp=exp,
                    mode=TARGET_MODE,
                )

                qualifies_escape = bool(
                    rel <= CONVERGENCE_REL_L2
                    and
                    rm["target_mode_residual_energy_share"]
                    <= CONVERGENCE_TARGET_SHARE
                )

                current_state = None

                if qualifies_escape:
                    current_state = stage18.capture_state(exp)

                    if escape_streak == 0:
                        escape_candidate_epoch = epoch
                        escape_candidate_state = copy.deepcopy(
                            current_state
                        )

                    escape_streak += 1
                else:
                    escape_streak = 0
                    escape_candidate_epoch = None
                    escape_candidate_state = None

                if (
                    escape_onset < 0
                    and escape_streak >= CERTIFY_POINTS
                ):
                    escape_onset = int(escape_candidate_epoch)
                    escape_confirmation = epoch

                    # Preserve the exact escape-onset state for Stage 26.
                    torch.save(
                        {
                            "seed": seed,
                            "epoch": escape_onset,
                            "base_mode": BASE_MODE,
                            "target_mode": TARGET_MODE,
                            "model_state_dict":
                                copy.deepcopy(
                                    escape_candidate_state["model"]
                                ),
                            "optimizer_state_dict":
                                copy.deepcopy(
                                    escape_candidate_state["optimizer"]
                                ),
                        },
                        run_dir / "certified_escape_onset_state.pt",
                    )

                tracking_rows.append(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "relative_l2_error": rel,
                        **rm,

                        "adam_target_uphill_cosine":
                            probe["adam_target_uphill_cosine"],

                        "adam_candidate_target_uphill":
                            probe["adam_candidate_target_uphill"],

                        "escape_streak":
                            escape_streak,
                    }
                )

            if (
                epoch > RECONSTRUCT_EPOCH
                and epoch % MOBILITY_AUDIT_INTERVAL == 0
            ):

                audit = stage24.light_audit(
                    stage18=stage18,
                    stage20=stage20,
                    exp=exp,
                    seed=seed,
                    epoch=epoch,
                    kind="EXTENSION_SURVEILLANCE_250",
                )

                mobility_rows.append(audit)

                if recovery_onset < 0:
                    recovered = bool(
                        float(audit["mu_raw"])
                        / max(conflict_mu, 1.0e-300)
                        >= MOBILITY_RECOVERY_RATIO
                    )

                    if recovered:
                        if recovery_streak == 0:
                            recovery_candidate_epoch = epoch

                        recovery_streak += 1
                    else:
                        recovery_streak = 0
                        recovery_candidate_epoch = None

                    if recovery_streak >= MOBILITY_RECOVERY_POINTS:
                        recovery_onset = int(
                            recovery_candidate_epoch
                        )

                        # Save current state. If candidate was 4000 this file
                        # records the certification state, while the event
                        # onset remains correctly reported as 4000.
                        torch.save(
                            {
                                "seed": seed,
                                "recovery_onset_epoch":
                                    recovery_onset,
                                "certification_epoch":
                                    epoch,
                                "base_mode": BASE_MODE,
                                "target_mode": TARGET_MODE,
                                "model_state_dict":
                                    copy.deepcopy(
                                        exp.model.state_dict()
                                    ),
                                "optimizer_state_dict":
                                    copy.deepcopy(
                                        exp.optimizer.state_dict()
                                    ),
                            },
                            run_dir / "mobility_recovery_certification_state.pt",
                        )

            if escape_onset >= 0 and epoch >= escape_confirmation:
                break

            if epoch < MAX_EPOCH:
                exp.train_step()

        recovery_before_escape = bool(
            recovery_onset >= 0
            and
            escape_onset >= 0
            and
            recovery_onset < escape_onset
        )

        extension_rows.append(
            {
                "seed": seed,

                "stage24_conflict_onset_epoch":
                    int(stage24_row["b1_conflict_onset_epoch"]),

                "stage24_conflict_release_onset_epoch":
                    int(stage24_row["b1_conflict_release_onset_epoch"]),

                "stage24_conflict_mu":
                    conflict_mu,

                "inherited_or_extended_recovery_onset_epoch":
                    recovery_onset,

                "new_certified_escape":
                    escape_onset >= 0,

                "new_escape_onset_epoch":
                    escape_onset,

                "new_escape_confirmation_epoch":
                    escape_confirmation,

                "recovery_precedes_escape":
                    recovery_before_escape,
            }
        )

        print(
            f"  recovery={recovery_onset}, "
            f"escape={escape_onset}, "
            f"confirm={escape_confirmation}, "
            f"recovery<escape={recovery_before_escape}"
        )

    # =========================================================================
    # Cumulative closure accounting
    # =========================================================================
    write_csv(
        out_dir / "epoch4000_replay_checks.csv",
        replay_rows,
    )
    write_csv(
        out_dir / "extension_tracking_metrics.csv",
        tracking_rows,
    )
    write_csv(
        out_dir / "extension_mobility_audits.csv",
        mobility_rows,
    )
    write_csv(
        out_dir / "extension_run_summary.csv",
        extension_rows,
    )

    C1 = bool(
        len(replay_rows) == 3
        and all(bool(r["pass"]) for r in replay_rows)
    )

    new_escape_count = sum(
        int(bool(r["new_certified_escape"]))
        for r in extension_rows
    )

    C2 = bool(new_escape_count >= 2)

    cumulative_rows = []

    # Existing completed Stage-24 runs.
    for row in pf["completed_stage24"]:
        cumulative_rows.append(
            {
                "seed": int(row["seed"]),
                "certified_escape": True,
                "escape_onset_epoch":
                    int(row["b1_escape_onset_epoch"]),
                "recovery_onset_epoch":
                    int(row["b1_mobility_recovery_onset_epoch"]),
                "recovery_precedes_escape":
                    bool_from_csv(row["b1_recovery_precedes_escape"]),
                "source": "STAGE24",
            }
        )

    # Extended censored runs.
    for row in extension_rows:
        cumulative_rows.append(
            {
                "seed": int(row["seed"]),
                "certified_escape":
                    bool(row["new_certified_escape"]),
                "escape_onset_epoch":
                    int(row["new_escape_onset_epoch"]),
                "recovery_onset_epoch":
                    int(
                        row[
                            "inherited_or_extended_recovery_onset_epoch"
                        ]
                    ),
                "recovery_precedes_escape":
                    bool(row["recovery_precedes_escape"]),
                "source": "STAGE25",
            }
        )

    write_csv(
        out_dir / "cumulative_stage24_p5_closure.csv",
        cumulative_rows,
    )

    escaped = [
        r for r in cumulative_rows
        if bool(r["certified_escape"])
    ]

    recovery_before_escape_count = sum(
        int(bool(r["recovery_precedes_escape"]))
        for r in escaped
    )

    C3 = bool(
        len(escaped) >= 4
        and
        recovery_before_escape_count
        >= math.ceil(0.80 * len(escaped))
    )

    closure = bool(C1 and C2 and C3)

    if closure:
        route_class = (
            "stage24_horizon_censoring_closed_base_mode_specificity_all_gates_closed"
        )
        next_route = (
            "stage26R_readonly_paired_tangent_eigenspace_mechanism_localization"
        )
    else:
        route_class = (
            "stage24_horizon_censoring_not_yet_closed"
        )
        next_route = (
            "stage26R_extend_only_remaining_censored_runs"
        )

    decision = {
        "continued_seeds":
            list(EXTEND_SEEDS),

        "C1_epoch4000_exact_reconstruction":
            C1,

        "new_escape_count":
            new_escape_count,

        "C2_new_escape_closure":
            C2,

        "cumulative_b1_escape_count":
            len(escaped),

        "cumulative_recovery_before_escape_count":
            recovery_before_escape_count,

        "C3_stage24_P5_closed":
            C3,

        "stage24_all_five_specificity_gates_closed":
            closure,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "This stage only resolves Stage-24 right-censoring. A PASS closes "
            "the paired b=1 versus b=2 phenotype comparison but does not by "
            "itself identify the tangent-space mechanism."
        ),
    }

    write_json(out_dir / "decision.json", decision)

    plot_extension(
        tracking_rows,
        out_dir / "censored_b1_extension_relL2.png",
    )

    # =========================================================================
    # Console
    # =========================================================================
    elapsed = time.perf_counter() - global_start

    lines = []

    lines.append("=" * 178)
    lines.append(
        "VPINN — STAGE 25R MINIMAL HORIZON CLOSURE SUMMARY"
    )
    lines.append("=" * 178)

    lines.append(
        "seed | replay4000 | recovery onset | escape onset | escape confirm | recovery<escape"
    )
    lines.append("-" * 178)

    replay_map = {
        int(r["seed"]): r
        for r in replay_rows
    }

    for row in extension_rows:
        seed = int(row["seed"])

        lines.append(
            f"{seed:4d} | "
            f"{replay_map[seed]['max_abs_difference']:.3e} | "
            f"{int(row['inherited_or_extended_recovery_onset_epoch']):14d} | "
            f"{int(row['new_escape_onset_epoch']):12d} | "
            f"{int(row['new_escape_confirmation_epoch']):14d} | "
            f"{str(row['recovery_precedes_escape'])}"
        )

    lines.append("-" * 178)

    lines.append(
        f"C1 epoch-4000 reconstruction         : "
        f"{sum(int(r['pass']) for r in replay_rows)}/3 -> {C1}"
    )

    lines.append(
        f"C2 new escapes                       : "
        f"{new_escape_count}/3 -> {C2}"
    )

    lines.append(
        f"cumulative b1 escapes                : "
        f"{len(escaped)}/5"
    )

    lines.append(
        f"recovery before escape               : "
        f"{recovery_before_escape_count}/{len(escaped)}"
    )

    lines.append(
        f"C3 Stage-24 P5 closure               : {C3}"
    )

    lines.append(
        f"ALL STAGE-24 SPECIFICITY GATES CLOSED: {closure}"
    )

    lines.append(
        f"route class                           : {route_class}"
    )

    lines.append(
        f"next route                            : {next_route}"
    )

    lines.append(
        f"elapsed seconds                       : {elapsed:.2f}"
    )

    lines.append("=" * 178)
    lines.append(
        "Guardrail: no new science is promoted here; this is censoring closure."
    )
    lines.append("=" * 178)

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
