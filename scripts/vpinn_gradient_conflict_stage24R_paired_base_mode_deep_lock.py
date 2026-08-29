#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 24R
Paired Base-Mode Deep-Lock Specificity Audit
=============================================

Scientific status
-----------------
Stage 22 used paired seeds {15,...,19} and found for target m=9:

    b=1 : certified conflict 5/5
    b=2 : certified conflict 5/5

but stopped both cells at conflict.

Stage 23 repaired the b=2 censoring and established:

    exact Stage-22 replay          10/10
    conflict release              10/10
    certified escape              10/10
    persistent deep mobility lock  0/10

For the b=2,m=9 paired subset specifically, conflict release occurred only
100-150 epochs after conflict onset and all five runs escaped.

Thus Stage 23 proved that a three-point target-uphill Adam conflict can be a
short transient and is NOT sufficient to define the pathological VPINN lock.

The paired b=1,m=9 trajectories for the SAME seeds remain censored.

Stage 24R repairs exactly that remaining censoring.

Question
--------
For the SAME target m=9 and SAME initialized networks, does changing only the
matched low-frequency base component

    b=2 -> b=1

change the phenotype from

    transient optimizer opposition

to

    persistent basis-invariant residual-mobility lock?

No new seeds.
No new target frequency.
No optimizer intervention.
No architecture change.
No threshold change.

Experiment
----------
Reconstruct each Stage-22 b=1,m=9 certified-conflict onset exactly for

    seeds {15,16,17,18,19}

and continue ordinary Adam through at most epoch 4000.

The manufactured family remains

    u* = (1/b) sin(b*pi*x) + (1.05/9) sin(9*pi*x),

so the base weak scale is exactly 1 and target weak scale exactly 1.05 for
both b=1 and b=2.

Tracking
--------
Adam target alignment every 25 epochs.

Basis-invariant full J/K mobility surveillance every 250 epochs.

Inherited quantities:

    mu = r^T K r / (||r||^2 tr K)

    deep-collapse threshold = 1e-6

    mobility-recovery ratio = 100x.

Certified optimizer-conflict release
------------------------------------
After the Stage-22 conflict onset:

    <g_T, Delta_Adam> <= 0

for THREE consecutive 25-epoch probes.

Certified escape
----------------
Inherited unchanged:

    relL2 <= 1e-2
    AND target residual share <= 0.20

for THREE consecutive observations.

Persistent deep mobility lock
-----------------------------
While unresolved and target-localized,

    relL2 > 1e-2
    target share >= 0.80,

TWO consecutive 250-epoch surveillance audits must satisfy

    mu <= 1e-6.

This is identical to Stage 23.

Certified mobility recovery
---------------------------
Only after persistent deep lock is certified, mobility recovery is certified
when TWO consecutive 250-epoch surveillance audits satisfy

    mu / mu_conflict >= 100.

The 100x threshold is inherited from Stage 20.

Paired b=2 controls
-------------------
Stage-23 b=2,m=9 outcomes for the exact same seeds are read only.

For every seed compare:

    conflict-onset mobility
    conflict->release delay
    conflict->escape delay
    persistent-deep-lock status.

Censor-aware comparison
-----------------------
If a b=1 run has not released/escaped by epoch 4000, its observed delay is a
right-censored lower bound

    4000 - conflict_onset.

A censored b=1 delay is counted as "at least 2x slower than b=2" only if this
lower bound is already >= 2 * paired b=2 delay.

Precommitted gates
------------------

P1 — EXACT B=1 STAGE-22 REPLAY
    5/5 onset replays PASS to <=1e-10.

P2 — PERSISTENT DEEP LOCK REPLICATION
    persistent deep mobility lock in >=4/5 b=1,m=9 runs.

P3 — CONFLICT-PERSISTENCE CONTRAST
    b=1 conflict->release delay is at least 2x the paired b=2 delay
    in >=4/5 seeds, using the censor-aware rule above.

P4 — ESCAPE-DELAY CONTRAST
    b=1 conflict->escape delay is greater than paired b=2 delay
    in >=4/5 seeds, again censor-aware.

P5 — MOBILITY RECOVERY PRECEDES ESCAPE
    at least 4 b=1 runs certify escape by epoch 4000,
    and among escaped runs >=80% certify mobility recovery before escape.

STRONG BASE-MODE DEEP-LOCK SPECIFICITY:
    P1 & P2 & P3 & P4 & P5.

Interpretation if PASS
----------------------
For fixed target m=9 and paired initialization, the b=1 low-mode background
produces a qualitatively different long-lived state than b=2:

    transient target-uphill conflict is common,
    but the pathological plateau is the persistent low-mobility state.

This does NOT yet prove why b=1 causes the collapse.

Next if PASS
------------
Stage 25R = read-only tangent-eigenspace mechanism localization, comparing
paired b=1 deep-lock and b=2 transient states before any new training sweep.

If P2 passes but P5 is horizon-censored:
    extend only censored b=1 runs.

If P2 fails:
    do not claim base-mode specificity; route to seed/state heterogeneity.

Novelty guardrail
-----------------
Effective-rank / residual-NTK spectral collapse has appeared in recent
hard-constraint PINN literature. The candidate contribution here is NOT
"low effective rank is new." The sharper object is the VPINN-specific,
basis-invariant residual-to-tangent mobility and its separation of transient
gradient conflict from a persistent weak-residual lock.
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
SEEDS = (15, 16, 17, 18, 19)

MAX_EPOCH = 4000
TRACK_INTERVAL = 25
MOBILITY_AUDIT_INTERVAL = 250

ACTIVE_TARGET_SHARE = 0.80
CONVERGENCE_REL_L2 = 1.0e-2
CONVERGENCE_TARGET_SHARE = 0.20
CERTIFY_POINTS = 3

MOBILITY_COLLAPSE_THRESHOLD = 1.0e-6
MOBILITY_LOCK_POINTS = 2

MOBILITY_RECOVERY_RATIO = 100.0
MOBILITY_RECOVERY_POINTS = 2

REPLAY_TOL = 1.0e-10


# =============================================================================
# CLI / generic helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-24R paired b=1 vs b=2 deep-lock specificity audit."
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
        "--stage22-dir",
        default="vpinn_gradient_conflict_stage22R_symmetry_sector_swap",
    )
    p.add_argument(
        "--stage23-script",
        default="vpinn_gradient_conflict_stage23R_censoring_repair_continuation.py",
    )
    p.add_argument(
        "--stage23-dir",
        default="vpinn_gradient_conflict_stage23R_censoring_repair_continuation",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage24R_paired_base_mode_deep_lock",
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
            [{k: row.get(k, None) for k in fields} for row in rows]
        )


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Preflight
# =============================================================================

def preflight(
    stage3_script: Path,
    stage18_script: Path,
    stage19_script: Path,
    stage20_script: Path,
    stage22_script: Path,
    stage22_dir: Path,
    stage23_script: Path,
    stage23_dir: Path,
) -> dict:

    s22_manifest_path = stage22_dir / "manifest.json"
    s22_run_path = stage22_dir / "run_summary.csv"
    s22_audit_path = stage22_dir / "event_kernel_audits.csv"

    s23_manifest_path = stage23_dir / "manifest.json"
    s23_decision_path = stage23_dir / "decision.json"
    s23_run_path = stage23_dir / "run_summary.csv"

    for p in (
        s22_manifest_path,
        s22_run_path,
        s22_audit_path,
        s23_manifest_path,
        s23_decision_path,
        s23_run_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    s22m = read_json(s22_manifest_path)
    s23m = read_json(s23_manifest_path)
    s23d = read_json(s23_decision_path)

    shas = {
        "s3": sha256_file(stage3_script),
        "s18": sha256_file(stage18_script),
        "s19": sha256_file(stage19_script),
        "s20": sha256_file(stage20_script),
        "s22": sha256_file(stage22_script),
        "s23": sha256_file(stage23_script),
    }

    if s22m.get("stage3_solver_sha256") != shas["s3"]:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 22.")
    if s22m.get("stage18_script_sha256") != shas["s18"]:
        raise RuntimeError("Stage-18 SHA mismatch against Stage 22.")
    if s22m.get("stage19_script_sha256") != shas["s19"]:
        raise RuntimeError("Stage-19 SHA mismatch against Stage 22.")
    if s22m.get("stage20_script_sha256") != shas["s20"]:
        raise RuntimeError("Stage-20 SHA mismatch against Stage 22.")
    if s22m.get("stage22r_script_sha256") != shas["s22"]:
        raise RuntimeError("Stage-22 SHA mismatch.")

    if s23m.get("stage22_script_sha256") != shas["s22"]:
        raise RuntimeError("Stage-22 SHA mismatch against Stage 23.")
    if s23m.get("stage23r_script_sha256") != shas["s23"]:
        raise RuntimeError("Stage-23 SHA mismatch.")

    if not bool(s23d.get("transient_conflict_phenotype_supported", False)):
        raise RuntimeError(
            "Stage 23 did not establish the transient b=2 conflict phenotype."
        )

    if int(s23d.get("persistent_deep_lock_count", -1)) != 0:
        raise RuntimeError("Expected Stage-23 b=2 deep-lock count = 0/10.")

    s22_runs = read_csv(s22_run_path)
    s22_audits = read_csv(s22_audit_path)
    s23_runs = read_csv(s23_run_path)

    onset_map = {}
    expected_audit_map = {}

    for row in s22_runs:
        if (
            int(row["base_mode"]) == BASE_MODE
            and int(row["target_mode"]) == TARGET_MODE
        ):
            seed = int(row["seed"])

            if str(row["certified_conflict"]).lower() != "true":
                raise RuntimeError(
                    f"Expected Stage-22 b=1,m=9 conflict for seed={seed}."
                )

            onset_map[seed] = int(float(row["conflict_onset_epoch"]))

    for row in s22_audits:
        if (
            int(row["base_mode"]) == BASE_MODE
            and int(row["target_mode"]) == TARGET_MODE
            and row["audit_kind"] == "CERTIFIED_CONFLICT_ONSET"
        ):
            expected_audit_map[int(row["seed"])] = row

    if set(onset_map) != set(SEEDS) or set(expected_audit_map) != set(SEEDS):
        raise RuntimeError(
            "Incomplete Stage-22 b=1,m=9 onset/audit records."
        )

    b2_map = {}

    for row in s23_runs:
        if (
            int(row["base_mode"]) == 2
            and int(row["target_mode"]) == TARGET_MODE
        ):
            seed = int(row["seed"])
            b2_map[seed] = {
                "conflict_onset_epoch":
                    int(row["stage22_conflict_onset_epoch"]),

                "release_onset_epoch":
                    int(row["conflict_release_onset_epoch"]),

                "escape_onset_epoch":
                    int(row["escape_onset_epoch"]),

                "release_delay":
                    int(row["conflict_to_release_delay"]),

                "escape_delay":
                    int(row["conflict_to_escape_delay"]),

                "deep_lock":
                    str(
                        row["persistent_deep_mobility_lock"]
                    ).lower() == "true",

                "conflict_mu":
                    float(row["stage22_conflict_mu"]),
            }

    if set(b2_map) != set(SEEDS):
        raise RuntimeError("Incomplete Stage-23 paired b=2,m=9 controls.")

    if any(v["deep_lock"] for v in b2_map.values()):
        raise RuntimeError("Unexpected b=2,m=9 deep lock in Stage 23.")

    return {
        **shas,
        "onset_map": onset_map,
        "expected_audit_map": expected_audit_map,
        "b2_map": b2_map,
    }


# =============================================================================
# Light J/K audit
# =============================================================================

def light_audit(
    stage18,
    stage20,
    exp,
    seed: int,
    epoch: int,
    kind: str,
) -> dict:

    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    kernel = stage18.residual_jacobian(exp)

    adam = stage18.predict_and_decompose_adam(
        exp=exp,
        J=kernel["J"],
        r=kernel["r"],
        params=kernel["params"],
        target_index=TARGET_MODE - 1,
    )

    inv = stage20.kernel_invariants(
        r=kernel["r"].cpu().numpy(),
        K=kernel["K"].cpu().numpy(),
        KD=adam["K_D"].cpu().numpy(),
        rotation_seed=240000 + 100*seed + epoch,
    )

    return {
        "seed": seed,
        "base_mode": BASE_MODE,
        "target_mode": TARGET_MODE,
        "epoch": epoch,
        "audit_kind": kind,

        "relative_l2_error": rel,
        **rm,

        "adam_target_uphill_cosine":
            adam["adam_target_uphill_cosine"],

        "adam_candidate_target_uphill":
            adam["adam_candidate_target_uphill"],

        **inv,
    }


# =============================================================================
# Exact onset reconstruction
# =============================================================================

def reconstruct_onset(
    stage3,
    stage18,
    stage19,
    stage20,
    stage22,
    device,
    seed: int,
    onset_epoch: int,
    expected: dict,
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

    for _ in range(onset_epoch):
        exp.train_step()

    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    probe = stage19.cheap_probe(
        exp=exp,
        mode=TARGET_MODE,
    )

    audit = light_audit(
        stage18=stage18,
        stage20=stage20,
        exp=exp,
        seed=seed,
        epoch=onset_epoch,
        kind="RECONSTRUCTED_CONFLICT_ONSET",
    )

    diffs = {
        "relative_l2_error":
            abs(rel - float(expected["relative_l2_error"])),

        "target_share":
            abs(
                float(rm["target_mode_residual_energy_share"])
                - float(expected["target_mode_residual_energy_share"])
            ),

        "adam_target_uphill_cosine":
            abs(
                float(probe["adam_target_uphill_cosine"])
                - float(expected["adam_target_uphill_cosine"])
            ),

        "mu_raw":
            abs(float(audit["mu_raw"]) - float(expected["mu_raw"])),
    }

    gap = max(diffs.values())

    if gap > REPLAY_TOL:
        raise RuntimeError(
            f"b=1,m=9 replay failed seed={seed}: "
            f"gap={gap:.3e}, diffs={diffs}"
        )

    replay = {
        "seed": seed,
        "onset_epoch": onset_epoch,
        "max_abs_difference": gap,
        "pass": True,
        **{f"gap_{k}": v for k, v in diffs.items()},
    }

    return exp, audit, replay


# =============================================================================
# Censor-aware paired comparisons
# =============================================================================

def at_least_factor_slower(
    event_epoch: int,
    horizon: int,
    onset_epoch: int,
    paired_delay: int,
    factor: float,
) -> bool:

    observed_delay = (
        event_epoch - onset_epoch
        if event_epoch >= 0
        else horizon - onset_epoch
    )

    return bool(
        observed_delay >= factor * paired_delay
    )


def strictly_slower(
    event_epoch: int,
    horizon: int,
    onset_epoch: int,
    paired_delay: int,
) -> bool:

    observed_delay = (
        event_epoch - onset_epoch
        if event_epoch >= 0
        else horizon - onset_epoch
    )

    return bool(
        observed_delay > paired_delay
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_paired_delays(rows: List[dict], path: Path) -> None:
    seeds = [int(r["seed"]) for r in rows]
    x = np.arange(len(seeds))
    width = 0.36

    b1 = [
        float(r["b1_conflict_to_release_observed_or_lower_bound"])
        for r in rows
    ]

    b2 = [
        float(r["b2_conflict_to_release_delay"])
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.3))

    ax.bar(x - width/2, b2, width, label="b=2 transient")
    ax.bar(x + width/2, b1, width, label="b=1")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Paired seed")
    ax.set_ylabel("Conflict-to-release delay (epochs)")
    ax.set_title("Same target, same initialization: does b=1 sustain conflict longer?")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_mobility(audits: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    for seed in SEEDS:
        rr = [
            r for r in audits
            if int(r["seed"]) == seed
        ]
        rr.sort(key=lambda x: int(x["epoch"]))

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r["mu_raw"]) for r in rr],
            marker="o",
            markersize=3,
            linewidth=1.1,
            label=f"seed {seed}",
        )

    ax.axhline(
        MOBILITY_COLLAPSE_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label="deep-collapse threshold",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Basis-invariant residual mobility μ")
    ax.set_title("b=1,m=9 mobility persistence and recovery")
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
    stage22_dir = resolve(args.stage22_dir)
    stage23_script = resolve(args.stage23_script)
    stage23_dir = resolve(args.stage23_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage18_script=stage18_script,
        stage19_script=stage19_script,
        stage20_script=stage20_script,
        stage22_script=stage22_script,
        stage22_dir=stage22_dir,
        stage23_script=stage23_script,
        stage23_dir=stage23_dir,
    )

    stage3 = load_module(stage3_script, "vpinn_stage3_stage24R")
    stage18 = load_module(stage18_script, "vpinn_stage18_stage24R")
    stage19 = load_module(stage19_script, "vpinn_stage19_stage24R")
    stage20 = load_module(stage20_script, "vpinn_stage20_stage24R")
    stage22 = load_module(stage22_script, "vpinn_stage22_stage24R")

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
        "stage23_script_sha256": pf["s23"],
        "stage24r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "paired_base_mode_deep_lock_specificity",

            "base_mode":
                BASE_MODE,

            "target_mode":
                TARGET_MODE,

            "paired_seeds":
                list(SEEDS),

            "paired_control":
                "Stage-23 b=2,m=9 same seed",

            "max_epoch":
                MAX_EPOCH,

            "mobility_audit_interval":
                MOBILITY_AUDIT_INTERVAL,

            "persistent_deep_lock":
                "2 consecutive unresolved/localized 250-grid audits with mu<=1e-6",

            "mobility_recovery":
                "2 consecutive post-deep-lock 250-grid audits with mu/mu_conflict>=100",

            "P1":
                "5/5 exact Stage-22 b=1,m=9 replay",

            "P2":
                "persistent deep lock >=4/5",

            "P3":
                "b1 conflict-release delay >=2x paired b2 in >=4/5",

            "P4":
                "b1 conflict-escape delay > paired b2 in >=4/5",

            "P5":
                ">=4 b1 escapes; recovery before escape in >=80% escaped",

            "no_new_training_factor":
                True,

            "optimizer_intervention":
                False,
        },
    }

    write_json(out_dir / "manifest.json", manifest)

    print("=" * 178)
    print(
        "VPINN — STAGE 24R PAIRED BASE-MODE DEEP-LOCK SPECIFICITY AUDIT"
    )
    print("=" * 178)
    print(f"device                    : {device}")
    print(f"continued cell            : b=1, m=9")
    print(f"paired control            : Stage-23 b=2, m=9")
    print(f"paired seeds              : {list(SEEDS)}")
    print(f"max epoch                 : {MAX_EPOCH}")
    print("optimizer intervention    : NONE")
    print("=" * 178)

    tracking_rows = []
    audit_rows = []
    replay_rows = []
    run_rows = []

    global_start = time.perf_counter()

    for seed in SEEDS:

        run_dir = (
            out_dir
            / f"seed_{seed:03d}"
            / "base_01_target_09"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        onset_epoch = pf["onset_map"][seed]
        expected = pf["expected_audit_map"][seed]
        b2 = pf["b2_map"][seed]

        exp, onset_audit, replay = reconstruct_onset(
            stage3=stage3,
            stage18=stage18,
            stage19=stage19,
            stage20=stage20,
            stage22=stage22,
            device=device,
            seed=seed,
            onset_epoch=onset_epoch,
            expected=expected,
            out_dir=run_dir,
        )

        replay_rows.append(replay)
        audit_rows.append(onset_audit)

        onset_mu = float(onset_audit["mu_raw"])

        release_streak = 0
        release_candidate_epoch = None
        release_onset = -1
        release_confirmation = -1

        escape_streak = 0
        escape_candidate_epoch = None
        escape_onset = -1
        escape_confirmation = -1

        deep_streak = 0
        deep_candidate_epoch = None
        deep_lock_onset = -1

        recovery_streak = 0
        recovery_candidate_epoch = None
        mobility_recovery_onset = -1

        last_audit_epoch = onset_epoch

        print()
        print("-" * 178)
        print(
            f"seed={seed}: onset={onset_epoch}, "
            f"mu0={onset_mu:.6e}, "
            f"paired b2 release delay={b2['release_delay']}, "
            f"paired b2 escape delay={b2['escape_delay']}"
        )

        for epoch in range(onset_epoch, MAX_EPOCH + 1):

            if epoch % TRACK_INTERVAL == 0:

                rm = exp.residual_metrics()
                rel = exp.relative_l2_error()

                probe = stage19.cheap_probe(
                    exp=exp,
                    mode=TARGET_MODE,
                )

                nonuphill = bool(
                    not probe["adam_candidate_target_uphill"]
                )

                if release_onset < 0:
                    if nonuphill:
                        if release_streak == 0:
                            release_candidate_epoch = epoch
                        release_streak += 1
                    else:
                        release_streak = 0
                        release_candidate_epoch = None

                    if release_streak >= CERTIFY_POINTS:
                        release_onset = int(release_candidate_epoch)
                        release_confirmation = epoch

                qualifies_escape = bool(
                    rel <= CONVERGENCE_REL_L2
                    and
                    rm["target_mode_residual_energy_share"]
                    <= CONVERGENCE_TARGET_SHARE
                )

                if escape_onset < 0:
                    if qualifies_escape:
                        if escape_streak == 0:
                            escape_candidate_epoch = epoch
                        escape_streak += 1
                    else:
                        escape_streak = 0
                        escape_candidate_epoch = None

                    if escape_streak >= CERTIFY_POINTS:
                        escape_onset = int(escape_candidate_epoch)
                        escape_confirmation = epoch

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

                        "release_streak":
                            release_streak,

                        "escape_streak":
                            escape_streak,
                    }
                )

            if (
                epoch > onset_epoch
                and epoch % MOBILITY_AUDIT_INTERVAL == 0
            ):
                audit = light_audit(
                    stage18=stage18,
                    stage20=stage20,
                    exp=exp,
                    seed=seed,
                    epoch=epoch,
                    kind="SURVEILLANCE_250",
                )

                audit_rows.append(audit)
                last_audit_epoch = epoch

                deep_eligible = bool(
                    audit["relative_l2_error"] > CONVERGENCE_REL_L2
                    and
                    audit["target_mode_residual_energy_share"]
                    >= ACTIVE_TARGET_SHARE
                )

                collapsed = bool(
                    deep_eligible
                    and
                    float(audit["mu_raw"])
                    <= MOBILITY_COLLAPSE_THRESHOLD
                )

                if deep_lock_onset < 0:
                    if collapsed:
                        if deep_streak == 0:
                            deep_candidate_epoch = epoch
                        deep_streak += 1
                    else:
                        deep_streak = 0
                        deep_candidate_epoch = None

                    if deep_streak >= MOBILITY_LOCK_POINTS:
                        deep_lock_onset = int(deep_candidate_epoch)

                # Mobility recovery is only meaningful after a persistent
                # deep lock has already been certified.
                if (
                    deep_lock_onset >= 0
                    and mobility_recovery_onset < 0
                ):
                    recovered = bool(
                        float(audit["mu_raw"])
                        / max(onset_mu, 1.0e-300)
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
                        mobility_recovery_onset = int(
                            recovery_candidate_epoch
                        )

            if escape_onset >= 0 and epoch >= escape_confirmation:
                if epoch != last_audit_epoch:
                    audit_rows.append(
                        light_audit(
                            stage18=stage18,
                            stage20=stage20,
                            exp=exp,
                            seed=seed,
                            epoch=epoch,
                            kind="ESCAPE_CONFIRMATION_STATE",
                        )
                    )
                break

            if epoch < MAX_EPOCH:
                exp.train_step()

        if escape_onset < 0 and MAX_EPOCH != last_audit_epoch:
            audit_rows.append(
                light_audit(
                    stage18=stage18,
                    stage20=stage20,
                    exp=exp,
                    seed=seed,
                    epoch=MAX_EPOCH,
                    kind="HORIZON_STATE",
                )
            )

        b1_release_observed = (
            release_onset - onset_epoch
            if release_onset >= 0
            else MAX_EPOCH - onset_epoch
        )

        b1_escape_observed = (
            escape_onset - onset_epoch
            if escape_onset >= 0
            else MAX_EPOCH - onset_epoch
        )

        release_2x = at_least_factor_slower(
            event_epoch=release_onset,
            horizon=MAX_EPOCH,
            onset_epoch=onset_epoch,
            paired_delay=b2["release_delay"],
            factor=2.0,
        )

        escape_slower = strictly_slower(
            event_epoch=escape_onset,
            horizon=MAX_EPOCH,
            onset_epoch=onset_epoch,
            paired_delay=b2["escape_delay"],
        )

        recovery_before_escape = bool(
            mobility_recovery_onset >= 0
            and
            escape_onset >= 0
            and
            mobility_recovery_onset < escape_onset
        )

        run_rows.append(
            {
                "seed": seed,

                "b1_conflict_onset_epoch":
                    onset_epoch,

                "b1_conflict_mu":
                    onset_mu,

                "b1_certified_conflict_release":
                    release_onset >= 0,

                "b1_conflict_release_onset_epoch":
                    release_onset,

                "b1_conflict_to_release_observed_or_lower_bound":
                    b1_release_observed,

                "b1_certified_escape":
                    escape_onset >= 0,

                "b1_escape_onset_epoch":
                    escape_onset,

                "b1_conflict_to_escape_observed_or_lower_bound":
                    b1_escape_observed,

                "b1_persistent_deep_lock":
                    deep_lock_onset >= 0,

                "b1_deep_lock_onset_epoch":
                    deep_lock_onset,

                "b1_mobility_recovery_onset_epoch":
                    mobility_recovery_onset,

                "b1_recovery_precedes_escape":
                    recovery_before_escape,

                "b2_conflict_mu":
                    b2["conflict_mu"],

                "b2_conflict_to_release_delay":
                    b2["release_delay"],

                "b2_conflict_to_escape_delay":
                    b2["escape_delay"],

                "paired_release_delay_b1_at_least_2x_b2":
                    release_2x,

                "paired_escape_delay_b1_greater_than_b2":
                    escape_slower,
            }
        )

        print(
            f"  b1 release={release_onset}, "
            f"escape={escape_onset}, "
            f"deep={deep_lock_onset}, "
            f"recovery={mobility_recovery_onset}, "
            f"release>=2xb2={release_2x}, "
            f"escape>b2={escape_slower}"
        )

    # =========================================================================
    # Persist
    # =========================================================================
    write_csv(out_dir / "tracking_metrics.csv", tracking_rows)
    write_csv(out_dir / "mobility_audits.csv", audit_rows)
    write_csv(out_dir / "stage22_b1_replay_checks.csv", replay_rows)
    write_csv(out_dir / "paired_run_summary.csv", run_rows)

    # =========================================================================
    # Gates
    # =========================================================================
    P1 = bool(
        len(replay_rows) == 5
        and all(bool(r["pass"]) for r in replay_rows)
    )

    deep_count = sum(
        int(bool(r["b1_persistent_deep_lock"]))
        for r in run_rows
    )
    P2 = bool(deep_count >= 4)

    release_contrast_count = sum(
        int(bool(r["paired_release_delay_b1_at_least_2x_b2"]))
        for r in run_rows
    )
    P3 = bool(release_contrast_count >= 4)

    escape_contrast_count = sum(
        int(bool(r["paired_escape_delay_b1_greater_than_b2"]))
        for r in run_rows
    )
    P4 = bool(escape_contrast_count >= 4)

    escaped = [
        r for r in run_rows
        if bool(r["b1_certified_escape"])
    ]

    recovery_before_escape_count = sum(
        int(bool(r["b1_recovery_precedes_escape"]))
        for r in escaped
    )

    P5 = bool(
        len(escaped) >= 4
        and
        recovery_before_escape_count
        >= math.ceil(0.80 * len(escaped))
    )

    strong = bool(P1 and P2 and P3 and P4 and P5)

    if strong:
        route_class = (
            "paired_base_mode_deep_lock_specificity_supported"
        )
        next_route = (
            "stage25R_readonly_tangent_eigenspace_mechanism_localization"
        )
    elif P1 and P2 and not P5:
        route_class = (
            "deep_lock_replication_passes_but_recovery_or_escape_is_horizon_censored"
        )
        next_route = (
            "stage25R_extend_only_censored_b1_runs"
        )
    else:
        route_class = (
            "paired_base_mode_deep_lock_specificity_not_cleanly_supported"
        )
        next_route = (
            "stage25R_seed_state_heterogeneity_audit"
        )

    decision = {
        "paired_seeds": list(SEEDS),

        "P1_exact_b1_stage22_replay":
            P1,

        "b1_persistent_deep_lock_count":
            deep_count,

        "P2_persistent_deep_lock_replication":
            P2,

        "paired_release_delay_2x_count":
            release_contrast_count,

        "P3_conflict_persistence_contrast":
            P3,

        "paired_escape_delay_greater_count":
            escape_contrast_count,

        "P4_escape_delay_contrast":
            P4,

        "b1_certified_escape_count":
            len(escaped),

        "b1_recovery_precedes_escape_count":
            recovery_before_escape_count,

        "P5_mobility_recovery_precedes_escape":
            P5,

        "strong_base_mode_deep_lock_specificity":
            strong,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A PASS separates the long-lived b=1,m=9 mobility-collapse "
            "phenotype from the paired b=2,m=9 transient-conflict phenotype "
            "under identical target frequency and paired initialization. "
            "It does not yet identify the causal network feature that makes "
            "b=1 structurally different."
        ),
    }

    write_json(out_dir / "decision.json", decision)

    plot_paired_delays(
        run_rows,
        out_dir / "paired_conflict_release_delays.png",
    )

    plot_mobility(
        audit_rows,
        out_dir / "b1_m9_mobility_persistence.png",
    )

    # =========================================================================
    # Console
    # =========================================================================
    elapsed = time.perf_counter() - global_start

    lines = []
    lines.append("=" * 180)
    lines.append(
        "VPINN — STAGE 24R PAIRED BASE-MODE DEEP-LOCK SPECIFICITY SUMMARY"
    )
    lines.append("=" * 180)

    lines.append(
        "seed | b1 onset | b1 release | b1 escape | deep | recovery | "
        "b2 release delay | b1 delay/lb | >=2x | b2 escape delay | b1 escape delay/lb | slower"
    )
    lines.append("-" * 180)

    for r in run_rows:
        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['b1_conflict_onset_epoch']):8d} | "
            f"{int(r['b1_conflict_release_onset_epoch']):10d} | "
            f"{int(r['b1_escape_onset_epoch']):9d} | "
            f"{str(r['b1_persistent_deep_lock']):5s} | "
            f"{int(r['b1_mobility_recovery_onset_epoch']):8d} | "
            f"{int(r['b2_conflict_to_release_delay']):16d} | "
            f"{int(r['b1_conflict_to_release_observed_or_lower_bound']):11d} | "
            f"{str(r['paired_release_delay_b1_at_least_2x_b2']):5s} | "
            f"{int(r['b2_conflict_to_escape_delay']):15d} | "
            f"{int(r['b1_conflict_to_escape_observed_or_lower_bound']):18d} | "
            f"{str(r['paired_escape_delay_b1_greater_than_b2'])}"
        )

    lines.append("-" * 180)

    lines.append(
        f"P1 exact b1 replay                    : "
        f"{sum(int(r['pass']) for r in replay_rows)}/5 -> {P1}"
    )
    lines.append(
        f"P2 persistent deep lock               : "
        f"{deep_count}/5 -> {P2}"
    )
    lines.append(
        f"P3 b1 release delay >=2x b2           : "
        f"{release_contrast_count}/5 -> {P3}"
    )
    lines.append(
        f"P4 b1 escape delay > b2               : "
        f"{escape_contrast_count}/5 -> {P4}"
    )
    lines.append(
        f"P5 recovery before escape             : "
        f"{recovery_before_escape_count}/{len(escaped)} "
        f"with {len(escaped)}/5 escapes -> {P5}"
    )
    lines.append(
        f"STRONG BASE-MODE DEEP-LOCK SPECIFICITY: {strong}"
    )
    lines.append(
        f"route class                            : {route_class}"
    )
    lines.append(
        f"next route                             : {next_route}"
    )
    lines.append(
        f"elapsed seconds                        : {elapsed:.2f}"
    )

    lines.append("=" * 180)
    lines.append(
        "Guardrail: target-uphill Adam conflict is not the lock definition. "
        "The deep phenotype requires persistent basis-invariant mobility collapse."
    )
    lines.append("=" * 180)

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
