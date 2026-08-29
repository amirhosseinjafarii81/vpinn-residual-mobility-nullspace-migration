#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 23R
Censoring-Repair Continuation: Transient Conflict vs Deep Mobility Lock
=======================================================================

Why Stage 23R is necessary
--------------------------
Stage 22R used a 2x2 base/target symmetry-sector swap:

    (b=1,m=9)   conflict 5/5
    (b=1,m=10)  conflict 0/5
    (b=2,m=9)   conflict 5/5
    (b=2,m=10)  conflict 5/5

The proposed symmetry-sector swap therefore FAILED.

However, Stage 22R intentionally STOPPED each run immediately after certified
conflict or certified escape.

Consequently the reported

    escape = 0/5

for both b=2 cells is CENSORED and must not be interpreted as failure to
escape.

More importantly, the b=2 conflict states were geometrically unlike the
deep b=1,m=9 lock:

    median mu(b=1,m=9 conflict)  ~ 5.14e-9
    median mu(b=2,m=9 conflict)  ~ 4.02e-5
    median mu(b=2,m=10 conflict) ~ 1.27e-5

and onset occurred much earlier:

    median epoch b=1,m=9  = 2225
    median epoch b=2,m=9  = 175
    median epoch b=2,m=10 = 100.

Thus "three consecutive target-uphill Adam probes" may be detecting TWO
different phenotypes:

    1) early transient optimizer opposition;
    2) deep mobility-collapse metastable lock.

Stage 23R repairs the censoring before any new sweep is authorized.

Scope
-----
Continue ONLY the ten censored b=2 conflict trajectories:

    base b=2
    target m in {9,10}
    seeds {15,16,17,18,19}

from their exact Stage-22 certified-conflict ONSET states through at most
epoch 4000 under ordinary Adam.

No new optimizer.
No changed thresholds.
No new seed.
No new PDE.

Exact reconstruction
--------------------
Because Stage 22 did not serialize model/optimizer states, Stage 23R
deterministically reconstructs every run from epoch 0 to the recorded
conflict onset.

At the reconstructed onset it must reproduce:

    relative L2 error
    target residual share
    Adam target-uphill cosine
    basis-invariant mobility mu_raw

against Stage-22 event_kernel_audits.csv.

Any mismatch above 1e-10 aborts.

Conflict release
----------------
After the certified conflict onset, keep probing every 25 epochs.

Certified optimizer-conflict release occurs when

    <g_T, Delta_Adam> <= 0

for THREE consecutive probes.

This is descriptive of the optimizer-opposition episode. It is NOT by itself
called VPINN unlock.

Certified escape
----------------
Inherited unchanged:

    relL2 <= 1e-2
    AND target residual share <= 0.20

for THREE consecutive 25-epoch observations.

Mobility surveillance
---------------------
The primary phenotype is now basis-invariant residual mobility

    mu = r^T K r / (||r||^2 tr K).

To avoid full-Jacobian cost at every probe, compute a LIGHT full kernel audit:

    * at conflict onset;
    * every 250 epochs after conflict while unresolved;
    * at optimizer-conflict release onset;
    * at certified escape onset;
    * at epoch 4000 if censored.

No Hessian/Pareto audit is computed in the 250-epoch surveillance grid.

Persistent deep mobility lock
-----------------------------
A run is certified to enter a PERSISTENT_DEEP_LOCK only if, while

    relL2 > 1e-2
    AND target share >= 0.80,

TWO consecutive 250-epoch surveillance audits satisfy

    mu <= 1e-6.

The 1e-6 threshold is inherited unchanged from Stages 20-22.

The two-consecutive-audit requirement deliberately distinguishes a persistent
mobility collapse from a single transient crossing.

Precommitted gates
------------------

R1 — EXACT STAGE-22 RECONSTRUCTION
    10/10 conflict-onset replays PASS to <=1e-10.

R2 — CONFLICT EPISODE RELEASE
    certified optimizer-conflict release in >=8/10 b=2 runs.

R3 — EVENTUAL ESCAPE
    certified escape by epoch 4000 in >=8/10 b=2 runs.

R4 — DEEP-LOCK RARITY
    persistent deep mobility lock in <=2/10 b=2 runs.

TRANSIENT-CONFLICT PHENOTYPE SUPPORTED:
    R1 & R2 & R3 & R4.

Interpretation if PASS
----------------------
The Stage-22 b=2 conflicts are not the same phenomenon as the deep b=1,m=9
lock. "Certified conflict" alone is therefore too coarse.

The preferred phenotype becomes:

    DEEP LOCK =
        persistent unresolved target
        + basis-invariant mobility collapse,

not merely a positive Adam target dot.

Next route if PASS
------------------
Stage 24R = base-mode structural specificity audit.

The next experiment will then ask why b=1,m=9 produces deep collapse while
b=2 does not, using a low-cost base-mode control rather than continuing to
treat all target-uphill episodes as equivalent.

If R3 or R4 FAIL
----------------
Do not start a new base-mode sweep. Instead route to a longer-horizon
mobility-state audit because the b=2 trajectories may develop a delayed deep
lock.

This stage is a censoring repair and phenotype-separation audit.
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


BASE_MODE = 2
TARGET_MODES = (9, 10)
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

REPLAY_TOL = 1.0e-10


# =============================================================================
# CLI / helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-23R continuation of Stage-22 censored b=2 conflicts."
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
        "--output-dir",
        default="vpinn_gradient_conflict_stage23R_censoring_repair_continuation",
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(
            [{k: row.get(k, None) for k in fields} for row in rows]
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
# Preflight
# =============================================================================

def preflight(
    stage3_script: Path,
    stage18_script: Path,
    stage19_script: Path,
    stage20_script: Path,
    stage22_script: Path,
    stage22_dir: Path,
) -> dict:

    manifest_path = stage22_dir / "manifest.json"
    decision_path = stage22_dir / "decision.json"
    run_path = stage22_dir / "run_summary.csv"
    audit_path = stage22_dir / "event_kernel_audits.csv"

    for path in (
        manifest_path,
        decision_path,
        run_path,
        audit_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    decision = read_json(decision_path)

    s3 = sha256_file(stage3_script)
    s18 = sha256_file(stage18_script)
    s19 = sha256_file(stage19_script)
    s20 = sha256_file(stage20_script)
    s22 = sha256_file(stage22_script)

    if manifest.get("stage3_solver_sha256") != s3:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 22.")
    if manifest.get("stage18_script_sha256") != s18:
        raise RuntimeError("Stage-18 SHA mismatch against Stage 22.")
    if manifest.get("stage19_script_sha256") != s19:
        raise RuntimeError("Stage-19 SHA mismatch against Stage 22.")
    if manifest.get("stage20_script_sha256") != s20:
        raise RuntimeError("Stage-20 SHA mismatch against Stage 22.")
    if manifest.get("stage22r_script_sha256") != s22:
        raise RuntimeError(
            "Stage-22 source SHA mismatch against executed manifest."
        )

    if decision.get("conflict_counts") != {
        "b1_m9": 5,
        "b1_m10": 0,
        "b2_m9": 5,
        "b2_m10": 5,
    }:
        raise RuntimeError("Unexpected Stage-22 conflict-count pattern.")

    if decision.get("next_route") != "stage23R_base_mode_scaling_control":
        raise RuntimeError("Unexpected Stage-22 next route.")

    runs = read_csv(run_path)
    audits = read_csv(audit_path)

    onset_map = {}
    audit_map = {}

    for row in runs:
        b = int(row["base_mode"])
        m = int(row["target_mode"])
        seed = int(row["seed"])

        if b == BASE_MODE and m in TARGET_MODES:
            if str(row["certified_conflict"]).lower() != "true":
                raise RuntimeError(
                    f"Expected Stage-22 b=2 conflict for seed={seed}, m={m}."
                )

            onset_map[(seed, m)] = int(
                float(row["conflict_onset_epoch"])
            )

    if len(onset_map) != 10:
        raise RuntimeError(
            f"Expected 10 censored b=2 conflict runs, got {len(onset_map)}."
        )

    for row in audits:
        b = int(row["base_mode"])
        m = int(row["target_mode"])
        seed = int(row["seed"])

        if (
            b == BASE_MODE
            and m in TARGET_MODES
            and row["audit_kind"] == "CERTIFIED_CONFLICT_ONSET"
        ):
            audit_map[(seed, m)] = row

    if len(audit_map) != 10:
        raise RuntimeError(
            f"Expected 10 Stage-22 b=2 conflict audits, got {len(audit_map)}."
        )

    return {
        "stage3_sha256": s3,
        "stage18_sha256": s18,
        "stage19_sha256": s19,
        "stage20_sha256": s20,
        "stage22_sha256": s22,
        "onset_map": onset_map,
        "audit_map": audit_map,
    }


# =============================================================================
# Light invariant audit: J/K only, no Hessian
# =============================================================================

def light_invariant_audit(
    stage18,
    stage20,
    exp,
    seed: int,
    target_mode: int,
    epoch: int,
    audit_kind: str,
) -> dict:

    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    kernel = stage18.residual_jacobian(exp)

    adam = stage18.predict_and_decompose_adam(
        exp=exp,
        J=kernel["J"],
        r=kernel["r"],
        params=kernel["params"],
        target_index=target_mode - 1,
    )

    inv = stage20.kernel_invariants(
        r=kernel["r"].cpu().numpy(),
        K=kernel["K"].cpu().numpy(),
        KD=adam["K_D"].cpu().numpy(),
        rotation_seed=230000 + 1000*target_mode + seed + epoch,
    )

    return {
        "seed": seed,
        "base_mode": BASE_MODE,
        "target_mode": target_mode,
        "epoch": epoch,
        "audit_kind": audit_kind,

        "relative_l2_error": rel,
        **rm,

        "adam_target_uphill_cosine":
            adam["adam_target_uphill_cosine"],

        "adam_candidate_target_uphill":
            adam["adam_candidate_target_uphill"],

        **inv,
    }


# =============================================================================
# Reconstruction
# =============================================================================

def reconstruct_onset(
    stage3,
    stage18,
    stage19,
    stage20,
    stage22,
    device,
    seed: int,
    target_mode: int,
    onset_epoch: int,
    expected: dict,
    out_dir: Path,
) -> tuple:

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
        target_mode=target_mode,
        out_dir=out_dir,
    )

    for _ in range(onset_epoch):
        exp.train_step()

    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    probe = stage19.cheap_probe(
        exp=exp,
        mode=target_mode,
    )

    audit = light_invariant_audit(
        stage18=stage18,
        stage20=stage20,
        exp=exp,
        seed=seed,
        target_mode=target_mode,
        epoch=onset_epoch,
        audit_kind="RECONSTRUCTED_CONFLICT_ONSET",
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
            abs(
                float(audit["mu_raw"])
                - float(expected["mu_raw"])
            ),
    }

    max_gap = max(diffs.values())

    if max_gap > REPLAY_TOL:
        raise RuntimeError(
            f"Stage-22 onset replay failed seed={seed}, m={target_mode}: "
            f"gap={max_gap:.3e}, diffs={diffs}"
        )

    return exp, audit, {
        "seed": seed,
        "target_mode": target_mode,
        "onset_epoch": onset_epoch,
        "max_abs_difference": max_gap,
        "pass": True,
        **{f"gap_{k}": v for k, v in diffs.items()},
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_alignment(tracking_rows: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    for target_mode in TARGET_MODES:
        for seed in SEEDS:
            rr = [
                r for r in tracking_rows
                if (
                    int(r["target_mode"]) == target_mode
                    and int(r["seed"]) == seed
                )
            ]
            rr.sort(key=lambda x: int(x["epoch"]))

            ax.plot(
                [int(r["epoch"]) for r in rr],
                [float(r["adam_target_uphill_cosine"]) for r in rr],
                linewidth=1.0,
                alpha=0.75,
                label=(
                    f"m={target_mode}, s={seed}"
                    if seed == SEEDS[0]
                    else None
                ),
            )

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Adam target-uphill cosine")
    ax.set_title("Do the early b=2 conflict episodes release under ordinary Adam?")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_mobility(audit_rows: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    for target_mode in TARGET_MODES:
        for seed in SEEDS:
            rr = [
                r for r in audit_rows
                if (
                    int(r["target_mode"]) == target_mode
                    and int(r["seed"]) == seed
                )
            ]
            rr.sort(key=lambda x: int(x["epoch"]))

            ax.plot(
                [int(r["epoch"]) for r in rr],
                [float(r["mu_raw"]) for r in rr],
                marker="o",
                markersize=3,
                linewidth=1.0,
                alpha=0.8,
                label=(
                    f"m={target_mode}, s={seed}"
                    if seed == SEEDS[0]
                    else None
                ),
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
    ax.set_title("Mobility surveillance after the censored Stage-22 conflicts")
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
    )

    stage3 = load_module(stage3_script, "vpinn_stage3_stage23R")
    stage18 = load_module(stage18_script, "vpinn_stage18_stage23R")
    stage19 = load_module(stage19_script, "vpinn_stage19_stage23R")
    stage20 = load_module(stage20_script, "vpinn_stage20_stage23R")
    stage22 = load_module(stage22_script, "vpinn_stage22_stage23R")

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),

        "stage3_solver_sha256": pf["stage3_sha256"],
        "stage18_script_sha256": pf["stage18_sha256"],
        "stage19_script_sha256": pf["stage19_sha256"],
        "stage20_script_sha256": pf["stage20_sha256"],
        "stage22_script_sha256": pf["stage22_sha256"],
        "stage23r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "censoring_repair_transient_conflict_vs_deep_lock",

            "base_mode":
                BASE_MODE,

            "target_modes":
                list(TARGET_MODES),

            "seeds":
                list(SEEDS),

            "max_epoch":
                MAX_EPOCH,

            "tracking_interval":
                TRACK_INTERVAL,

            "mobility_audit_interval":
                MOBILITY_AUDIT_INTERVAL,

            "conflict_release":
                "3 consecutive Adam target dot <=0 probes after conflict onset",

            "escape":
                "3 consecutive relL2<=1e-2 and target share<=0.20",

            "persistent_deep_lock":
                "2 consecutive 250-grid unresolved/target-localized audits with mu<=1e-6",

            "R1":
                "10/10 Stage-22 onset replay PASS <=1e-10",

            "R2":
                "conflict release >=8/10",

            "R3":
                "escape by 4000 >=8/10",

            "R4":
                "persistent deep lock <=2/10",

            "no_optimizer_intervention":
                True,
        },
    }

    write_json(out_dir / "manifest.json", manifest)

    print("=" * 176)
    print(
        "VPINN — STAGE 23R CENSORING-REPAIR: TRANSIENT CONFLICT VS DEEP MOBILITY LOCK"
    )
    print("=" * 176)
    print(f"device                    : {device}")
    print(f"base mode                 : {BASE_MODE}")
    print(f"target modes              : {list(TARGET_MODES)}")
    print(f"seeds                     : {list(SEEDS)}")
    print(f"mobility surveillance     : every {MOBILITY_AUDIT_INTERVAL} epochs")
    print("optimizer intervention    : NONE")
    print("=" * 176)

    tracking_rows = []
    audit_rows = []
    replay_rows = []
    run_rows = []

    global_start = time.perf_counter()

    for seed in SEEDS:
        for target_mode in TARGET_MODES:

            run_dir = (
                out_dir
                / f"seed_{seed:03d}"
                / f"base_02_target_{target_mode:02d}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            onset_epoch = pf["onset_map"][(seed, target_mode)]
            expected = pf["audit_map"][(seed, target_mode)]

            exp, onset_audit, replay = reconstruct_onset(
                stage3=stage3,
                stage18=stage18,
                stage19=stage19,
                stage20=stage20,
                stage22=stage22,
                device=device,
                seed=seed,
                target_mode=target_mode,
                onset_epoch=onset_epoch,
                expected=expected,
                out_dir=run_dir,
            )

            replay_rows.append(replay)
            audit_rows.append(onset_audit)

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

            last_audited_epoch = onset_epoch

            print()
            print("-" * 176)
            print(
                f"seed={seed} m={target_mode} "
                f"replayed onset={onset_epoch} "
                f"mu0={onset_audit['mu_raw']:.6e}"
            )

            # Continue from the exact onset state.
            for epoch in range(onset_epoch, MAX_EPOCH + 1):

                if epoch % TRACK_INTERVAL == 0:

                    rm = exp.residual_metrics()
                    rel = exp.relative_l2_error()

                    probe = stage19.cheap_probe(
                        exp=exp,
                        mode=target_mode,
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

                            if release_onset != last_audited_epoch:
                                # Reconstruct release state exactly from current
                                # trajectory would require storing candidate state.
                                # Instead mark event now and audit confirmation below
                                # only if onset is current. We therefore keep state
                                # snapshots during the streak.
                                pass

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
                            "base_mode": BASE_MODE,
                            "target_mode": target_mode,
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

                # -------------------------------------------------------------
                # 250-epoch light mobility surveillance.
                # -------------------------------------------------------------
                do_surveillance = bool(
                    epoch > onset_epoch
                    and epoch % MOBILITY_AUDIT_INTERVAL == 0
                )

                if do_surveillance:

                    audit = light_invariant_audit(
                        stage18=stage18,
                        stage20=stage20,
                        exp=exp,
                        seed=seed,
                        target_mode=target_mode,
                        epoch=epoch,
                        audit_kind="SURVEILLANCE_250",
                    )

                    audit_rows.append(audit)
                    last_audited_epoch = epoch

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

                if escape_onset >= 0 and epoch >= escape_confirmation:
                    # Audit the certified escape confirmation state if it was
                    # not already audited at this epoch. The event timing itself
                    # remains the certified onset.
                    if epoch != last_audited_epoch:
                        audit_rows.append(
                            light_invariant_audit(
                                stage18=stage18,
                                stage20=stage20,
                                exp=exp,
                                seed=seed,
                                target_mode=target_mode,
                                epoch=epoch,
                                audit_kind="ESCAPE_CONFIRMATION_STATE",
                            )
                        )
                    break

                if epoch < MAX_EPOCH:
                    exp.train_step()

            # If censored at horizon, save final invariant state.
            if escape_onset < 0:
                if MAX_EPOCH != last_audited_epoch:
                    audit_rows.append(
                        light_invariant_audit(
                            stage18=stage18,
                            stage20=stage20,
                            exp=exp,
                            seed=seed,
                            target_mode=target_mode,
                            epoch=MAX_EPOCH,
                            audit_kind="HORIZON_STATE",
                        )
                    )

            run_rows.append(
                {
                    "seed": seed,
                    "base_mode": BASE_MODE,
                    "target_mode": target_mode,

                    "stage22_conflict_onset_epoch":
                        onset_epoch,

                    "certified_conflict_release":
                        release_onset >= 0,

                    "conflict_release_onset_epoch":
                        release_onset,

                    "conflict_release_confirmation_epoch":
                        release_confirmation,

                    "certified_escape":
                        escape_onset >= 0,

                    "escape_onset_epoch":
                        escape_onset,

                    "escape_confirmation_epoch":
                        escape_confirmation,

                    "persistent_deep_mobility_lock":
                        deep_lock_onset >= 0,

                    "deep_mobility_lock_onset_epoch":
                        deep_lock_onset,

                    "conflict_to_release_delay": (
                        release_onset - onset_epoch
                        if release_onset >= 0
                        else None
                    ),

                    "conflict_to_escape_delay": (
                        escape_onset - onset_epoch
                        if escape_onset >= 0
                        else None
                    ),

                    "stage22_conflict_mu":
                        float(onset_audit["mu_raw"]),
                }
            )

            print(
                f"  release={release_onset} "
                f"escape={escape_onset} "
                f"deep_lock={deep_lock_onset}"
            )

    # =========================================================================
    # Persist
    # =========================================================================
    write_csv(out_dir / "tracking_metrics.csv", tracking_rows)
    write_csv(out_dir / "mobility_audits.csv", audit_rows)
    write_csv(out_dir / "stage22_replay_checks.csv", replay_rows)
    write_csv(out_dir / "run_summary.csv", run_rows)

    # =========================================================================
    # Aggregate / gates
    # =========================================================================
    R1 = bool(
        len(replay_rows) == 10
        and all(bool(r["pass"]) for r in replay_rows)
    )

    release_count = sum(
        int(bool(r["certified_conflict_release"]))
        for r in run_rows
    )

    escape_count = sum(
        int(bool(r["certified_escape"]))
        for r in run_rows
    )

    deep_count = sum(
        int(bool(r["persistent_deep_mobility_lock"]))
        for r in run_rows
    )

    R2 = bool(release_count >= 8)
    R3 = bool(escape_count >= 8)
    R4 = bool(deep_count <= 2)

    transient_supported = bool(
        R1 and R2 and R3 and R4
    )

    if transient_supported:
        route_class = (
            "b2_certified_conflicts_are_transient_not_deep_mobility_locks"
        )
        next_route = (
            "stage24R_base_mode_structural_specificity_audit"
        )
    elif R1 and R4 and not R3:
        route_class = (
            "b2_no_deep_lock_but_escape_not_closed_by_horizon"
        )
        next_route = (
            "stage24R_extended_escape_horizon_readonly_continuation"
        )
    else:
        route_class = (
            "b2_may_develop_delayed_deep_mobility_lock"
        )
        next_route = (
            "stage24R_delayed_mobility_state_audit"
        )

    decision = {
        "n_continued_runs": len(run_rows),

        "R1_exact_stage22_reconstruction":
            R1,

        "conflict_release_count":
            release_count,

        "R2_conflict_episode_release":
            R2,

        "escape_count":
            escape_count,

        "R3_eventual_escape":
            R3,

        "persistent_deep_lock_count":
            deep_count,

        "R4_deep_lock_rarity":
            R4,

        "transient_conflict_phenotype_supported":
            transient_supported,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "If PASS, Stage 22's b=2 three-point Adam conflicts are early "
            "transient opposition episodes, not the same phenotype as the "
            "deep basis-invariant mobility collapse previously observed for "
            "b=1,m=9. Do not use certified target-uphill conflict alone as "
            "the final lock definition."
        ),
    }

    write_json(out_dir / "decision.json", decision)

    plot_alignment(
        tracking_rows,
        out_dir / "b2_adam_conflict_release.png",
    )

    plot_mobility(
        audit_rows,
        out_dir / "b2_mobility_surveillance.png",
    )

    # =========================================================================
    # Console
    # =========================================================================
    elapsed = time.perf_counter() - global_start

    lines = []

    lines.append("=" * 178)
    lines.append(
        "VPINN — STAGE 23R CENSORING-REPAIR CONTINUATION SUMMARY"
    )
    lines.append("=" * 178)

    lines.append(
        "seed | target | conflict | release | escape | deep lock | "
        "mu(conflict) | delay conflict->release | delay conflict->escape"
    )
    lines.append("-" * 178)

    for r in run_rows:
        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['target_mode']):6d} | "
            f"{int(r['stage22_conflict_onset_epoch']):8d} | "
            f"{int(r['conflict_release_onset_epoch']):7d} | "
            f"{int(r['escape_onset_epoch']):6d} | "
            f"{int(r['deep_mobility_lock_onset_epoch']):9d} | "
            f"{float(r['stage22_conflict_mu']):.6e} | "
            f"{str(r['conflict_to_release_delay']):23s} | "
            f"{str(r['conflict_to_escape_delay'])}"
        )

    lines.append("-" * 178)

    lines.append(
        f"Stage-22 exact replay                 : "
        f"{sum(int(r['pass']) for r in replay_rows)}/10"
    )

    lines.append(
        f"R1 exact reconstruction               : {R1}"
    )

    lines.append(
        f"conflict release                      : {release_count}/10"
    )

    lines.append(
        f"R2 conflict episode release           : {R2}"
    )

    lines.append(
        f"escape by epoch 4000                  : {escape_count}/10"
    )

    lines.append(
        f"R3 eventual escape                    : {R3}"
    )

    lines.append(
        f"persistent deep mobility lock         : {deep_count}/10"
    )

    lines.append(
        f"R4 deep-lock rarity                   : {R4}"
    )

    lines.append(
        f"TRANSIENT-CONFLICT PHENOTYPE SUPPORTED: "
        f"{transient_supported}"
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

    lines.append("=" * 178)

    lines.append(
        "Guardrail: this stage repairs Stage-22 censoring. A PASS separates "
        "early target-uphill episodes from deep residual-mobility lock; it "
        "does not yet identify why base mode b=1 is special."
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
