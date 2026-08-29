#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 21R
Interleaved-Parity Frequency Transfer of Basis-Invariant Residual Mobility
==========================================================================

Scientific status entering Stage 21R
------------------------------------
Discovery set, odd target modes m={3,5,7,9}, seeds {0,...,4}:

    certified conflict counts:
        N3=0/5, N5=0/5, N7=0/5, N9=4/5.

Held-out m=9 seeds {5,...,9}:

    certified conflict     = 5/5
    certified escape       = 4/5 by epoch 4000
    release before escape  = 4/4 escaped seeds

At every held-out m=9 conflict state,

    mu_raw
      = r^T K r / (||r||^2 tr K)
      = ||J^T r||^2 / (||r||^2 ||J||_F^2)

was <= 1e-6 (actual values ~1e-9 to 1e-8).

For all seeds with both conflict and later parity release,

    mu_release / mu_conflict >= 100

(actual held-out ratios were ~3.9e2 to 1.46e4).

All event audits preserved mu_raw and the kernel eigenvalues under arbitrary
orthogonal test-coordinate rotations.

Therefore the preferred mechanistic object is now the BASIS-INVARIANT
residual-mobility collapse, while the m<->m+2 parity ladder is treated as a
coordinate-level marker that may or may not transfer.

Stage-21 question
-----------------
Does the conflict/mobility-collapse mechanism transfer to a COMPLETELY NEW
parity class and localize a frequency transition?

Use:

    target modes = {6,8,10}
    held-out seeds = {10,11,12,13,14}

No mode and no seed in this confirmatory block has been used in Stages
19-20.

The exact-solution family remains matched:

    u*_m = sin(pi x) + a_m sin(m pi x)
    a_m  = 0.15*7/m
    a_m*m = 1.05.

Thus target weak scale is fixed while frequency is varied.

Ordinary Adam only.
No optimizer intervention.

Tracking
--------
Every 25 epochs through at most epoch 4000.

Localization starts when

    target residual-energy share >= 0.80
    AND relL2 > 1e-2.

After localization starts, cheap probes continue through escape so that a
parity-ladder release can be observed.

Certified conflict
------------------
While still mechanism-active:

    target share >=0.80
    AND relL2>1e-2,

the exact inherited Adam candidate must satisfy

    <g_T, Delta_Adam> > 0

for THREE consecutive 25-epoch probes.

Certified parity anti-lock
--------------------------
After localization:

    C_sq(m,m+2) <= -0.95

for THREE consecutive probes.

Certified parity release
------------------------
After anti-lock:

    C_sq(m,m+2) > -0.95

for THREE consecutive probes.

Certified escape
----------------
Inherited unchanged:

    relL2 <= 1e-2
    AND target share <= 0.20

for THREE consecutive 25-epoch observations.

Full finite-width audits
------------------------
Full J/K/Adam/Pareto-curvature geometry is computed only at:

    * certified conflict onset, if present;
    * parity-release onset, but only if conflict has already occurred;
    * certified escape onset.

At every full audit compute:

    mu_raw
    mu_AdamMetric
    lambda_max(K)/tr(K)
    effective rank
    squared residual alignment with top kernel eigenvector
    deterministic orthogonal-rotation invariance.

Primary confirmatory gates
--------------------------

G1 — INTERLEAVED FREQUENCY TRANSITION
    Let N6,N8,N10 be certified-conflict seed counts.

    Require:
        N6 <= N8 <= N10
        N6 <= 1
        N10 >= 4
        N10 - N6 >= 3.

    N8 is deliberately unconstrained beyond monotone ordering: the experiment
    is allowed to reveal whether the transition lies below, at, or above m=8.

G2 — INVARIANT MOBILITY COLLAPSE TRANSFER
    Among ALL newly observed certified-conflict states:

        mu_raw(conflict) <= 1e-6

    in >=80%.

    This threshold is inherited unchanged from Stage 20.

G3 — MOBILITY RECOVERY
    Among runs with BOTH certified conflict and later certified parity release,
    require:
        at least 3 such runs,
        and mu_release/mu_conflict >=100 in >=80%.

G4 — RELEASE PRECEDES ESCAPE
    Among runs with BOTH certified conflict and certified escape:
        at least 3 such runs,
        and release onset < escape onset in >=80%.

G5 — ROTATION INVARIANCE
    Every full audit must preserve mu_raw and raw-kernel eigenvalues under the
    deterministic orthogonal coordinate rotation to <=1e-10.

Secondary parity-class gate
---------------------------
For each even target mode m={6,8,10}, define P_m as the fraction of all
post-localization cheap probes satisfying

    C_sq(m,m+2) <= -0.95.

EVEN-PARITY LADDER TRANSFER requires

    P_m >=0.80 for all three modes.

This is SECONDARY because the pairwise ladder is basis-dependent.

Decision
--------
A) G1+G2+G5 PASS:
       invariant frequency-selective mobility-collapse mechanism supported.

   If G3+G4 also PASS:
       unlock/recovery dynamics transfer too.

   If the secondary parity gate also PASS:
       parity-ladder marker transfers across odd/even parity classes.

   Next:
       Stage 22R = second PDE / test-basis robustness audit.

B) G1 fails but G2 passes:
       mobility collapse transfers to conflict states, but no clean frequency
       transition. Route to architecture/problem-family mobility audit.

C) G2 fails:
       invariant mobility-collapse hypothesis does not cleanly transfer.
       Route to kernel-spectrum alternative audit.

No universal VPINN law or novelty claim is authorized by this stage alone.
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


MODES = (6, 8, 10)
SEEDS = (10, 11, 12, 13, 14)

MAX_EPOCH = 4000
TRACK_INTERVAL = 25

LOCALIZE_SHARE = 0.80
CONVERGENCE_REL_L2 = 1.0e-2
CONVERGENCE_TARGET_SHARE = 0.20
CERTIFY_POINTS = 3

PARITY_THRESHOLD = -0.95
PARITY_TRANSFER_FRACTION = 0.80

MOBILITY_COLLAPSE_THRESHOLD = 1.0e-6
MOBILITY_RECOVERY_RATIO = 100.0
ROTATION_TOL = 1.0e-10


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-21R even-parity frequency transfer of invariant mobility."
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
        "--stage20-dir",
        default="vpinn_gradient_conflict_stage20R_heldout_mobility_unlock",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage21R_even_frequency_mobility_transfer",
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
    stage20_dir: Path,
) -> dict:

    manifest_path = stage20_dir / "manifest.json"
    decision_path = stage20_dir / "decision.json"

    if not manifest_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError("Stage-20 manifest/decision missing.")

    manifest = read_json(manifest_path)
    decision = read_json(decision_path)

    s3_sha = sha256_file(stage3_script)
    s18_sha = sha256_file(stage18_script)
    s19_sha = sha256_file(stage19_script)
    s20_sha = sha256_file(stage20_script)

    if manifest.get("stage3_solver_sha256") != s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 20.")

    if manifest.get("stage18_script_sha256") != s18_sha:
        raise RuntimeError("Stage-18 SHA mismatch against Stage 20.")

    if manifest.get("stage19_script_sha256") != s19_sha:
        raise RuntimeError("Stage-19 SHA mismatch against Stage 20.")

    if manifest.get("stage20r_script_sha256") != s20_sha:
        raise RuntimeError(
            "Stage-20 source SHA mismatch against its executed manifest."
        )

    if not bool(decision.get("strong_invariant_unlock_support", False)):
        raise RuntimeError(
            "Stage 20 did not authorize invariant mobility frequency transfer."
        )

    if decision.get("next_route") != (
        "stage21R_interleaved_parity_frequency_transfer_invariant_mobility"
    ):
        raise RuntimeError("Unexpected Stage-20 next route.")

    if int(decision.get("certified_conflict_count", -1)) != 5:
        raise RuntimeError("Expected held-out m=9 conflict 5/5.")

    if int(decision.get("conflict_mobility_collapse_count", -1)) != 5:
        raise RuntimeError("Expected held-out m=9 mobility collapse 5/5.")

    if not bool(decision.get("rotation_invariance_all_pass", False)):
        raise RuntimeError("Stage-20 rotation invariance was not all PASS.")

    return {
        "stage3_sha256": s3_sha,
        "stage18_sha256": s18_sha,
        "stage19_sha256": s19_sha,
        "stage20_sha256": s20_sha,
        "stage20_decision": decision,
    }


# =============================================================================
# Event state helper
# =============================================================================

def update_streak(
    condition: bool,
    streak: int,
    candidate_epoch,
    candidate_state,
    epoch: int,
    state,
):
    if condition:
        if streak == 0:
            candidate_epoch = epoch
            candidate_state = copy.deepcopy(state)

        streak += 1

    else:
        streak = 0
        candidate_epoch = None
        candidate_state = None

    return streak, candidate_epoch, candidate_state


# =============================================================================
# Plotting
# =============================================================================

def plot_conflict_counts(mode_summary: List[dict], path: Path) -> None:
    modes = [int(r["mode"]) for r in mode_summary]
    counts = [int(r["certified_conflict_count"]) for r in mode_summary]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    ax.bar([str(m) for m in modes], counts)

    ax.set_ylim(0, 5.5)
    ax.set_xlabel("New even target frequency m")
    ax.set_ylabel("Certified-conflict seeds out of 5")
    ax.set_title("Interleaved-parity localization of the conflict transition")

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_mobility_events(audit_rows: List[dict], path: Path) -> None:
    event_order = [
        "CERTIFIED_CONFLICT_ONSET",
        "PARITY_RELEASE_ONSET",
        "CERTIFIED_ESCAPE_ONSET",
    ]

    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    for mode in MODES:
        rr_mode = [
            r for r in audit_rows
            if int(r["mode"]) == mode
        ]

        medians = []

        for event in event_order:
            vals = [
                float(r["mu_raw"])
                for r in rr_mode
                if r["audit_kind"] == event
            ]

            medians.append(
                float(np.median(vals))
                if vals
                else np.nan
            )

        if np.any(np.isfinite(medians)):
            ax.plot(
                range(len(event_order)),
                medians,
                marker="o",
                linewidth=1.6,
                label=f"m={mode}",
            )

    ax.set_yscale("log")
    ax.set_xticks(range(len(event_order)))
    ax.set_xticklabels(["Conflict", "Release", "Escape"])
    ax.set_ylabel("Median basis-invariant residual mobility μ")
    ax.set_title("Does mobility collapse/recovery transfer to the even-frequency family?")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_parity_fractions(mode_summary: List[dict], path: Path) -> None:
    modes = [int(r["mode"]) for r in mode_summary]

    vals = [
        100.0 * float(r["parity_anti_probe_fraction"])
        for r in mode_summary
    ]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    ax.bar([str(m) for m in modes], vals)
    ax.axhline(
        100.0 * PARITY_TRANSFER_FRACTION,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_ylim(0.0, 105.0)
    ax.set_xlabel("New even target frequency m")
    ax.set_ylabel("Post-localization anti-aligned probes (%)")
    ax.set_title("Secondary check: does the m↔m+2 ladder transfer to even parity?")

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
    stage20_dir = resolve(args.stage20_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage18_script=stage18_script,
        stage19_script=stage19_script,
        stage20_script=stage20_script,
        stage20_dir=stage20_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage21R",
    )

    stage18 = load_module(
        stage18_script,
        "vpinn_stage18_stage21R",
    )

    stage19 = load_module(
        stage19_script,
        "vpinn_stage19_stage21R",
    )

    stage20 = load_module(
        stage20_script,
        "vpinn_stage20_stage21R",
    )

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
        "stage21r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "interleaved_even_frequency_invariant_mobility_transfer",

            "modes":
                list(MODES),

            "new_seeds":
                list(SEEDS),

            "matched_amplitude":
                "a_m=0.15*7/m; a_m*m=1.05",

            "max_epoch":
                MAX_EPOCH,

            "track_interval":
                TRACK_INTERVAL,

            "conflict":
                "3 consecutive mechanism-active target-uphill Adam probes",

            "anti_lock":
                "3 consecutive post-localization C_sq(m,m+2)<=-0.95",

            "release":
                "after anti-lock, 3 consecutive C_sq(m,m+2)>-0.95",

            "escape":
                "3 consecutive relL2<=1e-2 and target share<=0.20",

            "G1_frequency_transition":
                "N6<=N8<=N10, N6<=1, N10>=4, N10-N6>=3",

            "G2_mobility_collapse":
                "mu_conflict<=1e-6 in >=80% new conflict states",

            "G3_mobility_recovery":
                ">=3 conflict+release runs; >=80% recovery ratio>=100",

            "G4_release_before_escape":
                ">=3 conflict+escape runs; >=80% release before escape",

            "G5_rotation":
                "all event audits invariant <=1e-10",

            "secondary_parity_transfer":
                "P_m>=0.80 for m=6,8,10",

            "optimizer_intervention":
                False,
        },
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 174)
    print(
        "VPINN — STAGE 21R INTERLEAVED-PARITY FREQUENCY TRANSFER OF INVARIANT MOBILITY"
    )
    print("=" * 174)

    print(f"device                    : {device}")
    print(f"new modes                 : {list(MODES)}")
    print(f"new seeds                 : {list(SEEDS)}")
    print(f"maximum epoch             : {MAX_EPOCH}")
    print("optimizer intervention    : NONE")
    print("=" * 174)

    tracking_rows = []
    run_rows = []
    audit_rows = []

    global_start = time.perf_counter()

    for seed in SEEDS:
        for mode in MODES:

            run_dir = (
                out_dir
                / f"seed_{seed:03d}"
                / f"mode_{mode:02d}"
            )

            run_dir.mkdir(parents=True, exist_ok=True)

            exp = stage18.make_experiment(
                stage3=stage3,
                device=device,
                seed=seed,
                mode=mode,
                out_dir=run_dir,
            )

            localized = False

            conflict_streak = 0
            conflict_candidate_epoch = None
            conflict_candidate_state = None
            conflict_onset = -1
            conflict_confirmation = -1
            conflict_state = None

            anti_streak = 0
            anti_candidate_epoch = None
            anti_candidate_state = None
            anti_onset = -1
            anti_confirmation = -1

            release_streak = 0
            release_candidate_epoch = None
            release_candidate_state = None
            release_onset = -1
            release_confirmation = -1
            release_state = None

            escape_streak = 0
            escape_candidate_epoch = None
            escape_candidate_state = None
            escape_onset = -1
            escape_confirmation = -1
            escape_state = None

            postlocal_probe_count = 0
            parity_anti_count = 0

            print()
            print("-" * 174)
            print(
                f"seed={seed} mode={mode} "
                f"a={exp.amplitude:.12g} "
                f"a*m={exp.amplitude*mode:.12g}"
            )

            for epoch in range(MAX_EPOCH + 1):

                if epoch % TRACK_INTERVAL == 0:

                    rm = exp.residual_metrics()
                    rel = exp.relative_l2_error()

                    if (
                        not localized
                        and
                        rm["target_mode_residual_energy_share"]
                        >= LOCALIZE_SHARE
                        and
                        rel > CONVERGENCE_REL_L2
                    ):
                        localized = True

                    probe = None
                    current_state = None

                    if localized and escape_onset < 0:
                        probe = stage19.cheap_probe(
                            exp=exp,
                            mode=mode,
                        )

                        current_state = stage18.capture_state(exp)

                        postlocal_probe_count += 1

                        if (
                            probe["signed_sqgrad_cos_m_plus_2"]
                            <= PARITY_THRESHOLD
                        ):
                            parity_anti_count += 1

                        mechanism_active = bool(
                            rm["target_mode_residual_energy_share"]
                            >= LOCALIZE_SHARE
                            and
                            rel > CONVERGENCE_REL_L2
                        )

                        # -----------------------------------------------------
                        # Conflict
                        # -----------------------------------------------------
                        if conflict_onset < 0:
                            (
                                conflict_streak,
                                conflict_candidate_epoch,
                                conflict_candidate_state,
                            ) = update_streak(
                                condition=bool(
                                    mechanism_active
                                    and
                                    probe["adam_candidate_target_uphill"]
                                ),
                                streak=conflict_streak,
                                candidate_epoch=conflict_candidate_epoch,
                                candidate_state=conflict_candidate_state,
                                epoch=epoch,
                                state=current_state,
                            )

                            if conflict_streak >= CERTIFY_POINTS:
                                conflict_onset = int(
                                    conflict_candidate_epoch
                                )
                                conflict_confirmation = epoch
                                conflict_state = copy.deepcopy(
                                    conflict_candidate_state
                                )

                        # -----------------------------------------------------
                        # Anti-lock
                        # -----------------------------------------------------
                        if anti_onset < 0:
                            (
                                anti_streak,
                                anti_candidate_epoch,
                                anti_candidate_state,
                            ) = update_streak(
                                condition=bool(
                                    probe[
                                        "signed_sqgrad_cos_m_plus_2"
                                    ]
                                    <= PARITY_THRESHOLD
                                ),
                                streak=anti_streak,
                                candidate_epoch=anti_candidate_epoch,
                                candidate_state=anti_candidate_state,
                                epoch=epoch,
                                state=current_state,
                            )

                            if anti_streak >= CERTIFY_POINTS:
                                anti_onset = int(
                                    anti_candidate_epoch
                                )
                                anti_confirmation = epoch

                        # -----------------------------------------------------
                        # Release after anti-lock
                        # -----------------------------------------------------
                        if anti_onset >= 0 and release_onset < 0:
                            (
                                release_streak,
                                release_candidate_epoch,
                                release_candidate_state,
                            ) = update_streak(
                                condition=bool(
                                    probe[
                                        "signed_sqgrad_cos_m_plus_2"
                                    ]
                                    > PARITY_THRESHOLD
                                ),
                                streak=release_streak,
                                candidate_epoch=release_candidate_epoch,
                                candidate_state=release_candidate_state,
                                epoch=epoch,
                                state=current_state,
                            )

                            if release_streak >= CERTIFY_POINTS:
                                release_onset = int(
                                    release_candidate_epoch
                                )
                                release_confirmation = epoch
                                release_state = copy.deepcopy(
                                    release_candidate_state
                                )

                    # ---------------------------------------------------------
                    # Escape
                    # ---------------------------------------------------------
                    qualifies_escape = bool(
                        rel <= CONVERGENCE_REL_L2
                        and
                        rm["target_mode_residual_energy_share"]
                        <= CONVERGENCE_TARGET_SHARE
                    )

                    state_for_escape = (
                        stage18.capture_state(exp)
                        if qualifies_escape
                        else None
                    )

                    (
                        escape_streak,
                        escape_candidate_epoch,
                        escape_candidate_state,
                    ) = update_streak(
                        condition=qualifies_escape,
                        streak=escape_streak,
                        candidate_epoch=escape_candidate_epoch,
                        candidate_state=escape_candidate_state,
                        epoch=epoch,
                        state=state_for_escape,
                    )

                    if (
                        escape_onset < 0
                        and escape_streak >= CERTIFY_POINTS
                    ):
                        escape_onset = int(
                            escape_candidate_epoch
                        )
                        escape_confirmation = epoch
                        escape_state = copy.deepcopy(
                            escape_candidate_state
                        )

                    tracking_rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "epoch": epoch,
                            "relative_l2_error": rel,
                            **rm,

                            "localized": localized,

                            "adam_target_uphill_cosine": (
                                probe["adam_target_uphill_cosine"]
                                if probe is not None
                                else None
                            ),

                            "adam_candidate_target_uphill": (
                                probe["adam_candidate_target_uphill"]
                                if probe is not None
                                else None
                            ),

                            "signed_sqgrad_cos_m_plus_2": (
                                probe["signed_sqgrad_cos_m_plus_2"]
                                if probe is not None
                                else None
                            ),

                            "conflict_streak": conflict_streak,
                            "anti_lock_streak": anti_streak,
                            "release_streak": release_streak,
                        }
                    )

                    if escape_onset >= 0:
                        break

                if epoch < MAX_EPOCH:
                    exp.train_step()

            # -----------------------------------------------------------------
            # Full event audits
            # -----------------------------------------------------------------
            event_specs = []

            if conflict_state is not None:
                event_specs.append(
                    (
                        "CERTIFIED_CONFLICT_ONSET",
                        conflict_onset,
                        conflict_state,
                    )
                )

            if (
                conflict_state is not None
                and release_state is not None
            ):
                event_specs.append(
                    (
                        "PARITY_RELEASE_ONSET",
                        release_onset,
                        release_state,
                    )
                )

            if escape_state is not None:
                event_specs.append(
                    (
                        "CERTIFIED_ESCAPE_ONSET",
                        escape_onset,
                        escape_state,
                    )
                )

            local_audits = []

            for idx, (kind, event_epoch, state) in enumerate(event_specs):

                event_dir = run_dir / kind.lower()
                event_dir.mkdir(parents=True, exist_ok=True)

                audit = stage19.audit_saved_state(
                    stage18=stage18,
                    stage3=stage3,
                    device=device,
                    seed=seed,
                    mode=mode,
                    epoch=event_epoch,
                    state=state,
                    out_dir=event_dir,
                    audit_kind=kind,
                )

                npz = (
                    event_dir
                    / f"{kind.lower()}_kernels.npz"
                )

                inv = stage20.load_invariants_from_npz(
                    path=npz,
                    rotation_seed=210000
                    + 10000*mode
                    + 100*seed
                    + idx,
                )

                row = {
                    **audit,
                    **inv,
                }

                audit_rows.append(row)
                local_audits.append(row)

            conflict_audit = next(
                (
                    r for r in local_audits
                    if r["audit_kind"]
                    == "CERTIFIED_CONFLICT_ONSET"
                ),
                None,
            )

            release_audit = next(
                (
                    r for r in local_audits
                    if r["audit_kind"]
                    == "PARITY_RELEASE_ONSET"
                ),
                None,
            )

            recovery_ratio = (
                float(release_audit["mu_raw"])
                / max(
                    float(conflict_audit["mu_raw"]),
                    1.0e-300,
                )
                if (
                    conflict_audit is not None
                    and release_audit is not None
                )
                else None
            )

            run_rows.append(
                {
                    "seed": seed,
                    "mode": mode,

                    "localized": localized,

                    "certified_conflict":
                        conflict_onset >= 0,

                    "conflict_onset_epoch":
                        conflict_onset,

                    "conflict_confirmation_epoch":
                        conflict_confirmation,

                    "certified_anti_lock":
                        anti_onset >= 0,

                    "anti_lock_onset_epoch":
                        anti_onset,

                    "certified_parity_release":
                        release_onset >= 0,

                    "parity_release_onset_epoch":
                        release_onset,

                    "certified_escape":
                        escape_onset >= 0,

                    "escape_onset_epoch":
                        escape_onset,

                    "escape_confirmation_epoch":
                        escape_confirmation,

                    "release_precedes_escape":
                        bool(
                            conflict_onset >= 0
                            and
                            release_onset >= 0
                            and
                            escape_onset >= 0
                            and
                            release_onset < escape_onset
                        ),

                    "postlocal_probe_count":
                        postlocal_probe_count,

                    "parity_anti_probe_count":
                        parity_anti_count,

                    "parity_anti_probe_fraction":
                        (
                            parity_anti_count
                            / postlocal_probe_count
                            if postlocal_probe_count
                            else None
                        ),

                    "conflict_mu_raw": (
                        float(conflict_audit["mu_raw"])
                        if conflict_audit is not None
                        else None
                    ),

                    "release_mu_raw": (
                        float(release_audit["mu_raw"])
                        if release_audit is not None
                        else None
                    ),

                    "release_over_conflict_mu_ratio":
                        recovery_ratio,

                    "mobility_collapse_at_conflict": (
                        bool(
                            float(conflict_audit["mu_raw"])
                            <= MOBILITY_COLLAPSE_THRESHOLD
                        )
                        if conflict_audit is not None
                        else None
                    ),

                    "mobility_recovery_100x": (
                        bool(
                            recovery_ratio
                            >= MOBILITY_RECOVERY_RATIO
                        )
                        if recovery_ratio is not None
                        else None
                    ),
                }
            )

            print(
                f"  conflict={conflict_onset} "
                f"release={release_onset} "
                f"escape={escape_onset} "
                f"muC={run_rows[-1]['conflict_mu_raw']} "
                f"muR={run_rows[-1]['release_mu_raw']} "
                f"ratio={recovery_ratio}"
            )

    # =========================================================================
    # Aggregate
    # =========================================================================
    write_csv(
        out_dir / "tracking_metrics.csv",
        tracking_rows,
    )

    write_csv(
        out_dir / "run_summary.csv",
        run_rows,
    )

    write_csv(
        out_dir / "event_kernel_audits.csv",
        audit_rows,
    )

    mode_summary = []

    for mode in MODES:

        rr = [
            r for r in run_rows
            if int(r["mode"]) == mode
        ]

        conflicts = [
            r for r in rr
            if bool(r["certified_conflict"])
        ]

        conflict_release = [
            r for r in rr
            if (
                bool(r["certified_conflict"])
                and bool(r["certified_parity_release"])
            )
        ]

        conflict_escape = [
            r for r in rr
            if (
                bool(r["certified_conflict"])
                and bool(r["certified_escape"])
            )
        ]

        mode_summary.append(
            {
                "mode": mode,

                "certified_conflict_count":
                    len(conflicts),

                "certified_escape_count":
                    sum(
                        int(bool(r["certified_escape"]))
                        for r in rr
                    ),

                "median_conflict_onset_epoch": (
                    float(
                        np.median([
                            int(r["conflict_onset_epoch"])
                            for r in conflicts
                        ])
                    )
                    if conflicts
                    else None
                ),

                "median_conflict_mu_raw": (
                    float(
                        np.median([
                            float(r["conflict_mu_raw"])
                            for r in conflicts
                        ])
                    )
                    if conflicts
                    else None
                ),

                "mobility_collapse_count":
                    sum(
                        int(bool(r["mobility_collapse_at_conflict"]))
                        for r in conflicts
                    ),

                "conflict_release_count":
                    len(conflict_release),

                "mobility_recovery_100x_count":
                    sum(
                        int(bool(r["mobility_recovery_100x"]))
                        for r in conflict_release
                    ),

                "conflict_escape_count":
                    len(conflict_escape),

                "release_precedes_escape_count":
                    sum(
                        int(bool(r["release_precedes_escape"]))
                        for r in conflict_escape
                    ),

                "parity_anti_probe_fraction": (
                    sum(
                        int(r["parity_anti_probe_count"])
                        for r in rr
                    )
                    /
                    max(
                        sum(
                            int(r["postlocal_probe_count"])
                            for r in rr
                        ),
                        1,
                    )
                ),
            }
        )

    write_csv(
        out_dir / "mode_summary.csv",
        mode_summary,
    )

    counts = {
        int(r["mode"]):
            int(r["certified_conflict_count"])
        for r in mode_summary
    }

    # -------------------------------------------------------------------------
    # Gate G1
    # -------------------------------------------------------------------------
    G1 = bool(
        counts[6] <= counts[8] <= counts[10]
        and counts[6] <= 1
        and counts[10] >= 4
        and counts[10] - counts[6] >= 3
    )

    all_conflict_runs = [
        r for r in run_rows
        if bool(r["certified_conflict"])
    ]

    collapse_count = sum(
        int(bool(r["mobility_collapse_at_conflict"]))
        for r in all_conflict_runs
    )

    G2 = bool(
        all_conflict_runs
        and
        collapse_count
        >= math.ceil(
            0.80 * len(all_conflict_runs)
        )
    )

    conflict_release_runs = [
        r for r in run_rows
        if (
            bool(r["certified_conflict"])
            and bool(r["certified_parity_release"])
            and r["release_over_conflict_mu_ratio"]
            is not None
        )
    ]

    recovery_count = sum(
        int(bool(r["mobility_recovery_100x"]))
        for r in conflict_release_runs
    )

    G3 = bool(
        len(conflict_release_runs) >= 3
        and
        recovery_count
        >= math.ceil(
            0.80 * len(conflict_release_runs)
        )
    )

    conflict_escape_runs = [
        r for r in run_rows
        if (
            bool(r["certified_conflict"])
            and bool(r["certified_escape"])
        )
    ]

    release_before_escape_count = sum(
        int(bool(r["release_precedes_escape"]))
        for r in conflict_escape_runs
    )

    G4 = bool(
        len(conflict_escape_runs) >= 3
        and
        release_before_escape_count
        >= math.ceil(
            0.80 * len(conflict_escape_runs)
        )
    )

    G5 = bool(
        audit_rows
        and all(
            bool(r["rotation_invariance_pass"])
            for r in audit_rows
        )
    )

    parity_by_mode = {
        int(r["mode"]):
            float(r["parity_anti_probe_fraction"])
        for r in mode_summary
    }

    parity_transfer = all(
        parity_by_mode[m]
        >= PARITY_TRANSFER_FRACTION
        for m in MODES
    )

    invariant_frequency_support = bool(
        G1 and G2 and G5
    )

    unlock_transfer = bool(
        G3 and G4
    )

    if invariant_frequency_support:

        if unlock_transfer:
            route_class = (
                "interleaved_frequency_invariant_mobility_unlock_supported"
            )
        else:
            route_class = (
                "interleaved_frequency_mobility_collapse_supported_recovery_censored_or_mixed"
            )

        next_route = (
            "stage22R_second_problem_or_test_basis_robustness_audit"
        )

    elif G2:
        route_class = (
            "mobility_collapse_transfers_without_clean_frequency_transition"
        )

        next_route = (
            "stage22R_architecture_problem_family_mobility_audit"
        )

    else:
        route_class = (
            "invariant_mobility_collapse_not_cleanly_transferred"
        )

        next_route = (
            "stage22R_kernel_spectrum_alternative_audit"
        )

    decision = {
        "new_modes":
            list(MODES),

        "new_seeds":
            list(SEEDS),

        "conflict_counts_by_mode":
            counts,

        "G1_interleaved_frequency_transition":
            G1,

        "new_conflict_state_count":
            len(all_conflict_runs),

        "new_conflict_mobility_collapse_count":
            collapse_count,

        "G2_invariant_mobility_collapse_transfer":
            G2,

        "conflict_release_run_count":
            len(conflict_release_runs),

        "mobility_recovery_100x_count":
            recovery_count,

        "G3_mobility_recovery":
            G3,

        "conflict_escape_run_count":
            len(conflict_escape_runs),

        "release_precedes_escape_count":
            release_before_escape_count,

        "G4_release_precedes_escape":
            G4,

        "G5_rotation_invariance":
            G5,

        "parity_anti_probe_fraction_by_mode":
            parity_by_mode,

        "secondary_even_parity_ladder_transfer":
            parity_transfer,

        "invariant_frequency_selective_support":
            invariant_frequency_support,

        "unlock_recovery_transfer":
            unlock_transfer,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A successful Stage 21R would establish cross-parity, held-out "
            "frequency transfer of the basis-invariant mobility-collapse "
            "diagnostic in this 1D tanh VPINN family. It would still require "
            "a second PDE/test-basis or architecture control before any broad "
            "VPINN mechanism claim."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    plot_conflict_counts(
        mode_summary,
        out_dir / "new_even_conflict_counts.png",
    )

    if audit_rows:
        plot_mobility_events(
            audit_rows,
            out_dir / "new_even_mobility_events.png",
        )

    plot_parity_fractions(
        mode_summary,
        out_dir / "new_even_parity_transfer.png",
    )

    # =========================================================================
    # Console
    # =========================================================================
    elapsed = time.perf_counter() - global_start

    lines = []

    lines.append("=" * 178)
    lines.append(
        "VPINN — STAGE 21R INTERLEAVED-PARITY INVARIANT MOBILITY TRANSFER SUMMARY"
    )
    lines.append("=" * 178)

    lines.append(
        "mode | conflict | escape | median conflict epoch | median mu(conflict) | "
        "collapse | recovery>=100 | release<escape | parity fraction"
    )
    lines.append("-" * 178)

    for r in mode_summary:
        lines.append(
            f"{int(r['mode']):4d} | "
            f"{int(r['certified_conflict_count']):4d}/5   | "
            f"{int(r['certified_escape_count']):4d}/5 | "
            f"{str(r['median_conflict_onset_epoch']):21s} | "
            f"{str(r['median_conflict_mu_raw']):19s} | "
            f"{int(r['mobility_collapse_count']):4d} | "
            f"{int(r['mobility_recovery_100x_count']):4d}/"
            f"{int(r['conflict_release_count']):<4d} | "
            f"{int(r['release_precedes_escape_count']):4d}/"
            f"{int(r['conflict_escape_count']):<4d} | "
            f"{float(r['parity_anti_probe_fraction']):.6f}"
        )

    lines.append("-" * 178)

    lines.append(
        f"new conflict counts N6,N8,N10       : {counts}"
    )

    lines.append(
        f"G1 frequency transition             : {G1}"
    )

    lines.append(
        f"G2 invariant mobility collapse      : "
        f"{collapse_count}/{len(all_conflict_runs)} -> {G2}"
    )

    lines.append(
        f"G3 mobility recovery                : "
        f"{recovery_count}/{len(conflict_release_runs)} -> {G3}"
    )

    lines.append(
        f"G4 release before escape            : "
        f"{release_before_escape_count}/{len(conflict_escape_runs)} -> {G4}"
    )

    lines.append(
        f"G5 rotation invariance              : "
        f"{sum(int(r['rotation_invariance_pass']) for r in audit_rows)}/"
        f"{len(audit_rows)} -> {G5}"
    )

    lines.append(
        f"even parity fractions               : {parity_by_mode}"
    )

    lines.append(
        f"secondary parity transfer           : {parity_transfer}"
    )

    lines.append(
        f"INVARIANT FREQUENCY SUPPORT         : "
        f"{invariant_frequency_support}"
    )

    lines.append(
        f"UNLOCK/RECOVERY TRANSFER            : "
        f"{unlock_transfer}"
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

    lines.append("=" * 178)

    lines.append(
        "Guardrail: the pairwise parity ladder is secondary. The basis-invariant "
        "mobility-collapse/recovery mechanism is the primary candidate result."
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
