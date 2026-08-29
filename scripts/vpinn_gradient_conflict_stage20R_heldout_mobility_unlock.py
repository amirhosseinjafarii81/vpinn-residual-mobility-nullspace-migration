#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 20R
Held-Out Residual-Mobility Collapse / Unlock Audit
==================================================

Why this stage exists
---------------------
Stage 19R established on discovery seeds {0,1,2,3,4}:

    N3 = 0/5 certified optimizer conflict
    N5 = 0/5
    N7 = 0/5
    N9 = 4/5

and persistent m <-> m+2 squared-residual-gradient anti-alignment.

A post-Stage-19 read-only inspection exposed a deeper, BASIS-INVARIANT
candidate mechanism at m=9:

    K = J J^T,

    mu_K
      = (r^T K r) / (||r||^2 tr K)
      = ||J^T r||^2 / (||r||^2 ||J||_F^2).

mu_K measures how much of the remaining residual energy lies in directions
that can actually generate parameter gradient under the current finite-width
residual Jacobian. It is invariant under every orthogonal rotation of the
test-coordinate system:

    r' = Q r,
    J' = Q J,
    K' = Q K Q^T.

On the four Stage-19 m=9 conflict states, mu_K was ~1e-9 to 1e-8, while the
non-conflict/releasing seed had a value orders of magnitude larger.

This observation was generated AFTER Stage 19 and therefore must NOT be
validated on the same five seeds.

Stage 20R uses HELD-OUT seeds

    {5,6,7,8,9}

at m=9 only.

No optimizer intervention is introduced.

Questions
---------
Q1. Does persistent actual Adam target opposition replicate in held-out seeds?

Q2. Does certified m<->m+2 anti-alignment eventually RELEASE before successful
    escape?

Q3. At certified conflict, does the basis-invariant residual mobility mu_K
    collapse?

Q4. When the parity ladder releases, does mu_K recover by orders of magnitude?

Q5. Are mu_K and the kernel spectrum numerically invariant under an arbitrary
    orthogonal rotation of test coordinates, while pairwise coordinate
    patterns are not?

Ordinary-Adam trajectory
------------------------
Mode:
    m = 9

Seeds:
    5,6,7,8,9

Matched amplitude:
    a_9 = 0.15*7/9

Track every 25 epochs through at most epoch 4000.

The 4000 horizon is precommitted because the previous five m=9 seeds all had
historical certified escape confirmation by epoch 3650.

Localization / probe window
---------------------------
The cheap geometry probe starts once

    target residual-energy share >= 0.80
    AND relative L2 > 1e-2.

After localization has started, the cheap probe continues every 25 epochs
until certified escape or epoch 4000, even if target share later falls below
0.80. This is necessary to observe an unlock/release event.

Certified conflict
------------------
While still mechanism-active

    target share >= 0.80
    AND relL2 > 1e-2,

certified conflict requires

    <g_T, Delta_Adam> > 0

for THREE consecutive 25-epoch probes.

The conflict onset is the first point of that run.

Certified parity anti-lock
--------------------------
After localization, anti-lock is certified when

    C_sq(m,m+2) <= -0.95

for THREE consecutive probes.

Certified parity release
------------------------
Only after anti-lock has been certified, release is certified when

    C_sq(m,m+2) > -0.95

for THREE consecutive probes.

The release onset is the first point of that 3-point run.

Certified escape
----------------
Inherited exactly:

    relative L2 <= 1e-2
    AND target residual share <= 0.20

for THREE consecutive 25-epoch observations.

Full finite-width audits
------------------------
Save exact states and compute full J/K/Adam geometry only at:

    * certified conflict onset, if present;
    * certified parity-release onset, if present;
    * certified escape onset, if present.

No Hessian is computed at arbitrary epochs. Stage-18's read-only audit
computes Pareto curvature only when inherited Adam is target-uphill.

Basis-invariant diagnostics
---------------------------
For each full-audit state compute:

    mu_raw =
      r^T K r / (||r||^2 tr K)

    mu_AdamMetric =
      r^T K_D r / (||r||^2 tr K_D)

    top_fraction =
      lambda_max(K) / tr K

    effective_rank =
      tr(K)^2 / tr(K^2)

    residual_top_alignment_sq =
      |<r/||r||, e_max(K)>|^2

Also apply a deterministic random orthogonal Q and verify:

    eigenvalues(QKQ^T) = eigenvalues(K)
    mu(Qr,QKQ^T)       = mu(r,K)

to numerical precision.

Precommitted validation gates
-----------------------------

G1 — HELD-OUT CONFLICT REPLICATION
    certified conflict in >=4/5 seeds.

G2 — RELEASE PRECEDES ESCAPE
    at least 4 held-out seeds certify escape by epoch 4000,
    AND among escaped seeds >=80% have certified parity release strictly
    before certified escape onset.

G3 — BASIS-INVARIANT MOBILITY COLLAPSE
    among conflict seeds, >=80% satisfy

        mu_raw(conflict) <= 1e-6.

    The 1e-6 threshold is deliberately two orders of magnitude looser than
    the ~1e-8 discovery values.

G4 — MOBILITY RECOVERY
    among seeds having BOTH conflict and later parity release, >=80% satisfy

        mu_raw(release) / mu_raw(conflict) >= 100.

    The factor 100 is deliberately far below the ~1.8e4 discovery contrast.

G5 — ROTATION INVARIANCE
    all full audits must preserve raw-kernel eigenvalues and mu_raw under
    deterministic orthogonal test-coordinate rotation to <=1e-10.

STRONG INVARIANT UNLOCK SUPPORT:
    G1 and G2 and G3 and G4 and G5 all PASS.

Decision
--------
If STRONG:
    Stage 21R = interleaved-parity frequency transfer of the INVARIANT
                mobility-collapse mechanism using new modes {6,8,10}.

If conflict replicates but mobility recovery fails:
    Stage 21R = invariant kernel-spectrum alternative audit.

If conflict itself does not replicate:
    Stage 21R = seed-heterogeneity / initialization audit.

No universal or novelty claim is authorized by Stage 20R alone.
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
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


MODE = 9
SEEDS = (5, 6, 7, 8, 9)

MAX_EPOCH = 4000
TRACK_INTERVAL = 25

LOCALIZE_SHARE = 0.80
CONVERGENCE_REL_L2 = 1.0e-2
CONVERGENCE_TARGET_SHARE = 0.20
CERTIFY_POINTS = 3

PARITY_THRESHOLD = -0.95

MOBILITY_COLLAPSE_THRESHOLD = 1.0e-6
MOBILITY_RECOVERY_RATIO = 100.0
ROTATION_TOL = 1.0e-10


# =============================================================================
# CLI and helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-20R held-out residual-mobility unlock audit."
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
        "--stage19-dir",
        default="vpinn_gradient_conflict_stage19R_temporal_conflict_parity",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage20R_heldout_mobility_unlock",
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
    stage19_dir: Path,
) -> dict:

    manifest_path = stage19_dir / "manifest.json"
    decision_path = stage19_dir / "decision.json"

    if not manifest_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError("Stage-19 manifest/decision missing.")

    manifest = read_json(manifest_path)
    decision = read_json(decision_path)

    s3_sha = sha256_file(stage3_script)
    s18_sha = sha256_file(stage18_script)
    s19_sha = sha256_file(stage19_script)

    if manifest.get("stage3_solver_sha256") != s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 19.")

    if manifest.get("stage18_script_sha256") != s18_sha:
        raise RuntimeError("Stage-18 SHA mismatch against Stage 19.")

    if manifest.get("stage19r_script_sha256") != s19_sha:
        raise RuntimeError(
            "Stage-19 source SHA mismatch against its executed manifest."
        )

    if not bool(decision.get("strong_frequency_selective_conflict", False)):
        raise RuntimeError(
            "Stage 19 did not establish the discovery-set frequency-selective "
            "conflict prerequisite."
        )

    if not bool(decision.get("persistent_parity_ladder", False)):
        raise RuntimeError(
            "Stage 19 did not establish the discovery-set parity-ladder "
            "prerequisite."
        )

    if decision.get("certified_conflict_counts_by_mode", {}).get("9") != 4:
        raise RuntimeError("Expected Stage-19 m=9 conflict count = 4/5.")

    # Discovery-set invariant mobility reference from the four conflict
    # kernels. This is DESCRIPTIVE only and is never used to tune Stage-20
    # thresholds.
    discovery_mobility = []

    for seed in (0, 1, 2, 3):
        npz = (
            stage19_dir
            / f"seed_{seed:03d}"
            / "mode_09"
            / "certified_conflict_onset_kernels.npz"
        )

        if not npz.is_file():
            raise FileNotFoundError(npz)

        data = np.load(npz)
        r = data["residuals"]
        K = data["raw_kernel"]

        discovery_mobility.append(
            normalized_mobility(r, K)
        )

    return {
        "stage3_sha256": s3_sha,
        "stage18_sha256": s18_sha,
        "stage19_sha256": s19_sha,

        "discovery_conflict_mobility":
            discovery_mobility,

        "discovery_conflict_mobility_median":
            float(np.median(discovery_mobility)),
    }


# =============================================================================
# Invariant kernel diagnostics
# =============================================================================

def normalized_mobility(r: np.ndarray, K: np.ndarray) -> float:
    rr = float(np.dot(r, r))
    tr = float(np.trace(K))

    if rr <= 0.0 or tr <= 0.0:
        return float("nan")

    return float(
        np.dot(r, K @ r)
        / (rr * tr)
    )


def kernel_invariants(
    r: np.ndarray,
    K: np.ndarray,
    KD: np.ndarray,
    rotation_seed: int,
) -> dict:

    Ksym = 0.5 * (K + K.T)
    KDsym = 0.5 * (KD + KD.T)

    eig = np.linalg.eigvalsh(Ksym)
    eig = np.clip(eig, 0.0, None)

    eigD = np.linalg.eigvalsh(KDsym)
    eigD = np.clip(eigD, 0.0, None)

    tr = float(eig.sum())
    trD = float(eigD.sum())

    sq = float(np.dot(eig, eig))
    sqD = float(np.dot(eigD, eigD))

    top_fraction = (
        float(eig[-1] / tr)
        if tr > 0.0
        else float("nan")
    )

    top_fraction_D = (
        float(eigD[-1] / trD)
        if trD > 0.0
        else float("nan")
    )

    effective_rank = (
        tr * tr / sq
        if sq > 0.0
        else float("nan")
    )

    effective_rank_D = (
        trD * trD / sqD
        if sqD > 0.0
        else float("nan")
    )

    mu = normalized_mobility(r, Ksym)
    muD = normalized_mobility(r, KDsym)

    w, V = np.linalg.eigh(Ksym)
    top_vec = V[:, np.argmax(w)]

    rn = r / max(np.linalg.norm(r), 1.0e-300)

    residual_top_alignment_sq = float(
        np.dot(rn, top_vec) ** 2
    )

    # -------------------------------------------------------------
    # Deterministic orthogonal test-coordinate rotation.
    # -------------------------------------------------------------
    rng = np.random.default_rng(rotation_seed)
    A = rng.normal(size=(r.size, r.size))
    Q, _ = np.linalg.qr(A)

    rp = Q @ r
    Kp = Q @ Ksym @ Q.T

    eig_p = np.linalg.eigvalsh(
        0.5 * (Kp + Kp.T)
    )
    eig_p = np.clip(eig_p, 0.0, None)

    mu_p = normalized_mobility(rp, Kp)

    eig_rotation_gap = float(
        np.max(
            np.abs(
                np.sort(eig)
                - np.sort(eig_p)
            )
        )
    )

    mobility_rotation_gap = float(
        abs(mu - mu_p)
    )

    return {
        "mu_raw":
            mu,

        "mu_adam_metric":
            muD,

        "top_eigenvalue_fraction":
            top_fraction,

        "adam_top_eigenvalue_fraction":
            top_fraction_D,

        "effective_rank":
            effective_rank,

        "adam_effective_rank":
            effective_rank_D,

        "residual_top_alignment_sq":
            residual_top_alignment_sq,

        "rotation_eigenvalue_max_abs_gap":
            eig_rotation_gap,

        "rotation_mobility_abs_gap":
            mobility_rotation_gap,

        "rotation_invariance_pass":
            bool(
                eig_rotation_gap <= ROTATION_TOL
                and
                mobility_rotation_gap <= ROTATION_TOL
            ),
    }


def load_invariants_from_npz(
    path: Path,
    rotation_seed: int,
) -> dict:

    data = np.load(path)

    return kernel_invariants(
        r=data["residuals"],
        K=data["raw_kernel"],
        KD=data["adam_current_kernel"],
        rotation_seed=rotation_seed,
    )


# =============================================================================
# Event logic
# =============================================================================

def update_three_point_streak(
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

    return (
        streak,
        candidate_epoch,
        candidate_state,
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_temporal(rows: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    for seed in SEEDS:
        rr = [
            r for r in rows
            if int(r["seed"]) == seed
            and r.get("probe_started", False)
            and r.get("signed_sqgrad_cos_m_plus_2") is not None
        ]

        rr.sort(key=lambda x: int(x["epoch"]))

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r["signed_sqgrad_cos_m_plus_2"]) for r in rr],
            linewidth=1.2,
            label=f"seed {seed}",
        )

    ax.axhline(
        PARITY_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label="anti-lock/release threshold",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("signed cosine C_sq(9,11)")
    ax.set_title("Held-out m=9 parity-ladder lock and release")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_event_mobility(audits: List[dict], path: Path) -> None:
    event_order = [
        "CERTIFIED_CONFLICT_ONSET",
        "PARITY_RELEASE_ONSET",
        "CERTIFIED_ESCAPE_ONSET",
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    for seed in SEEDS:
        rr = [
            r for r in audits
            if int(r["seed"]) == seed
        ]

        points = []

        for event in event_order:
            match = [
                r for r in rr
                if r["audit_kind"] == event
            ]

            if match:
                points.append(
                    (
                        event_order.index(event),
                        float(match[0]["mu_raw"]),
                    )
                )

        if points:
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                linewidth=1.3,
                label=f"seed {seed}",
            )

    ax.set_yscale("log")
    ax.set_xticks(range(len(event_order)))
    ax.set_xticklabels(
        ["Conflict", "Parity release", "Escape"]
    )
    ax.set_ylabel("basis-invariant residual mobility μ")
    ax.set_title("Does residual mobility recover when the m=9 lock releases?")
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

    stage3_script = Path(args.stage3_script)
    if not stage3_script.is_absolute():
        stage3_script = root / stage3_script

    stage18_script = Path(args.stage18_script)
    if not stage18_script.is_absolute():
        stage18_script = root / stage18_script

    stage19_script = Path(args.stage19_script)
    if not stage19_script.is_absolute():
        stage19_script = root / stage19_script

    stage19_dir = Path(args.stage19_dir)
    if not stage19_dir.is_absolute():
        stage19_dir = root / stage19_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage18_script=stage18_script,
        stage19_script=stage19_script,
        stage19_dir=stage19_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage20R",
    )

    stage18 = load_module(
        stage18_script,
        "vpinn_stage18_stage20R",
    )

    stage19 = load_module(
        stage19_script,
        "vpinn_stage19_stage20R",
    )

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),

        "stage3_solver_sha256":
            pf["stage3_sha256"],

        "stage18_script_sha256":
            pf["stage18_sha256"],

        "stage19_script_sha256":
            pf["stage19_sha256"],

        "stage20r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "heldout_residual_mobility_collapse_unlock",

            "mode":
                MODE,

            "heldout_seeds":
                list(SEEDS),

            "max_epoch":
                MAX_EPOCH,

            "track_interval":
                TRACK_INTERVAL,

            "conflict":
                "3 consecutive active probes with Adam target dot > 0",

            "anti_lock":
                "3 consecutive post-localization probes C_sq(9,11)<=-0.95",

            "release":
                "after anti-lock, 3 consecutive probes C_sq(9,11)>-0.95",

            "escape":
                "3 consecutive relL2<=1e-2 and target share<=0.20",

            "mobility":
                "mu = r^T K r / (||r||^2 tr K)",

            "gates": {
                "G1":
                    "held-out conflict >=4/5",

                "G2":
                    ">=4 escapes and release precedes escape in >=80% escaped",

                "G3":
                    "mu_conflict<=1e-6 in >=80% conflict seeds",

                "G4":
                    "mu_release/mu_conflict>=100 in >=80% seeds with both",

                "G5":
                    "all event audits rotation-invariant to <=1e-10",
            },

            "no_optimizer_intervention":
                True,
        },

        "discovery_reference": {
            "stage19_conflict_mu_values":
                pf["discovery_conflict_mobility"],

            "stage19_conflict_mu_median":
                pf["discovery_conflict_mobility_median"],
        },
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 170)
    print(
        "VPINN — STAGE 20R HELD-OUT RESIDUAL-MOBILITY COLLAPSE / UNLOCK AUDIT"
    )
    print("=" * 170)

    print(f"device                    : {device}")
    print(f"mode                      : {MODE}")
    print(f"held-out seeds            : {list(SEEDS)}")
    print(f"maximum epoch             : {MAX_EPOCH}")
    print(
        "discovery conflict mu med : "
        f"{pf['discovery_conflict_mobility_median']:.6e}"
    )
    print("optimizer intervention    : NONE")
    print("=" * 170)

    tracking_rows = []
    run_rows = []
    audit_rows = []

    global_start = time.perf_counter()

    for seed in SEEDS:

        run_dir = (
            out_dir
            / f"seed_{seed:03d}"
            / "mode_09"
        )

        run_dir.mkdir(parents=True, exist_ok=True)

        exp = stage18.make_experiment(
            stage3=stage3,
            device=device,
            seed=seed,
            mode=MODE,
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

        print()
        print("-" * 170)
        print(
            f"seed={seed} mode=9 "
            f"a={exp.amplitude:.12g} "
            f"a*m={exp.amplitude*MODE:.12g}"
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

                if localized and escape_onset < 0:
                    probe = stage19.cheap_probe(
                        exp=exp,
                        mode=MODE,
                    )

                    current_state = stage18.capture_state(exp)

                    mechanism_active = bool(
                        rm["target_mode_residual_energy_share"]
                        >= LOCALIZE_SHARE
                        and
                        rel > CONVERGENCE_REL_L2
                    )

                    # ---------------------------------------------
                    # Conflict certification.
                    # ---------------------------------------------
                    if conflict_onset < 0:
                        (
                            conflict_streak,
                            conflict_candidate_epoch,
                            conflict_candidate_state,
                        ) = update_three_point_streak(
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

                    # ---------------------------------------------
                    # Anti-lock certification.
                    # ---------------------------------------------
                    if anti_onset < 0:
                        (
                            anti_streak,
                            anti_candidate_epoch,
                            anti_candidate_state,
                        ) = update_three_point_streak(
                            condition=bool(
                                probe["signed_sqgrad_cos_m_plus_2"]
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

                    # ---------------------------------------------
                    # Release only after anti-lock exists.
                    # ---------------------------------------------
                    if anti_onset >= 0 and release_onset < 0:
                        (
                            release_streak,
                            release_candidate_epoch,
                            release_candidate_state,
                        ) = update_three_point_streak(
                            condition=bool(
                                probe["signed_sqgrad_cos_m_plus_2"]
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

                # -------------------------------------------------
                # Escape certification independent of probe state.
                # -------------------------------------------------
                qualifies_escape = bool(
                    rel <= CONVERGENCE_REL_L2
                    and
                    rm["target_mode_residual_energy_share"]
                    <= CONVERGENCE_TARGET_SHARE
                )

                current_state_for_escape = (
                    stage18.capture_state(exp)
                    if qualifies_escape
                    else None
                )

                (
                    escape_streak,
                    escape_candidate_epoch,
                    escape_candidate_state,
                ) = update_three_point_streak(
                    condition=qualifies_escape,
                    streak=escape_streak,
                    candidate_epoch=escape_candidate_epoch,
                    candidate_state=escape_candidate_state,
                    epoch=epoch,
                    state=current_state_for_escape,
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
                        "epoch": epoch,

                        "relative_l2_error": rel,
                        **rm,

                        "probe_started": localized,

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

                        "conflict_streak":
                            conflict_streak,

                        "anti_lock_streak":
                            anti_streak,

                        "release_streak":
                            release_streak,
                    }
                )

                if escape_onset >= 0:
                    break

            if epoch < MAX_EPOCH:
                exp.train_step()

        # ---------------------------------------------------------------------
        # Event-state full audits.
        # ---------------------------------------------------------------------
        event_specs = []

        if conflict_state is not None:
            event_specs.append(
                (
                    "CERTIFIED_CONFLICT_ONSET",
                    conflict_onset,
                    conflict_state,
                )
            )

        if release_state is not None:
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

        seed_audits = []

        for idx, (kind, event_epoch, state) in enumerate(event_specs):

            event_dir = run_dir / kind.lower()
            event_dir.mkdir(parents=True, exist_ok=True)

            audit = stage19.audit_saved_state(
                stage18=stage18,
                stage3=stage3,
                device=device,
                seed=seed,
                mode=MODE,
                epoch=event_epoch,
                state=state,
                out_dir=event_dir,
                audit_kind=kind,
            )

            npz = (
                event_dir
                / f"{kind.lower()}_kernels.npz"
            )

            inv = load_invariants_from_npz(
                path=npz,
                rotation_seed=20000 + 100*seed + idx,
            )

            row = {
                **audit,
                **inv,
            }

            audit_rows.append(row)
            seed_audits.append(row)

        # ---------------------------------------------------------------------
        # Run-level recovery ratio.
        # ---------------------------------------------------------------------
        conflict_audit = next(
            (
                r for r in seed_audits
                if r["audit_kind"]
                == "CERTIFIED_CONFLICT_ONSET"
            ),
            None,
        )

        release_audit = next(
            (
                r for r in seed_audits
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

                "localized":
                    localized,

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

                "anti_lock_confirmation_epoch":
                    anti_confirmation,

                "certified_parity_release":
                    release_onset >= 0,

                "parity_release_onset_epoch":
                    release_onset,

                "parity_release_confirmation_epoch":
                    release_confirmation,

                "certified_escape":
                    escape_onset >= 0,

                "escape_onset_epoch":
                    escape_onset,

                "escape_confirmation_epoch":
                    escape_confirmation,

                "release_precedes_escape":
                    bool(
                        release_onset >= 0
                        and escape_onset >= 0
                        and release_onset < escape_onset
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
            f"  conflict={conflict_onset} | "
            f"release={release_onset} | "
            f"escape={escape_onset} | "
            f"mu_conflict="
            f"{run_rows[-1]['conflict_mu_raw']} | "
            f"mu_release="
            f"{run_rows[-1]['release_mu_raw']} | "
            f"ratio={recovery_ratio}"
        )

    # -------------------------------------------------------------------------
    # Persist.
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Gates.
    # -------------------------------------------------------------------------
    conflict_runs = [
        r for r in run_rows
        if bool(r["certified_conflict"])
    ]

    escaped_runs = [
        r for r in run_rows
        if bool(r["certified_escape"])
    ]

    both_runs = [
        r for r in run_rows
        if (
            r["release_over_conflict_mu_ratio"]
            is not None
        )
    ]

    conflict_count = len(conflict_runs)
    escape_count = len(escaped_runs)

    release_before_escape_count = sum(
        int(bool(r["release_precedes_escape"]))
        for r in escaped_runs
    )

    collapse_count = sum(
        int(bool(r["mobility_collapse_at_conflict"]))
        for r in conflict_runs
    )

    recovery_count = sum(
        int(bool(r["mobility_recovery_100x"]))
        for r in both_runs
    )

    rotation_pass = all(
        bool(r["rotation_invariance_pass"])
        for r in audit_rows
    )

    G1 = bool(
        conflict_count >= 4
    )

    G2 = bool(
        escape_count >= 4
        and
        release_before_escape_count
        >= math.ceil(0.80 * escape_count)
    )

    G3 = bool(
        conflict_count > 0
        and
        collapse_count
        >= math.ceil(0.80 * conflict_count)
    )

    G4 = bool(
        len(both_runs) > 0
        and
        recovery_count
        >= math.ceil(0.80 * len(both_runs))
    )

    G5 = bool(
        audit_rows
        and rotation_pass
    )

    strong = bool(
        G1 and G2 and G3 and G4 and G5
    )

    if strong:
        route_class = (
            "heldout_basis_invariant_mobility_unlock_supported"
        )

        next_route = (
            "stage21R_interleaved_parity_frequency_transfer_invariant_mobility"
        )

    elif G1 and G3:
        route_class = (
            "heldout_conflict_and_mobility_collapse_without_clean_recovery"
        )

        next_route = (
            "stage21R_invariant_kernel_spectrum_alternative_audit"
        )

    else:
        route_class = (
            "heldout_m9_mechanism_not_stably_replicated"
        )

        next_route = (
            "stage21R_seed_initialization_heterogeneity_audit"
        )

    decision = {
        "heldout_seeds":
            list(SEEDS),

        "certified_conflict_count":
            conflict_count,

        "certified_escape_count":
            escape_count,

        "release_precedes_escape_count":
            release_before_escape_count,

        "conflict_mobility_collapse_count":
            collapse_count,

        "both_conflict_and_release_count":
            len(both_runs),

        "mobility_recovery_100x_count":
            recovery_count,

        "rotation_invariance_all_pass":
            rotation_pass,

        "G1_heldout_conflict_replication":
            G1,

        "G2_release_precedes_escape":
            G2,

        "G3_mobility_collapse":
            G3,

        "G4_mobility_recovery":
            G4,

        "G5_rotation_invariance":
            G5,

        "strong_invariant_unlock_support":
            strong,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "mu is basis-invariant under orthogonal test-coordinate rotation, "
            "but Stage 20R remains one finite-width architecture and one 1D "
            "problem family. A successful held-out replication authorizes "
            "frequency/parity controls, not a universal VPINN theorem."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    plot_temporal(
        tracking_rows,
        out_dir / "heldout_parity_lock_release.png",
    )

    if audit_rows:
        plot_event_mobility(
            audit_rows,
            out_dir / "basis_invariant_mobility_events.png",
        )

    # -------------------------------------------------------------------------
    # Console.
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - global_start

    lines = []

    lines.append("=" * 174)
    lines.append(
        "VPINN — STAGE 20R HELD-OUT BASIS-INVARIANT MOBILITY UNLOCK SUMMARY"
    )
    lines.append("=" * 174)

    lines.append(
        "seed | conflict | release | escape | mu(conflict) | mu(release) | "
        "release/conflict | release<escape"
    )
    lines.append("-" * 174)

    for r in run_rows:
        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['conflict_onset_epoch']):8d} | "
            f"{int(r['parity_release_onset_epoch']):7d} | "
            f"{int(r['escape_onset_epoch']):6d} | "
            f"{str(r['conflict_mu_raw']):12s} | "
            f"{str(r['release_mu_raw']):11s} | "
            f"{str(r['release_over_conflict_mu_ratio']):16s} | "
            f"{str(r['release_precedes_escape'])}"
        )

    lines.append("-" * 174)

    lines.append(
        f"held-out certified conflict          : "
        f"{conflict_count}/5"
    )

    lines.append(
        f"held-out certified escape            : "
        f"{escape_count}/5"
    )

    lines.append(
        f"release precedes escape              : "
        f"{release_before_escape_count}/"
        f"{escape_count if escape_count else 0}"
    )

    lines.append(
        f"mu(conflict)<=1e-6                   : "
        f"{collapse_count}/"
        f"{conflict_count if conflict_count else 0}"
    )

    lines.append(
        f"mu(release)/mu(conflict)>=100        : "
        f"{recovery_count}/"
        f"{len(both_runs)}"
    )

    lines.append(
        f"rotation invariance                  : "
        f"{sum(int(r['rotation_invariance_pass']) for r in audit_rows)}/"
        f"{len(audit_rows)} PASS"
    )

    lines.append(
        f"G1 held-out conflict                 : {G1}"
    )
    lines.append(
        f"G2 release before escape             : {G2}"
    )
    lines.append(
        f"G3 mobility collapse                 : {G3}"
    )
    lines.append(
        f"G4 mobility recovery                 : {G4}"
    )
    lines.append(
        f"G5 rotation invariance               : {G5}"
    )

    lines.append(
        f"STRONG INVARIANT UNLOCK SUPPORT      : "
        f"{strong}"
    )

    lines.append(
        f"route class                           : "
        f"{route_class}"
    )

    lines.append(
        f"next route                            : "
        f"{next_route}"
    )

    lines.append(
        f"elapsed seconds                       : "
        f"{elapsed:.2f}"
    )

    lines.append("=" * 174)

    lines.append(
        "Guardrail: if this PASSes, the scientifically preferred object is the "
        "basis-invariant residual mobility collapse, not the coordinate-specific "
        "pairwise parity ladder by itself."
    )

    lines.append("=" * 174)

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
