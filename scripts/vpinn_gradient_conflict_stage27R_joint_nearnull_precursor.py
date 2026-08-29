#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 27R
Temporal Joint Near-Null Co-Alignment Precursor Audit
======================================================

Scientific status entering Stage 27R
------------------------------------
Stage 26R rejected both one-factor explanations of the paired mobility gap.

For paired seeds {15,...,19}, target m=9:

    A = b=1 certified-conflict state -> later deep lock
    B = b=2 certified-conflict state -> transient conflict

Stage 26 found:

    Q1 mobility reproduction              PASS 10/10
    Q2 residual-only cross-swap signature FAIL 0/5
    Q3 residual Shapley dominance         FAIL 0/5
    kernel Shapley dominance              only 2/5

and routed to:

    stage27R_joint_residual_kernel_precursor_audit.

The endpoint cross-swap is unusually informative:

    deep residual r_A under transient kernel K_B is highly mobile;
    transient residual r_B under deep kernel K_A is also mobile;
    only the ACTUAL pair (K_A,r_A) is nearly immobile.

Therefore the candidate mechanism is not "bad residual" or "bad kernel"
alone. It is a PAIR-SPECIFIC NEAR-NULL CO-ALIGNMENT between the evolving
weak residual and the finite-width residual tangent kernel.

Stage 27R temporalizes that interaction.

No new experiment
-----------------
This is a deterministic replay / read-only geometry audit of the existing
b=1,m=9 trajectories for paired seeds {15,...,19}.

No new:
    seed
    target frequency
    base mode
    optimizer
    architecture
    PDE
    training rule
    threshold.

Each seed is replayed ONCE from epoch 0 to its exact Stage-22 certified
conflict onset.

Sparse full-Jacobian audits
---------------------------
Cheap residual metrics are read every 25 epochs.

Define t_loc as the FIRST 25-grid point satisfying

    target residual-energy share >= 0.80
    AND relL2 > 1e-2.

At t_loc compute the first full J/K audit.

After t_loc compute full J/K only:
    * at every global 250-epoch point,
    * and at the exact Stage-22 conflict onset.

Thus each seed is replayed once and only a sparse set of expensive Jacobians
is formed. No Hessian is used.

Core invariant
--------------
    mu(K,r) = r^T K r / (||r||^2 tr K),
    K = J J^T.

For every audited epoch t, relative to the localized baseline 0=t_loc,
compute four algebraic mobilities:

    mu_00 = mu(K_0, r_0)
    mu_tt = mu(K_t, r_t)
    mu_t0 = mu(K_t, r_0)
    mu_0t = mu(K_0, r_t)

and the exact pair-interaction statistic

    Psi_t =
      log [ mu_t0 * mu_0t / (mu_tt * mu_00) ].

Psi=0 at baseline.

A large positive Psi means the CURRENT kernel and CURRENT residual are much
less mobile together than either cross-swapped pairing. This is a direct
read-only signature of pair-specific co-alignment.

The endpoint Stage-26 paired-state interaction was enormous, so Stage 27
precommits conservative temporal levels:

    INTERACTION_PRECURSOR = 10x  -> Psi >= log(10)
    STRONG_INTERACTION    = 100x -> Psi >= log(100).

These thresholds are fixed before Stage-27 results.

Current-kernel notch ratio
--------------------------
Also compute

    N_t = mu(K_t,r_t) / mu(K_t,r_0).

N_t << 1 means the current kernel responds far less to its own co-evolved
residual than to the original localized residual direction.

This is descriptive; no new post-hoc threshold is used for routing.

Temporal deep-collapse certification
------------------------------------
Use the inherited deep-collapse threshold

    mu <= 1e-6.

A PERSISTENT PRE-CONFLICT COLLAPSE is certified only when TWO consecutive
full-Jacobian audits, while

    relL2 > 1e-2
    AND target share >= 0.80,

satisfy mu<=1e-6.

The event onset is the first audit of that two-point run.

This is the same persistence logic used in Stages 23-24.

Exact log-Shapley path decomposition
------------------------------------
Between localized baseline (K_0,r_0) and each audited state (K_t,r_t):

    phi_K =
      1/2 [ log mu(K_t,r_0)-log mu(K_0,r_0)
          + log mu(K_t,r_t)-log mu(K_0,r_t) ]

    phi_r =
      1/2 [ log mu(K_0,r_t)-log mu(K_0,r_0)
          + log mu(K_t,r_t)-log mu(K_t,r_0) ].

Exactly:

    phi_K + phi_r = log(mu_tt/mu_00).

For a mobility DROP, define positive drop contributions

    D_K = -phi_K,
    D_r = -phi_r.

This remains algebraic, not causal.

Precommitted gates
------------------

T1 — EXACT CONFLICT-ONSET REPRODUCTION
    At the final audited state, reproduce Stage-22:
        relL2
        target share
        Adam target-uphill cosine
        mu
    to <=1e-10 in 5/5.

T2 — MOBILE LOCALIZED BASELINE
    mu_00 > 1e-6 in >=4/5 seeds.

T3 — PERSISTENT MOBILITY COLLAPSE PRECEDES OPTIMIZER CONFLICT
    persistent pre-conflict collapse certified in >=4/5 seeds
    AND its onset epoch is STRICTLY earlier than the Stage-22 certified
    Adam-conflict onset in those seeds.

T4 — STRONG PAIR-SPECIFIC INTERACTION AT CONFLICT
    exp(Psi_conflict) >= 100 in >=4/5 seeds.

JOINT NEAR-NULL PRECURSOR SUPPORTED:
    T1 & T2 & T3 & T4.

Secondary temporal diagnostic
-----------------------------
For each seed also record:
    first audit with exp(Psi)>=10,
    first persistent mobility-collapse onset,
    optimizer-conflict onset.

If the 10x interaction event precedes persistent collapse, that is a useful
precursor ordering, but it is NOT required for the primary gate because the
250-epoch audit grid may coarsen their ordering.

Decision routes
---------------
A) T1&T2&T3&T4 PASS:
       Stage 28R = independent problem/test-space robustness of the
                   basis-invariant near-null co-alignment mechanism.

B) T1&T2&T4 PASS but T3 fails only on temporal ordering:
       Stage 28R = targeted denser read-only localization around the
                   first collapse interval. No new sweep.

C) T4 fails:
       Stage 28R = alternative joint-geometry diagnostic audit.

Guardrail
---------
K/r cross-swaps and Psi are algebraic counterfactuals. Temporal ordering can
show that collapse precedes optimizer conflict on the replayed trajectories,
but this stage still does not prove an intervention-level causal mechanism.
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
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


SEEDS = (15, 16, 17, 18, 19)
BASE_MODE = 1
TARGET_MODE = 9

TRACK_INTERVAL = 25
FULL_AUDIT_INTERVAL = 250

LOCALIZE_SHARE = 0.80
CONVERGENCE_REL_L2 = 1.0e-2

MOBILITY_COLLAPSE_THRESHOLD = 1.0e-6
COLLAPSE_CERTIFY_POINTS = 2

INTERACTION_PRECURSOR_FACTOR = 10.0
STRONG_INTERACTION_FACTOR = 100.0

REPLAY_TOL = 1.0e-10


# =============================================================================
# CLI / generic helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-27R temporal joint near-null co-alignment precursor audit."
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
        "--stage26-script",
        default="vpinn_gradient_conflict_stage26R_tangent_eigenspace_localization.py",
    )
    p.add_argument(
        "--stage26-dir",
        default="vpinn_gradient_conflict_stage26R_tangent_eigenspace_localization",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage27R_joint_nearnull_precursor",
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
    stage26_script: Path,
    stage26_dir: Path,
) -> dict:

    s22_manifest_path = stage22_dir / "manifest.json"
    s22_run_path = stage22_dir / "run_summary.csv"
    s22_audit_path = stage22_dir / "event_kernel_audits.csv"

    s26_manifest_path = stage26_dir / "manifest.json"
    s26_decision_path = stage26_dir / "decision.json"

    for path in (
        s22_manifest_path,
        s22_run_path,
        s22_audit_path,
        s26_manifest_path,
        s26_decision_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    s22m = read_json(s22_manifest_path)
    s26m = read_json(s26_manifest_path)
    s26d = read_json(s26_decision_path)

    shas = {
        "s3": sha256_file(stage3_script),
        "s18": sha256_file(stage18_script),
        "s19": sha256_file(stage19_script),
        "s20": sha256_file(stage20_script),
        "s22": sha256_file(stage22_script),
        "s26": sha256_file(stage26_script),
    }

    checks22 = (
        ("stage3_solver_sha256", "s3"),
        ("stage18_script_sha256", "s18"),
        ("stage19_script_sha256", "s19"),
        ("stage20_script_sha256", "s20"),
        ("stage22r_script_sha256", "s22"),
    )

    for key, skey in checks22:
        if s22m.get(key) != shas[skey]:
            raise RuntimeError(f"Stage-22 provenance SHA mismatch: {key}")

    if s26m.get("stage22_script_sha256") != shas["s22"]:
        raise RuntimeError("Stage-22 SHA mismatch against Stage 26.")

    if s26m.get("stage26r_script_sha256") != shas["s26"]:
        raise RuntimeError("Stage-26 source SHA mismatch.")

    if s26d.get("next_route") != (
        "stage27R_joint_residual_kernel_precursor_audit"
    ):
        raise RuntimeError("Unexpected Stage-26 next route.")

    if s26d.get("route_class") != (
        "deep_lock_mobility_contrast_requires_joint_residual_kernel_geometry"
    ):
        raise RuntimeError("Unexpected Stage-26 route class.")

    if not bool(s26d.get("Q1_actual_mobility_reproduction", False)):
        raise RuntimeError("Stage-26 mobility reproduction did not pass.")

    runs = read_csv(s22_run_path)
    audits = read_csv(s22_audit_path)

    conflict_onset = {}
    expected = {}

    for row in runs:
        if (
            int(row["base_mode"]) == BASE_MODE
            and int(row["target_mode"]) == TARGET_MODE
            and int(row["seed"]) in SEEDS
        ):
            if str(row["certified_conflict"]).lower() != "true":
                raise RuntimeError(
                    f"Expected Stage-22 conflict seed={row['seed']}."
                )
            conflict_onset[int(row["seed"])] = int(
                float(row["conflict_onset_epoch"])
            )

    for row in audits:
        if (
            int(row["base_mode"]) == BASE_MODE
            and int(row["target_mode"]) == TARGET_MODE
            and int(row["seed"]) in SEEDS
            and row["audit_kind"] == "CERTIFIED_CONFLICT_ONSET"
        ):
            expected[int(row["seed"])] = row

    if set(conflict_onset) != set(SEEDS):
        raise RuntimeError("Incomplete Stage-22 conflict onset map.")

    if set(expected) != set(SEEDS):
        raise RuntimeError("Incomplete Stage-22 expected audit map.")

    return {
        **shas,
        "conflict_onset": conflict_onset,
        "expected": expected,
        "stage26_decision": s26d,
    }


# =============================================================================
# Geometry helpers
# =============================================================================

def symmetrize(K: np.ndarray) -> np.ndarray:
    return 0.5 * (K + K.T)


def mobility(K: np.ndarray, r: np.ndarray) -> float:
    K = symmetrize(K)

    rr = float(np.dot(r, r))
    tr = float(np.trace(K))
    num = float(np.dot(r, K @ r))

    if rr <= 0.0 or tr <= 0.0:
        raise RuntimeError("Invalid mobility denominator.")

    if num < -1.0e-12 * max(1.0, rr * abs(tr)):
        raise RuntimeError(f"Materially negative r^T K r: {num:.6e}")

    return max(num, 0.0) / (rr * tr)


def eigensystem(K: np.ndarray):
    K = symmetrize(K)

    vals, vecs = np.linalg.eigh(K)
    order = np.argsort(vals)[::-1]

    vals = vals[order]
    vecs = vecs[:, order]

    scale = max(float(np.max(np.abs(vals))), 1.0)

    if float(np.min(vals)) < -1.0e-10 * scale:
        raise RuntimeError(
            f"Materially negative eigenvalue: {np.min(vals):.6e}"
        )

    vals = np.clip(vals, 0.0, None)

    return vals, vecs


def actual_metrics(K: np.ndarray, r: np.ndarray) -> dict:
    K = symmetrize(K)
    mu = mobility(K, r)

    vals, vecs = eigensystem(K)
    tr = float(vals.sum())

    rn = r / max(float(np.linalg.norm(r)), 1.0e-300)
    coeff2 = (vecs.T @ rn) ** 2

    return {
        "mu_raw":
            mu,

        "top_eigenvalue_fraction":
            float(vals[0] / tr),

        "effective_rank":
            float(
                tr * tr
                / max(float(np.dot(vals, vals)), 1.0e-300)
            ),

        "residual_top_eigenvector_energy":
            float(coeff2[0]),
    }


def paired_shape_metrics(
    K0: np.ndarray,
    r0: np.ndarray,
    Kt: np.ndarray,
    rt: np.ndarray,
) -> dict:

    K0s = symmetrize(K0)
    Kts = symmetrize(Kt)

    K0n = K0s / float(np.trace(K0s))
    Ktn = Kts / float(np.trace(Kts))

    r0n = r0 / max(float(np.linalg.norm(r0)), 1.0e-300)
    rtn = rt / max(float(np.linalg.norm(rt)), 1.0e-300)

    vals0, vecs0 = eigensystem(K0s)
    valst, vecst = eigensystem(Kts)

    return {
        "normalized_kernel_shape_distance_from_localization":
            float(
                np.linalg.norm(
                    Ktn - K0n,
                    ord="fro",
                )
            ),

        "residual_direction_abs_cosine_to_localization":
            float(abs(np.dot(r0n, rtn))),

        "top1_eigenvector_abs_cosine_to_localization":
            float(abs(np.dot(vecs0[:, 0], vecst[:, 0]))),
    }


def log_shapley_drop(
    mu_00: float,
    mu_0t: float,
    mu_t0: float,
    mu_tt: float,
) -> dict:

    floor = 1.0e-300

    f00 = math.log(max(mu_00, floor))
    f0t = math.log(max(mu_0t, floor))
    ft0 = math.log(max(mu_t0, floor))
    ftt = math.log(max(mu_tt, floor))

    phi_K = 0.5 * (
        (ft0 - f00)
        +
        (ftt - f0t)
    )

    phi_r = 0.5 * (
        (f0t - f00)
        +
        (ftt - ft0)
    )

    total = ftt - f00

    return {
        "log_mobility_change_from_localization":
            total,

        "log_shapley_kernel_change":
            phi_K,

        "log_shapley_residual_change":
            phi_r,

        "positive_drop_contribution_kernel":
            -phi_K,

        "positive_drop_contribution_residual":
            -phi_r,

        "log_shapley_closure_gap":
            abs((phi_K + phi_r) - total),
    }


def full_geometry_audit(
    stage18,
    exp,
    seed: int,
    epoch: int,
    kind: str,
) -> dict:

    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    kernel = stage18.residual_jacobian(exp)

    r = kernel["r"].cpu().numpy()
    K = kernel["K"].cpu().numpy()

    metrics = actual_metrics(K, r)

    return {
        "seed": seed,
        "epoch": epoch,
        "audit_kind": kind,

        "relative_l2_error": rel,
        **rm,
        **metrics,

        "_r": r,
        "_K": K,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_mobility(rows: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))

    for seed in SEEDS:
        rr = [
            r for r in rows
            if int(r["seed"]) == seed
        ]

        rr.sort(key=lambda x: int(x["epoch"]))

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r["mu_raw"]) for r in rr],
            marker="o",
            markersize=3,
            linewidth=1.2,
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
    ax.set_title("When does persistent residual mobility collapse emerge before Adam conflict?")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_interaction(rows: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))

    for seed in SEEDS:
        rr = [
            r for r in rows
            if int(r["seed"]) == seed
        ]

        rr.sort(key=lambda x: int(x["epoch"]))

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r["interaction_factor"]) for r in rr],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"seed {seed}",
        )

    ax.axhline(
        INTERACTION_PRECURSOR_FACTOR,
        linestyle="--",
        linewidth=1.0,
        label="10x interaction precursor",
    )

    ax.axhline(
        STRONG_INTERACTION_FACTOR,
        linestyle=":",
        linewidth=1.0,
        label="100x strong interaction",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("K/r pair-interaction factor")
    ax.set_title("Emergence of pair-specific near-null co-alignment")
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
    stage26_script = resolve(args.stage26_script)
    stage26_dir = resolve(args.stage26_dir)
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
        stage26_script=stage26_script,
        stage26_dir=stage26_dir,
    )

    stage3 = load_module(stage3_script, "vpinn_stage3_stage27R")
    stage18 = load_module(stage18_script, "vpinn_stage18_stage27R")
    stage19 = load_module(stage19_script, "vpinn_stage19_stage27R")
    stage22 = load_module(stage22_script, "vpinn_stage22_stage27R")

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
        "stage26_script_sha256": pf["s26"],
        "stage27r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "temporal_joint_nearnull_coalignment_precursor",

            "seeds":
                list(SEEDS),

            "base_mode":
                BASE_MODE,

            "target_mode":
                TARGET_MODE,

            "localization":
                "first 25-grid target share>=0.80 and relL2>1e-2",

            "full_geometry_grid":
                "localization + global 250-grid + exact Stage-22 conflict onset",

            "mobility_threshold":
                MOBILITY_COLLAPSE_THRESHOLD,

            "persistent_collapse":
                "2 consecutive eligible full audits mu<=1e-6",

            "interaction_precursor_factor":
                INTERACTION_PRECURSOR_FACTOR,

            "strong_interaction_factor":
                STRONG_INTERACTION_FACTOR,

            "T1":
                "5/5 exact conflict-onset reproduction <=1e-10",

            "T2":
                "localized baseline mu>1e-6 in >=4/5",

            "T3":
                "persistent mobility collapse strictly before conflict in >=4/5",

            "T4":
                "interaction factor at conflict >=100 in >=4/5",

            "no_new_training_experiment":
                True,

            "optimizer_intervention":
                False,
        },
    }

    write_json(out_dir / "manifest.json", manifest)

    print("=" * 180)
    print(
        "VPINN — STAGE 27R TEMPORAL JOINT NEAR-NULL CO-ALIGNMENT PRECURSOR AUDIT"
    )
    print("=" * 180)
    print(f"device                    : {device}")
    print(f"seeds                     : {list(SEEDS)}")
    print(f"cell                      : b=1,m=9")
    print(f"full geometry             : localization + every 250 + conflict onset")
    print("new training experiment   : NONE (deterministic replay only)")
    print("=" * 180)

    tracking_rows = []
    geometry_rows = []
    seed_summary_rows = []
    reproduction_rows = []

    for seed in SEEDS:

        conflict_epoch = pf["conflict_onset"][seed]
        expected = pf["expected"][seed]

        run_dir = (
            out_dir
            / f"seed_{seed:03d}"
            / "base_01_target_09"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = stage22.make_config(
            stage3=stage3,
            seed=seed,
            device=device,
            out_dir=run_dir,
        )

        exp = stage22.SymmetrySwapExperiment(
            stage3=stage3,
            cfg=cfg,
            device=device,
            base_mode=BASE_MODE,
            target_mode=TARGET_MODE,
            out_dir=run_dir,
        )

        localized = False
        localization_epoch = -1

        audits = []

        print()
        print("-" * 180)
        print(
            f"seed={seed} conflict onset={conflict_epoch}"
        )

        for epoch in range(conflict_epoch + 1):

            if epoch % TRACK_INTERVAL == 0:

                rm = exp.residual_metrics()
                rel = exp.relative_l2_error()

                probe = stage19.cheap_probe(
                    exp=exp,
                    mode=TARGET_MODE,
                )

                active = bool(
                    rm["target_mode_residual_energy_share"]
                    >= LOCALIZE_SHARE
                    and
                    rel > CONVERGENCE_REL_L2
                )

                if active and not localized:
                    localized = True
                    localization_epoch = epoch

                tracking_rows.append(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "relative_l2_error": rel,
                        **rm,
                        "mechanism_active": active,
                        "adam_target_uphill_cosine":
                            probe["adam_target_uphill_cosine"],
                        "adam_candidate_target_uphill":
                            probe["adam_candidate_target_uphill"],
                    }
                )

                do_full = bool(
                    (
                        localized
                        and epoch == localization_epoch
                    )
                    or
                    (
                        localized
                        and epoch % FULL_AUDIT_INTERVAL == 0
                    )
                    or
                    epoch == conflict_epoch
                )

                # Conflict onset may not lie on the 25-grid in general.
                # Current Stage-22 onsets do, but the separate clause below
                # keeps the contract explicit.
                if do_full:
                    if not audits or int(audits[-1]["epoch"]) != epoch:
                        audits.append(
                            full_geometry_audit(
                                stage18=stage18,
                                exp=exp,
                                seed=seed,
                                epoch=epoch,
                                kind=(
                                    "LOCALIZED_BASELINE"
                                    if epoch == localization_epoch
                                    else
                                    "CERTIFIED_CONFLICT_ONSET"
                                    if epoch == conflict_epoch
                                    else
                                    "SURVEILLANCE_250"
                                ),
                            )
                        )

            # Exact conflict onset audit even if not on tracking grid.
            if epoch == conflict_epoch:
                if not audits or int(audits[-1]["epoch"]) != epoch:
                    audits.append(
                        full_geometry_audit(
                            stage18=stage18,
                            exp=exp,
                            seed=seed,
                            epoch=epoch,
                            kind="CERTIFIED_CONFLICT_ONSET",
                        )
                    )

            if epoch < conflict_epoch:
                exp.train_step()

        if not localized:
            raise RuntimeError(
                f"Seed {seed}: never reached localization before conflict."
            )

        if not audits:
            raise RuntimeError(f"Seed {seed}: no geometry audits.")

        audits.sort(key=lambda x: int(x["epoch"]))

        baseline = audits[0]

        if int(baseline["epoch"]) != localization_epoch:
            raise RuntimeError("Baseline audit is not localization state.")

        final = audits[-1]

        if int(final["epoch"]) != conflict_epoch:
            raise RuntimeError("Final audit is not conflict onset.")

        # -------------------------------------------------------------
        # Exact Stage-22 conflict-onset reproduction.
        # -------------------------------------------------------------
        final_probe = stage19.cheap_probe(
            exp=exp,
            mode=TARGET_MODE,
        )

        repro_diffs = {
            "relative_l2_error":
                abs(
                    float(final["relative_l2_error"])
                    - float(expected["relative_l2_error"])
                ),

            "target_share":
                abs(
                    float(
                        final[
                            "target_mode_residual_energy_share"
                        ]
                    )
                    - float(
                        expected[
                            "target_mode_residual_energy_share"
                        ]
                    )
                ),

            "adam_target_uphill_cosine":
                abs(
                    float(
                        final_probe[
                            "adam_target_uphill_cosine"
                        ]
                    )
                    - float(
                        expected[
                            "adam_target_uphill_cosine"
                        ]
                    )
                ),

            "mu_raw":
                abs(
                    float(final["mu_raw"])
                    - float(expected["mu_raw"])
                ),
        }

        max_gap = max(repro_diffs.values())

        if max_gap > REPLAY_TOL:
            raise RuntimeError(
                f"Seed {seed}: conflict replay failed "
                f"gap={max_gap:.3e}, diffs={repro_diffs}"
            )

        reproduction_rows.append(
            {
                "seed": seed,
                "conflict_epoch": conflict_epoch,
                "max_abs_difference": max_gap,
                "pass": True,
                **{f"gap_{k}": v for k, v in repro_diffs.items()},
            }
        )

        # -------------------------------------------------------------
        # Cross-swap each temporal state against localization baseline.
        # -------------------------------------------------------------
        K0 = baseline["_K"]
        r0 = baseline["_r"]
        mu00 = mobility(K0, r0)

        rows_seed = []

        for audit in audits:

            Kt = audit["_K"]
            rt = audit["_r"]

            mu_tt = mobility(Kt, rt)
            mu_t0 = mobility(Kt, r0)
            mu_0t = mobility(K0, rt)

            floor = 1.0e-300

            log_interaction = (
                math.log(max(mu_t0, floor))
                +
                math.log(max(mu_0t, floor))
                -
                math.log(max(mu_tt, floor))
                -
                math.log(max(mu00, floor))
            )

            interaction_factor = math.exp(
                min(log_interaction, 700.0)
            )

            notch_ratio = (
                mu_tt
                / max(mu_t0, floor)
            )

            shapley = log_shapley_drop(
                mu_00=mu00,
                mu_0t=mu_0t,
                mu_t0=mu_t0,
                mu_tt=mu_tt,
            )

            shape = paired_shape_metrics(
                K0=K0,
                r0=r0,
                Kt=Kt,
                rt=rt,
            )

            row = {
                key: value
                for key, value in audit.items()
                if not key.startswith("_")
            }

            row.update(
                {
                    "localization_epoch":
                        localization_epoch,

                    "conflict_epoch":
                        conflict_epoch,

                    "mu_00_localized_actual":
                        mu00,

                    "mu_tt_current_actual":
                        mu_tt,

                    "mu_t0_current_kernel_localized_residual":
                        mu_t0,

                    "mu_0t_localized_kernel_current_residual":
                        mu_0t,

                    "log_pair_interaction":
                        log_interaction,

                    "interaction_factor":
                        interaction_factor,

                    "current_kernel_self_notch_ratio":
                        notch_ratio,

                    "interaction_precursor_10x":
                        bool(
                            interaction_factor
                            >= INTERACTION_PRECURSOR_FACTOR
                        ),

                    "strong_interaction_100x":
                        bool(
                            interaction_factor
                            >= STRONG_INTERACTION_FACTOR
                        ),

                    **shapley,
                    **shape,
                }
            )

            geometry_rows.append(row)
            rows_seed.append(row)

        # -------------------------------------------------------------
        # Persist raw K/r temporal geometry for future read-only audit.
        # -------------------------------------------------------------
        np.savez_compressed(
            run_dir / "temporal_geometry.npz",
            epochs=np.asarray(
                [int(a["epoch"]) for a in audits],
                dtype=np.int64,
            ),
            residuals=np.stack(
                [a["_r"] for a in audits],
                axis=0,
            ),
            raw_kernels=np.stack(
                [a["_K"] for a in audits],
                axis=0,
            ),
        )

        # -------------------------------------------------------------
        # Temporal event extraction.
        # -------------------------------------------------------------
        first_interaction_10x = -1

        for row in rows_seed:
            if bool(row["interaction_precursor_10x"]):
                first_interaction_10x = int(row["epoch"])
                break

        collapse_streak = 0
        collapse_candidate = None
        persistent_collapse_onset = -1

        for row in rows_seed:

            eligible = bool(
                float(row["relative_l2_error"])
                > CONVERGENCE_REL_L2
                and
                float(
                    row["target_mode_residual_energy_share"]
                )
                >= LOCALIZE_SHARE
            )

            collapsed = bool(
                eligible
                and
                float(row["mu_tt_current_actual"])
                <= MOBILITY_COLLAPSE_THRESHOLD
            )

            if collapsed:
                if collapse_streak == 0:
                    collapse_candidate = int(row["epoch"])

                collapse_streak += 1

                if collapse_streak >= COLLAPSE_CERTIFY_POINTS:
                    persistent_collapse_onset = int(
                        collapse_candidate
                    )
                    break

            else:
                collapse_streak = 0
                collapse_candidate = None

        final_row = rows_seed[-1]

        baseline_mobile = bool(
            mu00 > MOBILITY_COLLAPSE_THRESHOLD
        )

        collapse_precedes_conflict = bool(
            persistent_collapse_onset >= 0
            and
            persistent_collapse_onset < conflict_epoch
        )

        strong_interaction_at_conflict = bool(
            float(final_row["interaction_factor"])
            >= STRONG_INTERACTION_FACTOR
        )

        seed_summary_rows.append(
            {
                "seed": seed,

                "localization_epoch":
                    localization_epoch,

                "localized_mu":
                    mu00,

                "localized_baseline_mobile":
                    baseline_mobile,

                "first_interaction_10x_epoch":
                    first_interaction_10x,

                "persistent_mobility_collapse_onset_epoch":
                    persistent_collapse_onset,

                "certified_adam_conflict_onset_epoch":
                    conflict_epoch,

                "persistent_collapse_precedes_conflict":
                    collapse_precedes_conflict,

                "interaction_10x_precedes_or_equals_collapse":
                    bool(
                        first_interaction_10x >= 0
                        and
                        persistent_collapse_onset >= 0
                        and
                        first_interaction_10x
                        <= persistent_collapse_onset
                    ),

                "conflict_mu":
                    float(final_row["mu_tt_current_actual"]),

                "conflict_interaction_factor":
                    float(final_row["interaction_factor"]),

                "strong_interaction_at_conflict":
                    strong_interaction_at_conflict,

                "conflict_notch_ratio":
                    float(
                        final_row[
                            "current_kernel_self_notch_ratio"
                        ]
                    ),

                "conflict_kernel_drop_contribution":
                    float(
                        final_row[
                            "positive_drop_contribution_kernel"
                        ]
                    ),

                "conflict_residual_drop_contribution":
                    float(
                        final_row[
                            "positive_drop_contribution_residual"
                        ]
                    ),

                "n_full_audits":
                    len(rows_seed),
            }
        )

        print(
            f"  loc={localization_epoch}, "
            f"mu_loc={mu00:.6e}, "
            f"interaction10x={first_interaction_10x}, "
            f"collapse={persistent_collapse_onset}, "
            f"conflict={conflict_epoch}, "
            f"I_conf={float(final_row['interaction_factor']):.3e}, "
            f"replay={max_gap:.3e}"
        )

    # =========================================================================
    # Persist
    # =========================================================================
    write_csv(
        out_dir / "tracking_metrics.csv",
        tracking_rows,
    )

    write_csv(
        out_dir / "temporal_joint_geometry.csv",
        geometry_rows,
    )

    write_csv(
        out_dir / "seed_event_summary.csv",
        seed_summary_rows,
    )

    write_csv(
        out_dir / "stage22_conflict_reproduction.csv",
        reproduction_rows,
    )

    # =========================================================================
    # Gates
    # =========================================================================
    T1 = bool(
        len(reproduction_rows) == 5
        and all(bool(r["pass"]) for r in reproduction_rows)
    )

    baseline_mobile_count = sum(
        int(bool(r["localized_baseline_mobile"]))
        for r in seed_summary_rows
    )

    T2 = bool(
        baseline_mobile_count >= 4
    )

    collapse_precedes_count = sum(
        int(bool(r["persistent_collapse_precedes_conflict"]))
        for r in seed_summary_rows
    )

    T3 = bool(
        collapse_precedes_count >= 4
    )

    strong_interaction_count = sum(
        int(bool(r["strong_interaction_at_conflict"]))
        for r in seed_summary_rows
    )

    T4 = bool(
        strong_interaction_count >= 4
    )

    precursor_order_count = sum(
        int(
            bool(
                r[
                    "interaction_10x_precedes_or_equals_collapse"
                ]
            )
        )
        for r in seed_summary_rows
    )

    joint_precursor = bool(
        T1 and T2 and T3 and T4
    )

    if joint_precursor:

        route_class = (
            "persistent_mobility_collapse_precedes_optimizer_conflict_with_strong_joint_nearnull_coalignment"
        )

        next_route = (
            "stage28R_independent_problem_testspace_robustness"
        )

    elif T1 and T2 and T4 and not T3:

        route_class = (
            "joint_nearnull_interaction_supported_but_temporal_ordering_underresolved"
        )

        next_route = (
            "stage28R_targeted_dense_readonly_collapse_localization"
        )

    else:

        route_class = (
            "strong_joint_nearnull_precursor_not_cleanly_supported"
        )

        next_route = (
            "stage28R_alternative_joint_geometry_diagnostic"
        )

    decision = {
        "seeds":
            list(SEEDS),

        "T1_conflict_onset_reproduction":
            T1,

        "localized_baseline_mobile_count":
            baseline_mobile_count,

        "T2_mobile_localized_baseline":
            T2,

        "persistent_collapse_precedes_conflict_count":
            collapse_precedes_count,

        "T3_persistent_collapse_precedes_conflict":
            T3,

        "strong_interaction_at_conflict_count":
            strong_interaction_count,

        "T4_strong_pair_interaction_at_conflict":
            T4,

        "interaction10x_precedes_or_equals_collapse_count":
            precursor_order_count,

        "joint_nearnull_precursor_supported":
            joint_precursor,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A PASS shows temporal precedence of the persistent low-mobility "
            "state relative to certified Adam target conflict and a strong "
            "algebraic K/r interaction. It does not prove intervention-level "
            "causality."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    plot_mobility(
        geometry_rows,
        out_dir / "mobility_collapse_before_conflict.png",
    )

    plot_interaction(
        geometry_rows,
        out_dir / "joint_interaction_precursor.png",
    )

    # =========================================================================
    # Console
    # =========================================================================
    lines = []

    lines.append("=" * 184)
    lines.append(
        "VPINN — STAGE 27R TEMPORAL JOINT NEAR-NULL PRECURSOR SUMMARY"
    )
    lines.append("=" * 184)

    lines.append(
        "seed | localized | mu(localized) | interaction10x | persistent collapse | "
        "Adam conflict | collapse<conflict | interaction(conflict)"
    )

    lines.append("-" * 184)

    for r in seed_summary_rows:
        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['localization_epoch']):9d} | "
            f"{float(r['localized_mu']):.6e} | "
            f"{int(r['first_interaction_10x_epoch']):14d} | "
            f"{int(r['persistent_mobility_collapse_onset_epoch']):19d} | "
            f"{int(r['certified_adam_conflict_onset_epoch']):13d} | "
            f"{str(r['persistent_collapse_precedes_conflict']):17s} | "
            f"{float(r['conflict_interaction_factor']):.6e}"
        )

    lines.append("-" * 184)

    lines.append(
        f"T1 conflict reproduction              : "
        f"{sum(int(r['pass']) for r in reproduction_rows)}/5 -> {T1}"
    )

    lines.append(
        f"T2 mobile localized baseline          : "
        f"{baseline_mobile_count}/5 -> {T2}"
    )

    lines.append(
        f"T3 persistent collapse before conflict: "
        f"{collapse_precedes_count}/5 -> {T3}"
    )

    lines.append(
        f"T4 strong joint interaction at conflict: "
        f"{strong_interaction_count}/5 -> {T4}"
    )

    lines.append(
        f"10x interaction <= collapse onset     : "
        f"{precursor_order_count}/5"
    )

    lines.append(
        f"JOINT NEAR-NULL PRECURSOR SUPPORTED   : "
        f"{joint_precursor}"
    )

    lines.append(
        f"route class                            : "
        f"{route_class}"
    )

    lines.append(
        f"next route                             : "
        f"{next_route}"
    )

    lines.append("=" * 184)

    lines.append(
        "Guardrail: temporal precedence plus algebraic interaction is stronger "
        "than endpoint correlation, but still not an intervention-level causal proof."
    )

    lines.append("=" * 184)

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
