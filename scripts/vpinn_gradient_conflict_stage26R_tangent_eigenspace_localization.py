#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 26R
Read-Only Paired Tangent-Eigenspace Mechanism Localization
==========================================================

Scientific status
-----------------
Stage 25 closed the remaining Stage-24 horizon censoring:

    epoch-4000 reconstruction   3/3 exact
    new escapes                 3/3
    cumulative b=1 escapes      5/5
    certified recovery<escape   4/5
    all Stage-24 specificity gates closed

The paired phenotype is therefore established for seeds {15,...,19},
target m=9:

    b=1 : persistent deep residual-mobility lock
    b=2 : transient optimizer conflict, no persistent deep lock

The next question is NOT whether the phenotype exists.

The next question is:

    What part of the tangent geometry creates the ~orders-of-magnitude
    mobility gap?

Recall

    mu(K,r) = r^T K r / (||r||^2 tr K),
    K = J J^T.

Because mu normalizes both residual energy and kernel trace, it depends only
on:

    * the SHAPE / eigenspaces of K,
    * the DIRECTION of r in weak-test space.

Stage 26R separates these two factors without any new training.

Paired states
-------------
For every seed {15,...,19}, use the exact Stage-22 certified-conflict-onset
kernels for:

    A = (b=1, m=9): later deep-lock phenotype
    B = (b=2, m=9): early transient-conflict phenotype.

These have paired initialization and the same target mode / target weak scale.

No trajectory is rerun.

2x2 algebraic cross-swap
------------------------
For each paired seed compute:

    mu_AA = mu(K_A, r_A)   actual deep state
    mu_AB = mu(K_A, r_B)   transient residual in deep-state kernel
    mu_BA = mu(K_B, r_A)   deep residual in transient-state kernel
    mu_BB = mu(K_B, r_B)   actual transient state

Interpretation:

If

    mu_BA <= 1e-6

then the deep residual direction remains effectively immobile even when
evaluated in the b=2 transient kernel.

If

    mu_AB > 1e-6

then the transient residual direction remains mobile even when evaluated in
the b=1 deep kernel.

That pattern localizes the contrast primarily to RESIDUAL DIRECTION /
TANGENT-EIGENSPACE PLACEMENT rather than merely to the kernel spectrum.

The threshold 1e-6 is inherited unchanged from Stages 20-25.

Exact log-Shapley decomposition
-------------------------------
Let

    f(K,r) = log mu(K,r).

For the transition A -> B, define the symmetric two-factor Shapley
contributions:

    phi_K =
      1/2 [ f(K_B,r_A)-f(K_A,r_A)
          + f(K_B,r_B)-f(K_A,r_B) ]

    phi_r =
      1/2 [ f(K_A,r_B)-f(K_A,r_A)
          + f(K_B,r_B)-f(K_B,r_A) ].

Then exactly:

    phi_K + phi_r = log(mu_BB / mu_AA).

This is an ALGEBRAIC decomposition, not a causal intervention.

Residual-direction dominance is declared for a seed only if:

    phi_r > 0
    AND phi_r >= 2 * |phi_K|.

Kernel-shape dominance is declared only if:

    phi_K > 0
    AND phi_K >= 2 * |phi_r|.

Otherwise the seed is MIXED.

Eigen-subspace diagnostics
--------------------------
For each actual state compute:

    s1 = lambda_max(K)/tr(K)

    rho_max =
      r^T K r / (||r||^2 lambda_max(K))
      = mu / s1

so that exactly

    mu = s1 * rho_max.

This separates top-spectrum concentration from residual placement relative to
the top spectral scale.

Also compute, for kernel-trace subspaces capturing 90%, 95%, and 99% of
tr(K):

    d_q = minimal number of leading eigenvectors capturing q of trace(K)
    E_q = residual-energy fraction lying in that subspace.

These are invariant under a common orthogonal rotation of weak-test
coordinates.

Paired eigenspace comparisons
-----------------------------
Because the two states use the same 24 weak tests, compute:

    |e1_A^T e1_B|

and principal-angle cosines between the top-2 and top-3 eigenspaces.

Also compute:

    |rhat_A^T rhat_B|

and normalized kernel-shape distance

    ||K_A/tr(K_A) - K_B/tr(K_B)||_F.

All are invariant under simultaneous orthogonal test-coordinate rotation.

Precommitted route gates
------------------------

Q1 — DATA / MOBILITY REPRODUCTION
    Recomputed mu_AA and mu_BB must match Stage-22 event audit values
    to <=1e-10 for all 10 actual states.

Q2 — CROSS-SWAP RESIDUAL-DIRECTION SIGNATURE
    In >=4/5 paired seeds BOTH:
        mu_BA <= 1e-6
        mu_AB >  1e-6.

Q3 — LOG-SHAPLEY RESIDUAL DOMINANCE
    residual-direction dominance in >=4/5 seeds.

RESIDUAL-EIGENSPACE LOCALIZATION SUPPORTED:
    Q1 & Q2 & Q3.

If Q1 passes but Q2/Q3 instead show kernel-shape dominance in >=4/5:
    route to temporal kernel-deformation precursor audit.

Otherwise:
    route to joint residual+kernel precursor audit.

Next stage if residual localization PASS
----------------------------------------
Stage 27R = Temporal Residual-Rotation Precursor Audit.

It will use only a sparse set of pre-lock checkpoints / deterministic
read-only reconstructions to ask whether r rotates into low-response tangent
eigenspaces BEFORE the deep lock becomes certified.

No optimizer rescue sweep is authorized.

Guardrail
---------
Cross-swapping K and r across two training states is a read-only algebraic
counterfactual. It localizes which mathematical factor carries the mobility
contrast; it does not by itself establish causal dynamics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (15, 16, 17, 18, 19)
TARGET_MODE = 9

DEEP_BASE = 1
TRANSIENT_BASE = 2

MOBILITY_COLLAPSE_THRESHOLD = 1.0e-6
REPRO_TOL = 1.0e-10

TRACE_LEVELS = (0.90, 0.95, 0.99)


# =============================================================================
# CLI / I/O
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-26R read-only tangent-eigenspace mechanism localization."
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
        "--stage25-script",
        default="vpinn_gradient_conflict_stage25R_minimal_horizon_closure.py",
    )

    p.add_argument(
        "--stage25-dir",
        default="vpinn_gradient_conflict_stage25R_minimal_horizon_closure",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage26R_tangent_eigenspace_localization",
    )

    return p.parse_args()


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
            [{key: row.get(key, None) for key in fields} for row in rows]
        )


# =============================================================================
# Preflight / data loading
# =============================================================================

def conflict_npz_path(
    stage22_dir: Path,
    seed: int,
    base_mode: int,
) -> Path:

    return (
        stage22_dir
        / f"seed_{seed:03d}"
        / f"base_{base_mode:02d}_target_{TARGET_MODE:02d}"
        / "certified_conflict_onset"
        / "certified_conflict_onset_kernels.npz"
    )


def preflight(
    stage22_script: Path,
    stage22_dir: Path,
    stage25_script: Path,
    stage25_dir: Path,
) -> dict:

    s22_manifest_path = stage22_dir / "manifest.json"
    s22_audit_path = stage22_dir / "event_kernel_audits.csv"

    s25_manifest_path = stage25_dir / "manifest.json"
    s25_decision_path = stage25_dir / "decision.json"

    for p in (
        s22_manifest_path,
        s22_audit_path,
        s25_manifest_path,
        s25_decision_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    s22_manifest = read_json(s22_manifest_path)
    s25_manifest = read_json(s25_manifest_path)
    s25_decision = read_json(s25_decision_path)

    s22_sha = sha256_file(stage22_script)
    s25_sha = sha256_file(stage25_script)

    if s22_manifest.get("stage22r_script_sha256") != s22_sha:
        raise RuntimeError(
            "Stage-22 source SHA mismatch against executed Stage-22 manifest."
        )

    if s25_manifest.get("stage22_script_sha256") != s22_sha:
        raise RuntimeError(
            "Stage-22 source SHA mismatch against Stage-25 provenance."
        )

    if s25_manifest.get("stage25r_script_sha256") != s25_sha:
        raise RuntimeError(
            "Stage-25 source SHA mismatch against executed Stage-25 manifest."
        )

    if not bool(
        s25_decision.get(
            "stage24_all_five_specificity_gates_closed",
            False,
        )
    ):
        raise RuntimeError(
            "Stage 25 did not close all Stage-24 specificity gates."
        )

    if s25_decision.get("next_route") != (
        "stage26R_readonly_paired_tangent_eigenspace_mechanism_localization"
    ):
        raise RuntimeError("Unexpected Stage-25 next route.")

    audits = read_csv(s22_audit_path)

    expected_mu = {}

    for row in audits:
        seed = int(row["seed"])
        b = int(row["base_mode"])
        m = int(row["target_mode"])

        if (
            seed in SEEDS
            and b in (DEEP_BASE, TRANSIENT_BASE)
            and m == TARGET_MODE
            and row["audit_kind"] == "CERTIFIED_CONFLICT_ONSET"
        ):
            expected_mu[(seed, b)] = float(row["mu_raw"])

    if len(expected_mu) != 10:
        raise RuntimeError(
            f"Expected 10 paired Stage-22 conflict audits; got {len(expected_mu)}."
        )

    for seed in SEEDS:
        for base_mode in (DEEP_BASE, TRANSIENT_BASE):
            path = conflict_npz_path(
                stage22_dir,
                seed,
                base_mode,
            )

            if not path.is_file():
                raise FileNotFoundError(path)

    return {
        "stage22_sha256": s22_sha,
        "stage25_sha256": s25_sha,
        "expected_mu": expected_mu,
        "stage25_decision": s25_decision,
    }


# =============================================================================
# Geometry
# =============================================================================

def symmetrize(K: np.ndarray) -> np.ndarray:
    return 0.5 * (K + K.T)


def mobility(K: np.ndarray, r: np.ndarray) -> float:
    K = symmetrize(K)

    rr = float(np.dot(r, r))
    tr = float(np.trace(K))
    num = float(np.dot(r, K @ r))

    if rr <= 0.0 or tr <= 0.0:
        raise RuntimeError(
            f"Invalid mobility denominator: ||r||2={rr}, trK={tr}"
        )

    # A materially negative Rayleigh quotient would contradict K=JJ^T and
    # indicates corrupted input rather than a meaningful state.
    if num < -1.0e-12 * max(1.0, abs(tr) * rr):
        raise RuntimeError(
            f"Materially negative r^T K r: {num:.6e}"
        )

    num = max(num, 0.0)

    return num / (rr * tr)


def eigensystem(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K = symmetrize(K)

    vals, vecs = np.linalg.eigh(K)

    order = np.argsort(vals)[::-1]

    vals = vals[order]
    vecs = vecs[:, order]

    # Clip only tiny negative numerical eigenvalues.
    scale = max(
        float(np.max(np.abs(vals))),
        1.0,
    )

    if float(np.min(vals)) < -1.0e-10 * scale:
        raise RuntimeError(
            f"Kernel has a materially negative eigenvalue: {np.min(vals):.6e}"
        )

    vals = np.clip(vals, 0.0, None)

    return vals, vecs


def actual_state_metrics(K: np.ndarray, r: np.ndarray) -> dict:
    K = symmetrize(K)

    mu = mobility(K, r)

    vals, vecs = eigensystem(K)

    tr = float(vals.sum())

    if tr <= 0.0:
        raise RuntimeError("Nonpositive kernel trace.")

    s1 = float(vals[0] / tr)

    rho_max = (
        mu / s1
        if s1 > 0.0
        else float("nan")
    )

    rn = r / max(float(np.linalg.norm(r)), 1.0e-300)

    coeff2 = (vecs.T @ rn) ** 2

    cumulative_trace = np.cumsum(vals) / tr
    cumulative_residual = np.cumsum(coeff2)

    out = {
        "mu_raw": mu,
        "top_eigenvalue_fraction": s1,
        "rho_max_normalized_rayleigh": rho_max,
        "effective_rank": (
            tr * tr
            / max(float(np.dot(vals, vals)), 1.0e-300)
        ),
        "residual_top_eigenvector_energy":
            float(coeff2[0]),
    }

    for q in TRACE_LEVELS:
        d = int(
            np.searchsorted(
                cumulative_trace,
                q,
                side="left",
            )
            + 1
        )

        d = min(d, len(vals))

        out[
            f"trace_rank_{int(round(100*q))}"
        ] = d

        out[
            f"residual_energy_in_trace_{int(round(100*q))}_subspace"
        ] = float(cumulative_residual[d - 1])

    return out


def principal_subspace_min_cosine(
    VA: np.ndarray,
    VB: np.ndarray,
    d: int,
) -> float:
    A = VA[:, :d]
    B = VB[:, :d]

    singular = np.linalg.svd(
        A.T @ B,
        compute_uv=False,
    )

    return float(np.min(np.clip(singular, 0.0, 1.0)))


def pair_invariant_comparisons(
    KA: np.ndarray,
    rA: np.ndarray,
    KB: np.ndarray,
    rB: np.ndarray,
) -> dict:

    KAn = symmetrize(KA)
    KBn = symmetrize(KB)

    trA = float(np.trace(KAn))
    trB = float(np.trace(KBn))

    if trA <= 0.0 or trB <= 0.0:
        raise RuntimeError("Nonpositive trace in paired comparison.")

    KAn = KAn / trA
    KBn = KBn / trB

    kernel_shape_distance = float(
        np.linalg.norm(
            KAn - KBn,
            ord="fro",
        )
    )

    rnA = rA / max(float(np.linalg.norm(rA)), 1.0e-300)
    rnB = rB / max(float(np.linalg.norm(rB)), 1.0e-300)

    residual_abs_cosine = float(
        abs(np.dot(rnA, rnB))
    )

    valsA, vecsA = eigensystem(KA)
    valsB, vecsB = eigensystem(KB)

    top1_abs_cosine = float(
        abs(np.dot(vecsA[:, 0], vecsB[:, 0]))
    )

    return {
        "normalized_kernel_shape_fro_distance":
            kernel_shape_distance,

        "paired_residual_direction_abs_cosine":
            residual_abs_cosine,

        "paired_top1_eigenvector_abs_cosine":
            top1_abs_cosine,

        "paired_top2_subspace_min_cosine":
            principal_subspace_min_cosine(
                vecsA,
                vecsB,
                2,
            ),

        "paired_top3_subspace_min_cosine":
            principal_subspace_min_cosine(
                vecsA,
                vecsB,
                3,
            ),
    }


def log_shapley(
    mu_AA: float,
    mu_AB: float,
    mu_BA: float,
    mu_BB: float,
) -> dict:

    floor = 1.0e-300

    fAA = math.log(max(mu_AA, floor))
    fAB = math.log(max(mu_AB, floor))
    fBA = math.log(max(mu_BA, floor))
    fBB = math.log(max(mu_BB, floor))

    phi_K = 0.5 * (
        (fBA - fAA)
        +
        (fBB - fAB)
    )

    phi_r = 0.5 * (
        (fAB - fAA)
        +
        (fBB - fBA)
    )

    total = fBB - fAA
    closure_gap = abs(
        (phi_K + phi_r) - total
    )

    residual_dominated = bool(
        phi_r > 0.0
        and
        phi_r >= 2.0 * abs(phi_K)
    )

    kernel_dominated = bool(
        phi_K > 0.0
        and
        phi_K >= 2.0 * abs(phi_r)
    )

    if residual_dominated:
        cls = "RESIDUAL_DIRECTION_DOMINATED"
    elif kernel_dominated:
        cls = "KERNEL_SHAPE_DOMINATED"
    else:
        cls = "MIXED"

    return {
        "log_mobility_total_contrast":
            total,

        "log_shapley_kernel":
            phi_K,

        "log_shapley_residual_direction":
            phi_r,

        "log_shapley_closure_gap":
            closure_gap,

        "shapley_class":
            cls,

        "residual_direction_dominated":
            residual_dominated,

        "kernel_shape_dominated":
            kernel_dominated,
    }


# =============================================================================
# Plots
# =============================================================================

def plot_crossswap(rows: List[dict], path: Path) -> None:
    seeds = [int(r["seed"]) for r in rows]
    x = np.arange(len(seeds))
    width = 0.19

    series = (
        ("mu_AA_deep_actual", "K_A,r_A"),
        ("mu_BA_deep_residual_transient_kernel", "K_B,r_A"),
        ("mu_AB_transient_residual_deep_kernel", "K_A,r_B"),
        ("mu_BB_transient_actual", "K_B,r_B"),
    )

    fig, ax = plt.subplots(figsize=(11.0, 5.8))

    offsets = (
        -1.5*width,
        -0.5*width,
        0.5*width,
        1.5*width,
    )

    for offset, (key, label) in zip(offsets, series):
        ax.bar(
            x + offset,
            [float(r[key]) for r in rows],
            width,
            label=label,
        )

    ax.axhline(
        MOBILITY_COLLAPSE_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label="deep-lock threshold",
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Paired seed")
    ax.set_ylabel("Normalized residual mobility μ")
    ax.set_title("Read-only K/r cross-swap: which factor carries the deep mobility collapse?")
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_shapley(rows: List[dict], path: Path) -> None:
    seeds = [int(r["seed"]) for r in rows]
    x = np.arange(len(seeds))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    ax.bar(
        x - width/2,
        [float(r["log_shapley_kernel"]) for r in rows],
        width,
        label="Kernel-shape contribution",
    )

    ax.bar(
        x + width/2,
        [float(r["log_shapley_residual_direction"]) for r in rows],
        width,
        label="Residual-direction contribution",
    )

    ax.axhline(0.0, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Paired seed")
    ax.set_ylabel("Contribution to log mobility increase A→B")
    ax.set_title("Exact two-factor log-Shapley decomposition")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_active_capture(
    actual_rows: List[dict],
    path: Path,
) -> None:

    deep = sorted(
        [
            r for r in actual_rows
            if int(r["base_mode"]) == DEEP_BASE
        ],
        key=lambda r: int(r["seed"]),
    )

    transient = sorted(
        [
            r for r in actual_rows
            if int(r["base_mode"]) == TRANSIENT_BASE
        ],
        key=lambda r: int(r["seed"]),
    )

    x = np.arange(len(SEEDS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    ax.bar(
        x - width/2,
        [
            float(
                r[
                    "residual_energy_in_trace_95_subspace"
                ]
            )
            for r in deep
        ],
        width,
        label="b=1 deep",
    )

    ax.bar(
        x + width/2,
        [
            float(
                r[
                    "residual_energy_in_trace_95_subspace"
                ]
            )
            for r in transient
        ],
        width,
        label="b=2 transient",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("Paired seed")
    ax.set_ylabel("Residual energy inside 95%-trace kernel subspace")
    ax.set_title("How much residual lies inside the kernel's dominant tangent subspace?")
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

    stage22_script = resolve(args.stage22_script)
    stage22_dir = resolve(args.stage22_dir)
    stage25_script = resolve(args.stage25_script)
    stage25_dir = resolve(args.stage25_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    pf = preflight(
        stage22_script=stage22_script,
        stage22_dir=stage22_dir,
        stage25_script=stage25_script,
        stage25_dir=stage25_dir,
    )

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,

        "stage22_script_sha256":
            pf["stage22_sha256"],

        "stage25_script_sha256":
            pf["stage25_sha256"],

        "stage26r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "readonly_paired_tangent_eigenspace_mechanism_localization",

            "paired_seeds":
                list(SEEDS),

            "target_mode":
                TARGET_MODE,

            "deep_state":
                "Stage-22 b=1,m=9 certified conflict onset",

            "transient_state":
                "Stage-22 b=2,m=9 certified conflict onset",

            "mobility_threshold":
                MOBILITY_COLLAPSE_THRESHOLD,

            "Q1":
                "10/10 actual mobility reproduction <=1e-10",

            "Q2":
                "in >=4/5, mu(K_B,r_A)<=1e-6 AND mu(K_A,r_B)>1e-6",

            "Q3":
                "residual-direction log-Shapley dominance in >=4/5",

            "no_training":
                True,

            "no_optimizer_intervention":
                True,
        },
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 178)
    print(
        "VPINN — STAGE 26R READ-ONLY PAIRED TANGENT-EIGENSPACE LOCALIZATION"
    )
    print("=" * 178)
    print(f"paired seeds              : {list(SEEDS)}")
    print(f"target mode               : {TARGET_MODE}")
    print("deep state                : b=1 Stage-22 conflict onset")
    print("transient state           : b=2 Stage-22 conflict onset")
    print("training                   : NONE")
    print("=" * 178)

    actual_rows = []
    pair_rows = []
    reproduction_rows = []

    for seed in SEEDS:

        data = {}

        for base_mode in (DEEP_BASE, TRANSIENT_BASE):

            path = conflict_npz_path(
                stage22_dir,
                seed,
                base_mode,
            )

            npz = np.load(path)

            r = np.asarray(
                npz["residuals"],
                dtype=np.float64,
            )

            K = np.asarray(
                npz["raw_kernel"],
                dtype=np.float64,
            )

            metrics = actual_state_metrics(K, r)

            expected = pf["expected_mu"][(seed, base_mode)]

            gap = abs(
                float(metrics["mu_raw"])
                - float(expected)
            )

            reproduction_rows.append(
                {
                    "seed": seed,
                    "base_mode": base_mode,
                    "expected_mu_stage22": expected,
                    "recomputed_mu": metrics["mu_raw"],
                    "abs_gap": gap,
                    "pass": bool(gap <= REPRO_TOL),
                }
            )

            if gap > REPRO_TOL:
                raise RuntimeError(
                    f"Mobility reproduction failed seed={seed}, "
                    f"base={base_mode}: gap={gap:.3e}"
                )

            actual_rows.append(
                {
                    "seed": seed,
                    "base_mode": base_mode,
                    "target_mode": TARGET_MODE,
                    "phenotype": (
                        "DEEP_LOCK"
                        if base_mode == DEEP_BASE
                        else "TRANSIENT_CONFLICT"
                    ),
                    **metrics,
                }
            )

            data[base_mode] = {
                "r": r,
                "K": K,
                "metrics": metrics,
            }

        A = data[DEEP_BASE]
        B = data[TRANSIENT_BASE]

        mu_AA = mobility(A["K"], A["r"])
        mu_AB = mobility(A["K"], B["r"])
        mu_BA = mobility(B["K"], A["r"])
        mu_BB = mobility(B["K"], B["r"])

        shapley = log_shapley(
            mu_AA=mu_AA,
            mu_AB=mu_AB,
            mu_BA=mu_BA,
            mu_BB=mu_BB,
        )

        pairgeom = pair_invariant_comparisons(
            KA=A["K"],
            rA=A["r"],
            KB=B["K"],
            rB=B["r"],
        )

        cross_signature = bool(
            mu_BA <= MOBILITY_COLLAPSE_THRESHOLD
            and
            mu_AB > MOBILITY_COLLAPSE_THRESHOLD
        )

        s1_ratio = (
            A["metrics"]["top_eigenvalue_fraction"]
            /
            max(
                B["metrics"]["top_eigenvalue_fraction"],
                1.0e-300,
            )
        )

        rho_ratio = (
            A["metrics"]["rho_max_normalized_rayleigh"]
            /
            max(
                B["metrics"]["rho_max_normalized_rayleigh"],
                1.0e-300,
            )
        )

        mobility_ratio = (
            mu_AA
            /
            max(mu_BB, 1.0e-300)
        )

        pair_rows.append(
            {
                "seed": seed,

                "mu_AA_deep_actual":
                    mu_AA,

                "mu_BA_deep_residual_transient_kernel":
                    mu_BA,

                "mu_AB_transient_residual_deep_kernel":
                    mu_AB,

                "mu_BB_transient_actual":
                    mu_BB,

                "deep_over_transient_actual_mobility":
                    mobility_ratio,

                "deep_over_transient_top_eigen_fraction":
                    s1_ratio,

                "deep_over_transient_rho_max":
                    rho_ratio,

                "crossswap_residual_direction_signature":
                    cross_signature,

                **shapley,
                **pairgeom,
            }
        )

        print()
        print(
            f"seed={seed}: "
            f"muAA={mu_AA:.6e}, "
            f"muBA={mu_BA:.6e}, "
            f"muAB={mu_AB:.6e}, "
            f"muBB={mu_BB:.6e}, "
            f"cross={cross_signature}, "
            f"Shapley={shapley['shapley_class']}"
        )

    # -------------------------------------------------------------------------
    # Persist detailed diagnostics.
    # -------------------------------------------------------------------------
    write_csv(
        out_dir / "actual_state_spectral_metrics.csv",
        actual_rows,
    )

    write_csv(
        out_dir / "paired_crossswap_decomposition.csv",
        pair_rows,
    )

    write_csv(
        out_dir / "stage22_mobility_reproduction.csv",
        reproduction_rows,
    )

    # -------------------------------------------------------------------------
    # Route gates.
    # -------------------------------------------------------------------------
    Q1 = bool(
        len(reproduction_rows) == 10
        and
        all(bool(r["pass"]) for r in reproduction_rows)
    )

    cross_count = sum(
        int(
            bool(
                r[
                    "crossswap_residual_direction_signature"
                ]
            )
        )
        for r in pair_rows
    )

    Q2 = bool(cross_count >= 4)

    residual_shapley_count = sum(
        int(
            bool(r["residual_direction_dominated"])
        )
        for r in pair_rows
    )

    kernel_shapley_count = sum(
        int(
            bool(r["kernel_shape_dominated"])
        )
        for r in pair_rows
    )

    Q3 = bool(
        residual_shapley_count >= 4
    )

    residual_localization = bool(
        Q1 and Q2 and Q3
    )

    kernel_localization = bool(
        Q1
        and kernel_shapley_count >= 4
        and not residual_localization
    )

    if residual_localization:
        route_class = (
            "deep_lock_mobility_contrast_localized_to_residual_tangent_eigenspace_placement"
        )

        next_route = (
            "stage27R_temporal_residual_rotation_precursor_audit"
        )

    elif kernel_localization:
        route_class = (
            "deep_lock_mobility_contrast_localized_to_kernel_shape_deformation"
        )

        next_route = (
            "stage27R_temporal_kernel_deformation_precursor_audit"
        )

    else:
        route_class = (
            "deep_lock_mobility_contrast_requires_joint_residual_kernel_geometry"
        )

        next_route = (
            "stage27R_joint_residual_kernel_precursor_audit"
        )

    decision = {
        "paired_seeds":
            list(SEEDS),

        "Q1_actual_mobility_reproduction":
            Q1,

        "crossswap_residual_signature_count":
            cross_count,

        "Q2_crossswap_residual_direction_signature":
            Q2,

        "residual_shapley_dominance_count":
            residual_shapley_count,

        "kernel_shapley_dominance_count":
            kernel_shapley_count,

        "Q3_log_shapley_residual_dominance":
            Q3,

        "residual_eigenspace_localization_supported":
            residual_localization,

        "kernel_shape_localization_supported":
            kernel_localization,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "The K/r cross-swap and log-Shapley analysis are algebraic "
            "read-only decompositions of paired states. They identify which "
            "factor carries the normalized mobility contrast but do not prove "
            "that factor dynamically caused the later training trajectory."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    plot_crossswap(
        pair_rows,
        out_dir / "paired_crossswap_mobility.png",
    )

    plot_shapley(
        pair_rows,
        out_dir / "log_shapley_mobility_decomposition.png",
    )

    plot_active_capture(
        actual_rows,
        out_dir / "residual_capture_in_kernel95_subspace.png",
    )

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------
    deep_rows = [
        r for r in actual_rows
        if int(r["base_mode"]) == DEEP_BASE
    ]

    transient_rows = [
        r for r in actual_rows
        if int(r["base_mode"]) == TRANSIENT_BASE
    ]

    def med(rows, key):
        return float(
            np.median(
                [float(r[key]) for r in rows]
            )
        )

    lines = []

    lines.append("=" * 180)
    lines.append(
        "VPINN — STAGE 26R READ-ONLY TANGENT-EIGENSPACE LOCALIZATION SUMMARY"
    )
    lines.append("=" * 180)

    lines.append(
        "seed | muAA deep | muBA deep-r/transient-K | "
        "muAB transient-r/deep-K | muBB transient | cross | Shapley class"
    )

    lines.append("-" * 180)

    for r in pair_rows:
        lines.append(
            f"{int(r['seed']):4d} | "
            f"{float(r['mu_AA_deep_actual']):.6e} | "
            f"{float(r['mu_BA_deep_residual_transient_kernel']):.6e} | "
            f"{float(r['mu_AB_transient_residual_deep_kernel']):.6e} | "
            f"{float(r['mu_BB_transient_actual']):.6e} | "
            f"{str(r['crossswap_residual_direction_signature']):5s} | "
            f"{r['shapley_class']}"
        )

    lines.append("-" * 180)

    lines.append(
        f"median deep mu                        : "
        f"{med(deep_rows, 'mu_raw'):.6e}"
    )

    lines.append(
        f"median transient mu                   : "
        f"{med(transient_rows, 'mu_raw'):.6e}"
    )

    lines.append(
        f"median deep top eigen fraction        : "
        f"{med(deep_rows, 'top_eigenvalue_fraction'):.6f}"
    )

    lines.append(
        f"median transient top eigen fraction   : "
        f"{med(transient_rows, 'top_eigenvalue_fraction'):.6f}"
    )

    lines.append(
        f"median deep rho_max                   : "
        f"{med(deep_rows, 'rho_max_normalized_rayleigh'):.6e}"
    )

    lines.append(
        f"median transient rho_max              : "
        f"{med(transient_rows, 'rho_max_normalized_rayleigh'):.6e}"
    )

    lines.append(
        f"Q1 mobility reproduction              : "
        f"{sum(int(r['pass']) for r in reproduction_rows)}/10 -> {Q1}"
    )

    lines.append(
        f"Q2 K/r cross-swap residual signature  : "
        f"{cross_count}/5 -> {Q2}"
    )

    lines.append(
        f"Q3 residual Shapley dominance         : "
        f"{residual_shapley_count}/5 -> {Q3}"
    )

    lines.append(
        f"kernel Shapley dominance              : "
        f"{kernel_shapley_count}/5"
    )

    lines.append(
        f"RESIDUAL-EIGENSPACE LOCALIZATION      : "
        f"{residual_localization}"
    )

    lines.append(
        f"route class                            : "
        f"{route_class}"
    )

    lines.append(
        f"next route                             : "
        f"{next_route}"
    )

    lines.append("=" * 180)

    lines.append(
        "Guardrail: read-only factor localization is not a causal training intervention."
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
