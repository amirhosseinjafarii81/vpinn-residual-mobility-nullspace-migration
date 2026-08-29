#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 12
Common Pareto-Blend Feasibility Audit
=====================================

Motivation
----------
Stage 10 showed that repeated full REFLECT fails because the reflected
displacement becomes target-descent but total-loss ascent.

Stage 11 projected the failed REFLECT displacement onto the zero-margin joint
descent cone

    <grad L, d> <= 0,
    <grad T, d> <= 0,

at each seed's earliest failure. The exact KKT projection was valid for all
active seeds, but in every active seed the TOTAL constraint was the only active
constraint and

    <grad L, d_cone> ~= 0.

Thus the minimum-distance zero-margin projection lands on the tangent boundary
of the total-loss level set. The Stage-11 line scans then showed positive
second-order total-loss curvature. In seeds 0,1,2 no tested positive step was
jointly safe. Seed 3 was classified as damped-safe only because tiny positive
total-loss changes fell inside the Stage-11 numerical nonincrease tolerance.

Therefore the next audit should NOT continue the zero-margin cone direction.
It should ask whether there is a SIMPLE INTERIOR Pareto direction that keeps
strict first-order descent for BOTH objectives.

Candidate and reflected endpoints
---------------------------------
At an earliest Stage-10 direction-conflict state, define:

    a = exact next inherited Adam displacement
    r = exact Stage-9 REFLECT displacement

and

    L(theta) = mean_k R_k(theta)^2
    T(theta) = R_9(theta)^2 / M.

At these states the desired endpoint pattern is:

    <grad L, a> < 0      (Adam protects total loss)
    <grad T, a> > 0      (Adam is target-uphill)

    <grad L, r> > 0      (REFLECT harms total loss)
    <grad T, r> < 0      (REFLECT repairs target loss)

Consider the one-parameter Pareto blend

    d(lambda) = (1-lambda) a + lambda r,
    lambda in [0,1].

Because both directional derivatives are affine in lambda, the STRICT joint
first-order descent interval can be computed analytically:

    <grad L, d(lambda)> < 0
    <grad T, d(lambda)> < 0.

No learning-rate sweep and no optimizer reset is involved.

Common-lambda rule
------------------
For each active seed, compute its strict feasible lambda interval.

Then compute the intersection across ALL active seeds:

    I_common = intersection_s I_s.

If the intersection is nonempty, choose exactly ONE common lambda:

    lambda_star = midpoint(I_common).

This is a deterministic rule. lambda_star is NOT selected by looking at exact
post-step losses.

The midpoint maximizes the minimum distance, in lambda-space, to the two
first-order feasibility boundaries of the common interval.

Read-only exact validation
--------------------------
At lambda_star, for each active seed:

  * verify strict first-order total descent;
  * verify strict first-order target descent;
  * evaluate exact full-step total and target losses;
  * perform an exact read-only line scan for

      alpha in {1/64,1/32,1/16,1/8,1/4,1/2,3/4,1};

  * record step norm relative to Adam and REFLECT;
  * record retained target-descent fraction relative to full REFLECT;
  * restore parameters exactly.

Classification
--------------
COMMON_FULL_SAFE:
    alpha=1 strictly decreases BOTH exact total and target loss.

COMMON_DAMPED_SAFE:
    alpha=1 is not jointly safe, but some alpha<1 strictly decreases both.

COMMON_FIRST_ORDER_ONLY:
    d(lambda_star) is strict first-order descent for both, but no scanned
    alpha produces strict exact decrease of both.

DEGENERATE_STEP:
    ||d(lambda_star)|| < 0.5 ||a||.

NO_COMMON_INTERVAL:
    the strict first-order intervals do not have a common intersection.

Primary route
-------------
Active denominator is the four Stage-10 earliest direction-conflict seeds.

If:
    * I_common is nonempty,
    * no active seed is DEGENERATE_STEP,
    * >=3/4 active seeds are COMMON_FULL_SAFE,

then authorize a bounded common-Pareto-blend continuation pilot.

Else if >=3/4 are COMMON_FULL_SAFE or COMMON_DAMPED_SAFE:
    authorize a backtracking common-Pareto-blend continuation pilot.

Otherwise:
    route to state-adaptive Pareto / strict-margin geometry.

This stage is an AUDIT only. No proposed continuation method is trained.
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


ALPHAS = (
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    3.0 / 4.0,
    1.0,
)

TARGET_MODE = 9


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-12 common Pareto-blend feasibility audit."
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
        "--stage11-script",
        default="vpinn_gradient_conflict_stage11_joint_descent_cone_audit.py",
    )
    p.add_argument(
        "--stage11-dir",
        default="vpinn_gradient_conflict_stage11_joint_descent_cone_audit",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage12_common_pareto_blend_audit",
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
    stage11_script: Path,
    stage11_dir: Path,
) -> dict:
    paths = {
        "s5_manifest": stage5_dir / "manifest.json",
        "s10_manifest": stage10_dir / "manifest.json",
        "s10_seed_summary": stage10_dir / "seed_summary.csv",
        "s11_manifest": stage11_dir / "manifest.json",
        "s11_decision": stage11_dir / "decision.json",
        "s11_seed_summary": stage11_dir / "seed_summary.csv",
        "s11_repro": stage11_dir / "stage10_reproduction_checks.csv",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    s5_manifest = read_json(paths["s5_manifest"])
    s10_manifest = read_json(paths["s10_manifest"])
    s11_manifest = read_json(paths["s11_manifest"])
    s11_decision = read_json(paths["s11_decision"])

    if s11_decision.get("next_route") != "mixed_pareto_geometry_audit":
        raise RuntimeError(
            "Stage 11 did not authorize mixed Pareto geometry audit."
        )

    if not bool(
        s11_decision.get("all_stage10_reproductions_pass", False)
    ):
        raise RuntimeError("Stage-11 Stage-10 reproductions are not all PASS.")

    if not bool(
        s11_decision.get("all_cone_kkt_checks_pass", False)
    ):
        raise RuntimeError("Stage-11 KKT checks are not all PASS.")

    actual_s3_sha = sha256_file(stage3_script)
    actual_s9_sha = sha256_file(stage9_script)
    actual_s11_sha = sha256_file(stage11_script)

    if s5_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 5.")

    if s10_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 10.")

    if s11_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 11.")

    if s10_manifest.get("stage9_script_sha256") != actual_s9_sha:
        raise RuntimeError("Stage-9 SHA mismatch against Stage 10.")

    if s11_manifest.get("stage9_script_sha256") != actual_s9_sha:
        raise RuntimeError("Stage-9 SHA mismatch against Stage 11.")

    if s11_manifest.get("stage11_script_sha256") != actual_s11_sha:
        raise RuntimeError("Stage-11 source SHA mismatch against its manifest.")

    repro_rows = read_csv(paths["s11_repro"])

    if not repro_rows or not all(
        str(r["pass"]).lower() == "true"
        for r in repro_rows
    ):
        raise RuntimeError("Stage-11 reproduction CSV is not all PASS.")

    s10_summary = read_csv(paths["s10_seed_summary"])
    s11_summary = read_csv(paths["s11_seed_summary"])

    active_targets = {}

    for row in s10_summary:
        seed = int(row["seed"])
        epoch = int(row["earliest_active_non_SAFE_FULL_epoch"])
        cls = row["earliest_failure_class"]

        if epoch >= 0:
            if cls != "DIRECTION_CONFLICT":
                raise RuntimeError(
                    f"Seed {seed} earliest Stage-10 failure is {cls}, "
                    "not DIRECTION_CONFLICT."
                )
            active_targets[seed] = epoch

    active11 = {
        int(r["seed"])
        for r in s11_summary
        if r["status"] == "ACTIVE_AUDIT"
    }

    if active11 != set(active_targets.keys()):
        raise RuntimeError(
            "Stage-10 and Stage-11 active seed sets do not match."
        )

    if len(active_targets) != 4:
        raise RuntimeError(
            f"Expected four active seeds, got {active_targets}."
        )

    for seed in range(5):
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing locked checkpoint: {checkpoint}")

    return {
        "active_targets": active_targets,
        "stage11_summary": s11_summary,
        "stage3_sha256": actual_s3_sha,
        "stage9_sha256": actual_s9_sha,
        "stage11_sha256": actual_s11_sha,
    }


# =============================================================================
# Strict affine interval
# =============================================================================

def strict_negative_interval(
    f0: float,
    f1: float,
    tol: float = 1.0e-15,
) -> Tuple[float, float]:
    """
    Return the open lambda interval inside [0,1] for

        f(lambda) = (1-lambda) f0 + lambda f1 < 0.

    Returned endpoints describe the mathematical boundary. Strictness is
    handled later by choosing the midpoint of a nonzero-width intersection.
    """
    slope = f1 - f0

    if abs(slope) <= tol * max(1.0, abs(f0), abs(f1)):
        if f0 < 0.0:
            return 0.0, 1.0
        return math.inf, -math.inf

    root = -f0 / slope

    if slope > 0.0:
        # f < 0 for lambda < root
        lo = 0.0
        hi = min(1.0, root)
    else:
        # f < 0 for lambda > root
        lo = max(0.0, root)
        hi = 1.0

    return lo, hi


def intersect_open_intervals(
    intervals: List[Tuple[float, float]],
) -> Tuple[float, float]:
    lo = max(x[0] for x in intervals)
    hi = min(x[1] for x in intervals)
    return lo, hi


# =============================================================================
# Exact loss scan
# =============================================================================

def loss_metrics(exp, target_mode: int = TARGET_MODE) -> dict:
    residuals = exp.weak_residuals().detach()
    energy = residuals.square()

    return {
        "total_loss": float(torch.mean(energy).item()),
        "target_loss": float(
            (energy[target_mode - 1] / energy.numel()).item()
        ),
    }


def set_displaced_params(
    params,
    base_parts,
    direction: torch.Tensor,
    alpha: float,
) -> None:
    offset = 0

    with torch.no_grad():
        for p, p0 in zip(params, base_parts):
            n = p.numel()

            p.copy_(
                p0
                + alpha
                * direction[offset:offset+n].reshape_as(p)
            )

            offset += n

    if offset != direction.numel():
        raise RuntimeError("Direction size mismatch.")


def restore_params(params, base_parts) -> None:
    with torch.no_grad():
        for p, p0 in zip(params, base_parts):
            p.copy_(p0)


def strict_line_scan(
    exp,
    params,
    direction: torch.Tensor,
) -> Tuple[List[dict], dict]:
    base_parts = [p.detach().clone() for p in params]
    pre = loss_metrics(exp)

    rows = []

    try:
        for alpha in ALPHAS:
            set_displaced_params(
                params=params,
                base_parts=base_parts,
                direction=direction,
                alpha=alpha,
            )

            post = loss_metrics(exp)

            dL = post["total_loss"] - pre["total_loss"]
            dT = post["target_loss"] - pre["target_loss"]

            # Strict actual improvement. No nonincrease tolerance is used for
            # the scientific classification. This avoids the Stage-11
            # boundary/tolerance ambiguity.
            joint_strict = bool(dL < 0.0 and dT < 0.0)

            rows.append(
                {
                    "alpha": float(alpha),
                    "pre_total_loss": pre["total_loss"],
                    "post_total_loss": post["total_loss"],
                    "total_loss_change": dL,
                    "pre_target_loss": pre["target_loss"],
                    "post_target_loss": post["target_loss"],
                    "target_loss_change": dT,
                    "joint_strict_improvement": joint_strict,
                }
            )

    finally:
        restore_params(params, base_parts)

    safe = [
        r for r in rows
        if r["joint_strict_improvement"]
    ]

    full = next(r for r in rows if r["alpha"] == 1.0)

    return rows, {
        "full_step_strictly_safe":
            bool(full["joint_strict_improvement"]),
        "max_strictly_safe_alpha": (
            max(r["alpha"] for r in safe)
            if safe else 0.0
        ),
        "any_strictly_safe_alpha": bool(safe),
        "full_total_loss_change":
            float(full["total_loss_change"]),
        "full_target_loss_change":
            float(full["target_loss_change"]),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_intervals(
    seed_rows: List[dict],
    lambda_star: float | None,
    path: Path,
) -> None:
    active = [
        r for r in seed_rows
        if r["status"] == "ACTIVE_AUDIT"
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    y = np.arange(len(active))

    for i, row in enumerate(active):
        lo = float(row["lambda_interval_lower"])
        hi = float(row["lambda_interval_upper"])

        ax.plot(
            [lo, hi],
            [i, i],
            linewidth=7,
            solid_capstyle="butt",
        )

    if lambda_star is not None:
        ax.axvline(
            lambda_star,
            linestyle="--",
            linewidth=1.5,
            label=f"common midpoint = {lambda_star:.6f}",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"seed {int(r['seed'])}" for r in active]
    )
    ax.set_xlim(0.45, 0.65)
    ax.set_xlabel("lambda")
    ax.set_title("Strict first-order joint-descent intervals")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_norm_ratios(
    seed_rows: List[dict],
    path: Path,
) -> None:
    active = [
        r for r in seed_rows
        if r["status"] == "ACTIVE_AUDIT"
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    seeds = [str(int(r["seed"])) for r in active]
    vals = [
        float(r["blend_over_adam_norm"])
        for r in active
    ]

    ax.bar(seeds, vals)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.axhline(0.5, linestyle=":", linewidth=1.0)

    ax.set_xlabel("Seed")
    ax.set_ylabel("||d(lambda*)|| / ||Adam candidate||")
    ax.set_title("Common Pareto-blend step magnitude")

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

    stage11_script = Path(args.stage11_script)
    if not stage11_script.is_absolute():
        stage11_script = root / stage11_script

    stage11_dir = Path(args.stage11_dir)
    if not stage11_dir.is_absolute():
        stage11_dir = root / stage11_dir

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
        stage11_script=stage11_script,
        stage11_dir=stage11_dir,
    )

    stage9 = load_module(
        stage9_script,
        "vpinn_stage9_replay_stage12",
    )

    stage11 = load_module(
        stage11_script,
        "vpinn_stage11_geometry_stage12",
    )

    stage3 = stage9.load_stage3_module(stage3_script)

    print("=" * 142)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 12 COMMON PARETO-BLEND FEASIBILITY AUDIT"
    )
    print("=" * 142)
    print(f"device                    : {device}")
    print(f"active earliest failures  : {pf['active_targets']}")
    print(
        "family                    : d(lambda)=(1-lambda) Adam + lambda REFLECT"
    )
    print(
        "lambda rule               : midpoint of common strict first-order interval"
    )
    print(
        "scientific exact safety   : strict decrease of BOTH exact losses"
    )
    print(f"Stage-3 SHA256            : {pf['stage3_sha256']}")
    print(f"Stage-9 SHA256            : {pf['stage9_sha256']}")
    print(f"Stage-11 SHA256           : {pf['stage11_sha256']}")
    print("=" * 142)

    # First pass: reconstruct all earliest failures and derive intervals.
    states = {}
    seed_rows: List[dict] = []

    stage11_by_seed = {
        int(r["seed"]): r
        for r in pf["stage11_summary"]
    }

    for seed in range(5):
        if seed not in pf["active_targets"]:
            seed_rows.append(
                {
                    "seed": seed,
                    "status": "NO_TRIGGER",
                    "failure_epoch": -1,
                }
            )
            continue

        failure_epoch = pf["active_targets"][seed]

        seed_dir = out_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        exp = stage9.make_experiment(
            stage3=stage3,
            device=device,
            seed=seed,
            out_dir=seed_dir,
        )

        stage9.load_locked_checkpoint(
            exp=exp,
            checkpoint_path=checkpoint,
            expected_seed=seed,
        )

        for _epoch in range(2500, failure_epoch):
            stage9.intervention_step(
                exp=exp,
                branch="REFLECT",
                target_mode=TARGET_MODE,
            )

        geo = stage11.current_geometry(
            exp=exp,
            target_mode=TARGET_MODE,
        )

        if not geo["reflect_active"]:
            raise RuntimeError(
                f"Seed {seed} earliest failure is no longer REFLECT-active."
            )

        # Reproduce Stage-11 source geometry exactly.
        old = stage11_by_seed[seed]

        source_checks = {
            "reflect_total_dot":
                abs(
                    float(old["reflect_total_directional_derivative"])
                    - geo["reflect_total_dot"]
                ),
            "reflect_target_dot":
                abs(
                    float(old["reflect_target_directional_derivative"])
                    - geo["reflect_target_dot"]
                ),
        }

        source_gap = max(source_checks.values())

        if source_gap > 1.0e-10:
            raise RuntimeError(
                f"Stage-11 source reproduction failed for seed {seed}: "
                f"{source_gap:.3e}"
            )

        gL = geo["g_total"]
        gT = geo["g_target"]
        adam = geo["adam_candidate"]
        reflect = geo["reflected"]

        adam_L = float(torch.dot(gL, adam).item())
        adam_T = float(torch.dot(gT, adam).item())

        reflect_L = float(torch.dot(gL, reflect).item())
        reflect_T = float(torch.dot(gT, reflect).item())

        endpoint_pattern_pass = bool(
            adam_L < 0.0
            and adam_T > 0.0
            and reflect_L > 0.0
            and reflect_T < 0.0
        )

        if not endpoint_pattern_pass:
            raise RuntimeError(
                f"Seed {seed} does not have the expected Adam/REFLECT "
                f"Pareto endpoint pattern."
            )

        interval_L = strict_negative_interval(
            adam_L,
            reflect_L,
        )

        interval_T = strict_negative_interval(
            adam_T,
            reflect_T,
        )

        local_interval = intersect_open_intervals(
            [interval_L, interval_T]
        )

        if not (local_interval[0] < local_interval[1]):
            raise RuntimeError(
                f"Seed {seed} has no local strict joint-descent blend interval."
            )

        states[seed] = {
            "exp": exp,
            "geo": geo,
            "failure_epoch": failure_epoch,
            "adam_L": adam_L,
            "adam_T": adam_T,
            "reflect_L": reflect_L,
            "reflect_T": reflect_T,
            "interval_L": interval_L,
            "interval_T": interval_T,
            "local_interval": local_interval,
            "source_gap": source_gap,
        }

        seed_rows.append(
            {
                "seed": seed,
                "status": "ACTIVE_AUDIT",
                "failure_epoch": failure_epoch,

                "adam_total_directional_derivative":
                    adam_L,
                "adam_target_directional_derivative":
                    adam_T,
                "reflect_total_directional_derivative":
                    reflect_L,
                "reflect_target_directional_derivative":
                    reflect_T,

                "lambda_total_lower":
                    interval_L[0],
                "lambda_total_upper":
                    interval_L[1],
                "lambda_target_lower":
                    interval_T[0],
                "lambda_target_upper":
                    interval_T[1],

                "lambda_interval_lower":
                    local_interval[0],
                "lambda_interval_upper":
                    local_interval[1],
                "lambda_interval_width":
                    local_interval[1] - local_interval[0],

                "stage11_source_reproduction_gap":
                    source_gap,
            }
        )

    active_intervals = [
        states[s]["local_interval"]
        for s in sorted(states.keys())
    ]

    common_lo, common_hi = intersect_open_intervals(
        active_intervals
    )

    common_exists = bool(common_lo < common_hi)

    if common_exists:
        lambda_star = 0.5 * (common_lo + common_hi)
    else:
        lambda_star = None

    print()
    print(
        f"common strict interval     : "
        f"({common_lo:.12f}, {common_hi:.12f})"
    )

    if lambda_star is not None:
        print(f"common midpoint lambda*    : {lambda_star:.12f}")
    else:
        print("common midpoint lambda*    : NONE")

    scan_rows: List[dict] = []

    # Second pass: exact common-lambda validation.
    if common_exists:
        for row in seed_rows:
            if row["status"] != "ACTIVE_AUDIT":
                continue

            seed = int(row["seed"])
            st = states[seed]

            exp = st["exp"]
            geo = st["geo"]

            adam = geo["adam_candidate"]
            reflect = geo["reflected"]
            gL = geo["g_total"]
            gT = geo["g_target"]

            blend = (
                (1.0 - lambda_star) * adam
                + lambda_star * reflect
            )

            blend_L = float(torch.dot(gL, blend).item())
            blend_T = float(torch.dot(gT, blend).item())

            if not (blend_L < 0.0 and blend_T < 0.0):
                raise RuntimeError(
                    f"Common lambda is not strict joint descent for seed {seed}."
                )

            adam_norm = float(
                torch.linalg.vector_norm(adam).item()
            )

            reflect_norm = float(
                torch.linalg.vector_norm(reflect).item()
            )

            blend_norm = float(
                torch.linalg.vector_norm(blend).item()
            )

            blend_over_adam = (
                blend_norm / max(adam_norm, 1.0e-300)
            )

            blend_over_reflect = (
                blend_norm / max(reflect_norm, 1.0e-300)
            )

            target_retention = (
                -blend_T
                / max(-st["reflect_T"], 1.0e-300)
            )

            total_retention = (
                -blend_L
                / max(-st["adam_L"], 1.0e-300)
            )

            line_rows, scan_summary = strict_line_scan(
                exp=exp,
                params=geo["params"],
                direction=blend,
            )

            degenerate = bool(blend_over_adam < 0.5)

            if degenerate:
                cls = "DEGENERATE_STEP"
            elif scan_summary["full_step_strictly_safe"]:
                cls = "COMMON_FULL_SAFE"
            elif scan_summary["any_strictly_safe_alpha"]:
                cls = "COMMON_DAMPED_SAFE"
            else:
                cls = "COMMON_FIRST_ORDER_ONLY"

            row.update(
                {
                    "lambda_star": lambda_star,
                    "common_interval_lower": common_lo,
                    "common_interval_upper": common_hi,
                    "common_interval_width":
                        common_hi - common_lo,

                    "blend_total_directional_derivative":
                        blend_L,
                    "blend_target_directional_derivative":
                        blend_T,

                    "blend_norm": blend_norm,
                    "blend_over_adam_norm":
                        blend_over_adam,
                    "blend_over_reflect_norm":
                        blend_over_reflect,

                    "target_descent_retention_vs_reflect":
                        target_retention,
                    "total_descent_retention_vs_adam":
                        total_retention,

                    "classification": cls,
                    "degenerate_step": degenerate,

                    **scan_summary,
                }
            )

            for sr in line_rows:
                scan_rows.append(
                    {
                        "seed": seed,
                        "failure_epoch":
                            st["failure_epoch"],
                        "lambda_star":
                            lambda_star,
                        "classification":
                            cls,
                        **sr,
                    }
                )

            print()
            print("-" * 142)
            print(
                f"SEED {seed} @ {st['failure_epoch']} | "
                f"class={cls}"
            )
            print(
                f"local interval="
                f"({st['local_interval'][0]:.12f}, "
                f"{st['local_interval'][1]:.12f})"
            )
            print(
                f"blend dL={blend_L:+.6e}, "
                f"dT={blend_T:+.6e}"
            )
            print(
                f"norm/Adam={blend_over_adam:.6f}, "
                f"target keep={target_retention:.6f}, "
                f"total keep={total_retention:.6f}"
            )
            print(
                f"full exact dL="
                f"{scan_summary['full_total_loss_change']:+.6e}, "
                f"full exact dT="
                f"{scan_summary['full_target_loss_change']:+.6e}"
            )
            print(
                f"max strictly safe alpha="
                f"{scan_summary['max_strictly_safe_alpha']:.6f}"
            )

    write_csv(out_dir / "seed_summary.csv", seed_rows)
    write_csv(out_dir / "line_scan_metrics.csv", scan_rows)

    active_rows = [
        r for r in seed_rows
        if r["status"] == "ACTIVE_AUDIT"
    ]

    n_active = len(active_rows)
    group_need = max(1, math.ceil(0.75 * n_active))

    full_safe = sum(
        int(r.get("classification") == "COMMON_FULL_SAFE")
        for r in active_rows
    )

    damped_safe = sum(
        int(r.get("classification") == "COMMON_DAMPED_SAFE")
        for r in active_rows
    )

    first_order_only = sum(
        int(
            r.get("classification")
            == "COMMON_FIRST_ORDER_ONLY"
        )
        for r in active_rows
    )

    degenerate_count = sum(
        int(r.get("classification") == "DEGENERATE_STEP")
        for r in active_rows
    )

    safe_total = full_safe + damped_safe

    if (
        common_exists
        and degenerate_count == 0
        and full_safe >= group_need
    ):
        route_class = "common_interior_pareto_full_step_viable"
        next_route = "bounded_common_pareto_blend_continuation_pilot"
    elif (
        common_exists
        and degenerate_count == 0
        and safe_total >= group_need
    ):
        route_class = "common_interior_pareto_backtracking_viable"
        next_route = "backtracking_common_pareto_blend_continuation_pilot"
    else:
        route_class = "common_blend_not_sufficient"
        next_route = "state_adaptive_strict_margin_pareto_audit"

    target_keep_vals = [
        float(r["target_descent_retention_vs_reflect"])
        for r in active_rows
        if "target_descent_retention_vs_reflect" in r
    ]

    total_keep_vals = [
        float(r["total_descent_retention_vs_adam"])
        for r in active_rows
        if "total_descent_retention_vs_adam" in r
    ]

    norm_vals = [
        float(r["blend_over_adam_norm"])
        for r in active_rows
        if "blend_over_adam_norm" in r
    ]

    decision = {
        "n_active_seeds": n_active,
        "active_seeds": sorted(states.keys()),
        "group_required_count": group_need,

        "common_interval_exists": common_exists,
        "common_interval_lower": common_lo,
        "common_interval_upper": common_hi,
        "common_interval_width":
            common_hi - common_lo,
        "lambda_star": lambda_star,

        "full_safe_count": full_safe,
        "damped_safe_count": damped_safe,
        "first_order_only_count": first_order_only,
        "degenerate_step_count": degenerate_count,

        "median_target_descent_retention_vs_reflect": (
            float(np.median(target_keep_vals))
            if target_keep_vals else None
        ),
        "median_total_descent_retention_vs_adam": (
            float(np.median(total_keep_vals))
            if total_keep_vals else None
        ),
        "median_blend_over_adam_norm": (
            float(np.median(norm_vals))
            if norm_vals else None
        ),

        "route_class": route_class,
        "next_route": next_route,

        "interpretation_guardrail": (
            "The common lambda is derived solely from the intersection of "
            "strict first-order Pareto intervals at the earliest failure "
            "states. Exact post-step losses are used only for validation, "
            "not for selecting lambda_star. A local audit PASS does not yet "
            "establish faster escape; only the bounded continuation may test it."
        ),
    }

    write_json(out_dir / "decision.json", decision)

    precommitment = {
        "stage": "common_pareto_blend_feasibility",
        "family":
            "d(lambda)=(1-lambda) AdamCandidate + lambda REFLECT",
        "lambda_selection":
            "midpoint of common strict first-order joint-descent interval",
        "selection_uses_exact_post_step_losses": False,
        "strict_exact_safety":
            "both total loss and target loss must strictly decrease",
        "line_scan_alphas": list(ALPHAS),
        "degeneracy_guard":
            "||d(lambda*)|| / ||AdamCandidate|| >= 0.5",
        "group_gate": "3/4 active seeds",
        "next_routes": {
            "full_safe":
                "bounded_common_pareto_blend_continuation_pilot",
            "damped_safe":
                "backtracking_common_pareto_blend_continuation_pilot",
            "fail":
                "state_adaptive_strict_margin_pareto_audit",
        },
    }

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),

        "stage3_solver_sha256":
            pf["stage3_sha256"],
        "stage9_script_sha256":
            pf["stage9_sha256"],
        "stage11_script_sha256":
            pf["stage11_sha256"],
        "stage12_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "stage10_dir": str(stage10_dir),
        "stage11_dir": str(stage11_dir),

        "precommitment": precommitment,
    }

    write_json(out_dir / "manifest.json", manifest)

    plot_intervals(
        seed_rows,
        lambda_star,
        out_dir / "strict_pareto_intervals.png",
    )

    if common_exists:
        plot_norm_ratios(
            seed_rows,
            out_dir / "common_blend_norm_ratio.png",
        )

    print()
    print("=" * 142)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 12 COMMON PARETO-BLEND SUMMARY"
    )
    print("=" * 142)
    print(
        "seed | epoch | local lambda interval        | class                    | "
        "norm/Adam | target keep | total keep | max safe alpha"
    )
    print("-" * 142)

    for row in seed_rows:
        if row["status"] != "ACTIVE_AUDIT":
            print(
                f"{int(row['seed']):4d} | "
                f"{-1:5d} | "
                f"{'-':28s} | "
                f"{'NO_TRIGGER':24s} | "
                f"{'-':9s} | {'-':11s} | {'-':10s} | {'-'}"
            )
            continue

        print(
            f"{int(row['seed']):4d} | "
            f"{int(row['failure_epoch']):5d} | "
            f"({row['lambda_interval_lower']:.9f},"
            f"{row['lambda_interval_upper']:.9f}) | "
            f"{str(row.get('classification','NO_COMMON')):24s} | "
            f"{float(row.get('blend_over_adam_norm',float('nan'))):9.6f} | "
            f"{float(row.get('target_descent_retention_vs_reflect',float('nan'))):11.6f} | "
            f"{float(row.get('total_descent_retention_vs_adam',float('nan'))):10.6f} | "
            f"{float(row.get('max_strictly_safe_alpha',0.0)):.6f}"
        )

    print("-" * 142)
    print(
        f"common strict interval               : "
        f"({common_lo:.12f}, {common_hi:.12f})"
    )
    print(f"lambda_star                          : {lambda_star}")
    print(f"COMMON_FULL_SAFE                     : {full_safe}/{n_active}")
    print(f"COMMON_DAMPED_SAFE                   : {damped_safe}/{n_active}")
    print(f"COMMON_FIRST_ORDER_ONLY              : {first_order_only}/{n_active}")
    print(f"DEGENERATE_STEP                      : {degenerate_count}/{n_active}")
    print(
        f"median target descent retained       : "
        f"{decision['median_target_descent_retention_vs_reflect']}"
    )
    print(
        f"median total descent retained        : "
        f"{decision['median_total_descent_retention_vs_adam']}"
    )
    print(
        f"median blend/Adam norm               : "
        f"{decision['median_blend_over_adam_norm']}"
    )
    print(f"route class                          : {route_class}")
    print(f"next route                           : {next_route}")
    print("=" * 142)
    print(
        "Guardrail: this is a local Pareto-feasibility audit, not yet an "
        "escape-time result."
    )
    print("=" * 142)

    (out_dir / "console_summary.txt").write_text(
        "\n".join(
            [
                "Stage 12 completed.",
                f"common_interval=({common_lo},{common_hi})",
                f"lambda_star={lambda_star}",
                f"full_safe={full_safe}/{n_active}",
                f"damped_safe={damped_safe}/{n_active}",
                f"route={next_route}",
            ]
        ),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
