#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 10
Early-Horizon Local Feasibility Audit:
Curvature-Limited vs Direction-Conflicted REFLECT
==================================================

Background
----------
Stage 8 showed that a ONE-STEP norm-preserving reflection of the target-uphill
Adam displacement improves R_9^2 without increasing the total VPINN loss.

Stage 9 showed that REPEATING the same reflection does NOT accelerate escape:
  * REFLECT accelerated 0/5 seeds;
  * several REFLECT trajectories sharply reduced R_9^2 while exploding
    non-target weak residuals and solution error.

The next scientifically necessary question is:

    Why does a locally successful one-step reflection become unstable
    under continuation?

Two distinct mechanisms must be separated.

A) CURVATURE / STEP-SIZE FAILURE
--------------------------------
The reflected direction is still first-order descent for BOTH:
    * total VPINN loss L,
    * target loss T = R_9^2 / M,

but the full alpha=1 step is too large. A smaller alpha is jointly safe.

B) MULTI-OBJECTIVE DIRECTION CONFLICT
-------------------------------------
The reflected direction is target descent but first-order ASCENT for the
total VPINN objective:

    <grad L, Delta_R> >= 0.

Then merely shrinking alpha is not the right fundamental fix. The direction
itself violates the total-objective descent half-space.

Design
------
Replay the exact Stage-9 REFLECT trajectory from each saved epoch-2500 state
only through epoch 2600.

Probe epochs:
    every 5 epochs from 2500 through 2600.

At each probe, BEFORE the next intervention step:

  1) predict the exact next Adam displacement Delta_A from the current
     optimizer state;
  2) construct the exact Stage-9 REFLECT displacement Delta_R;
  3) compute first-order directional derivatives:
         dT = <grad T, Delta_R>
         dL = <grad L, Delta_R>
  4) perform a READ-ONLY exact line scan along Delta_R:
         alpha in {1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 3/4, 1}
     measuring the actual total VPINN loss and target loss;
  5) restore parameters exactly;
  6) advance one real Stage-9 REFLECT step.

No optimizer state is changed by the line scan.

Local classification at an ACTIVE REFLECT probe
------------------------------------------------
SAFE_FULL:
    alpha=1 does not increase either total or target loss.

CURVATURE_LIMITED:
    dL < 0 and dT < 0,
    alpha=1 is unsafe,
    but some alpha < 1 is jointly safe.

DIRECTION_CONFLICT:
    dT < 0 but dL >= 0.
    REFLECT is target-descent but total-objective non-descent already at
    first order.

UNRESOLVED_HIGH_CURVATURE:
    dL < 0 and dT < 0,
    alpha=1 unsafe,
    and none of the precommitted grid alphas is jointly safe.

INACTIVE:
    inherited Adam step is already target-nonuphill, so Stage-9 REFLECT is
    the identity.

Primary route decision
----------------------
For each seed, record the EARLIEST active probe that is not SAFE_FULL.

If >=3/5 seeds have earliest failure = CURVATURE_LIMITED:
    authorize a bounded trust-region REFLECT pilot.

If >=3/5 seeds have earliest failure = DIRECTION_CONFLICT:
    authorize a joint total/target descent-cone projection audit.

Otherwise:
    route to a mixed local-geometry audit.

Why earliest failure?
---------------------
Later geometry can be a consequence of an already-diverged trajectory.
The earliest local violation is the least-confounded mechanistic diagnostic.

Reproducibility
---------------
* Stage-9 decision must explicitly route to local_curvature_trust_region_audit.
* Stage-3 and Stage-9 source SHA256 values must match their manifests.
* Exact Stage-9 REFLECT states at epochs 2525, 2550, 2575, and 2600 are
  reproduced and compared to Stage-9 aggregate_trajectories.csv.
* Any replay drift above 1e-10 aborts the run.

This stage is an audit only. It does not train a new proposed method.
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

PROBE_START = 2500
PROBE_END = 2600
PROBE_INTERVAL = 5
TARGET_MODE = 9


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stage-10 local feasibility audit for repeated reflected Adam."
        )
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
        "--stage9-dir",
        default="vpinn_gradient_conflict_stage9_reflected_adam_continuation",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage10_local_feasibility_audit",
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


def load_module(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Python source not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {path}")

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
    stage9_script: Path,
    stage9_dir: Path,
) -> dict:
    paths = {
        "s9_manifest": stage9_dir / "manifest.json",
        "s9_decision": stage9_dir / "decision.json",
        "s9_aggregate": stage9_dir / "aggregate_trajectories.csv",
        "s9_first_step": stage9_dir / "first_step_stage8_reproduction.json",
        "s5_manifest": stage5_dir / "manifest.json",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    s9_manifest = read_json(paths["s9_manifest"])
    s9_decision = read_json(paths["s9_decision"])
    s9_first_step = read_json(paths["s9_first_step"])
    s5_manifest = read_json(paths["s5_manifest"])

    if s9_decision.get("next_route") != "local_curvature_trust_region_audit":
        raise RuntimeError(
            "Stage 9 did not route to local_curvature_trust_region_audit."
        )

    if bool(s9_decision.get("reflect_group_gate_pass", True)):
        raise RuntimeError(
            "Stage 9 unexpectedly passed the REFLECT escape-time group gate."
        )

    if not bool(
        s9_decision.get("all_first_step_stage8_reproductions_pass", False)
    ):
        raise RuntimeError("Stage-9 first-step reproductions are not all PASS.")

    if not bool(s9_first_step.get("all_pass", False)):
        raise RuntimeError(
            "Stage-9 first_step_stage8_reproduction.json is not PASS."
        )

    actual_s3_sha = sha256_file(stage3_script)
    actual_s9_sha = sha256_file(stage9_script)

    if s5_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 5.")

    if s9_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 9.")

    if s9_manifest.get("stage9_script_sha256") != actual_s9_sha:
        raise RuntimeError(
            "Stage-9 source SHA mismatch against its result manifest."
        )

    aggregate = read_csv(paths["s9_aggregate"])

    for seed in range(5):
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing locked checkpoint: {checkpoint}")

    return {
        "seeds": [0, 1, 2, 3, 4],
        "stage3_sha256": actual_s3_sha,
        "stage9_sha256": actual_s9_sha,
        "aggregate": aggregate,
    }


# =============================================================================
# Adam candidate + reflected direction, read only
# =============================================================================

def predict_candidate_geometry(exp, target_mode: int = TARGET_MODE) -> dict:
    residuals = exp.weak_residuals()
    params = tuple(p for p in exp.model.parameters() if p.requires_grad)

    t = target_mode - 1
    M = residuals.numel()

    target_loss = residuals[t].square() / M
    total_loss = residuals.square().mean()

    gt_parts = torch.autograd.grad(
        target_loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )

    g_parts = torch.autograd.grad(
        total_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    gt = flatten(gt_parts).detach()
    g = flatten(g_parts).detach()

    group = exp.optimizer.param_groups[0]

    if float(group.get("weight_decay", 0.0)) != 0.0:
        raise RuntimeError("Expected Adam weight_decay=0.")
    if bool(group.get("amsgrad", False)):
        raise RuntimeError("Expected Adam amsgrad=False.")
    if bool(group.get("maximize", False)):
        raise RuntimeError("Expected Adam maximize=False.")

    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    lr = float(group["lr"])

    candidate_parts = []
    offset = 0

    for p in params:
        n = p.numel()
        gp = g[offset:offset+n].reshape_as(p)
        offset += n

        state = exp.optimizer.state[p]

        m_old = state["exp_avg"]
        v_old = state["exp_avg_sq"]

        step_raw = state["step"]
        step_old = (
            int(step_raw.item())
            if torch.is_tensor(step_raw)
            else int(step_raw)
        )

        m_new = beta1 * m_old + (1.0 - beta1) * gp
        v_new = beta2 * v_old + (1.0 - beta2) * gp.square()

        step_new = step_old + 1
        bc1 = 1.0 - beta1 ** step_new
        bc2 = 1.0 - beta2 ** step_new

        denom = (v_new.sqrt() / math.sqrt(bc2)).add(eps)
        delta = -lr / bc1 * m_new / denom

        candidate_parts.append(delta.reshape(-1))

    candidate = flatten(candidate_parts)

    gt2 = torch.dot(gt, gt)
    target_dot_candidate = torch.dot(gt, candidate)

    if float(gt2.item()) <= 0.0:
        raise RuntimeError("Target gradient has zero norm.")

    active = bool(target_dot_candidate.item() > 0.0)

    if active:
        component = (target_dot_candidate / gt2) * gt
        reflected = candidate - 2.0 * component
    else:
        reflected = candidate.detach().clone()

    candidate_norm = torch.linalg.vector_norm(candidate)
    reflected_norm = torch.linalg.vector_norm(reflected)

    if active:
        ratio = float(
            (
                reflected_norm
                / torch.clamp(candidate_norm, min=1.0e-300)
            ).item()
        )

        if abs(ratio - 1.0) > 1.0e-10:
            raise RuntimeError("REFLECT norm identity failed.")

    d_target = float(torch.dot(gt, reflected).item())
    d_total = float(torch.dot(g, reflected).item())

    return {
        "params": params,
        "gt": gt,
        "g": g,
        "candidate": candidate,
        "reflected": reflected,
        "active": active,
        "candidate_norm": float(candidate_norm.item()),
        "reflected_norm": float(reflected_norm.item()),
        "target_directional_derivative": d_target,
        "total_directional_derivative": d_total,
        "target_gradient_norm": float(torch.linalg.vector_norm(gt).item()),
        "total_gradient_norm": float(torch.linalg.vector_norm(g).item()),
    }


# =============================================================================
# Exact read-only line scan
# =============================================================================

def residual_loss_metrics(exp, target_mode: int = TARGET_MODE) -> dict:
    residuals = exp.weak_residuals().detach()
    energy = residuals.square()

    return {
        "total_loss": float(torch.mean(energy).item()),
        "target_loss": float(
            (energy[target_mode - 1] / energy.numel()).item()
        ),
    }


def apply_temp_displacement(
    params,
    base_parts,
    displacement: torch.Tensor,
    alpha: float,
) -> None:
    offset = 0

    with torch.no_grad():
        for p, p0 in zip(params, base_parts):
            n = p.numel()
            p.copy_(
                p0
                + alpha
                * displacement[offset:offset+n].reshape_as(p)
            )
            offset += n

    if offset != displacement.numel():
        raise RuntimeError("Temporary displacement size mismatch.")


def restore_params(params, base_parts) -> None:
    with torch.no_grad():
        for p, p0 in zip(params, base_parts):
            p.copy_(p0)


def exact_line_scan(exp, geo: dict) -> tuple[list[dict], dict]:
    params = geo["params"]
    reflected = geo["reflected"]

    base_parts = [p.detach().clone() for p in params]
    base = residual_loss_metrics(exp)

    tol_total = 1.0e-12 * max(1.0, abs(base["total_loss"]))
    tol_target = 1.0e-12 * max(1.0, abs(base["target_loss"]))

    rows = []

    try:
        for alpha in ALPHAS:
            apply_temp_displacement(
                params=params,
                base_parts=base_parts,
                displacement=reflected,
                alpha=alpha,
            )

            after = residual_loss_metrics(exp)

            total_change = after["total_loss"] - base["total_loss"]
            target_change = after["target_loss"] - base["target_loss"]

            joint_safe = bool(
                total_change <= tol_total
                and target_change <= tol_target
            )

            if geo["total_directional_derivative"] < 0.0:
                predicted_total_decrease = (
                    -alpha * geo["total_directional_derivative"]
                )
                rho_total = (
                    (base["total_loss"] - after["total_loss"])
                    / max(predicted_total_decrease, 1.0e-300)
                )
            else:
                rho_total = float("nan")

            if geo["target_directional_derivative"] < 0.0:
                predicted_target_decrease = (
                    -alpha * geo["target_directional_derivative"]
                )
                rho_target = (
                    (base["target_loss"] - after["target_loss"])
                    / max(predicted_target_decrease, 1.0e-300)
                )
            else:
                rho_target = float("nan")

            rows.append(
                {
                    "alpha": float(alpha),
                    "pre_total_loss": base["total_loss"],
                    "post_total_loss": after["total_loss"],
                    "total_loss_change": total_change,
                    "pre_target_loss": base["target_loss"],
                    "post_target_loss": after["target_loss"],
                    "target_loss_change": target_change,
                    "joint_safe": joint_safe,
                    "rho_total": rho_total,
                    "rho_target": rho_target,
                }
            )
    finally:
        restore_params(params, base_parts)

    safe_alphas = [
        r["alpha"] for r in rows if r["joint_safe"]
    ]

    max_safe_alpha = max(safe_alphas) if safe_alphas else 0.0
    full_safe = bool(
        next(r for r in rows if r["alpha"] == 1.0)["joint_safe"]
    )

    return rows, {
        "pre_total_loss": base["total_loss"],
        "pre_target_loss": base["target_loss"],
        "full_step_joint_safe": full_safe,
        "max_joint_safe_alpha": max_safe_alpha,
        "any_damped_joint_safe": bool(
            any(
                r["joint_safe"] and r["alpha"] < 1.0
                for r in rows
            )
        ),
    }


# =============================================================================
# Local classification
# =============================================================================

def classify_local(geo: dict, scan_summary: dict) -> str:
    if not geo["active"]:
        return "INACTIVE"

    dT = geo["target_directional_derivative"]
    dL = geo["total_directional_derivative"]

    # Reflection must be target-descent when active.
    if dT >= 0.0:
        return "INVALID_TARGET_GEOMETRY"

    if scan_summary["full_step_joint_safe"]:
        return "SAFE_FULL"

    if dL >= 0.0:
        return "DIRECTION_CONFLICT"

    if scan_summary["any_damped_joint_safe"]:
        return "CURVATURE_LIMITED"

    return "UNRESOLVED_HIGH_CURVATURE"


# =============================================================================
# Replay verification
# =============================================================================

def reference_map(rows: List[dict]) -> dict:
    return {
        (int(r["seed"]), str(r["branch"]), int(r["epoch"])): r
        for r in rows
    }


def verify_stage9_state(
    seed: int,
    epoch: int,
    state: dict,
    last_step_diag: dict,
    ref_map: dict,
    tolerance: float = 1.0e-10,
) -> dict:
    key = (seed, "REFLECT", epoch)

    if key not in ref_map:
        raise RuntimeError(
            f"Stage-9 REFLECT reference missing: seed={seed}, epoch={epoch}."
        )

    old = ref_map[key]

    comparisons = {
        "relative_l2_error": (
            float(old["relative_l2_error"]),
            float(state["relative_l2_error"]),
        ),
        "vpinn_loss": (
            float(old["vpinn_loss"]),
            float(state["vpinn_loss"]),
        ),
        "target_loss_unscaled": (
            float(old["target_loss_unscaled"]),
            float(state["target_loss_unscaled"]),
        ),
        "target_share": (
            float(old["target_mode_residual_energy_share"]),
            float(state["target_mode_residual_energy_share"]),
        ),
        "candidate_alignment": (
            float(old["candidate_adam_target_descent_alignment"]),
            float(
                last_step_diag[
                    "candidate_adam_target_descent_alignment"
                ]
            ),
        ),
        "applied_alignment": (
            float(old["applied_target_descent_alignment"]),
            float(last_step_diag["applied_target_descent_alignment"]),
        ),
        "candidate_norm": (
            float(old["candidate_adam_update_norm"]),
            float(last_step_diag["candidate_adam_update_norm"]),
        ),
        "applied_norm": (
            float(old["applied_update_norm"]),
            float(last_step_diag["applied_update_norm"]),
        ),
    }

    diffs = {
        k: abs(a - b)
        for k, (a, b) in comparisons.items()
    }

    max_diff = max(diffs.values())

    return {
        "seed": seed,
        "epoch": epoch,
        "max_abs_difference": max_diff,
        "field_abs_differences": diffs,
        "pass": bool(max_diff <= tolerance),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_safe_alpha(probes: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for seed in sorted(set(int(r["seed"]) for r in probes)):
        rr = [
            r for r in probes
            if int(r["seed"]) == seed
        ]
        rr.sort(key=lambda x: int(x["epoch"]))

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r["max_joint_safe_alpha"]) for r in rr],
            marker="o",
            markersize=3,
            label=f"seed {seed}",
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Largest jointly safe alpha on precommitted grid")
    ax.set_title("Local safe fraction of the REFLECT displacement")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_directional_derivative(
    probes: List[dict],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for seed in sorted(set(int(r["seed"]) for r in probes)):
        rr = [
            r for r in probes
            if int(r["seed"]) == seed
            and bool(r["intervention_active"])
        ]
        rr.sort(key=lambda x: int(x["epoch"]))

        if not rr:
            continue

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r["total_directional_derivative"]) for r in rr],
            marker="o",
            markersize=3,
            label=f"seed {seed}",
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("<grad L, Delta_REFLECT>")
    ax.set_title("Does REFLECT remain a total-objective descent direction?")
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

    stage9_script = Path(args.stage9_script)
    if not stage9_script.is_absolute():
        stage9_script = root / stage9_script

    stage9_dir = Path(args.stage9_dir)
    if not stage9_dir.is_absolute():
        stage9_dir = root / stage9_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage5_dir=stage5_dir,
        stage9_script=stage9_script,
        stage9_dir=stage9_dir,
    )

    stage9 = load_module(
        stage9_script,
        "vpinn_stage9_replay_stage10",
    )

    stage3 = stage9.load_stage3_module(stage3_script)

    ref_map = reference_map(pf["aggregate"])

    precommitment = {
        "stage":
            "early_horizon_local_feasibility_curvature_vs_direction_audit",
        "trajectory":
            "exact Stage-9 REFLECT replay",
        "seeds": pf["seeds"],
        "probe_window": [PROBE_START, PROBE_END],
        "probe_interval": PROBE_INTERVAL,
        "line_scan_alphas": list(ALPHAS),
        "joint_safety": (
            "exact total VPINN loss nonincrease AND "
            "exact target loss R9^2/M nonincrease"
        ),
        "classification": {
            "SAFE_FULL":
                "alpha=1 jointly safe",
            "CURVATURE_LIMITED":
                "dL<0,dT<0; alpha=1 unsafe; some alpha<1 jointly safe",
            "DIRECTION_CONFLICT":
                "dT<0 but dL>=0",
            "UNRESOLVED_HIGH_CURVATURE":
                "dL<0,dT<0; alpha=1 unsafe; no scanned alpha safe",
            "INACTIVE":
                "candidate Adam step already target-nonuphill",
        },
        "primary_seed_marker":
            "earliest active probe whose class is not SAFE_FULL",
        "decision_routes": {
            "curvature_earliest_in_at_least_3_seeds":
                "bounded_trust_region_reflect_pilot",
            "direction_conflict_earliest_in_at_least_3_seeds":
                "joint_total_target_descent_cone_projection_audit",
            "otherwise":
                "mixed_local_geometry_audit",
        },
        "no_new_training_method_is_tested": True,
    }

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),
        "stage3_solver_sha256": pf["stage3_sha256"],
        "stage9_script_sha256": pf["stage9_sha256"],
        "stage9_dir": str(stage9_dir),
        "stage10_script_sha256": sha256_file(Path(__file__).resolve()),
        "precommitment": precommitment,
    }
    write_json(out_dir / "manifest.json", manifest)

    print("=" * 130)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 10 EARLY-HORIZON LOCAL FEASIBILITY AUDIT"
    )
    print("=" * 130)
    print(f"device                  : {device}")
    print(f"seeds                   : {pf['seeds']}")
    print(
        f"REFLECT replay window   : {PROBE_START}..{PROBE_END}"
    )
    print(f"probe every             : {PROBE_INTERVAL} epochs")
    print(f"line-scan alphas        : {list(ALPHAS)}")
    print(
        "question                 : curvature/step-size vs direction conflict"
    )
    print(f"Stage-3 SHA256          : {pf['stage3_sha256']}")
    print(f"Stage-9 SHA256          : {pf['stage9_sha256']}")
    print("=" * 130)

    probe_rows: List[dict] = []
    scan_rows: List[dict] = []
    replay_checks: List[dict] = []
    seed_summaries: List[dict] = []

    start = time.perf_counter()

    for seed in pf["seeds"]:
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

        print()
        print("-" * 130)
        print(f"SEED {seed}")

        earliest_failure_epoch = -1
        earliest_failure_class = None
        earliest_failure_safe_alpha = None

        last_step_diag = None

        for epoch in range(PROBE_START, PROBE_END + 1):
            # Verify exact Stage-9 replay on the existing 25-epoch grid.
            if epoch > PROBE_START and epoch % 25 == 0:
                state = stage9.state_metrics(exp)

                check = verify_stage9_state(
                    seed=seed,
                    epoch=epoch,
                    state=state,
                    last_step_diag=last_step_diag,
                    ref_map=ref_map,
                    tolerance=1.0e-10,
                )

                replay_checks.append(check)

                if not check["pass"]:
                    raise RuntimeError(
                        f"Stage-9 replay drift seed={seed}, epoch={epoch}: "
                        f"{check['max_abs_difference']:.3e}"
                    )

            is_probe = (
                (epoch - PROBE_START) % PROBE_INTERVAL == 0
            )

            if is_probe:
                geo = predict_candidate_geometry(exp)

                line_rows, scan_summary = exact_line_scan(exp, geo)

                local_class = classify_local(
                    geo=geo,
                    scan_summary=scan_summary,
                )

                row = {
                    "seed": seed,
                    "epoch": epoch,
                    "intervention_active": geo["active"],
                    "local_class": local_class,
                    "candidate_update_norm":
                        geo["candidate_norm"],
                    "reflected_update_norm":
                        geo["reflected_norm"],
                    "target_directional_derivative":
                        geo["target_directional_derivative"],
                    "total_directional_derivative":
                        geo["total_directional_derivative"],
                    "target_gradient_norm":
                        geo["target_gradient_norm"],
                    "total_gradient_norm":
                        geo["total_gradient_norm"],
                    **scan_summary,
                }

                probe_rows.append(row)

                for sr in line_rows:
                    scan_rows.append(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "intervention_active":
                                geo["active"],
                            "local_class": local_class,
                            **sr,
                        }
                    )

                if (
                    geo["active"]
                    and local_class != "SAFE_FULL"
                    and earliest_failure_epoch < 0
                ):
                    earliest_failure_epoch = epoch
                    earliest_failure_class = local_class
                    earliest_failure_safe_alpha = (
                        scan_summary["max_joint_safe_alpha"]
                    )

                print(
                    f"epoch={epoch} | "
                    f"active={geo['active']} | "
                    f"class={local_class:24s} | "
                    f"dL={geo['total_directional_derivative']:+.3e} | "
                    f"dT={geo['target_directional_derivative']:+.3e} | "
                    f"alpha_safe_max="
                    f"{scan_summary['max_joint_safe_alpha']:.5f}"
                )

            if epoch == PROBE_END:
                break

            # Advance the exact Stage-9 REFLECT trajectory.
            last_step_diag = stage9.intervention_step(
                exp=exp,
                branch="REFLECT",
                target_mode=TARGET_MODE,
            )

        summary = {
            "seed": seed,
            "earliest_active_non_SAFE_FULL_epoch":
                earliest_failure_epoch,
            "earliest_failure_class":
                earliest_failure_class,
            "earliest_failure_max_joint_safe_alpha":
                earliest_failure_safe_alpha,
            "n_active_probes": sum(
                int(r["intervention_active"])
                for r in probe_rows
                if int(r["seed"]) == seed
            ),
            "n_SAFE_FULL": sum(
                int(r["local_class"] == "SAFE_FULL")
                for r in probe_rows
                if int(r["seed"]) == seed
            ),
            "n_CURVATURE_LIMITED": sum(
                int(r["local_class"] == "CURVATURE_LIMITED")
                for r in probe_rows
                if int(r["seed"]) == seed
            ),
            "n_DIRECTION_CONFLICT": sum(
                int(r["local_class"] == "DIRECTION_CONFLICT")
                for r in probe_rows
                if int(r["seed"]) == seed
            ),
            "n_UNRESOLVED_HIGH_CURVATURE": sum(
                int(
                    r["local_class"]
                    == "UNRESOLVED_HIGH_CURVATURE"
                )
                for r in probe_rows
                if int(r["seed"]) == seed
            ),
            "n_INACTIVE": sum(
                int(r["local_class"] == "INACTIVE")
                for r in probe_rows
                if int(r["seed"]) == seed
            ),
        }

        seed_summaries.append(summary)
        write_json(seed_dir / "summary.json", summary)

    write_csv(out_dir / "probe_metrics.csv", probe_rows)
    write_csv(out_dir / "line_scan_metrics.csv", scan_rows)
    write_csv(out_dir / "seed_summary.csv", seed_summaries)
    write_csv(out_dir / "stage9_replay_checks.csv", replay_checks)

    curvature_earliest = sum(
        int(
            r["earliest_failure_class"]
            == "CURVATURE_LIMITED"
        )
        for r in seed_summaries
    )

    direction_earliest = sum(
        int(
            r["earliest_failure_class"]
            == "DIRECTION_CONFLICT"
        )
        for r in seed_summaries
    )

    unresolved_earliest = sum(
        int(
            r["earliest_failure_class"]
            == "UNRESOLVED_HIGH_CURVATURE"
        )
        for r in seed_summaries
    )

    no_failure = sum(
        int(r["earliest_failure_class"] is None)
        for r in seed_summaries
    )

    if curvature_earliest >= 3:
        next_route = "bounded_trust_region_reflect_pilot"
        route_class = "curvature_step_size_failure_dominant"
    elif direction_earliest >= 3:
        next_route = "joint_total_target_descent_cone_projection_audit"
        route_class = "multiobjective_direction_conflict_dominant"
    else:
        next_route = "mixed_local_geometry_audit"
        route_class = "mixed_or_unresolved_local_failure"

    decision = {
        "n_seeds": len(seed_summaries),
        "earliest_failure_curvature_limited_count":
            curvature_earliest,
        "earliest_failure_direction_conflict_count":
            direction_earliest,
        "earliest_failure_unresolved_high_curvature_count":
            unresolved_earliest,
        "no_early_failure_count": no_failure,
        "all_stage9_replay_checks_pass":
            all(bool(r["pass"]) for r in replay_checks),
        "route_class": route_class,
        "next_route": next_route,
        "interpretation_guardrail": (
            "This stage identifies the earliest local feasibility failure "
            "along the already-defined Stage-9 REFLECT trajectory. "
            "CURVATURE_LIMITED supports step-size control; "
            "DIRECTION_CONFLICT means damping alone is not the fundamental "
            "fix because REFLECT violates total-objective descent at first order."
        ),
    }
    write_json(out_dir / "decision.json", decision)

    plot_safe_alpha(
        probe_rows,
        out_dir / "max_joint_safe_alpha.png",
    )

    plot_directional_derivative(
        probe_rows,
        out_dir / "reflect_total_directional_derivative.png",
    )

    elapsed = time.perf_counter() - start

    lines = []
    lines.append("=" * 142)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 10 LOCAL FEASIBILITY SUMMARY"
    )
    lines.append("=" * 142)
    lines.append(
        "seed | earliest failure | class                      | max safe alpha | "
        "SAFE | CURV | DIR | HIGH-CURV | inactive"
    )
    lines.append("-" * 142)

    for r in seed_summaries:
        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['earliest_active_non_SAFE_FULL_epoch']):16d} | "
            f"{str(r['earliest_failure_class']):26s} | "
            f"{str(r['earliest_failure_max_joint_safe_alpha']):14s} | "
            f"{int(r['n_SAFE_FULL']):4d} | "
            f"{int(r['n_CURVATURE_LIMITED']):5d} | "
            f"{int(r['n_DIRECTION_CONFLICT']):3d} | "
            f"{int(r['n_UNRESOLVED_HIGH_CURVATURE']):9d} | "
            f"{int(r['n_INACTIVE']):8d}"
        )

    lines.append("-" * 142)
    lines.append(
        f"Stage-9 replay checks             : "
        f"{sum(int(r['pass']) for r in replay_checks)}/"
        f"{len(replay_checks)} PASS"
    )
    lines.append(
        f"earliest CURVATURE_LIMITED         : "
        f"{curvature_earliest}/5"
    )
    lines.append(
        f"earliest DIRECTION_CONFLICT        : "
        f"{direction_earliest}/5"
    )
    lines.append(
        f"earliest UNRESOLVED_HIGH_CURVATURE : "
        f"{unresolved_earliest}/5"
    )
    lines.append(
        f"no early active failure            : "
        f"{no_failure}/5"
    )
    lines.append(
        f"route class                        : "
        f"{route_class}"
    )
    lines.append(
        f"next route                         : "
        f"{next_route}"
    )
    lines.append(
        f"elapsed seconds                    : {elapsed:.2f}"
    )
    lines.append("=" * 142)
    lines.append(
        "Guardrail: do not launch a trust-region continuation unless the "
        "earliest failures actually support a step-size/curvature mechanism."
    )
    lines.append("=" * 142)

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
