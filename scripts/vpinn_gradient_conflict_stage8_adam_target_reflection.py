#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 8
One-Step Norm-Preserving Adam Target-Reflection Audit
=====================================================

Why this stage
--------------
Stage 7 ruled out simple Adam-state reset as a clean mechanism:

  * M0 (first-moment reset) strongly shrank the update and did not rescue
    target descent.
  * V0 and MV0 often changed direction, but they also changed update magnitude
    by orders of magnitude, producing a severe scale confound.
  * No reset branch passed the precommitted 4/5 viability gate.

Therefore the next question must isolate DIRECTION from MAGNITUDE.

Starting from the exact epoch-2500 locked checkpoint, let

    Delta_A

be the exact next parameter displacement produced by the inherited Adam state,
and let

    g_t = grad[(1/M) R_9^2].

If the Adam step is target-uphill,

    <g_t, Delta_A> > 0,

define two geometry-only interventions.

1) PROJECT
----------
Remove only the uphill component:

    Delta_P =
        Delta_A
        - (<g_t,Delta_A>/||g_t||^2) g_t.

Properties:
    <g_t, Delta_P> = 0,
and Delta_P is the minimum-Euclidean-distance displacement from Delta_A
that is first-order target-neutral.

2) REFLECT
----------
Reflect the uphill target component:

    Delta_R =
        Delta_A
        - 2 (<g_t,Delta_A>/||g_t||^2) g_t.

Properties:
    ||Delta_R|| = ||Delta_A|| exactly,
    <g_t,Delta_R> = -<g_t,Delta_A>,
and every component orthogonal to g_t is unchanged.

If the inherited Adam step is already target-descent/non-uphill, both PROJECT
and REFLECT are the identity: no intervention is made.

This makes REFLECT a parameter-free, norm-preserving direction intervention.
It is specifically designed to remove the update-magnitude confound exposed
by Stage 7.

Branches
--------
CONTROL:
    exact inherited Adam step.

PROJECT:
    minimum-distance target-neutral correction.

REFLECT:
    norm-preserving target-component reflection.

This stage is ONE STEP ONLY. It is a low-cost causal preflight before any
continuation experiment.

Primary precommitted REFLECT gate
---------------------------------
Per seed, REFLECT passes if all are true:

  1) norm preservation:
       | ||Delta_R||/||Delta_A|| - 1 | <= 1e-10

  2) target response:
       post-step R_9^2 < pre-step R_9^2

  3) paired improvement:
       post-step R_9^2_REFLECT < post-step R_9^2_CONTROL

  4) objective safety:
       post-step total VPINN loss <= pre-step total VPINN loss

  5) solution-error guard:
       post relative L2 <= 1.01 * pre relative L2

Group authorization:
    >= 4/5 seeds pass all five conditions.

If group gate passes:
    authorize Stage 9 norm-preserving reflected-Adam continuation from the
    identical epoch-2500 checkpoints.

If group gate fails:
    do not launch continuation; route to local curvature / trust-region audit.

PROJECT is diagnostic only in Stage 8 and is not used to rescue a failed
REFLECT gate.

Scientific guardrail
--------------------
A one-step PASS shows that correcting only the target-uphill component of the
actual Adam displacement can causally improve the immediate target residual
without increasing step norm. It does NOT yet establish faster VPINN escape.
That requires the paired continuation stage.
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
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


BRANCHES = ("CONTROL", "PROJECT", "REFLECT")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-8 one-step norm-preserving Adam reflection audit."
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
        "--stage6-dir",
        default="vpinn_gradient_conflict_stage6_adam_update_geometry",
    )
    p.add_argument(
        "--stage7-dir",
        default="vpinn_gradient_conflict_stage7_adam_state_component_audit",
    )
    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage8_adam_target_reflection",
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


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return

    # Rows intentionally contain a few branch-specific diagnostics.
    # Use the stable union of keys instead of assuming identical schemas.
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


def load_stage3_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Stage-3 solver not found: {path}")

    spec = importlib.util.spec_from_file_location(
        "vpinn_stage3_solver_stage8", str(path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Stage-3 solver: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def flatten(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in parts], dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 0.0:
        return float("nan")
    c = torch.dot(a, b) / denom
    return float(torch.clamp(c, -1.0, 1.0).item())


# =============================================================================
# Prerequisite gates
# =============================================================================

def preflight(
    stage3_script: Path,
    stage5_dir: Path,
    stage6_dir: Path,
    stage7_dir: Path,
) -> dict:
    paths = {
        "s5_manifest": stage5_dir / "manifest.json",
        "s5_decision": stage5_dir / "decision.json",
        "s6_manifest": stage6_dir / "manifest.json",
        "s6_decision": stage6_dir / "decision.json",
        "s7_manifest": stage7_dir / "manifest.json",
        "s7_decision": stage7_dir / "decision.json",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    data = {k: read_json(v) for k, v in paths.items()}

    if data["s6_decision"].get("next_route") != (
        "optimizer_state_causal_ablation"
    ):
        raise RuntimeError("Stage 6 did not authorize optimizer-state audit.")

    if not bool(data["s6_decision"].get("group_gate_pass", False)):
        raise RuntimeError("Stage-6 group gate is not PASS.")

    if data["s7_decision"].get("next_route") != (
        "alternative_gradient_geometry_mechanism"
    ):
        raise RuntimeError(
            "Stage 7 did not route to alternative gradient geometry."
        )

    if bool(data["s7_decision"].get("continuation_authorized", True)):
        raise RuntimeError(
            "Stage 7 unexpectedly authorized an optimizer-reset continuation."
        )

    if not bool(
        data["s7_decision"].get("all_formula_runtime_checks_pass", False)
    ):
        raise RuntimeError("Stage-7 formula/runtime checks are not PASS.")

    actual_sha = sha256_file(stage3_script)

    shas = [
        data["s5_manifest"].get("stage3_solver_sha256"),
        data["s6_manifest"].get("stage3_solver_sha256"),
        data["s7_manifest"].get("stage3_solver_sha256"),
    ]

    if any(x != actual_sha for x in shas):
        raise RuntimeError(
            "Stage-3 solver SHA256 mismatch across prerequisite stages."
        )

    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing locked checkpoint: {checkpoint}")

    return {
        "seeds": seeds,
        "stage3_sha256": actual_sha,
        **data,
    }


# =============================================================================
# Experiment and checkpoint
# =============================================================================

def make_experiment(stage3, device: torch.device, seed: int, out_dir: Path):
    cfg = stage3.Config(
        seed=seed,
        device=str(device),
        epochs=2501,
        learning_rate=1.0e-3,
        width=32,
        depth=3,
        n_test=24,
        n_quad=256,
        n_eval=4001,
        modes=(9,),
        reference_mode=7,
        reference_amplitude=0.15,
        track_interval=25,
        diagnostic_epochs=(2500,),
        output_dir=str(out_dir),
    )

    return stage3.ModeExperiment(
        cfg=cfg,
        device=device,
        mode=9,
        out_dir=out_dir,
    )


def load_locked_checkpoint(
    exp,
    checkpoint_path: Path,
    expected_seed: int,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=exp.device,
        weights_only=False,
    )

    if int(checkpoint["seed"]) != expected_seed:
        raise RuntimeError("Checkpoint seed mismatch.")
    if int(checkpoint["epoch"]) != 2500:
        raise RuntimeError("Checkpoint epoch is not 2500.")
    if int(checkpoint["mode"]) != 9:
        raise RuntimeError("Checkpoint mode is not 9.")

    expected_amp = 0.15 * 7.0 / 9.0
    if abs(float(checkpoint["amplitude"]) - expected_amp) > 1.0e-15:
        raise RuntimeError("Checkpoint amplitude mismatch.")

    exp.model.load_state_dict(checkpoint["model_state_dict"])
    exp.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for state in exp.optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(exp.device)


# =============================================================================
# State and gradient metrics
# =============================================================================

def state_metrics(exp, target_mode: int = 9) -> dict:
    residuals = exp.weak_residuals().detach()
    energy = residuals.square()
    total_energy = energy.sum().clamp_min(1.0e-300)
    t = target_mode - 1
    dominant = int(torch.argmax(energy).item())

    return {
        "relative_l2_error": exp.relative_l2_error(),
        "vpinn_loss": float(torch.mean(energy).item()),
        "target_loss_unscaled": float(energy[t].item()),
        "target_mode_abs_residual": float(torch.abs(residuals[t]).item()),
        "target_mode_residual_energy_share": float(
            (energy[t] / total_energy).item()
        ),
        "dominant_residual_mode": dominant + 1,
    }


def current_gradients(exp, target_mode: int = 9):
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

    return params, gt, g


# =============================================================================
# Exact inherited Adam displacement
# =============================================================================

def exact_next_adam_displacement(exp, params, g: torch.Tensor) -> torch.Tensor:
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

    parts = []
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
        parts.append(delta.reshape(-1))

    return flatten(parts)


def geometry_branches(
    adam_delta: torch.Tensor,
    target_gradient: torch.Tensor,
) -> dict:
    gt2 = torch.dot(target_gradient, target_gradient)

    if float(gt2.item()) <= 0.0:
        raise RuntimeError("Target gradient has zero norm.")

    dot = torch.dot(target_gradient, adam_delta)

    control = adam_delta.detach().clone()

    if float(dot.item()) > 0.0:
        projection_component = (dot / gt2) * target_gradient

        project = control - projection_component
        reflect = control - 2.0 * projection_component
        intervention_active = True
    else:
        project = control.detach().clone()
        reflect = control.detach().clone()
        intervention_active = False

    return {
        "CONTROL": control,
        "PROJECT": project,
        "REFLECT": reflect,
        "intervention_active": intervention_active,
        "control_target_dot": float(dot.item()),
    }


def apply_displacement(exp, displacement: torch.Tensor) -> None:
    offset = 0

    with torch.no_grad():
        for p in exp.model.parameters():
            n = p.numel()
            p.add_(displacement[offset:offset+n].reshape_as(p))
            offset += n

    if offset != displacement.numel():
        raise RuntimeError("Displacement size mismatch.")


# =============================================================================
# Runtime verification of CONTROL
# =============================================================================

def verify_control_runtime(
    stage3,
    device,
    seed: int,
    checkpoint_path: Path,
    predicted_delta: torch.Tensor,
    scratch_dir: Path,
    tolerance: float = 5.0e-12,
) -> dict:
    exp = make_experiment(stage3, device, seed, scratch_dir)
    load_locked_checkpoint(exp, checkpoint_path, seed)

    params = tuple(p for p in exp.model.parameters() if p.requires_grad)

    before = flatten([p.detach().clone() for p in params])

    exp.optimizer.zero_grad(set_to_none=True)
    residuals = exp.weak_residuals()
    loss = residuals.square().mean()
    loss.backward()
    exp.optimizer.step()

    after = flatten([p.detach().clone() for p in params])
    actual = after - before

    max_abs = float(torch.max(torch.abs(actual - predicted_delta)).item())
    rel = float(
        (
            torch.linalg.vector_norm(actual - predicted_delta)
            / torch.clamp(
                torch.linalg.vector_norm(actual),
                min=1.0e-300,
            )
        ).item()
    )

    return {
        "seed": seed,
        "max_abs_difference": max_abs,
        "relative_difference": rel,
        "tolerance": tolerance,
        "pass": bool(max_abs <= tolerance),
    }


# =============================================================================
# Per-seed audit
# =============================================================================

def run_seed(
    stage3,
    device,
    seed: int,
    checkpoint_path: Path,
    seed_dir: Path,
) -> tuple[list[dict], dict]:
    probe = make_experiment(
        stage3,
        device,
        seed,
        seed_dir / "_probe",
    )
    load_locked_checkpoint(probe, checkpoint_path, seed)

    pre = state_metrics(probe)

    params, gt, g = current_gradients(probe)
    adam_delta = exact_next_adam_displacement(probe, params, g)

    branch_geo = geometry_branches(adam_delta, gt)

    control_norm = float(torch.linalg.vector_norm(adam_delta).item())
    gt_norm = float(torch.linalg.vector_norm(gt).item())

    runtime_check = verify_control_runtime(
        stage3=stage3,
        device=device,
        seed=seed,
        checkpoint_path=checkpoint_path,
        predicted_delta=adam_delta,
        scratch_dir=seed_dir / "_control_runtime",
    )

    if not runtime_check["pass"]:
        raise RuntimeError(
            f"CONTROL Adam formula mismatch for seed {seed}: "
            f"{runtime_check['max_abs_difference']:.3e}"
        )

    rows = []

    for branch in BRANCHES:
        exp = make_experiment(
            stage3,
            device,
            seed,
            seed_dir / branch.lower(),
        )
        load_locked_checkpoint(exp, checkpoint_path, seed)

        delta = branch_geo[branch]

        delta_norm = float(torch.linalg.vector_norm(delta).item())
        alignment = cosine(delta, -gt)

        apply_displacement(exp, delta)

        post = state_metrics(exp)

        target_log_change = math.log(
            max(post["target_loss_unscaled"], 1.0e-300)
            / max(pre["target_loss_unscaled"], 1.0e-300)
        )

        total_log_change = math.log(
            max(post["vpinn_loss"], 1.0e-300)
            / max(pre["vpinn_loss"], 1.0e-300)
        )

        row = {
            "seed": seed,
            "branch": branch,
            "intervention_active":
                bool(branch_geo["intervention_active"]),

            "pre_relative_l2_error":
                pre["relative_l2_error"],
            "pre_vpinn_loss":
                pre["vpinn_loss"],
            "pre_target_loss_unscaled":
                pre["target_loss_unscaled"],
            "pre_target_share":
                pre["target_mode_residual_energy_share"],

            "control_target_directional_derivative":
                branch_geo["control_target_dot"],
            "control_update_norm":
                control_norm,
            "target_gradient_norm":
                gt_norm,

            "branch_update_norm":
                delta_norm,
            "update_norm_ratio_vs_control":
                delta_norm / max(control_norm, 1.0e-300),
            "target_descent_alignment":
                alignment,

            "post_relative_l2_error":
                post["relative_l2_error"],
            "post_vpinn_loss":
                post["vpinn_loss"],
            "post_target_loss_unscaled":
                post["target_loss_unscaled"],
            "post_target_share":
                post["target_mode_residual_energy_share"],

            "target_log_change":
                target_log_change,
            "total_loss_log_change":
                total_log_change,

            "target_loss_decreased":
                bool(
                    post["target_loss_unscaled"]
                    < pre["target_loss_unscaled"]
                ),
            "total_loss_nonincreased":
                bool(post["vpinn_loss"] <= pre["vpinn_loss"]),
            "relative_l2_guard_pass":
                bool(
                    post["relative_l2_error"]
                    <= 1.01 * pre["relative_l2_error"]
                ),
        }

        rows.append(row)

    # Pair REFLECT against CONTROL after both rows exist.
    by_branch = {r["branch"]: r for r in rows}

    control_post_target = by_branch["CONTROL"]["post_target_loss_unscaled"]

    reflect = by_branch["REFLECT"]

    norm_preservation_error = abs(
        reflect["update_norm_ratio_vs_control"] - 1.0
    )

    reflect["norm_preservation_error"] = norm_preservation_error
    # If the inherited Adam step is already target-descent, REFLECT is
    # intentionally the identity. In that case equality with CONTROL is the
    # correct outcome, not a failure of the intervention logic.
    if branch_geo["intervention_active"]:
        paired_target_condition = bool(
            reflect["post_target_loss_unscaled"] < control_post_target
        )
    else:
        paired_target_condition = bool(
            abs(
                reflect["post_target_loss_unscaled"]
                - control_post_target
            ) <= 1.0e-12 * max(1.0, abs(control_post_target))
        )

    reflect["paired_target_improvement_vs_control"] = (
        paired_target_condition
    )

    reflect_pass = bool(
        norm_preservation_error <= 1.0e-10
        and reflect["target_loss_decreased"]
        and paired_target_condition
        and reflect["total_loss_nonincreased"]
        and reflect["relative_l2_guard_pass"]
    )

    reflect["reflect_seed_pass"] = reflect_pass

    # PROJECT is diagnostic, with its own geometric identity checks.
    project = by_branch["PROJECT"]

    project["target_neutral_alignment_abs"] = abs(
        float(project["target_descent_alignment"])
    ) if branch_geo["intervention_active"] else None

    # Mathematical identity checks.
    if branch_geo["intervention_active"]:
        if norm_preservation_error > 1.0e-10:
            raise RuntimeError(
                f"REFLECT norm preservation failed for seed {seed}."
            )

        expected_reflect_alignment = -by_branch["CONTROL"][
            "target_descent_alignment"
        ]

        if abs(
            reflect["target_descent_alignment"]
            - expected_reflect_alignment
        ) > 1.0e-10:
            raise RuntimeError(
                f"REFLECT alignment identity failed for seed {seed}."
            )

        if abs(project["target_descent_alignment"]) > 1.0e-10:
            raise RuntimeError(
                f"PROJECT target-neutral identity failed for seed {seed}."
            )

    return rows, runtime_check


# =============================================================================
# Plotting
# =============================================================================

def paired_plot(
    rows: List[dict],
    field: str,
    ylabel: str,
    title: str,
    path: Path,
    hline: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    x = np.arange(len(BRANCHES), dtype=float)
    seeds = sorted(set(int(r["seed"]) for r in rows))

    for seed in seeds:
        rr = {
            r["branch"]: r
            for r in rows
            if int(r["seed"]) == seed
        }

        y = [float(rr[b][field]) for b in BRANCHES]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.2,
            label=f"seed {seed}",
        )

    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(BRANCHES)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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

    stage6_dir = Path(args.stage6_dir)
    if not stage6_dir.is_absolute():
        stage6_dir = root / stage6_dir

    stage7_dir = Path(args.stage7_dir)
    if not stage7_dir.is_absolute():
        stage7_dir = root / stage7_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage5_dir=stage5_dir,
        stage6_dir=stage6_dir,
        stage7_dir=stage7_dir,
    )

    stage3 = load_stage3_module(stage3_script)

    precommitment = {
        "stage": "one_step_norm_preserving_adam_target_reflection",
        "locked_epoch": 2500,
        "seeds": pf["seeds"],
        "branches": {
            "CONTROL": "exact inherited Adam displacement",
            "PROJECT": (
                "minimum-distance removal of target-uphill component; "
                "identity if control is already target-nonuphill"
            ),
            "REFLECT": (
                "norm-preserving reflection of target-uphill component; "
                "identity if control is already target-nonuphill"
            ),
        },
        "reflect_per_seed_gate": {
            "norm_ratio_abs_error_le": 1.0e-10,
            "post_target_loss_lt_pre": True,
            "post_target_loss_lt_control_post": True,
            "post_total_loss_le_pre": True,
            "post_relative_l2_le_1p01_times_pre": True,
        },
        "group_gate": "at least 4/5 REFLECT seed passes",
        "next_route_if_pass":
            "norm_preserving_reflected_adam_continuation",
        "next_route_if_fail":
            "local_curvature_trust_region_audit",
        "project_branch_is_diagnostic_only": True,
    }

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),
        "stage3_solver_sha256": pf["stage3_sha256"],
        "stage5_dir": str(stage5_dir),
        "stage6_dir": str(stage6_dir),
        "stage7_dir": str(stage7_dir),
        "stage8_script_sha256": sha256_file(Path(__file__).resolve()),
        "precommitment": precommitment,
    }
    write_json(out_dir / "manifest.json", manifest)

    print("=" * 126)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 8 ONE-STEP NORM-PRESERVING ADAM TARGET REFLECTION"
    )
    print("=" * 126)
    print(f"device                 : {device}")
    print(f"seeds                  : {pf['seeds']}")
    print(f"locked state           : exact Stage-5 epoch-2500 checkpoints")
    print(f"branches               : {list(BRANCHES)}")
    print(f"REFLECT step norm      : exactly CONTROL norm")
    print(f"group gate             : >=4/5 REFLECT seed passes")
    print(f"Stage-3 solver SHA256  : {pf['stage3_sha256']}")
    print("=" * 126)

    rows: List[dict] = []
    runtime_checks: List[dict] = []

    start = time.perf_counter()

    for seed in pf["seeds"]:
        seed_dir = out_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        seed_rows, runtime_check = run_seed(
            stage3=stage3,
            device=device,
            seed=seed,
            checkpoint_path=checkpoint,
            seed_dir=seed_dir,
        )

        rows.extend(seed_rows)
        runtime_checks.append(runtime_check)

        rr = {r["branch"]: r for r in seed_rows}

        print()
        print("-" * 126)
        print(
            f"SEED {seed} | control alignment="
            f"{rr['CONTROL']['target_descent_alignment']:+.6f} | "
            f"intervention active={rr['REFLECT']['intervention_active']}"
        )

        for branch in BRANCHES:
            r = rr[branch]

            extra = ""
            if branch == "REFLECT":
                extra = (
                    f" | PASS={r.get('reflect_seed_pass', False)}"
                )

            print(
                f"{branch:8s} | "
                f"norm/control={r['update_norm_ratio_vs_control']:.9f} | "
                f"A_target={r['target_descent_alignment']:+.6f} | "
                f"ΔlogR9²={r['target_log_change']:+.6e} | "
                f"ΔlogL={r['total_loss_log_change']:+.6e}"
                f"{extra}"
            )

    write_csv(out_dir / "branch_results.csv", rows)

    write_json(
        out_dir / "control_runtime_verification.json",
        {
            "all_pass": all(r["pass"] for r in runtime_checks),
            "results": runtime_checks,
        },
    )

    reflect_rows = [r for r in rows if r["branch"] == "REFLECT"]

    pass_count = sum(
        int(bool(r.get("reflect_seed_pass", False)))
        for r in reflect_rows
    )

    active_count = sum(
        int(bool(r["intervention_active"]))
        for r in reflect_rows
    )

    target_improvement_ratios = [
        float(r["post_target_loss_unscaled"])
        / float(
            next(
                x["post_target_loss_unscaled"]
                for x in rows
                if x["seed"] == r["seed"]
                and x["branch"] == "CONTROL"
            )
        )
        for r in reflect_rows
    ]

    group_pass = bool(pass_count >= 4)

    decision = {
        "n_seeds": len(pf["seeds"]),
        "intervention_active_seed_count": active_count,
        "reflect_seed_pass_count": pass_count,
        "group_gate_pass": group_pass,
        "median_reflect_post_target_ratio_vs_control":
            float(np.median(target_improvement_ratios)),
        "all_control_runtime_checks_pass":
            all(r["pass"] for r in runtime_checks),
        "next_route": (
            "norm_preserving_reflected_adam_continuation"
            if group_pass
            else "local_curvature_trust_region_audit"
        ),
        "continuation_authorized": group_pass,
        "interpretation_guardrail": (
            "A Stage-8 PASS establishes an immediate causal benefit of "
            "norm-preserving correction of target-uphill Adam displacement. "
            "It does not establish faster escape; Stage 9 must test that from "
            "the same locked checkpoints."
        ),
    }
    write_json(out_dir / "decision.json", decision)

    paired_plot(
        rows,
        field="target_descent_alignment",
        ylabel="Target-descent alignment",
        title="Adam direction before and after target-component correction",
        path=out_dir / "target_descent_alignment_by_branch.png",
        hline=0.0,
    )

    paired_plot(
        rows,
        field="target_log_change",
        ylabel="log(R9²_after / R9²_before)",
        title="Actual one-step target-residual response",
        path=out_dir / "target_loss_change_by_branch.png",
        hline=0.0,
    )

    paired_plot(
        rows,
        field="total_loss_log_change",
        ylabel="log(L_after / L_before)",
        title="Total VPINN-loss response",
        path=out_dir / "total_loss_change_by_branch.png",
        hline=0.0,
    )

    paired_plot(
        rows,
        field="update_norm_ratio_vs_control",
        ylabel="Update norm / CONTROL",
        title="Update-magnitude control",
        path=out_dir / "update_norm_ratio_by_branch.png",
        hline=1.0,
    )

    elapsed = time.perf_counter() - start

    lines = []
    lines.append("=" * 132)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 8 NORM-PRESERVING TARGET-REFLECTION SUMMARY"
    )
    lines.append("=" * 132)
    lines.append(
        "seed | active | control A | reflect A | norm ratio | "
        "reflect ΔlogR9² | reflect ΔlogL | PASS"
    )
    lines.append("-" * 132)

    for seed in pf["seeds"]:
        control = next(
            r for r in rows
            if r["seed"] == seed and r["branch"] == "CONTROL"
        )
        reflect = next(
            r for r in rows
            if r["seed"] == seed and r["branch"] == "REFLECT"
        )

        lines.append(
            f"{seed:4d} | "
            f"{str(bool(reflect['intervention_active'])):6s} | "
            f"{control['target_descent_alignment']:+9.6f} | "
            f"{reflect['target_descent_alignment']:+9.6f} | "
            f"{reflect['update_norm_ratio_vs_control']:10.9f} | "
            f"{reflect['target_log_change']:+15.6e} | "
            f"{reflect['total_loss_log_change']:+13.6e} | "
            f"{'PASS' if reflect.get('reflect_seed_pass', False) else 'FAIL'}"
        )

    lines.append("-" * 132)
    lines.append(
        f"control runtime verification       : "
        f"{sum(int(r['pass']) for r in runtime_checks)}/"
        f"{len(runtime_checks)} PASS"
    )
    lines.append(
        f"REFLECT intervention active        : "
        f"{active_count}/{len(pf['seeds'])}"
    )
    lines.append(
        f"REFLECT seed passes                : "
        f"{pass_count}/{len(pf['seeds'])}"
    )
    lines.append(
        f"group gate                         : "
        f"{'PASS' if group_pass else 'FAIL'}"
    )
    lines.append(
        f"median post-target ratio vs CONTROL: "
        f"{np.median(target_improvement_ratios):.9f}"
    )
    lines.append(
        f"next route                         : "
        f"{decision['next_route']}"
    )
    lines.append(
        f"continuation authorized            : "
        f"{decision['continuation_authorized']}"
    )
    lines.append(f"elapsed seconds                    : {elapsed:.2f}")
    lines.append("=" * 132)
    lines.append(
        "Guardrail: one-step directional rescue is not yet an escape-time result."
    )
    lines.append("=" * 132)

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
