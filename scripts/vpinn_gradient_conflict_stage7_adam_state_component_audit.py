#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 7
One-Step Adam-State Component Causal Audit
==========================================

Scientific motivation
---------------------
Stage 6 showed, for all five seeds:

    Adam target-descent unlock
        precedes
    residual release
        precedes
    certified escape.

That temporal ordering is highly reproducible, but it does not identify WHICH
part of Adam's state is responsible.

At epoch 2500, branch from the exact SAME:
    * neural-network parameters,
    * PDE/test-space state,
    * optimizer state,

and change only selected Adam-state components for ONE step.

Branches
--------
CONTROL:
    original Adam state

M0:
    exp_avg <- 0
    exp_avg_sq unchanged
    step unchanged

V0:
    exp_avg unchanged
    exp_avg_sq <- 0
    step unchanged

MV0:
    exp_avg <- 0
    exp_avg_sq <- 0
    step unchanged

Keeping `step` unchanged is deliberate. We test accumulated first/second moment
memory without introducing an additional bias-correction/time-reset confound.

For each seed/branch we:
  1) predict the exact next Adam parameter displacement;
  2) measure target-descent alignment of that displacement;
  3) apply exactly one real PyTorch Adam step;
  4) verify predicted vs actual displacement;
  5) measure the actual one-step change in R_9^2 and total VPINN loss.

Primary branch-viability criteria (per seed)
--------------------------------------------
A non-control branch is "viable" if all are true:
  * predicted target-descent alignment > 0;
  * actual R_9^2 decreases after the step;
  * actual post-step R_9^2 is lower than CONTROL post-step R_9^2;
  * update-norm ratio vs CONTROL is in [0.25, 4.0].

Group eligibility:
    >= 4/5 seeds viable.

Route selection
---------------
1) Eligible single-component branches (M0, V0) are preferred over MV0.
2) If both M0 and V0 are eligible, select the one with smaller
   median |log(update_norm_ratio_vs_control)|, i.e. the smaller scale
   perturbation relative to CONTROL.
3) If neither single-component branch is eligible but MV0 is, select MV0.
4) If no branch is eligible, do NOT launch a long continuation. Route back to
   gradient-level / alternative mechanism analysis.

Why one step first?
-------------------
A multi-hundred-epoch ablation before this audit would waste compute and add
interpretive ambiguity. This stage is a bounded causal route-selection test.

This is NOT yet an escape-time intervention study.
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
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


BRANCHES = ("CONTROL", "M0", "V0", "MV0")
NONCONTROL = ("M0", "V0", "MV0")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-7 one-step Adam-state component causal audit."
    )
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument(
        "--stage3-script",
        type=str,
        default="vpinn_gradient_conflict_stage3_frequency_transfer.py",
    )
    p.add_argument(
        "--stage5-dir",
        type=str,
        default="vpinn_gradient_conflict_stage5_escape_time",
    )
    p.add_argument(
        "--stage6-dir",
        type=str,
        default="vpinn_gradient_conflict_stage6_adam_update_geometry",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="vpinn_gradient_conflict_stage7_adam_state_component_audit",
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


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_stage3_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Stage-3 solver not found: {path}")

    spec = importlib.util.spec_from_file_location(
        "vpinn_stage3_solver", str(path)
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
    value = torch.dot(a, b) / denom
    return float(torch.clamp(value, -1.0, 1.0).item())


# =============================================================================
# Preflight
# =============================================================================

def preflight(stage3_script: Path, stage5_dir: Path, stage6_dir: Path) -> dict:
    s5_manifest_path = stage5_dir / "manifest.json"
    s5_decision_path = stage5_dir / "decision.json"
    s6_manifest_path = stage6_dir / "manifest.json"
    s6_decision_path = stage6_dir / "decision.json"
    s6_formula_path = stage6_dir / "adam_formula_verification.json"

    required = [
        s5_manifest_path,
        s5_decision_path,
        s6_manifest_path,
        s6_decision_path,
        s6_formula_path,
    ]

    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    s5_manifest = read_json(s5_manifest_path)
    s5_decision = read_json(s5_decision_path)
    s6_manifest = read_json(s6_manifest_path)
    s6_decision = read_json(s6_decision_path)
    s6_formula = read_json(s6_formula_path)

    if s6_decision.get("next_route") != "optimizer_state_causal_ablation":
        raise RuntimeError(
            "Stage 6 did not authorize optimizer-state causal ablation."
        )

    if not bool(s6_decision.get("group_gate_pass", False)):
        raise RuntimeError("Stage-6 group gate is not PASS.")

    if not bool(s6_decision.get("all_historical_replay_checks_pass", False)):
        raise RuntimeError("Stage-6 historical replay checks are not all PASS.")

    if not bool(s6_formula.get("all_pass", False)):
        raise RuntimeError("Stage-6 Adam formula verification is not all PASS.")

    actual_stage3_sha = sha256_file(stage3_script)
    s5_sha = s5_manifest.get("stage3_solver_sha256")
    s6_sha = s6_manifest.get("stage3_solver_sha256")

    if not (actual_stage3_sha == s5_sha == s6_sha):
        raise RuntimeError(
            "Stage-3 solver identity mismatch across Stage 5 / Stage 6 / current."
        )

    seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        ckpt = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing locked checkpoint: {ckpt}")

    return {
        "seeds": seeds,
        "stage3_sha256": actual_stage3_sha,
        "stage5_manifest": s5_manifest,
        "stage6_manifest": s6_manifest,
        "stage6_decision": s6_decision,
    }


# =============================================================================
# Experiment and checkpoint
# =============================================================================

def make_experiment(stage3, device, seed: int, out_dir: Path):
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
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=exp.device,
        weights_only=False,
    )

    if int(checkpoint["seed"]) != expected_seed:
        raise RuntimeError("Checkpoint seed mismatch.")
    if int(checkpoint["epoch"]) != 2500:
        raise RuntimeError("Checkpoint is not epoch 2500.")
    if int(checkpoint["mode"]) != 9:
        raise RuntimeError("Checkpoint mode is not 9.")

    exp.model.load_state_dict(checkpoint["model_state_dict"])
    exp.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for state in exp.optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(exp.device)

    return checkpoint


def apply_branch_state_ablation(exp, branch: str) -> None:
    if branch not in BRANCHES:
        raise ValueError(f"Unknown branch: {branch}")

    if branch == "CONTROL":
        return

    reset_m = branch in ("M0", "MV0")
    reset_v = branch in ("V0", "MV0")

    for p in exp.model.parameters():
        state = exp.optimizer.state[p]

        if reset_m:
            if "exp_avg" not in state:
                raise RuntimeError("Adam exp_avg missing.")
            state["exp_avg"].zero_()

        if reset_v:
            if "exp_avg_sq" not in state:
                raise RuntimeError("Adam exp_avg_sq missing.")
            state["exp_avg_sq"].zero_()

        # IMPORTANT: state["step"] is deliberately preserved.


# =============================================================================
# Metrics
# =============================================================================

def state_metrics(exp, target_mode: int = 9) -> dict:
    residuals = exp.weak_residuals().detach()
    energy = residuals.square()
    total = energy.sum().clamp_min(1.0e-300)
    t = target_mode - 1
    dominant = int(torch.argmax(energy).item())

    return {
        "relative_l2_error": exp.relative_l2_error(),
        "vpinn_loss": float(torch.mean(energy).item()),
        "target_loss_unscaled": float(energy[t].item()),
        "target_mode_abs_residual": float(torch.abs(residuals[t]).item()),
        "target_mode_residual_energy_share": float(
            (energy[t] / total).item()
        ),
        "dominant_residual_mode": dominant + 1,
    }


def exact_gradients(exp, target_mode: int = 9):
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

    return params, gt_parts, g_parts


def predict_adam_update(exp, params, g_parts):
    """
    Exact next PyTorch Adam displacement for current optimizer state.
    Assumes Stage-3 default Adam: weight_decay=0, amsgrad=False, maximize=False.
    """
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

    delta_parts = []

    for p, gp in zip(params, g_parts):
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

        delta_parts.append(delta.reshape(-1))

    return flatten(delta_parts)


def run_one_step_branch(
    stage3,
    device,
    seed: int,
    branch: str,
    checkpoint_path: Path,
    branch_dir: Path,
) -> dict:
    exp = make_experiment(
        stage3=stage3,
        device=device,
        seed=seed,
        out_dir=branch_dir,
    )

    load_locked_checkpoint(
        exp,
        checkpoint_path,
        expected_seed=seed,
    )

    apply_branch_state_ablation(exp, branch)

    pre = state_metrics(exp)

    params, gt_parts, g_parts = exact_gradients(exp)
    gt = flatten(gt_parts).detach()
    g = flatten(g_parts).detach()

    predicted_delta = predict_adam_update(
        exp=exp,
        params=params,
        g_parts=g_parts,
    )

    predicted_alignment = cosine(predicted_delta, -gt)
    raw_alignment = cosine(-g, -gt)

    predicted_norm = float(
        torch.linalg.vector_norm(predicted_delta).item()
    )

    before = flatten(
        [p.detach().clone() for p in params]
    )

    # Real runtime Adam step.
    exp.optimizer.zero_grad(set_to_none=True)
    residuals = exp.weak_residuals()
    loss = residuals.square().mean()

    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss before one-step branch.")

    loss.backward()
    exp.optimizer.step()

    after = flatten(
        [p.detach().clone() for p in params]
    )

    actual_delta = after - before

    update_max_abs_diff = float(
        torch.max(torch.abs(actual_delta - predicted_delta)).item()
    )

    update_rel_diff = float(
        (
            torch.linalg.vector_norm(actual_delta - predicted_delta)
            / torch.clamp(
                torch.linalg.vector_norm(actual_delta),
                min=1.0e-300,
            )
        ).item()
    )

    if update_max_abs_diff > 5.0e-12:
        raise RuntimeError(
            f"Predicted-vs-actual Adam update mismatch: "
            f"{update_max_abs_diff:.3e}"
        )

    post = state_metrics(exp)

    target_log_change = math.log(
        max(post["target_loss_unscaled"], 1.0e-300)
        / max(pre["target_loss_unscaled"], 1.0e-300)
    )

    total_log_change = math.log(
        max(post["vpinn_loss"], 1.0e-300)
        / max(pre["vpinn_loss"], 1.0e-300)
    )

    return {
        "seed": seed,
        "branch": branch,

        "pre_relative_l2_error":
            pre["relative_l2_error"],
        "pre_vpinn_loss":
            pre["vpinn_loss"],
        "pre_target_loss_unscaled":
            pre["target_loss_unscaled"],
        "pre_target_share":
            pre["target_mode_residual_energy_share"],

        "predicted_target_descent_alignment":
            predicted_alignment,
        "raw_target_descent_alignment":
            raw_alignment,
        "predicted_update_norm":
            predicted_norm,

        "actual_update_norm":
            float(torch.linalg.vector_norm(actual_delta).item()),
        "predicted_actual_update_max_abs_diff":
            update_max_abs_diff,
        "predicted_actual_update_relative_diff":
            update_rel_diff,

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
        "total_loss_decreased":
            bool(post["vpinn_loss"] < pre["vpinn_loss"]),
    }


# =============================================================================
# Decision logic
# =============================================================================

def decorate_paired_results(rows: List[dict]) -> List[dict]:
    by_seed = {}

    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[
            str(row["branch"])
        ] = row

    decorated = []

    for seed, branches in by_seed.items():
        control = branches["CONTROL"]
        control_norm = float(control["actual_update_norm"])
        control_target_post = float(
            control["post_target_loss_unscaled"]
        )

        for branch in BRANCHES:
            row = dict(branches[branch])

            norm_ratio = (
                float(row["actual_update_norm"])
                / max(control_norm, 1.0e-300)
            )

            post_target_ratio_vs_control = (
                float(row["post_target_loss_unscaled"])
                / max(control_target_post, 1.0e-300)
            )

            row["update_norm_ratio_vs_control"] = norm_ratio
            row[
                "post_target_loss_ratio_vs_control"
            ] = post_target_ratio_vs_control

            if branch == "CONTROL":
                row["viable_seed"] = None
            else:
                row["viable_seed"] = bool(
                    float(
                        row[
                            "predicted_target_descent_alignment"
                        ]
                    ) > 0.0
                    and bool(row["target_loss_decreased"])
                    and post_target_ratio_vs_control < 1.0
                    and 0.25 <= norm_ratio <= 4.0
                )

            decorated.append(row)

    return decorated


def choose_route(rows: List[dict]) -> dict:
    branch_stats = {}

    for branch in NONCONTROL:
        rr = [r for r in rows if r["branch"] == branch]

        viable_count = sum(
            int(bool(r["viable_seed"])) for r in rr
        )

        ratios = np.array(
            [float(r["update_norm_ratio_vs_control"]) for r in rr],
            dtype=float,
        )

        scale_perturbation = float(
            np.median(np.abs(np.log(np.maximum(ratios, 1.0e-300))))
        )

        alignment = np.array(
            [
                float(r["predicted_target_descent_alignment"])
                for r in rr
            ]
        )

        target_advantage = np.array(
            [
                -math.log(
                    max(
                        float(r["post_target_loss_ratio_vs_control"]),
                        1.0e-300,
                    )
                )
                for r in rr
            ]
        )

        branch_stats[branch] = {
            "viable_seed_count": viable_count,
            "eligible_group": bool(viable_count >= 4),
            "median_abs_log_update_norm_ratio":
                scale_perturbation,
            "median_target_descent_alignment":
                float(np.median(alignment)),
            "median_target_log_advantage_vs_control":
                float(np.median(target_advantage)),
        }

    eligible_single = [
        b for b in ("M0", "V0")
        if branch_stats[b]["eligible_group"]
    ]

    selected = None
    reason = None

    if eligible_single:
        selected = min(
            eligible_single,
            key=lambda b:
                branch_stats[b][
                    "median_abs_log_update_norm_ratio"
                ],
        )
        reason = (
            "eligible single-component branch with the smaller "
            "median absolute log update-norm perturbation"
        )
    elif branch_stats["MV0"]["eligible_group"]:
        selected = "MV0"
        reason = (
            "no single-component branch passed; combined moment reset passed"
        )
    else:
        selected = None
        reason = "no optimizer-state branch passed the precommitted viability gate"

    if selected == "M0":
        next_route = "first_moment_reset_continuation"
    elif selected == "V0":
        next_route = "second_moment_reset_continuation"
    elif selected == "MV0":
        next_route = "combined_moment_reset_continuation"
    else:
        next_route = "alternative_gradient_geometry_mechanism"

    return {
        "branch_stats": branch_stats,
        "selected_branch": selected,
        "selection_reason": reason,
        "next_route": next_route,
        "continuation_authorized": bool(selected is not None),
    }


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

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage5_dir=stage5_dir,
        stage6_dir=stage6_dir,
    )

    stage3 = load_stage3_module(stage3_script)

    precommitment = {
        "stage": "one_step_adam_state_component_causal_audit",
        "locked_epoch": 2500,
        "seeds": pf["seeds"],
        "branches": {
            "CONTROL": "original Adam state",
            "M0": "exp_avg=0; exp_avg_sq preserved; step preserved",
            "V0": "exp_avg preserved; exp_avg_sq=0; step preserved",
            "MV0": "exp_avg=0; exp_avg_sq=0; step preserved",
        },
        "per_seed_viability": {
            "predicted_target_descent_alignment_gt": 0.0,
            "actual_target_loss_decreases": True,
            "post_target_loss_lower_than_control": True,
            "update_norm_ratio_vs_control_in": [0.25, 4.0],
        },
        "group_eligibility": "at least 4/5 viable seeds",
        "selection_rule": (
            "prefer eligible single-component branch; if both M0 and V0 "
            "eligible choose smaller median |log(update_norm_ratio)|; "
            "otherwise use MV0 only if eligible"
        ),
        "purpose": "route selection before any long continuation",
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
        "stage7_script_sha256": sha256_file(Path(__file__).resolve()),
        "precommitment": precommitment,
    }

    write_json(out_dir / "manifest.json", manifest)

    print("=" * 128)
    print(
        "VPINN GRADIENT GEOMETRY — STAGE 7 ONE-STEP ADAM-STATE COMPONENT CAUSAL AUDIT"
    )
    print("=" * 128)
    print(f"device                 : {device}")
    print(f"seeds                  : {pf['seeds']}")
    print(f"locked state           : exact epoch-2500 Stage-5 checkpoints")
    print(f"branches               : {list(BRANCHES)}")
    print(f"step reset             : NEVER (preserved in every branch)")
    print(f"group eligibility      : >=4/5 viable seeds")
    print(f"Stage-3 solver SHA256  : {pf['stage3_sha256']}")
    print("=" * 128)

    rows: List[dict] = []
    start = time.perf_counter()

    for seed in pf["seeds"]:
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        print()
        print("-" * 128)
        print(f"SEED {seed}")

        for branch in BRANCHES:
            branch_dir = (
                out_dir
                / f"seed_{seed:03d}"
                / branch.lower()
            )
            branch_dir.mkdir(parents=True, exist_ok=True)

            result = run_one_step_branch(
                stage3=stage3,
                device=device,
                seed=seed,
                branch=branch,
                checkpoint_path=checkpoint,
                branch_dir=branch_dir,
            )

            rows.append(result)

            print(
                f"{branch:7s} | "
                f"A={result['predicted_target_descent_alignment']:+.6f} | "
                f"Δlog R9²={result['target_log_change']:+.6e} | "
                f"||Δθ||={result['actual_update_norm']:.6e} | "
                f"formula gap="
                f"{result['predicted_actual_update_max_abs_diff']:.3e}"
            )

    decorated = decorate_paired_results(rows)
    write_csv(out_dir / "branch_results.csv", decorated)

    decision = choose_route(decorated)

    decision.update(
        {
            "all_formula_runtime_checks_pass": bool(
                all(
                    float(r["predicted_actual_update_max_abs_diff"])
                    <= 5.0e-12
                    for r in decorated
                )
            ),
            "interpretation_guardrail": (
                "This stage establishes only the immediate causal effect of "
                "Adam-state component ablations on one-step target correction. "
                "A selected branch must still be tested in a paired continuation "
                "before making an escape-time claim."
            ),
        }
    )

    write_json(out_dir / "decision.json", decision)

    # Plots.
    paired_plot(
        decorated,
        field="predicted_target_descent_alignment",
        ylabel="Predicted Adam target-descent alignment",
        title="One-step target-descent geometry after Adam-state ablation",
        path=out_dir / "target_descent_alignment_by_branch.png",
        hline=0.0,
    )

    paired_plot(
        decorated,
        field="target_log_change",
        ylabel="log(R9²_after / R9²_before)",
        title="Actual one-step target-residual change",
        path=out_dir / "target_loss_change_by_branch.png",
        hline=0.0,
    )

    paired_plot(
        decorated,
        field="update_norm_ratio_vs_control",
        ylabel="Update norm / CONTROL update norm",
        title="Step-scale perturbation induced by each state ablation",
        path=out_dir / "update_norm_ratio_by_branch.png",
        hline=1.0,
    )

    # Compact branch summary.
    branch_summary = []

    for branch in NONCONTROL:
        stats = decision["branch_stats"][branch]

        branch_summary.append(
            {
                "branch": branch,
                **stats,
            }
        )

    write_csv(out_dir / "branch_summary.csv", branch_summary)

    elapsed = time.perf_counter() - start

    lines = []
    lines.append("=" * 132)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 7 ONE-STEP ADAM-STATE COMPONENT SUMMARY"
    )
    lines.append("=" * 132)
    lines.append(
        "branch | viable seeds | eligible | median alignment | "
        "median target advantage | median |log norm ratio|"
    )
    lines.append("-" * 132)

    for branch in NONCONTROL:
        s = decision["branch_stats"][branch]
        lines.append(
            f"{branch:6s} | "
            f"{int(s['viable_seed_count']):12d} | "
            f"{str(bool(s['eligible_group'])):8s} | "
            f"{s['median_target_descent_alignment']:+16.6f} | "
            f"{s['median_target_log_advantage_vs_control']:+23.6e} | "
            f"{s['median_abs_log_update_norm_ratio']:22.6f}"
        )

    lines.append("-" * 132)
    lines.append(
        f"selected branch                   : "
        f"{decision['selected_branch']}"
    )
    lines.append(
        f"selection reason                  : "
        f"{decision['selection_reason']}"
    )
    lines.append(
        f"continuation authorized           : "
        f"{decision['continuation_authorized']}"
    )
    lines.append(
        f"next route                        : "
        f"{decision['next_route']}"
    )
    lines.append(
        f"formula/runtime checks            : "
        f"{'PASS' if decision['all_formula_runtime_checks_pass'] else 'FAIL'}"
    )
    lines.append(f"elapsed seconds                   : {elapsed:.2f}")
    lines.append("=" * 132)
    lines.append(
        "Guardrail: one-step causal rescue is route evidence, not yet an "
        "escape-time result."
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
