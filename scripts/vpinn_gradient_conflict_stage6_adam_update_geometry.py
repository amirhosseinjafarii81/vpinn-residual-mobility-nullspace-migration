#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 6
Adam Effective-Update Geometry Audit
====================================

Motivation
----------
Stages 1-5 established:
  * a reproducible high-frequency VPINN plateau,
  * strong target-vs-rest cancellation in the RAW loss gradients,
  * eventual finite-time escape for all five seeds.

But raw gradient geometry is not the actual optimization geometry under Adam.
Adam carries first/second-moment history and applies coordinatewise scaling.
Therefore, before any gradient-surgery intervention, we must measure the
ACTUAL next Adam parameter update implied by the current optimizer state.

This stage is observational only. It reproduces the exact Stage-5 baseline
continuations from the saved epoch-2500 checkpoints.

Key quantity
------------
Let:
    g_t = grad[(1/M) R_9^2]
    g   = grad[(1/M) sum_k R_k^2]

Let Δtheta_Adam be the exact next parameter displacement that PyTorch Adam
would apply from the CURRENT optimizer state and CURRENT total gradient.

Define Adam target-descent alignment:

    A_Adam =
        <Δtheta_Adam, -g_t>
        / (||Δtheta_Adam|| ||g_t||).

Interpretation:
    A_Adam > 0 : the next Adam update decreases target loss R_9^2 to
                 first order.
    A_Adam < 0 : the next Adam update increases target loss R_9^2 to
                 first order.

For comparison, define raw target-descent alignment:

    A_raw =
        < -g, -g_t > / (||g|| ||g_t||)
      = <g, g_t> / (||g|| ||g_t||).

We also measure:
  * cos(Δtheta_Adam, -g)        : optimizer rotation vs raw descent
  * ||Δtheta_Adam||/(lr||g||)  : effective step amplification
  * raw target-vs-rest cancellation
  * moment-history / current-gradient contribution ratio

Precommitted "Adam target-descent unlock"
-----------------------------------------
Tracking interval is 25 epochs.

An unlock onset is the earliest epoch t for which

    A_Adam(t) > 0,
    A_Adam(t+25) > 0,
    A_Adam(t+50) > 0.

Thus three consecutive observations are required, mirroring the Stage-5
escape-certification logic.

Primary temporal gate
---------------------
For each seed:
    unlock_onset <= Stage-5 residual_release_onset.

Group support:
    at least 4/5 seeds.

If this gate passes:
    next stage = optimizer-state causal ablation from identical epoch-2500
                 checkpoints.

If it fails:
    raw/Adam geometry is not a reliable precursor; do not perform optimizer
    state intervention yet.

Reproducibility
---------------
* exact Stage-5 checkpoints are loaded;
* Stage-3 solver SHA256 must match Stage-5 manifest;
* every 25 epochs the reproduced trajectory is compared against the
  historical Stage-5 baseline;
* any drift above 1e-10 aborts the audit;
* no training hyperparameter is changed.
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


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-6 Adam effective-update geometry audit."
    )
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--track-interval", type=int, default=25)
    p.add_argument(
        "--stage3-script",
        type=str,
        default="vpinn_gradient_conflict_stage3_frequency_transfer.py",
    )
    p.add_argument(
        "--stage5-script",
        type=str,
        default="vpinn_gradient_conflict_stage5_escape_time.py",
    )
    p.add_argument(
        "--stage5-dir",
        type=str,
        default="vpinn_gradient_conflict_stage5_escape_time",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="vpinn_gradient_conflict_stage6_adam_update_geometry",
    )
    args = p.parse_args()

    if args.track_interval != 25:
        raise ValueError(
            "Stage 6 is precommitted to the Stage-5 25-epoch tracking grid."
        )

    return args


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
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    denom = na * nb
    if float(denom.item()) <= 0.0:
        return float("nan")
    value = torch.dot(a, b) / denom
    return float(torch.clamp(value, -1.0, 1.0).item())


# =============================================================================
# Stage-5 preflight
# =============================================================================

def stage5_preflight(
    stage5_dir: Path,
    stage3_script: Path,
    stage5_script: Path,
) -> dict:
    decision_path = stage5_dir / "decision.json"
    manifest_path = stage5_dir / "manifest.json"
    summary_path = stage5_dir / "escape_time_summary.csv"
    aggregate_path = stage5_dir / "aggregate_postlock_metrics.csv"
    anchor_path = stage5_dir / "anchor_reproduction.json"

    required = [
        decision_path,
        manifest_path,
        summary_path,
        aggregate_path,
        anchor_path,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage-5 prerequisite files:\n  "
            + "\n  ".join(missing)
        )

    decision = read_json(decision_path)
    manifest = read_json(manifest_path)
    anchor = read_json(anchor_path)
    summary = read_csv(summary_path)
    aggregate = read_csv(aggregate_path)

    if decision.get("next_route") != (
        "causal_intervention_from_epoch2500_locked_states"
    ):
        raise RuntimeError(
            "Stage 5 did not authorize mechanism follow-up from epoch 2500."
        )

    if not bool(decision.get("next_route_authorized", False)):
        raise RuntimeError("Stage-5 next route is not authorized.")

    if not bool(anchor.get("all_pass", False)):
        raise RuntimeError("Stage-5 anchor reproduction is not PASS.")

    expected_sha = manifest.get("stage3_solver_sha256")
    actual_sha = sha256_file(stage3_script)

    if expected_sha != actual_sha:
        raise RuntimeError(
            "Stage-3 solver SHA256 mismatch. Refusing implementation drift.\n"
            f"Expected: {expected_sha}\n"
            f"Actual  : {actual_sha}"
        )

    if not stage5_script.is_file():
        raise FileNotFoundError(
            f"Stage-5 script not found: {stage5_script}"
        )

    expected_stage5_sha = manifest.get("stage5_script_sha256")
    actual_stage5_sha = sha256_file(stage5_script)

    if expected_stage5_sha != actual_stage5_sha:
        raise RuntimeError(
            "Stage-5 script SHA256 mismatch. Exact baseline replay requires "
            "the same Stage-5 implementation.\n"
            f"Expected: {expected_stage5_sha}\n"
            f"Actual  : {actual_stage5_sha}"
        )

    seeds = [int(r["seed"]) for r in summary]
    if seeds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Unexpected seed set/order: {seeds}")

    if any(str(r["censored_at_max_epoch"]).lower() == "true" for r in summary):
        raise RuntimeError("Stage 6 requires all Stage-5 seeds to have escaped.")

    return {
        "decision": decision,
        "manifest": manifest,
        "summary": summary,
        "aggregate": aggregate,
        "seeds": seeds,
        "stage3_sha256": actual_sha,
        "stage5_sha256": actual_stage5_sha,
    }


# =============================================================================
# Experiment reconstruction
# =============================================================================

def make_experiment(stage3, device, seed: int, out_dir: Path):
    cfg = stage3.Config(
        seed=seed,
        device=str(device),
        epochs=4000,
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
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

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
# Adam effective-update geometry
# =============================================================================

def compute_geometry(exp, target_mode: int = 9) -> dict:
    residuals = exp.weak_residuals()
    params = tuple(p for p in exp.model.parameters() if p.requires_grad)

    t = target_mode - 1
    M = residuals.numel()

    target_loss = residuals[t].square() / M
    rest_loss = (
        residuals.square().sum() - residuals[t].square()
    ) / M
    total_loss = residuals.square().mean()

    # Three exact autograd vectors only at sparse 25-epoch diagnostics.
    gt = flatten(
        torch.autograd.grad(
            target_loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
    ).detach()

    gr = flatten(
        torch.autograd.grad(
            rest_loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
    ).detach()

    g = flatten(
        torch.autograd.grad(
            total_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    ).detach()

    # Verify decomposition without using it to drive training.
    decomposition_gap = float(
        torch.linalg.vector_norm(g - (gt + gr)).item()
    )

    gt_norm = torch.linalg.vector_norm(gt)
    gr_norm = torch.linalg.vector_norm(gr)
    g_norm = torch.linalg.vector_norm(g)

    raw_target_rest_cosine = cosine(gt, gr)
    raw_target_descent_alignment = cosine(g, gt)
    raw_cancel = float(
        (
            g_norm
            / torch.clamp(gt_norm + gr_norm, min=1.0e-300)
        ).item()
    )

    # -------------------------------------------------------------------------
    # Predict the EXACT next PyTorch Adam parameter displacement from:
    #   current gradient g
    #   current exp_avg / exp_avg_sq / step
    #
    # Default Stage-3 Adam has:
    #   weight_decay=0, amsgrad=False, maximize=False
    # and no foreach/fused complications are relied upon here.
    # -------------------------------------------------------------------------
    group = exp.optimizer.param_groups[0]

    if float(group.get("weight_decay", 0.0)) != 0.0:
        raise RuntimeError("Stage 6 assumes Adam weight_decay=0.")
    if bool(group.get("amsgrad", False)):
        raise RuntimeError("Stage 6 assumes Adam amsgrad=False.")
    if bool(group.get("maximize", False)):
        raise RuntimeError("Stage 6 assumes Adam maximize=False.")

    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    lr = float(group["lr"])

    delta_parts = []
    history_parts = []
    current_parts = []
    pre_gt_parts = []
    pre_gr_parts = []
    pre_g_parts = []

    offset = 0

    for p in params:
        n = p.numel()

        gp = g[offset:offset+n].reshape_as(p)
        gtp = gt[offset:offset+n].reshape_as(p)
        grp = gr[offset:offset+n].reshape_as(p)

        offset += n

        state = exp.optimizer.state[p]

        if "exp_avg" not in state or "exp_avg_sq" not in state:
            raise RuntimeError(
                "Adam state is missing exp_avg/exp_avg_sq at epoch 2500+."
            )

        m_old = state["exp_avg"]
        v_old = state["exp_avg_sq"]

        step_old_raw = state["step"]
        step_old = (
            int(step_old_raw.item())
            if torch.is_tensor(step_old_raw)
            else int(step_old_raw)
        )

        m_new = beta1 * m_old + (1.0 - beta1) * gp
        v_new = beta2 * v_old + (1.0 - beta2) * gp.square()

        step_new = step_old + 1
        bias_correction1 = 1.0 - beta1 ** step_new
        bias_correction2 = 1.0 - beta2 ** step_new

        denom = (
            v_new.sqrt() / math.sqrt(bias_correction2)
        ).add(eps)

        step_size = lr / bias_correction1

        delta = -step_size * m_new / denom

        delta_parts.append(delta.reshape(-1))

        history_parts.append(
            (beta1 * m_old).reshape(-1)
        )
        current_parts.append(
            ((1.0 - beta1) * gp).reshape(-1)
        )

        # Same next-step diagonal preconditioner, applied separately to
        # target/rest/current gradients for geometry diagnostics.
        pre_gt_parts.append((gtp / denom).reshape(-1))
        pre_gr_parts.append((grp / denom).reshape(-1))
        pre_g_parts.append((gp / denom).reshape(-1))

    delta = flatten(delta_parts)
    history_term = flatten(history_parts)
    current_term = flatten(current_parts)
    pre_gt = flatten(pre_gt_parts)
    pre_gr = flatten(pre_gr_parts)
    pre_g = flatten(pre_g_parts)

    delta_norm = torch.linalg.vector_norm(delta)

    # Positive => actual next Adam step is first-order descent for R_9^2.
    adam_target_descent_alignment = cosine(delta, -gt)

    # Positive and near one => Adam step follows raw total descent.
    adam_raw_descent_alignment = cosine(delta, -g)

    amplification = float(
        (
            delta_norm
            / torch.clamp(lr * g_norm, min=1.0e-300)
        ).item()
    )

    history_norm = torch.linalg.vector_norm(history_term)
    current_norm = torch.linalg.vector_norm(current_term)

    history_over_current = float(
        (
            history_norm
            / torch.clamp(current_norm, min=1.0e-300)
        ).item()
    )

    pre_cancel = float(
        (
            torch.linalg.vector_norm(pre_g)
            / torch.clamp(
                torch.linalg.vector_norm(pre_gt)
                + torch.linalg.vector_norm(pre_gr),
                min=1.0e-300,
            )
        ).item()
    )

    return {
        "gradient_decomposition_gap": decomposition_gap,
        "raw_target_gradient_norm": float(gt_norm.item()),
        "raw_rest_gradient_norm": float(gr_norm.item()),
        "raw_total_gradient_norm": float(g_norm.item()),
        "raw_target_vs_rest_cosine": raw_target_rest_cosine,
        "raw_target_descent_alignment": raw_target_descent_alignment,
        "raw_cancellation_ratio": raw_cancel,
        "adam_next_step_norm": float(delta_norm.item()),
        "adam_target_descent_alignment":
            adam_target_descent_alignment,
        "adam_raw_descent_alignment":
            adam_raw_descent_alignment,
        "adam_step_amplification_over_lr_raw_gradient":
            amplification,
        "moment_history_norm": float(history_norm.item()),
        "current_gradient_contribution_norm":
            float(current_norm.item()),
        "moment_history_over_current_gradient":
            history_over_current,
        "moment_history_vs_current_cosine":
            cosine(history_term, current_term),
        "preconditioned_target_vs_rest_cosine":
            cosine(pre_gt, pre_gr),
        "preconditioned_cancellation_ratio":
            pre_cancel,
    }


# =============================================================================
# Historical reproduction
# =============================================================================

def historical_map(rows: List[dict]):
    return {
        (int(r["seed"]), int(r["epoch"])): r
        for r in rows
    }


def verify_historical_state(
    seed: int,
    epoch: int,
    state: dict,
    history_map: dict,
    tolerance: float = 1.0e-8,
) -> dict:
    key = (seed, epoch)

    if key not in history_map:
        raise RuntimeError(
            f"Historical Stage-5 row missing: seed={seed}, epoch={epoch}."
        )

    old = history_map[key]

    comparisons = {
        "relative_l2_error": (
            float(old["relative_l2_error"]),
            float(state["relative_l2_error"]),
        ),
        "vpinn_loss": (
            float(old["vpinn_loss"]),
            float(state["vpinn_loss"]),
        ),
        "target_mode_residual_energy_share": (
            float(old["target_mode_residual_energy_share"]),
            float(state["target_mode_residual_energy_share"]),
        ),
        "target_mode_abs_residual": (
            float(old["target_mode_abs_residual"]),
            float(state["target_mode_abs_residual"]),
        ),
    }

    diffs = {
        k: abs(a - b)
        for k, (a, b) in comparisons.items()
    }

    max_diff = max(diffs.values())

    return {
        "max_abs_difference": max_diff,
        "pass": bool(max_diff <= tolerance),
        "field_abs_differences": diffs,
    }


# =============================================================================
# One-step Adam formula verification
# =============================================================================

def verify_adam_prediction(
    stage3,
    device,
    checkpoint_path: Path,
    seed: int,
    expected_geometry: dict,
    scratch_dir: Path,
    tolerance: float = 5.0e-12,
) -> dict:
    """
    Clone epoch-2500 state, calculate the predicted Adam displacement, then
    perform ONE real baseline Adam step and compare actual parameter delta.

    This verifies that our read-only Adam formula matches the runtime optimizer.
    """
    exp = make_experiment(stage3, device, seed, scratch_dir)
    load_locked_checkpoint(exp, checkpoint_path, seed)

    # Recompute prediction, but also independently construct the vector here.
    residuals = exp.weak_residuals()
    params = tuple(p for p in exp.model.parameters() if p.requires_grad)
    total_loss = residuals.square().mean()

    g_parts = torch.autograd.grad(
        total_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    g = flatten(g_parts).detach()

    group = exp.optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    lr = float(group["lr"])

    predicted_parts = []
    offset = 0

    before = flatten(
        [p.detach().clone() for p in params]
    )

    for p in params:
        n = p.numel()
        gp = g[offset:offset+n].reshape_as(p)
        offset += n

        st = exp.optimizer.state[p]
        m = st["exp_avg"]
        v = st["exp_avg_sq"]
        step_raw = st["step"]
        step = (
            int(step_raw.item())
            if torch.is_tensor(step_raw)
            else int(step_raw)
        )

        m_new = beta1 * m + (1.0 - beta1) * gp
        v_new = beta2 * v + (1.0 - beta2) * gp.square()

        step_new = step + 1
        bc1 = 1.0 - beta1 ** step_new
        bc2 = 1.0 - beta2 ** step_new

        denom = (v_new.sqrt() / math.sqrt(bc2)).add(eps)
        predicted_parts.append(
            (-lr / bc1 * m_new / denom).reshape(-1)
        )

    predicted = flatten(predicted_parts)

    # Actual runtime step.
    exp.optimizer.zero_grad(set_to_none=True)
    residuals2 = exp.weak_residuals()
    loss2 = residuals2.square().mean()
    loss2.backward()
    exp.optimizer.step()

    after = flatten(
        [p.detach().clone() for p in params]
    )

    actual = after - before

    max_abs = float(
        torch.max(torch.abs(predicted - actual)).item()
    )
    rel = float(
        (
            torch.linalg.vector_norm(predicted - actual)
            / torch.clamp(
                torch.linalg.vector_norm(actual),
                min=1.0e-300,
            )
        ).item()
    )

    return {
        "seed": seed,
        "max_abs_parameter_update_difference": max_abs,
        "relative_update_difference": rel,
        "tolerance": tolerance,
        "pass": bool(max_abs <= tolerance),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_by_seed(
    rows: List[dict],
    key: str,
    ylabel: str,
    title: str,
    path: Path,
    hline: float | None = None,
    log_y: bool = False,
) -> None:
    seeds = sorted(set(int(r["seed"]) for r in rows))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for seed in seeds:
        rr = [r for r in rows if int(r["seed"]) == seed]
        rr.sort(key=lambda r: int(r["epoch"]))

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r[key]) for r in rr],
            marker="o",
            markersize=3,
            label=f"seed {seed}",
        )

    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1.2)

    if log_y:
        vals = [
            float(r[key])
            for r in rows
            if np.isfinite(float(r[key])) and float(r[key]) > 0
        ]
        if vals:
            ax.set_yscale("log")

    ax.set_xlabel("Epoch")
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

    stage5_script = Path(args.stage5_script)
    if not stage5_script.is_absolute():
        stage5_script = root / stage5_script

    stage5_dir = Path(args.stage5_dir)
    if not stage5_dir.is_absolute():
        stage5_dir = root / stage5_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    preflight = stage5_preflight(
        stage5_dir=stage5_dir,
        stage3_script=stage3_script,
        stage5_script=stage5_script,
    )

    stage3 = load_stage3_module(stage3_script)
    stage5 = load_stage3_module(stage5_script)

    hist = historical_map(preflight["aggregate"])

    baseline_summary = {
        int(r["seed"]): {
            "release_onset": int(r["release_onset_epoch"]),
            "escape_onset": int(r["certified_escape_onset_epoch"]),
            "escape_confirmation":
                int(r["certified_escape_confirmation_epoch"]),
        }
        for r in preflight["summary"]
    }

    precommitment = {
        "stage": "adam_effective_update_geometry_audit",
        "seeds": preflight["seeds"],
        "start_epoch": 2500,
        "end_epoch_per_seed":
            "Stage-5 certified escape confirmation epoch",
        "tracking_interval": 25,
        "training_trajectory": "exact Stage-5 baseline reproduction",
        "adam_target_descent_alignment":
            "<Delta_theta_Adam,-g_target>/(||Delta|| ||g_target||)",
        "unlock_definition": {
            "condition": "adam_target_descent_alignment > 0",
            "consecutive_observations": 3,
        },
        "primary_per_seed_temporal_gate":
            "unlock_onset <= Stage-5 residual_release_onset",
        "group_gate": "at least 4/5 seeds",
        "next_route_if_pass":
            "optimizer_state_causal_ablation",
        "next_route_if_fail":
            "alternative_escape_mechanism_audit",
    }

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),
        "stage3_solver_path": str(stage3_script),
        "stage3_solver_sha256": preflight["stage3_sha256"],
        "stage5_script_path": str(stage5_script),
        "stage5_script_sha256": preflight["stage5_sha256"],
        "stage5_dir": str(stage5_dir),
        "stage6_script_sha256": sha256_file(Path(__file__).resolve()),
        "precommitment": precommitment,
    }
    write_json(out_dir / "manifest.json", manifest)

    print("=" * 118)
    print("VPINN GRADIENT GEOMETRY — STAGE 6 ADAM EFFECTIVE-UPDATE GEOMETRY AUDIT")
    print("=" * 118)
    print(f"device                       : {device}")
    print(f"seeds                        : {preflight['seeds']}")
    print(f"start                        : exact Stage-5 epoch-2500 checkpoints")
    print(f"tracking interval            : 25")
    print(
        "unlock                        : A_Adam > 0 for 3 consecutive observations"
    )
    print(
        "primary gate                  : unlock onset <= residual-release onset"
    )
    print(f"Stage-3 solver SHA256        : {preflight['stage3_sha256']}")
    print("=" * 118)

    # -------------------------------------------------------------------------
    # Verify Adam formula once per seed at the locked state.
    # -------------------------------------------------------------------------
    formula_checks = []

    for seed in preflight["seeds"]:
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        exp_probe = make_experiment(
            stage3,
            device,
            seed,
            out_dir / "_formula_probe" / f"seed_{seed:03d}",
        )
        load_locked_checkpoint(exp_probe, checkpoint, seed)
        geometry = compute_geometry(exp_probe)

        check = verify_adam_prediction(
            stage3=stage3,
            device=device,
            checkpoint_path=checkpoint,
            seed=seed,
            expected_geometry=geometry,
            scratch_dir=(
                out_dir / "_formula_check" / f"seed_{seed:03d}"
            ),
            tolerance=5.0e-12,
        )

        formula_checks.append(check)

        print(
            f"Adam formula seed {seed}: "
            f"{'PASS' if check['pass'] else 'FAIL'} | "
            f"max update diff="
            f"{check['max_abs_parameter_update_difference']:.3e}"
        )

        if not check["pass"]:
            write_json(
                out_dir / "adam_formula_verification.json",
                {
                    "all_pass": False,
                    "results": formula_checks,
                },
            )
            raise RuntimeError(
                f"Adam update formula verification failed for seed {seed}."
            )

    write_json(
        out_dir / "adam_formula_verification.json",
        {
            "all_pass": True,
            "results": formula_checks,
        },
    )

    # -------------------------------------------------------------------------
    # Exact baseline replay + read-only optimizer geometry.
    # -------------------------------------------------------------------------
    all_rows: List[dict] = []
    seed_summaries: List[dict] = []
    reproduction_checks: List[dict] = []

    global_start = time.perf_counter()

    for seed in preflight["seeds"]:
        seed_dir = out_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        exp = make_experiment(stage3, device, seed, seed_dir)

        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )
        load_locked_checkpoint(exp, checkpoint, seed)

        end_epoch = baseline_summary[seed]["escape_confirmation"]
        release_epoch = baseline_summary[seed]["release_onset"]
        escape_epoch = baseline_summary[seed]["escape_onset"]

        print()
        print("-" * 118)
        print(
            f"SEED {seed} | release={release_epoch} | "
            f"escape={escape_epoch} | replay through {end_epoch}"
        )

        rows: List[dict] = []

        for epoch in range(2500, end_epoch + 1):
            if epoch % 25 == 0:
                # Reproduce Stage 5 EXACTLY: use the same measurement function
                # and later reuse the same residual graph for the optimizer
                # step. This avoids tiny long-horizon drift from changing the
                # floating-point/autograd execution order.
                stage5_metrics, stage5_residuals = stage5.measure(
                    exp=exp,
                    target_mode=9,
                    need_geometry=True,
                )
                rel = exp.relative_l2_error()

                geo = compute_geometry(exp)

                state = {
                    "seed": seed,
                    "epoch": epoch,
                    "relative_l2_error": rel,
                    **stage5_metrics,
                    **geo,
                }

                # Exact historical trajectory guard.
                check = verify_historical_state(
                    seed=seed,
                    epoch=epoch,
                    state=state,
                    history_map=hist,
                    tolerance=1.0e-8,
                )

                reproduction_checks.append(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "max_abs_difference":
                            check["max_abs_difference"],
                        "pass": check["pass"],
                    }
                )

                if not check["pass"]:
                    raise RuntimeError(
                        f"Stage-5 replay drift at seed={seed}, epoch={epoch}: "
                        f"{check['max_abs_difference']:.3e}"
                    )

                rows.append(state)
                all_rows.append(state)

                if epoch < end_epoch:
                    stage5.optimizer_step_from_residuals(
                        exp,
                        stage5_residuals,
                    )
            elif epoch < end_epoch:
                exp.train_step()

        write_csv(seed_dir / "adam_geometry_metrics.csv", rows)

        # -------------------------------------------------------------
        # Three-consecutive-positive Adam target-descent unlock.
        # -------------------------------------------------------------
        consecutive = 0
        candidate = -1
        unlock_onset = -1
        unlock_confirmation = -1

        for row in rows:
            positive = bool(
                float(row["adam_target_descent_alignment"]) > 0.0
            )

            if positive:
                if consecutive == 0:
                    candidate = int(row["epoch"])
                consecutive += 1
            else:
                consecutive = 0
                candidate = -1

            if consecutive >= 3:
                unlock_onset = candidate
                unlock_confirmation = int(row["epoch"])
                break

        temporal_pass = bool(
            unlock_onset >= 0
            and unlock_onset <= release_epoch
        )

        anchor = rows[0]

        summary = {
            "seed": seed,
            "adam_unlock_onset_epoch": unlock_onset,
            "adam_unlock_confirmation_epoch": unlock_confirmation,
            "stage5_residual_release_epoch": release_epoch,
            "stage5_escape_onset_epoch": escape_epoch,
            "unlock_to_release_lead_epochs": (
                release_epoch - unlock_onset
                if unlock_onset >= 0
                else None
            ),
            "unlock_to_escape_lead_epochs": (
                escape_epoch - unlock_onset
                if unlock_onset >= 0
                else None
            ),
            "temporal_gate_pass": temporal_pass,
            "epoch2500_raw_cancellation_ratio":
                float(anchor["raw_cancellation_ratio"]),
            "epoch2500_adam_target_descent_alignment":
                float(anchor["adam_target_descent_alignment"]),
            "epoch2500_adam_raw_descent_alignment":
                float(anchor["adam_raw_descent_alignment"]),
            "epoch2500_adam_amplification":
                float(
                    anchor[
                        "adam_step_amplification_over_lr_raw_gradient"
                    ]
                ),
            "epoch2500_moment_history_over_current":
                float(
                    anchor["moment_history_over_current_gradient"]
                ),
        }

        seed_summaries.append(summary)
        write_json(seed_dir / "summary.json", summary)

        print(
            f"  Adam unlock onset={unlock_onset} | "
            f"release={release_epoch} | "
            f"lead={summary['unlock_to_release_lead_epochs']} | "
            f"{'PASS' if temporal_pass else 'FAIL'}"
        )

    write_csv(out_dir / "aggregate_adam_geometry_metrics.csv", all_rows)
    write_csv(out_dir / "seed_summary.csv", seed_summaries)
    write_csv(out_dir / "trajectory_reproduction_checks.csv", reproduction_checks)

    pass_count = sum(
        int(r["temporal_gate_pass"])
        for r in seed_summaries
    )

    leads = [
        int(r["unlock_to_release_lead_epochs"])
        for r in seed_summaries
        if r["unlock_to_release_lead_epochs"] is not None
    ]

    escape_leads = [
        int(r["unlock_to_escape_lead_epochs"])
        for r in seed_summaries
        if r["unlock_to_escape_lead_epochs"] is not None
    ]

    group_pass = bool(pass_count >= 4)

    decision = {
        "n_seeds": len(seed_summaries),
        "temporal_gate_pass_count": pass_count,
        "group_gate_pass": group_pass,
        "unlock_precedes_or_equals_release_seeds": [
            int(r["seed"])
            for r in seed_summaries
            if r["temporal_gate_pass"]
        ],
        "median_unlock_to_release_lead_epochs": (
            float(np.median(leads)) if leads else None
        ),
        "median_unlock_to_escape_lead_epochs": (
            float(np.median(escape_leads))
            if escape_leads else None
        ),
        "all_historical_replay_checks_pass":
            all(bool(r["pass"]) for r in reproduction_checks),
        "all_adam_formula_checks_pass":
            all(bool(r["pass"]) for r in formula_checks),
        "next_route": (
            "optimizer_state_causal_ablation"
            if group_pass
            else "alternative_escape_mechanism_audit"
        ),
        "interpretation_guardrail": (
            "Temporal precedence of Adam target-descent unlock over residual "
            "release supports optimizer-aware mechanism follow-up, but does "
            "not prove that Adam state causes escape. The authorized next "
            "stage must intervene on optimizer state from identical model "
            "checkpoints."
        ),
    }
    write_json(out_dir / "decision.json", decision)

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    plot_by_seed(
        all_rows,
        key="adam_target_descent_alignment",
        ylabel="Adam target-descent alignment",
        title="When does the actual Adam step start descending R9²?",
        path=out_dir / "adam_target_descent_alignment.png",
        hline=0.0,
    )

    plot_by_seed(
        all_rows,
        key="raw_target_descent_alignment",
        ylabel="Raw target-descent alignment",
        title="Raw total-gradient direction relative to target descent",
        path=out_dir / "raw_target_descent_alignment.png",
        hline=0.0,
    )

    plot_by_seed(
        all_rows,
        key="adam_step_amplification_over_lr_raw_gradient",
        ylabel="||Δθ_Adam|| / (lr ||g_raw||)",
        title="Adam amplification of the nearly cancelled raw gradient",
        path=out_dir / "adam_effective_amplification.png",
        log_y=True,
    )

    plot_by_seed(
        all_rows,
        key="raw_cancellation_ratio",
        ylabel="Raw cancellation ratio",
        title="Raw target-vs-rest cancellation along baseline escape",
        path=out_dir / "raw_cancellation_ratio.png",
        log_y=True,
    )

    plot_by_seed(
        all_rows,
        key="moment_history_over_current_gradient",
        ylabel="||β1 m|| / ||(1-β1)g||",
        title="Adam first-moment memory relative to current gradient input",
        path=out_dir / "adam_moment_memory_ratio.png",
        log_y=True,
    )

    elapsed = time.perf_counter() - global_start

    lines = []
    lines.append("=" * 150)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 6 ADAM EFFECTIVE-UPDATE SUMMARY"
    )
    lines.append("=" * 150)
    lines.append(
        "seed | Adam unlock | residual release | escape onset | "
        "unlock->release | unlock->escape | A_Adam@2500 | amplification@2500 | PASS"
    )
    lines.append("-" * 150)

    for r in seed_summaries:
        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['adam_unlock_onset_epoch']):11d} | "
            f"{int(r['stage5_residual_release_epoch']):16d} | "
            f"{int(r['stage5_escape_onset_epoch']):12d} | "
            f"{str(r['unlock_to_release_lead_epochs']):15s} | "
            f"{str(r['unlock_to_escape_lead_epochs']):13s} | "
            f"{r['epoch2500_adam_target_descent_alignment']:12.6f} | "
            f"{r['epoch2500_adam_amplification']:18.1f} | "
            f"{'PASS' if r['temporal_gate_pass'] else 'FAIL'}"
        )

    lines.append("-" * 150)
    lines.append(
        f"Adam formula verification          : "
        f"{sum(int(r['pass']) for r in formula_checks)}/{len(formula_checks)} PASS"
    )
    lines.append(
        f"historical replay checks           : "
        f"{sum(int(r['pass']) for r in reproduction_checks)}/"
        f"{len(reproduction_checks)} PASS"
    )
    lines.append(
        f"unlock <= residual release         : "
        f"{pass_count}/{len(seed_summaries)}"
    )

    if leads:
        lines.append(
            f"median unlock->release lead        : "
            f"{np.median(leads):.1f} epochs"
        )

    if escape_leads:
        lines.append(
            f"median unlock->escape lead         : "
            f"{np.median(escape_leads):.1f} epochs"
        )

    lines.append(
        f"group gate                         : "
        f"{'PASS' if group_pass else 'FAIL'}"
    )
    lines.append(
        f"next route                         : "
        f"{decision['next_route']}"
    )
    lines.append(f"elapsed seconds                    : {elapsed:.2f}")
    lines.append("=" * 150)
    lines.append(
        "Guardrail: optimizer-aware temporal ordering is not causality. "
        "A model-state-matched optimizer-state intervention is required next."
    )
    lines.append("=" * 150)

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
