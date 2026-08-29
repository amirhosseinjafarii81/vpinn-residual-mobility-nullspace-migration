#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 17R
Finite-Width Weak-Test Interaction Kernel + Curvature Barrier Audit
===================================================================

SCIENTIFIC POSITION
-------------------
A July-2026 VPINN preprint already analyzes residual-gradient Gram / tangent
kernel dynamics in an NTK regime. Therefore Stage 17R does NOT claim that
G = J J^T itself is new.

Instead, Stage 17R targets the finite-width, optimizer-state-aware mechanism
that our experiments exposed:

  (A) function-space test orthogonality versus parameter-space weak-test
      interaction at finite width;

  (B) the ACTUAL Adam state-induced interaction metric, including
      second-moment preconditioning and first-moment memory;

  (C) the nonlinear curvature barrier that makes a direction with strict
      first-order Pareto descent fail in the exact finite step.

No optimizer rescue is attempted.

Core objects
------------
Let

    r(theta) = [R_1(theta), ..., R_M(theta)]^T

and

    J(theta) = d r / d theta  in R^{M x P}.

Raw residual-gradient interaction kernel:

    K = J J^T.

Normalized parameter-space interaction:

    Khat_ij = K_ij / sqrt(K_ii K_jj).

For squared residual gradients,

    g_i = grad(R_i^2) = 2 R_i grad R_i,

hence their cosine is

    C_ij = sign(R_i R_j) Khat_ij.

Thus orthogonality of test functions in function space does NOT automatically
imply orthogonality of their parameter gradients.

Target self/cross decomposition
-------------------------------
For target t=9 under raw gradient flow:

    r_t (K r)_t
      = r_t^2 K_tt
        + r_t sum_{j != t} K_tj r_j
      = SELF + CROSS.

SELF > 0 is the target's own correction term.
CROSS can oppose it.

Define the descriptive interference ratio

    I_raw = - CROSS / SELF.

I_raw > 1 means cross-test coupling overturns target self-correction in the
raw residual-kernel dynamics.

Adam-state-aware decomposition
------------------------------
For the exact next Adam step,

    m_new = beta1 m_old + (1-beta1) g
    v_new = beta2 v_old + (1-beta2) g^2,

with g = grad L = (2/M) J^T r.

Write the exact bias-corrected candidate step as

    Delta_A = Delta_current + Delta_history,

where Delta_current is the contribution of the current raw gradient and
Delta_history is the contribution of the stored first moment.

The current-gradient component induces the positive semidefinite
state-dependent kernel

    K_Adam = J D_t J^T,

where D_t is Adam's coordinatewise second-moment metric at that step.

This gives a finite-width, optimizer-state-aware weak-test interaction matrix.
The history contribution is kept separately as

    J Delta_history.

Curvature barrier
-----------------
At every Stage-16 adaptive safety-failure state, reconstruct the exact
pre-failure adaptive direction d and compute

    first = grad L^T d,

    kappa_total = d^T H_L d,

    kappa_GN = (2/M) ||J d||^2,

    kappa_NL = kappa_total - kappa_GN,

    Delta_2 = first + 0.5 kappa_total.

Compare Delta_2 with the exact nonlinear full-step loss change that triggered
the precommitted Stage-16 safety failure.

Classification is objective-aware because the Stage-16 safety gate fires
when EITHER total loss OR target loss increases beyond tolerance.

For each objective that actually triggers the safety gate:

SECOND_ORDER_CURVATURE_BARRIER:
    first-order directional derivative < 0,
    exact finite-step change > tolerance,
    second-order Taylor prediction > 0.

HIGHER_ORDER_BARRIER:
    first-order directional derivative < 0,
    exact finite-step change > tolerance,
    second-order Taylor prediction <= 0.

A seed is classified as SECOND_ORDER_EXPLAINS_ALL_TRIGGERING_OBJECTIVES only
when every objective that actually triggered the gate is explained at second
order.

Matched control
---------------
At the SAME global pre-failure epoch, reconstruct the ordinary-Adam CONTROL
branch from the identical Stage-16 branch state and compute the same finite-
width kernel diagnostics. This tells us whether the adaptive trajectory moved
into a qualitatively different parameter-space interaction geometry.

Primary route
-------------
If second order explains ALL safety-triggering objectives in >=3/4 adaptive
failure states, authorize:

    Stage 18R — frequency transfer of finite-width weak-test interaction
                and curvature mechanism across m in {3,5,7,9}.

Otherwise:
    perform a higher-order local remainder audit before frequency transfer.

This is a mechanism audit, not an optimizer-development stage.
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


TARGET_MODE = 9
TARGET_INDEX = TARGET_MODE - 1
REPLAY_TOL = 1.0e-10
LOSS_TOL_FACTOR = 1.0e-12
TEST_GRAM_ORDER = 512


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-17R finite-width weak-test interaction and curvature audit."
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
        "--stage15-script",
        default="vpinn_gradient_conflict_stage15_adaptive_midpoint_persistence.py",
    )

    p.add_argument(
        "--stage16-dir",
        default="vpinn_gradient_conflict_stage16_matched_escape_comparison",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage17R_kernel_curvature_audit",
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


def flatten(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([x.reshape(-1) for x in parts], dim=0)


def capture_state(exp) -> dict:
    return {
        "model": copy.deepcopy(exp.model.state_dict()),
        "optimizer": copy.deepcopy(exp.optimizer.state_dict()),
    }


def load_captured_state(exp, captured: dict) -> None:
    exp.model.load_state_dict(copy.deepcopy(captured["model"]))
    exp.optimizer.load_state_dict(copy.deepcopy(captured["optimizer"]))

    for state in exp.optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(exp.device)


# =============================================================================
# Preflight
# =============================================================================

def preflight(
    stage3_script: Path,
    stage5_dir: Path,
    stage9_script: Path,
    stage15_script: Path,
    stage16_dir: Path,
) -> dict:

    paths = {
        "s16_manifest": stage16_dir / "manifest.json",
        "s16_decision": stage16_dir / "decision.json",
        "s16_paired": stage16_dir / "paired_escape_summary.csv",
        "s16_traj": stage16_dir / "trajectory_metrics.csv",
        "s16_clones": stage16_dir / "matched_state_clone_checks.csv",
        "s16_replay": stage16_dir / "stage15_replay_checks.csv",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage-16 prerequisites:\n  " + "\n  ".join(missing)
        )

    manifest = read_json(paths["s16_manifest"])
    decision = read_json(paths["s16_decision"])

    if bool(decision.get("primary_group_gate_pass", True)):
        raise RuntimeError(
            "Stage 16 unexpectedly passed; Stage 17R was designed for the "
            "observed matched-state adaptive safety-failure result."
        )

    if int(decision.get("control_certified_escape_count", -1)) != 4:
        raise RuntimeError("Expected Stage-16 CONTROL escape 4/4.")

    if int(decision.get("adaptive_safety_failure_count", -1)) != 4:
        raise RuntimeError("Expected Stage-16 adaptive safety failure 4/4.")

    clones = read_csv(paths["s16_clones"])
    replay = read_csv(paths["s16_replay"])

    if not clones or not all(str(r["pass"]).lower() == "true" for r in clones):
        raise RuntimeError("Stage-16 clone checks are not all PASS.")

    if not replay or not all(str(r["pass"]).lower() == "true" for r in replay):
        raise RuntimeError("Stage-16 replay checks are not all PASS.")

    actual_s3 = sha256_file(stage3_script)
    actual_s9 = sha256_file(stage9_script)
    actual_s15 = sha256_file(stage15_script)

    if manifest.get("stage3_solver_sha256") != actual_s3:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 16.")

    if manifest.get("stage9_script_sha256") != actual_s9:
        raise RuntimeError("Stage-9 SHA mismatch against Stage 16.")

    if manifest.get("stage15_script_sha256") != actual_s15:
        raise RuntimeError(
            "Stage-15 source SHA mismatch against Stage 16. Use the exact "
            "Stage-15 source that generated the Stage-16 run."
        )

    paired = read_csv(paths["s16_paired"])

    failure_map = {}
    branch_map = {}

    for r in paired:
        seed = int(r["seed"])

        if r["adaptive_status"] != "SAFETY_FAILURE":
            raise RuntimeError(
                f"Seed {seed}: expected Stage-16 adaptive safety failure."
            )

        if r["adaptive_failure_type"] != "NONLINEAR_SAFETY_FAILURE":
            raise RuntimeError(
                f"Seed {seed}: expected NONLINEAR_SAFETY_FAILURE."
            )

        failure_map[seed] = int(float(r["adaptive_failure_epoch"]))
        branch_map[seed] = int(float(r["branch_epoch"]))

    expected_branch = {0: 2505, 1: 2505, 2: 2505, 3: 2510}

    if branch_map != expected_branch:
        raise RuntimeError(f"Unexpected Stage-16 branch map: {branch_map}")

    traj = read_csv(paths["s16_traj"])

    failure_rows = {}

    for r in traj:
        if (
            r["branch"] == "ADAPTIVE"
            and r["event"] == "NONLINEAR_SAFETY_FAILURE"
        ):
            failure_rows[int(r["seed"])] = r

    if set(failure_rows) != set(failure_map):
        raise RuntimeError("Incomplete Stage-16 adaptive failure-event rows.")

    for seed in range(4):
        checkpoint = (
            stage5_dir
            / f"seed_{seed:03d}"
            / "locked_state_epoch_2500.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    return {
        "branch_map": branch_map,
        "failure_map": failure_map,
        "failure_rows": failure_rows,
        "stage3_sha256": actual_s3,
        "stage9_sha256": actual_s9,
        "stage15_sha256": actual_s15,
    }


# =============================================================================
# Test-space Gram
# =============================================================================

def h1_test_gram(M: int, order: int = TEST_GRAM_ORDER) -> np.ndarray:
    """
    v_k = sqrt(2)/(k*pi) sin(k*pi*x)
    v'_k = sqrt(2) cos(k*pi*x)

    Compute the H_0^1 seminorm Gram numerically as an independent audit.
    """
    z, w = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (z + 1.0)
    ww = 0.5 * w

    k = np.arange(1, M + 1, dtype=float)[:, None]
    Vp = np.sqrt(2.0) * np.cos(k * np.pi * x[None, :])

    G = (Vp * ww[None, :]) @ Vp.T
    return G


# =============================================================================
# Residual Jacobian and kernel
# =============================================================================

def residual_jacobian(exp) -> dict:
    residuals = exp.weak_residuals()
    params = tuple(p for p in exp.model.parameters() if p.requires_grad)

    rows = []

    for i in range(residuals.numel()):
        parts = torch.autograd.grad(
            residuals[i],
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

        rows.append(flatten(parts).detach())

    J = torch.stack(rows, dim=0)
    r = residuals.detach()

    K = J @ J.T

    diag = torch.diag(K).clamp_min(1.0e-300)
    denom = torch.sqrt(diag[:, None] * diag[None, :])
    Kcorr = K / denom

    sign_outer = torch.sign(r[:, None] * r[None, :])
    squared_residual_grad_cos = sign_outer * Kcorr

    return {
        "params": params,
        "r": r,
        "J": J,
        "K": K,
        "Kcorr": Kcorr,
        "squared_residual_grad_cos": squared_residual_grad_cos,
    }


def kernel_summary(
    r: torch.Tensor,
    K: torch.Tensor,
    Kcorr: torch.Tensor,
    target_index: int = TARGET_INDEX,
) -> dict:

    t = target_index

    self_term = float((r[t] * r[t] * K[t, t]).item())

    cross_term = float(
        (
            r[t]
            * (
                torch.dot(K[t, :], r)
                - K[t, t] * r[t]
            )
        ).item()
    )

    total_term = self_term + cross_term

    interference_ratio = (
        -cross_term / max(abs(self_term), 1.0e-300)
    )

    offdiag = K - torch.diag(torch.diag(K))

    offdiag_frob_ratio = float(
        (
            torch.linalg.vector_norm(offdiag)
            /
            torch.clamp(
                torch.linalg.vector_norm(K),
                min=1.0e-300,
            )
        ).item()
    )

    row = Kcorr[t, :].detach().clone()
    row[t] = 0.0

    max_abs_target_corr = float(torch.max(torch.abs(row)).item())
    most_negative_target_corr = float(torch.min(row).item())
    most_positive_target_corr = float(torch.max(row).item())

    eigvals = torch.linalg.eigvalsh(K).clamp_min(0.0)
    trace = torch.sum(eigvals)
    sq = torch.sum(eigvals.square())

    effective_rank_participation = float(
        (
            trace.square()
            / torch.clamp(sq, min=1.0e-300)
        ).item()
    )

    positive = eigvals[eigvals > 1.0e-14 * torch.max(eigvals)]

    if positive.numel() >= 2:
        condition_positive = float(
            (torch.max(positive) / torch.min(positive)).item()
        )
    else:
        condition_positive = float("inf")

    return {
        "target_self_term": self_term,
        "target_cross_term": cross_term,
        "target_self_plus_cross": total_term,
        "target_interference_ratio": interference_ratio,
        "target_raw_kernel_predicts_uphill": bool(total_term < 0.0),

        "kernel_offdiag_frobenius_ratio": offdiag_frob_ratio,
        "target_max_abs_normalized_coupling": max_abs_target_corr,
        "target_most_negative_normalized_coupling": most_negative_target_corr,
        "target_most_positive_normalized_coupling": most_positive_target_corr,

        "kernel_trace": float(trace.item()),
        "kernel_effective_rank_participation": effective_rank_participation,
        "kernel_positive_condition_estimate": condition_positive,
    }


# =============================================================================
# Adam-state-aware decomposition
# =============================================================================

def adam_decomposition(
    exp,
    params,
    g_total: torch.Tensor,
    J: torch.Tensor,
    r: torch.Tensor,
) -> dict:

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

    D_parts = []
    current_parts = []
    history_parts = []
    candidate_parts = []

    offset = 0
    step_values = []

    for p in params:
        n = p.numel()
        gp = g_total[offset:offset+n].reshape_as(p)
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

        step_new = step_old + 1
        step_values.append(step_new)

        v_new = beta2 * v_old + (1.0 - beta2) * gp.square()

        bc1 = 1.0 - beta1 ** step_new
        bc2 = 1.0 - beta2 ** step_new

        Dp = 1.0 / (
            v_new.sqrt() / math.sqrt(bc2) + eps
        )

        coeff_current = (1.0 - beta1) / bc1
        coeff_history = beta1 / bc1

        delta_current = -lr * Dp * coeff_current * gp
        delta_history = -lr * Dp * coeff_history * m_old
        delta_candidate = delta_current + delta_history

        D_parts.append(Dp.reshape(-1))
        current_parts.append(delta_current.reshape(-1))
        history_parts.append(delta_history.reshape(-1))
        candidate_parts.append(delta_candidate.reshape(-1))

    if len(set(step_values)) != 1:
        raise RuntimeError(
            "Adam parameter groups have inconsistent step counters; "
            "Stage 17R decomposition assumes a common optimizer step."
        )

    D = flatten(D_parts)
    delta_current = flatten(current_parts)
    delta_history = flatten(history_parts)
    delta_candidate = flatten(candidate_parts)

    # State-dependent Adam current-gradient interaction kernel.
    JD = J * D[None, :]
    K_D = JD @ J.T

    diag = torch.diag(K_D).clamp_min(1.0e-300)
    K_D_corr = K_D / torch.sqrt(diag[:, None] * diag[None, :])

    t = TARGET_INDEX

    self_D = float((r[t] * r[t] * K_D[t, t]).item())
    cross_D = float(
        (
            r[t]
            * (
                torch.dot(K_D[t, :], r)
                - K_D[t, t] * r[t]
            )
        ).item()
    )

    interference_D = (
        -cross_D / max(abs(self_D), 1.0e-300)
    )

    dr_current = J @ delta_current
    dr_history = J @ delta_history
    dr_candidate = dr_current + dr_history

    M = r.numel()

    dT_current = float(
        ((2.0 / M) * r[t] * dr_current[t]).item()
    )
    dT_history = float(
        ((2.0 / M) * r[t] * dr_history[t]).item()
    )
    dT_candidate = dT_current + dT_history

    dL_current = float(
        ((2.0 / M) * torch.dot(r, dr_current)).item()
    )
    dL_history = float(
        ((2.0 / M) * torch.dot(r, dr_history)).item()
    )
    dL_candidate = dL_current + dL_history

    row = K_D_corr[t, :].detach().clone()
    row[t] = 0.0

    return {
        "D": D,
        "K_D": K_D,
        "K_D_corr": K_D_corr,

        "delta_current": delta_current,
        "delta_history": delta_history,
        "delta_candidate": delta_candidate,

        "adam_target_self_term": self_D,
        "adam_target_cross_term": cross_D,
        "adam_target_interference_ratio": interference_D,

        "adam_current_first_order_target_change": dT_current,
        "adam_history_first_order_target_change": dT_history,
        "adam_candidate_first_order_target_change": dT_candidate,

        "adam_current_first_order_total_change": dL_current,
        "adam_history_first_order_total_change": dL_history,
        "adam_candidate_first_order_total_change": dL_candidate,

        "adam_target_max_abs_normalized_coupling":
            float(torch.max(torch.abs(row)).item()),

        "adam_candidate_norm":
            float(torch.linalg.vector_norm(delta_candidate).item()),

        "adam_current_norm":
            float(torch.linalg.vector_norm(delta_current).item()),

        "adam_history_norm":
            float(torch.linalg.vector_norm(delta_history).item()),
    }


# =============================================================================
# Read-only adaptive direction
# =============================================================================

def read_only_adaptive_direction(
    exp,
    stage15,
    kernel: dict,
) -> dict:

    residuals = exp.weak_residuals()
    params = kernel["params"]
    r = kernel["r"]
    J = kernel["J"]

    M = residuals.numel()
    t = TARGET_INDEX

    target_loss = residuals[t].square() / M
    total_loss = residuals.square().mean()

    gt_parts = torch.autograd.grad(
        target_loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )

    gL_parts = torch.autograd.grad(
        total_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    gt = flatten(gt_parts).detach()
    gL = flatten(gL_parts).detach()

    candidate = stage15.predict_adam_candidate(
        exp=exp,
        params=params,
        g_total=gL,
    )

    candidate_target_dot = float(torch.dot(gt, candidate).item())
    candidate_total_dot = float(torch.dot(gL, candidate).item())

    if candidate_target_dot <= 0.0:
        raise RuntimeError(
            "Stage-16 adaptive failure state unexpectedly has an inherited "
            "Adam candidate that is target-nonuphill."
        )

    gt2 = torch.dot(gt, gt)

    reflected = candidate - 2.0 * (
        torch.dot(gt, candidate) / gt2
    ) * gt

    reflect_target_dot = float(torch.dot(gt, reflected).item())
    reflect_total_dot = float(torch.dot(gL, reflected).item())

    int_L = stage15.strict_negative_interval(
        candidate_total_dot,
        reflect_total_dot,
    )

    int_T = stage15.strict_negative_interval(
        candidate_target_dot,
        reflect_target_dot,
    )

    lo, hi = stage15.intersect_open_intervals(
        [int_L, int_T]
    )

    if not (
        math.isfinite(lo)
        and math.isfinite(hi)
        and lo < hi
    ):
        raise RuntimeError("No strict Pareto interval at Stage-16 failure state.")

    lam = 0.5 * (lo + hi)

    d = (1.0 - lam) * candidate + lam * reflected

    first_total = float(torch.dot(gL, d).item())
    first_target = float(torch.dot(gt, d).item())

    if not (first_total < 0.0 and first_target < 0.0):
        raise RuntimeError(
            "Adaptive midpoint is not strict first-order joint descent."
        )

    return {
        "direction": d.detach(),
        "g_total": gL,
        "g_target": gt,
        "candidate": candidate.detach(),
        "reflected": reflected.detach(),

        "lambda_mid": lam,
        "interval_lower": lo,
        "interval_upper": hi,
        "interval_width": hi - lo,

        "first_order_total": first_total,
        "first_order_target": first_target,

        "candidate_total_dot": candidate_total_dot,
        "candidate_target_dot": candidate_target_dot,
        "reflect_total_dot": reflect_total_dot,
        "reflect_target_dot": reflect_target_dot,
    }


# =============================================================================
# Curvature decomposition
# =============================================================================

def directional_hessian(
    scalar: torch.Tensor,
    params,
    d: torch.Tensor,
) -> float:

    grad_parts = torch.autograd.grad(
        scalar,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )

    g = flatten(grad_parts)

    directional_grad = torch.dot(g, d)

    hv_parts = torch.autograd.grad(
        directional_grad,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )

    hv = flatten(hv_parts).detach()

    return float(torch.dot(d, hv).item())


def curvature_decomposition(
    exp,
    kernel: dict,
    adaptive: dict,
) -> dict:

    d = adaptive["direction"]
    J = kernel["J"]
    params = kernel["params"]

    residuals = exp.weak_residuals()
    M = residuals.numel()
    t = TARGET_INDEX

    total_loss = residuals.square().mean()
    target_loss = residuals[t].square() / M

    kappa_total = directional_hessian(
        scalar=total_loss,
        params=params,
        d=d,
    )

    # Fresh graph for target Hessian.
    residuals_t = exp.weak_residuals()
    target_loss_fresh = residuals_t[t].square() / M

    kappa_target = directional_hessian(
        scalar=target_loss_fresh,
        params=params,
        d=d,
    )

    Jd = J @ d

    kappa_GN = float(
        ((2.0 / M) * torch.dot(Jd, Jd)).item()
    )

    kappa_NL = kappa_total - kappa_GN

    jt_d = float(torch.dot(J[t, :], d).item())

    kappa_target_GN = (2.0 / M) * (jt_d ** 2)
    kappa_target_NL = kappa_target - kappa_target_GN

    delta2_total = (
        adaptive["first_order_total"]
        + 0.5 * kappa_total
    )

    delta2_target = (
        adaptive["first_order_target"]
        + 0.5 * kappa_target
    )

    return {
        "kappa_total": kappa_total,
        "kappa_GN": kappa_GN,
        "kappa_NL": kappa_NL,

        "kappa_target_total": kappa_target,
        "kappa_target_GN": kappa_target_GN,
        "kappa_target_NL": kappa_target_NL,

        "second_order_predicted_total_change": delta2_total,
        "second_order_predicted_target_change": delta2_target,

        "GN_fraction_of_total_curvature": (
            kappa_GN / kappa_total
            if abs(kappa_total) > 1.0e-300
            else float("nan")
        ),
    }


# =============================================================================
# Model state metrics
# =============================================================================

def state_metrics(exp) -> dict:
    residuals = exp.weak_residuals().detach()
    energy = residuals.square()
    total_energy = energy.sum().clamp_min(1.0e-300)

    return {
        "relative_l2_error": exp.relative_l2_error(),
        "vpinn_loss": float(torch.mean(energy).item()),
        "target_loss": float(
            (energy[TARGET_INDEX] / energy.numel()).item()
        ),
        "target_share": float(
            (energy[TARGET_INDEX] / total_energy).item()
        ),
    }


# =============================================================================
# Reconstruction
# =============================================================================

def reconstruct_branch_state(
    stage3,
    stage9,
    stage15,
    stage5_dir: Path,
    device: torch.device,
    seed: int,
    branch_epoch: int,
    target_epoch: int,
    branch: str,
    out_dir: Path,
):
    """
    Reconstruct state at target_epoch, where target_epoch is a PRE-STEP epoch.
    """
    checkpoint = (
        stage5_dir
        / f"seed_{seed:03d}"
        / "locked_state_epoch_2500.pt"
    )

    exp = stage9.make_experiment(
        stage3=stage3,
        device=device,
        seed=seed,
        out_dir=out_dir,
    )

    stage9.load_locked_checkpoint(
        exp=exp,
        checkpoint_path=checkpoint,
        expected_seed=seed,
    )

    for _ in range(2500, branch_epoch):
        stage9.intervention_step(
            exp=exp,
            branch="REFLECT",
            target_mode=TARGET_MODE,
        )

    epoch = branch_epoch

    if branch == "CONTROL":
        while epoch < target_epoch:
            exp.optimizer.zero_grad(set_to_none=True)
            residuals = exp.weak_residuals()
            loss = residuals.square().mean()
            loss.backward()
            exp.optimizer.step()
            epoch += 1

    elif branch == "ADAPTIVE":
        while epoch < target_epoch:
            result = stage15.adaptive_midpoint_step(
                exp=exp,
                target_mode=TARGET_MODE,
            )

            if result["status"] != "OK":
                raise RuntimeError(
                    f"Seed {seed}: unexpected adaptive failure before "
                    f"target epoch {target_epoch}; got {result['status']} "
                    f"on step from epoch {epoch}."
                )

            epoch += 1

    else:
        raise ValueError(branch)

    return exp


# =============================================================================
# Failure replay
# =============================================================================

def replay_failure_one_step(
    stage3,
    stage9,
    stage15,
    device,
    source_exp,
    seed: int,
    expected_failure_row: dict,
    out_dir: Path,
) -> dict:

    captured = capture_state(source_exp)

    clone = stage9.make_experiment(
        stage3=stage3,
        device=device,
        seed=seed,
        out_dir=out_dir,
    )

    load_captured_state(clone, captured)

    result = stage15.adaptive_midpoint_step(
        exp=clone,
        target_mode=TARGET_MODE,
    )

    if result["status"] != "NONLINEAR_SAFETY_FAILURE":
        raise RuntimeError(
            f"Seed {seed}: expected Stage-16 NONLINEAR_SAFETY_FAILURE, "
            f"got {result['status']}."
        )

    post = state_metrics(clone)

    diffs = {
        "relative_l2_error": abs(
            post["relative_l2_error"]
            - float(expected_failure_row["relative_l2_error"])
        ),
        "vpinn_loss": abs(
            post["vpinn_loss"]
            - float(expected_failure_row["vpinn_loss"])
        ),
        "target_share": abs(
            post["target_share"]
            - float(
                expected_failure_row[
                    "target_mode_residual_energy_share"
                ]
            )
        ),
    }

    max_diff = max(diffs.values())

    if max_diff > REPLAY_TOL:
        raise RuntimeError(
            f"Seed {seed}: failure replay mismatch {max_diff:.3e}."
        )

    return {
        "result": result,
        "post": post,
        "max_replay_difference": max_diff,
        "field_differences": diffs,
    }


# =============================================================================
# Plots
# =============================================================================

def plot_interference(rows: List[dict], path: Path) -> None:
    adaptive = [r for r in rows if r["branch"] == "ADAPTIVE"]
    control = [r for r in rows if r["branch"] == "CONTROL"]

    seeds = [int(r["seed"]) for r in adaptive]
    x = np.arange(len(seeds))
    width = 0.36

    amap = {int(r["seed"]): r for r in adaptive}
    cmap = {int(r["seed"]): r for r in control}

    a = [float(amap[s]["target_interference_ratio"]) for s in seeds]
    c = [float(cmap[s]["target_interference_ratio"]) for s in seeds]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(x - width/2, c, width, label="Matched Adam control")
    ax.bar(x + width/2, a, width, label="Adaptive pre-failure")
    ax.axhline(1.0, linestyle="--", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("- target CROSS / target SELF")
    ax.set_title("Finite-width weak-test interference at matched epochs")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_curvature(
    rows: List[dict],
    path: Path,
    objective: str,
) -> None:
    seeds = [int(r["seed"]) for r in rows]
    x = np.arange(len(seeds))
    width = 0.26

    if objective == "total":
        first_key = "first_order_total"
        gn_key = "kappa_GN"
        nl_key = "kappa_NL"
        ylabel = "Contribution to second-order ΔL model"
        title = "Total objective: what overturns first-order descent?"
    elif objective == "target":
        first_key = "first_order_target"
        gn_key = "kappa_target_GN"
        nl_key = "kappa_target_NL"
        ylabel = "Contribution to second-order ΔT model"
        title = "Target objective: what overturns first-order descent?"
    else:
        raise ValueError(objective)

    first = [float(r[first_key]) for r in rows]
    half_gn = [0.5 * float(r[gn_key]) for r in rows]
    half_nl = [0.5 * float(r[nl_key]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.bar(x - width, first, width, label="First-order descent")
    ax.bar(x, half_gn, width, label="1/2 Gauss-Newton curvature")
    ax.bar(x + width, half_nl, width, label="1/2 nonlinear residual curvature")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_prediction(
    rows: List[dict],
    path: Path,
    objective: str,
) -> None:
    seeds = [int(r["seed"]) for r in rows]
    x = np.arange(len(seeds))
    width = 0.36

    if objective == "total":
        pred_key = "second_order_predicted_total_change"
        exact_key = "exact_total_loss_change"
        ylabel = "Total-loss change"
        title = "Total objective: second-order prediction vs exact step"
    elif objective == "target":
        pred_key = "second_order_predicted_target_change"
        exact_key = "exact_target_loss_change"
        ylabel = "Target-loss change"
        title = "Target objective: second-order prediction vs exact step"
    else:
        raise ValueError(objective)

    pred = [float(r[pred_key]) for r in rows]
    exact = [float(r[exact_key]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(x - width/2, pred, width, label="Second-order prediction")
    ax.bar(x + width/2, exact, width, label="Exact nonlinear step")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_target_kernel_heatmap(
    matrix: np.ndarray,
    seed: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0)

    ax.set_xlabel("Test mode j")
    ax.set_ylabel("Test mode i")
    ax.set_title(
        f"Seed {seed}: signed cosine of squared weak-residual gradients"
    )

    fig.colorbar(im, ax=ax)
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

    stage5_dir = Path(args.stage5_dir)
    if not stage5_dir.is_absolute():
        stage5_dir = root / stage5_dir

    stage9_script = Path(args.stage9_script)
    if not stage9_script.is_absolute():
        stage9_script = root / stage9_script

    stage15_script = Path(args.stage15_script)
    if not stage15_script.is_absolute():
        stage15_script = root / stage15_script

    stage16_dir = Path(args.stage16_dir)
    if not stage16_dir.is_absolute():
        stage16_dir = root / stage16_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage5_dir=stage5_dir,
        stage9_script=stage9_script,
        stage15_script=stage15_script,
        stage16_dir=stage16_dir,
    )

    stage9 = load_module(
        stage9_script,
        "vpinn_stage9_stage17R",
    )

    stage15 = load_module(
        stage15_script,
        "vpinn_stage15_stage17R",
    )

    stage3 = stage9.load_stage3_module(stage3_script)

    print("=" * 164)
    print(
        "VPINN — STAGE 17R FINITE-WIDTH WEAK-TEST INTERACTION + CURVATURE AUDIT"
    )
    print("=" * 164)
    print(f"device                    : {device}")
    print(f"branch epochs             : {pf['branch_map']}")
    print(f"adaptive failure epochs   : {pf['failure_map']}")
    print("new training              : NONE")
    print("optimizer rescue          : NONE")
    print("=" * 164)

    # Independent function-space Gram audit.
    test_gram = h1_test_gram(M=24)
    test_offdiag = test_gram - np.diag(np.diag(test_gram))

    test_gram_diag_error = float(
        np.max(np.abs(np.diag(test_gram) - 1.0))
    )
    test_gram_max_offdiag = float(
        np.max(np.abs(test_offdiag))
    )

    np.savez_compressed(
        out_dir / "test_space_gram.npz",
        gram=test_gram,
    )

    kernel_rows = []
    curvature_rows = []
    failure_replay_rows = []

    start = time.perf_counter()

    for seed in range(4):
        branch_epoch = pf["branch_map"][seed]
        failure_epoch = pf["failure_map"][seed]
        pre_failure_epoch = failure_epoch - 1

        print()
        print("-" * 164)
        print(
            f"SEED {seed}: branch={branch_epoch}, "
            f"adaptive failure={failure_epoch}, "
            f"audit pre-failure state={pre_failure_epoch}"
        )

        adaptive_exp = reconstruct_branch_state(
            stage3=stage3,
            stage9=stage9,
            stage15=stage15,
            stage5_dir=stage5_dir,
            device=device,
            seed=seed,
            branch_epoch=branch_epoch,
            target_epoch=pre_failure_epoch,
            branch="ADAPTIVE",
            out_dir=out_dir / f"seed_{seed:03d}" / "adaptive",
        )

        control_exp = reconstruct_branch_state(
            stage3=stage3,
            stage9=stage9,
            stage15=stage15,
            stage5_dir=stage5_dir,
            device=device,
            seed=seed,
            branch_epoch=branch_epoch,
            target_epoch=pre_failure_epoch,
            branch="CONTROL",
            out_dir=out_dir / f"seed_{seed:03d}" / "control",
        )

        # -------------------------------------------------------------
        # Exact adaptive failure replay.
        # -------------------------------------------------------------
        failure_replay = replay_failure_one_step(
            stage3=stage3,
            stage9=stage9,
            stage15=stage15,
            device=device,
            source_exp=adaptive_exp,
            seed=seed,
            expected_failure_row=pf["failure_rows"][seed],
            out_dir=out_dir / f"seed_{seed:03d}" / "failure_replay",
        )

        failure_result = failure_replay["result"]

        failure_replay_rows.append(
            {
                "seed": seed,
                "failure_epoch": failure_epoch,
                "max_replay_difference":
                    failure_replay["max_replay_difference"],
                "pass":
                    failure_replay["max_replay_difference"] <= REPLAY_TOL,
            }
        )

        # -------------------------------------------------------------
        # Kernel audits for both matched states.
        # -------------------------------------------------------------
        branch_cache = {}

        for branch, exp in (
            ("CONTROL", control_exp),
            ("ADAPTIVE", adaptive_exp),
        ):
            kdat = residual_jacobian(exp)

            residuals = exp.weak_residuals()
            total_loss = residuals.square().mean()
            params = kdat["params"]

            gL_parts = torch.autograd.grad(
                total_loss,
                params,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )

            gL = flatten(gL_parts).detach()

            raw_summary = kernel_summary(
                r=kdat["r"],
                K=kdat["K"],
                Kcorr=kdat["Kcorr"],
            )

            adam = adam_decomposition(
                exp=exp,
                params=params,
                g_total=gL,
                J=kdat["J"],
                r=kdat["r"],
            )

            sm = state_metrics(exp)

            row = {
                "seed": seed,
                "branch": branch,
                "epoch": pre_failure_epoch,

                "test_gram_diag_error":
                    test_gram_diag_error,
                "test_gram_max_abs_offdiag":
                    test_gram_max_offdiag,

                "relative_l2_error":
                    sm["relative_l2_error"],
                "vpinn_loss":
                    sm["vpinn_loss"],
                "target_loss":
                    sm["target_loss"],
                "target_share":
                    sm["target_share"],

                **raw_summary,

                "adam_target_self_term":
                    adam["adam_target_self_term"],
                "adam_target_cross_term":
                    adam["adam_target_cross_term"],
                "adam_target_interference_ratio":
                    adam["adam_target_interference_ratio"],

                "adam_current_first_order_target_change":
                    adam["adam_current_first_order_target_change"],
                "adam_history_first_order_target_change":
                    adam["adam_history_first_order_target_change"],
                "adam_candidate_first_order_target_change":
                    adam["adam_candidate_first_order_target_change"],

                "adam_current_first_order_total_change":
                    adam["adam_current_first_order_total_change"],
                "adam_history_first_order_total_change":
                    adam["adam_history_first_order_total_change"],
                "adam_candidate_first_order_total_change":
                    adam["adam_candidate_first_order_total_change"],

                "adam_target_max_abs_normalized_coupling":
                    adam["adam_target_max_abs_normalized_coupling"],

                "adam_candidate_norm":
                    adam["adam_candidate_norm"],
                "adam_current_norm":
                    adam["adam_current_norm"],
                "adam_history_norm":
                    adam["adam_history_norm"],
            }

            kernel_rows.append(row)

            branch_cache[branch] = {
                "exp": exp,
                "kernel": kdat,
                "adam": adam,
            }

            np.savez_compressed(
                out_dir
                / f"seed_{seed:03d}_{branch.lower()}_kernels.npz",
                residuals=kdat["r"].cpu().numpy(),
                raw_kernel=kdat["K"].cpu().numpy(),
                raw_kernel_corr=kdat["Kcorr"].cpu().numpy(),
                signed_squared_residual_gradient_cosine=
                    kdat["squared_residual_grad_cos"].cpu().numpy(),
                adam_current_kernel=adam["K_D"].cpu().numpy(),
                adam_current_kernel_corr=adam["K_D_corr"].cpu().numpy(),
            )

        # -------------------------------------------------------------
        # Curvature audit only at exact adaptive pre-failure state.
        # -------------------------------------------------------------
        adaptive_readonly = read_only_adaptive_direction(
            exp=adaptive_exp,
            stage15=stage15,
            kernel=branch_cache["ADAPTIVE"]["kernel"],
        )

        # Verify read-only midpoint geometry agrees with the actual failure step.
        lambda_gap = abs(
            adaptive_readonly["lambda_mid"]
            - float(failure_result["lambda_mid"])
        )

        if lambda_gap > REPLAY_TOL:
            raise RuntimeError(
                f"Seed {seed}: read-only lambda mismatch {lambda_gap:.3e}."
            )

        curv = curvature_decomposition(
            exp=adaptive_exp,
            kernel=branch_cache["ADAPTIVE"]["kernel"],
            adaptive=adaptive_readonly,
        )

        exact_dL = float(failure_result["total_loss_change"])
        exact_dT = float(failure_result["target_loss_change"])

        pre_total = float(failure_result["pre_total_loss"])
        pre_target = float(failure_result["pre_target_loss"])

        tol_total = LOSS_TOL_FACTOR * max(1.0, abs(pre_total))
        tol_target = LOSS_TOL_FACTOR * max(1.0, abs(pre_target))

        # Stage 15/16 defines NONLINEAR_SAFETY_FAILURE when EITHER
        # objective increases beyond tolerance. Do not assume the total
        # objective is necessarily the trigger.
        total_failure = bool(exact_dL > tol_total)
        target_failure = bool(exact_dT > tol_target)

        if not (total_failure or target_failure):
            raise RuntimeError(
                f"Seed {seed}: replay reports NONLINEAR_SAFETY_FAILURE, "
                "but neither exact objective exceeds its safety tolerance. "
                f"dL={exact_dL:+.6e}, tolL={tol_total:.6e}, "
                f"dT={exact_dT:+.6e}, tolT={tol_target:.6e}."
            )

        # The adaptive midpoint construction itself must still be strict
        # first-order descent for BOTH objectives.
        if not (
            adaptive_readonly["first_order_total"] < 0.0
            and adaptive_readonly["first_order_target"] < 0.0
        ):
            raise RuntimeError(
                f"Seed {seed}: adaptive midpoint is not strict first-order "
                "joint descent at the reconstructed pre-failure state."
            )

        def classify_objective(is_failure, second_order_prediction):
            if not is_failure:
                return "NO_EXACT_FAILURE"
            if second_order_prediction > 0.0:
                return "SECOND_ORDER_CURVATURE_BARRIER"
            return "HIGHER_ORDER_BARRIER"

        total_mechanism = classify_objective(
            total_failure,
            curv["second_order_predicted_total_change"],
        )

        target_mechanism = classify_objective(
            target_failure,
            curv["second_order_predicted_target_change"],
        )

        failed_mechanisms = [
            mechanism
            for is_failure, mechanism in (
                (total_failure, total_mechanism),
                (target_failure, target_mechanism),
            )
            if is_failure
        ]

        if all(
            mechanism == "SECOND_ORDER_CURVATURE_BARRIER"
            for mechanism in failed_mechanisms
        ):
            classification = "SECOND_ORDER_EXPLAINS_ALL_TRIGGERING_OBJECTIVES"
        elif any(
            mechanism == "SECOND_ORDER_CURVATURE_BARRIER"
            for mechanism in failed_mechanisms
        ):
            classification = "MIXED_SECOND_AND_HIGHER_ORDER_BARRIER"
        else:
            classification = "HIGHER_ORDER_BARRIER"

        second_order_remainder_total = (
            exact_dL
            - curv["second_order_predicted_total_change"]
        )

        second_order_remainder_target = (
            exact_dT
            - curv["second_order_predicted_target_change"]
        )

        row = {
            "seed": seed,
            "pre_failure_epoch": pre_failure_epoch,
            "failure_epoch": failure_epoch,

            "lambda_mid":
                adaptive_readonly["lambda_mid"],
            "interval_lower":
                adaptive_readonly["interval_lower"],
            "interval_upper":
                adaptive_readonly["interval_upper"],
            "interval_width":
                adaptive_readonly["interval_width"],

            "first_order_total":
                adaptive_readonly["first_order_total"],
            "first_order_target":
                adaptive_readonly["first_order_target"],

            **curv,

            "exact_total_loss_change":
                exact_dL,
            "exact_target_loss_change":
                exact_dT,

            "total_safety_tolerance":
                tol_total,
            "target_safety_tolerance":
                tol_target,

            "total_objective_triggered_failure":
                total_failure,
            "target_objective_triggered_failure":
                target_failure,

            "total_failure_mechanism":
                total_mechanism,
            "target_failure_mechanism":
                target_mechanism,

            "third_and_higher_total_remainder":
                second_order_remainder_total,

            "third_and_higher_target_remainder":
                second_order_remainder_target,

            "second_order_total_sign_matches_exact":
                bool(
                    (
                        curv["second_order_predicted_total_change"] == 0.0
                        and exact_dL == 0.0
                    )
                    or
                    (
                        curv["second_order_predicted_total_change"] * exact_dL
                        > 0.0
                    )
                ),

            "second_order_target_sign_matches_exact":
                bool(
                    (
                        curv["second_order_predicted_target_change"] == 0.0
                        and exact_dT == 0.0
                    )
                    or
                    (
                        curv["second_order_predicted_target_change"] * exact_dT
                        > 0.0
                    )
                ),

            "classification":
                classification,

            "failure_replay_max_abs_difference":
                failure_replay["max_replay_difference"],

            "read_only_lambda_gap":
                lambda_gap,
        }

        curvature_rows.append(row)

        print(
            f"  raw target interference "
            f"(adaptive/control) = "
            f"{kernel_rows[-1]['target_interference_ratio']:.6f} / "
            f"{kernel_rows[-2]['target_interference_ratio']:.6f}"
        )

        print(
            f"  Adam target first-order change: "
            f"current={branch_cache['ADAPTIVE']['adam']['adam_current_first_order_target_change']:+.3e}, "
            f"history={branch_cache['ADAPTIVE']['adam']['adam_history_first_order_target_change']:+.3e}"
        )

        print(
            f"  TOTAL : first={adaptive_readonly['first_order_total']:+.6e}, "
            f"1/2H={0.5*curv['kappa_total']:+.6e}, "
            f"second-order={curv['second_order_predicted_total_change']:+.6e}, "
            f"exact={exact_dL:+.6e}, "
            f"trigger={total_failure}, mechanism={total_mechanism}"
        )

        print(
            f"  TARGET: first={adaptive_readonly['first_order_target']:+.6e}, "
            f"1/2H={0.5*curv['kappa_target_total']:+.6e}, "
            f"second-order={curv['second_order_predicted_target_change']:+.6e}, "
            f"exact={exact_dT:+.6e}, "
            f"trigger={target_failure}, mechanism={target_mechanism}"
        )

        print(
            f"  TOTAL curvature decomposition: "
            f"GN={curv['kappa_GN']:+.6e}, "
            f"nonlinear={curv['kappa_NL']:+.6e}"
        )

        print(
            f"  TARGET curvature decomposition: "
            f"GN={curv['kappa_target_GN']:+.6e}, "
            f"nonlinear={curv['kappa_target_NL']:+.6e}, "
            f"overall class={classification}"
        )

        plot_target_kernel_heatmap(
            matrix=branch_cache["ADAPTIVE"]["kernel"][
                "squared_residual_grad_cos"
            ].cpu().numpy(),
            seed=seed,
            path=out_dir / f"seed_{seed:03d}_weak_test_interaction_heatmap.png",
        )

    # -----------------------------------------------------------------
    # Aggregate decision.
    # -----------------------------------------------------------------
    barrier_count = sum(
        int(
            r["classification"]
            == "SECOND_ORDER_EXPLAINS_ALL_TRIGGERING_OBJECTIVES"
        )
        for r in curvature_rows
    )

    mixed_count = sum(
        int(
            r["classification"]
            == "MIXED_SECOND_AND_HIGHER_ORDER_BARRIER"
        )
        for r in curvature_rows
    )

    higher_count = sum(
        int(r["classification"] == "HIGHER_ORDER_BARRIER")
        for r in curvature_rows
    )

    total_trigger_count = sum(
        int(bool(r["total_objective_triggered_failure"]))
        for r in curvature_rows
    )

    target_trigger_count = sum(
        int(bool(r["target_objective_triggered_failure"]))
        for r in curvature_rows
    )

    replay_pass = all(bool(r["pass"]) for r in failure_replay_rows)

    if barrier_count >= 3 and replay_pass:
        route_class = (
            "finite_width_second_order_barrier_reproducible"
        )
        next_route = (
            "stage18R_frequency_transfer_kernel_curvature_mechanism"
        )
    else:
        route_class = (
            "higher_order_or_mixed_failure_mechanism"
        )
        next_route = (
            "stage18R_higher_order_local_remainder_audit"
        )

    decision = {
        "test_space_gram_diag_error":
            test_gram_diag_error,

        "test_space_gram_max_abs_offdiag":
            test_gram_max_offdiag,

        "failure_replay_pass":
            replay_pass,

        "second_order_explains_all_triggering_objectives_count":
            barrier_count,

        "mixed_second_and_higher_order_barrier_count":
            mixed_count,

        "higher_order_barrier_count":
            higher_count,

        "total_objective_failure_trigger_count":
            total_trigger_count,

        "target_objective_failure_trigger_count":
            target_trigger_count,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "novelty_guardrail": (
            "Do not claim J J^T residual-kernel dynamics as new. "
            "The candidate contribution is the finite-width, actual-Adam-state "
            "weak-test interaction decomposition plus curvature breakdown of "
            "locally Pareto-safe directions, subject to literature validation "
            "and frequency transfer."
        ),
    }

    write_csv(
        out_dir / "kernel_branch_comparison.csv",
        kernel_rows,
    )

    write_csv(
        out_dir / "curvature_failure_decomposition.csv",
        curvature_rows,
    )

    write_csv(
        out_dir / "failure_replay_checks.csv",
        failure_replay_rows,
    )

    write_json(
        out_dir / "decision.json",
        decision,
    )

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
        "stage15_script_sha256":
            pf["stage15_sha256"],
        "stage17R_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "stage16_dir":
            str(stage16_dir),

        "precommitment": {
            "stage":
                "finite_width_weak_test_interaction_and_curvature_barrier_audit",

            "new_training":
                False,

            "optimizer_rescue":
                False,

            "primary_failure_classification": (
                "For every objective that actually triggers the Stage-16 "
                "safety gate, classify SECOND_ORDER_CURVATURE_BARRIER iff its "
                "first-order derivative is negative, its exact finite-step "
                "change is unsafe, and its second-order Taylor prediction is "
                "positive."
            ),

            "group_gate": (
                ">=3/4 seeds where second order explains ALL objectives that "
                "actually triggered the safety gate"
            ),

            "next_if_pass":
                "stage18R_frequency_transfer_kernel_curvature_mechanism",
        },
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    plot_interference(
        kernel_rows,
        out_dir / "raw_target_interference_control_vs_adaptive.png",
    )

    plot_curvature(
        curvature_rows,
        out_dir / "total_curvature_decomposition.png",
        objective="total",
    )

    plot_curvature(
        curvature_rows,
        out_dir / "target_curvature_decomposition.png",
        objective="target",
    )

    plot_prediction(
        curvature_rows,
        out_dir / "total_second_order_vs_exact_failure.png",
        objective="total",
    )

    plot_prediction(
        curvature_rows,
        out_dir / "target_second_order_vs_exact_failure.png",
        objective="target",
    )

    elapsed = time.perf_counter() - start

    lines = []
    lines.append("=" * 168)
    lines.append(
        "VPINN — STAGE 17R FINITE-WIDTH WEAK-TEST INTERACTION + CURVATURE SUMMARY"
    )
    lines.append("=" * 168)

    lines.append(
        "seed | triggers    | total mechanism                  | "
        "target mechanism                 | class"
    )
    lines.append("-" * 168)

    for r in curvature_rows:
        triggers = (
            ("L" if r["total_objective_triggered_failure"] else "")
            +
            ("T" if r["target_objective_triggered_failure"] else "")
        )

        lines.append(
            f"{int(r['seed']):4d} | "
            f"{triggers:11s} | "
            f"{r['total_failure_mechanism']:32s} | "
            f"{r['target_failure_mechanism']:32s} | "
            f"{r['classification']}"
        )

    lines.append("-" * 168)

    lines.append(
        f"function-space Gram max offdiag      : "
        f"{test_gram_max_offdiag:.3e}"
    )

    lines.append(
        f"Stage-16 failure replay              : "
        f"{sum(int(r['pass']) for r in failure_replay_rows)}/4 PASS"
    )

    lines.append(
        f"2nd-order explains all triggering obj: "
        f"{barrier_count}/4"
    )

    lines.append(
        f"MIXED 2nd/higher-order barrier       : "
        f"{mixed_count}/4"
    )

    lines.append(
        f"HIGHER_ORDER_BARRIER                 : "
        f"{higher_count}/4"
    )

    lines.append(
        f"total-objective failure triggers     : "
        f"{total_trigger_count}/4"
    )

    lines.append(
        f"target-objective failure triggers    : "
        f"{target_trigger_count}/4"
    )

    lines.append(
        f"route class                          : "
        f"{route_class}"
    )

    lines.append(
        f"next route                           : "
        f"{next_route}"
    )

    lines.append(
        f"elapsed seconds                      : "
        f"{elapsed:.2f}"
    )

    lines.append("=" * 168)

    lines.append(
        "Novelty guardrail: J J^T itself is not the novelty target. "
        "The target is finite-width, Adam-state-aware weak-test interaction "
        "plus curvature breakdown, followed by frequency transfer."
    )

    lines.append("=" * 168)

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
