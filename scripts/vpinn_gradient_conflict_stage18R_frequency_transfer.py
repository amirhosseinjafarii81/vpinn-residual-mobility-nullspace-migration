#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 18R
Multi-Seed Phase-Matched Frequency Transfer of Weak-Test Interaction
====================================================================

Scientific purpose
------------------
Stage 17R established, for the m=9 conflict-active trajectory:

  * H_0^1 test functions are orthogonal to numerical precision while their
    finite-width parameter gradients can be strongly coupled;

  * the actual Adam target motion can be produced by two distinct channels:
        - instantaneous second-moment-weighted interaction geometry;
        - stored first-moment/history;

  * all four final adaptive safety failures are explained at second order,
    with target-side failures strongly nonlinear-curvature dominated and the
    seed-2 total-loss failure Gauss-Newton dominated.

Stage 18R asks the next nontrivial question:

    DOES THIS MECHANISM TRANSFER SYSTEMATICALLY WITH TARGET FREQUENCY?

The target frequencies are

    m in {3, 5, 7, 9},

with five paired seeds

    seed in {0,1,2,3,4}.

The exact solution family and weak-scale matching are inherited unchanged
from Stage 3:

    u*_m = sin(pi x) + a_m sin(m pi x),
    a_m  = 0.15 * 7 / m,
    a_m m = 1.05.

Thus frequency is changed while the target weak/derivative scale is matched.

Why phase matching instead of a fixed epoch?
--------------------------------------------
A fixed epoch would compare different learning phases: low frequencies may
already be nearly resolved while m=9 remains locked.

For each (seed,mode), define the PHASE-MATCHED LOCK STATE as the FIRST point
on the precommitted global 25-epoch tracking grid satisfying BOTH

    target residual-energy share >= 0.98
    relative L2 error             >  1e-2.

At this state the target weak mode carries almost all residual energy while
the solution is still not converged.

The trajectory is ordinary Adam only. No intervention is trained.

Efficiency
----------
Each of the 20 trajectories runs only until its first phase-matched state,
or epoch 2500 if no such state occurs.

Expensive residual-Jacobian and Hessian diagnostics are computed exactly ONCE
per phase-matched trajectory. This is the lowest-cost experiment that can
test frequency transfer of the Stage-17R mechanism.

At the phase-matched state
--------------------------
Compute:

1) Raw finite-width weak-test interaction

       J_ij = d R_i / d theta_j
       K    = J J^T

   and normalized target cross-coupling.

2) Adam-state-aware interaction

       K_Adam = J D_t J^T,

   where D_t is the exact current Adam second-moment metric.

3) Exact Adam current/history decomposition

       Delta_A = Delta_current + Delta_history.

   Record the first-order target contributions of both components.

4) Adam target-uphill cosine

       U_Adam =
           <g_T, Delta_A> / (||g_T|| ||Delta_A||),

   where positive values mean the actual next Adam candidate is TARGET-UPHILL.

5) If U_Adam > 0, construct a READ-ONLY current Pareto midpoint between
   inherited Adam and its target reflection. No continuation is performed.

   Measure:
       * strict first-order total/target derivatives;
       * total and target Hessian directional curvatures;
       * Gauss-Newton versus nonlinear residual curvature;
       * exact full-step total/target changes;
       * whether second order predicts any finite-step safety failure.

No lambda tuning, line search, backtracking, reset, or optimizer rescue.

Stage-3 reproduction
--------------------
For seed 0, every Stage-18R tracked state up to the selected phase state is
compared with the already-existing Stage-3 paired frequency-transfer output.
Any difference above 1e-10 aborts the audit.

Primary frequency-transfer gate
-------------------------------
Coverage:
    each mode must have a valid phase-matched state in >=4/5 seeds.

Three precommitted paired endpoint trends compare m=9 against m=3:

    A) phase epoch is later:
           epoch_9 > epoch_3

    B) normalized target parameter-space coupling is stronger:
           max_{j != t}|Khat_tj|_9 > max_{j != t}|Khat_tj|_3

    C) Adam is more target-uphill:
           U_Adam,9 > U_Adam,3

A trend is sign-consistent if it holds in >=4/5 paired seeds.

STRONG FREQUENCY-ORDERED SUPPORT:
    coverage passes
    AND at least 2 of the 3 paired endpoint trends are sign-consistent.

This gate intentionally does NOT require all observables to be monotone at all
four frequencies; with only four discrete modes that would be an unnecessarily
strong postulate.

Secondary evidence
------------------
For every mode report:
    * phase-state coverage;
    * Adam target-uphill prevalence;
    * raw target interference;
    * Adam-metric target interference;
    * current/history target contribution;
    * Pareto-active prevalence;
    * exact Pareto finite-step unsafe prevalence;
    * second-order curvature-barrier prevalence.

Decision routes
---------------
If strong frequency-ordered support passes:
    Stage 19R = dense-frequency transition localization.

If coverage passes but only one/no endpoint trend passes, while m=9 still has
>=4/5 target-uphill prevalence:
    Stage 19R = high-frequency local-band mechanism audit.

Otherwise:
    do NOT advertise a frequency law; route to architecture/phase-heterogeneity
    audit.

This stage tests a mechanism. It does not claim novelty by itself.
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
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


MODES = (3, 5, 7, 9)
SEEDS = (0, 1, 2, 3, 4)

MAX_EPOCH = 2500
TRACK_INTERVAL = 25

PHASE_TARGET_SHARE = 0.98
CONVERGENCE_REL_L2 = 1.0e-2

REPRO_TOL = 1.0e-10
LOSS_TOL_FACTOR = 1.0e-12
ADAM_FORMULA_TOL = 5.0e-12


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-18R phase-matched multi-seed frequency transfer."
    )

    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")

    p.add_argument(
        "--stage3-script",
        default="vpinn_gradient_conflict_stage3_frequency_transfer.py",
    )

    p.add_argument(
        "--stage3-dir",
        default="vpinn_gradient_conflict_stage3_frequency_transfer",
    )

    p.add_argument(
        "--stage17r-dir",
        default="vpinn_gradient_conflict_stage17R_kernel_curvature_audit_v2",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage18R_frequency_transfer",
    )

    return p.parse_args()


# =============================================================================
# Generic helpers
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
        for key in row:
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


def restore_state(exp, state: dict) -> None:
    exp.model.load_state_dict(copy.deepcopy(state["model"]))
    exp.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))

    for item in exp.optimizer.state.values():
        for key, value in list(item.items()):
            if torch.is_tensor(value):
                item[key] = value.to(exp.device)


# =============================================================================
# Preflight
# =============================================================================

def preflight(
    stage3_script: Path,
    stage3_dir: Path,
    stage17r_dir: Path,
) -> dict:

    paths = {
        "s3_manifest": stage3_dir / "manifest.json",
        "s3_tracking": stage3_dir / "aggregate_tracking_metrics.csv",
        "s17_manifest": stage17r_dir / "manifest.json",
        "s17_decision": stage17r_dir / "decision.json",
        "s17_curvature": stage17r_dir / "curvature_failure_decomposition.csv",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing prerequisite files:\n  " + "\n  ".join(missing)
        )

    s3_manifest = read_json(paths["s3_manifest"])
    s17_manifest = read_json(paths["s17_manifest"])
    s17_decision = read_json(paths["s17_decision"])

    actual_s3_sha = sha256_file(stage3_script)

    if s3_manifest.get("script_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 source SHA mismatch against Stage-3 manifest.")

    if s17_manifest.get("stage3_solver_sha256") != actual_s3_sha:
        raise RuntimeError("Stage-3 source SHA mismatch against Stage-17R manifest.")

    if s17_decision.get("next_route") != (
        "stage18R_frequency_transfer_kernel_curvature_mechanism"
    ):
        raise RuntimeError(
            "Stage 17R did not authorize frequency-transfer mechanism audit."
        )

    if int(
        s17_decision.get(
            "second_order_explains_all_triggering_objectives_count",
            -1,
        )
    ) != 4:
        raise RuntimeError(
            "Stage 17R did not report 4/4 second-order explanation."
        )

    s3_tracking = read_csv(paths["s3_tracking"])

    anchor = {
        (int(r["mode"]), int(r["epoch"])): r
        for r in s3_tracking
    }

    for mode in MODES:
        if (mode, 0) not in anchor:
            raise RuntimeError(
                f"Stage-3 seed-0 tracking anchor missing mode={mode}, epoch=0."
            )

    return {
        "stage3_sha256": actual_s3_sha,
        "stage17r_manifest": s17_manifest,
        "stage3_anchor": anchor,
    }


# =============================================================================
# Stage-3 experiment construction
# =============================================================================

def make_experiment(stage3, device, seed: int, mode: int, out_dir: Path):
    cfg = stage3.Config(
        seed=seed,
        device=str(device),
        epochs=MAX_EPOCH,
        learning_rate=1.0e-3,
        width=32,
        depth=3,
        n_test=24,
        n_quad=256,
        n_eval=4001,
        modes=(mode,),
        reference_mode=7,
        reference_amplitude=0.15,
        track_interval=TRACK_INTERVAL,
        diagnostic_epochs=(0,),
        convergence_error_threshold=CONVERGENCE_REL_L2,
        localization_share_threshold=0.80,
        resolved_share_threshold=0.20,
        conflict_gamma_threshold=0.20,
        conflict_weighted_negative_threshold=0.50,
        active_relative_tol=1.0e-8,
        grad_absolute_eps=1.0e-300,
        output_dir=str(out_dir),
        dpi=220,
    )

    return stage3.ModeExperiment(
        cfg=cfg,
        device=device,
        mode=mode,
        out_dir=out_dir,
    )


# =============================================================================
# Residual Jacobian + kernel
# =============================================================================

def residual_jacobian(exp) -> dict:
    residuals = exp.weak_residuals()

    params = tuple(
        p for p in exp.model.parameters()
        if p.requires_grad
    )

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

    Kcorr = (
        K
        /
        torch.sqrt(
            diag[:, None] * diag[None, :]
        )
    )

    return {
        "params": params,
        "r": r,
        "J": J,
        "K": K,
        "Kcorr": Kcorr,
    }


def kernel_summary(
    r: torch.Tensor,
    K: torch.Tensor,
    Kcorr: torch.Tensor,
    target_index: int,
) -> dict:

    t = target_index

    self_term = float(
        (r[t].square() * K[t, t]).item()
    )

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

    row = Kcorr[t, :].detach().clone()
    row[t] = 0.0

    offdiag = K - torch.diag(torch.diag(K))

    eigvals = torch.linalg.eigvalsh(K).clamp_min(0.0)

    trace = torch.sum(eigvals)
    sq = torch.sum(eigvals.square())

    effective_rank = float(
        (
            trace.square()
            /
            torch.clamp(sq, min=1.0e-300)
        ).item()
    )

    return {
        "target_self_term": self_term,
        "target_cross_term": cross_term,
        "target_interference_ratio": interference_ratio,
        "raw_kernel_predicts_target_uphill": bool(total_term < 0.0),

        "target_max_abs_normalized_coupling":
            float(torch.max(torch.abs(row)).item()),

        "target_most_negative_normalized_coupling":
            float(torch.min(row).item()),

        "target_most_positive_normalized_coupling":
            float(torch.max(row).item()),

        "kernel_offdiag_frobenius_ratio":
            float(
                (
                    torch.linalg.vector_norm(offdiag)
                    /
                    torch.clamp(
                        torch.linalg.vector_norm(K),
                        min=1.0e-300,
                    )
                ).item()
            ),

        "kernel_effective_rank_participation":
            effective_rank,
    }


# =============================================================================
# Adam exact decomposition
# =============================================================================

def predict_and_decompose_adam(
    exp,
    J: torch.Tensor,
    r: torch.Tensor,
    params,
    target_index: int,
) -> dict:

    M = r.numel()
    t = target_index

    # Exact raw total and target gradients from J and r.
    g_total = (2.0 / M) * (J.T @ r)
    g_target = (2.0 / M) * r[t] * J[t, :]

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

        state = exp.optimizer.state.get(p, {})

        if "exp_avg" in state:
            m_old = state["exp_avg"]
            v_old = state["exp_avg_sq"]

            step_raw = state["step"]

            step_old = (
                int(step_raw.item())
                if torch.is_tensor(step_raw)
                else int(step_raw)
            )
        else:
            m_old = torch.zeros_like(p)
            v_old = torch.zeros_like(p)
            step_old = 0

        step_new = step_old + 1
        step_values.append(step_new)

        v_new = (
            beta2 * v_old
            + (1.0 - beta2) * gp.square()
        )

        bc1 = 1.0 - beta1 ** step_new
        bc2 = 1.0 - beta2 ** step_new

        Dp = 1.0 / (
            v_new.sqrt() / math.sqrt(bc2)
            + eps
        )

        coeff_current = (1.0 - beta1) / bc1
        coeff_history = beta1 / bc1

        d_current = (
            -lr * Dp * coeff_current * gp
        )

        d_history = (
            -lr * Dp * coeff_history * m_old
        )

        d_candidate = d_current + d_history

        D_parts.append(Dp.reshape(-1))
        current_parts.append(d_current.reshape(-1))
        history_parts.append(d_history.reshape(-1))
        candidate_parts.append(d_candidate.reshape(-1))

    if len(set(step_values)) != 1:
        raise RuntimeError("Inconsistent Adam step counters across parameters.")

    D = flatten(D_parts)
    d_current = flatten(current_parts)
    d_history = flatten(history_parts)
    d_candidate = flatten(candidate_parts)

    # Adam-current state-dependent interaction kernel.
    KD = (J * D[None, :]) @ J.T

    diag = torch.diag(KD).clamp_min(1.0e-300)

    KDcorr = (
        KD
        /
        torch.sqrt(
            diag[:, None] * diag[None, :]
        )
    )

    self_D = float(
        (r[t].square() * KD[t, t]).item()
    )

    cross_D = float(
        (
            r[t]
            * (
                torch.dot(KD[t, :], r)
                - KD[t, t] * r[t]
            )
        ).item()
    )

    adam_interference = (
        -cross_D / max(abs(self_D), 1.0e-300)
    )

    target_dot_current = float(
        torch.dot(g_target, d_current).item()
    )

    target_dot_history = float(
        torch.dot(g_target, d_history).item()
    )

    target_dot_candidate = float(
        torch.dot(g_target, d_candidate).item()
    )

    total_dot_current = float(
        torch.dot(g_total, d_current).item()
    )

    total_dot_history = float(
        torch.dot(g_total, d_history).item()
    )

    total_dot_candidate = float(
        torch.dot(g_total, d_candidate).item()
    )

    denom = (
        torch.linalg.vector_norm(g_target)
        *
        torch.linalg.vector_norm(d_candidate)
    ).clamp_min(1.0e-300)

    target_uphill_cosine = float(
        (
            torch.dot(g_target, d_candidate)
            / denom
        ).item()
    )

    row = KDcorr[t, :].detach().clone()
    row[t] = 0.0

    return {
        "g_total": g_total,
        "g_target": g_target,

        "D": D,
        "K_D": KD,
        "K_D_corr": KDcorr,

        "delta_current": d_current,
        "delta_history": d_history,
        "delta_candidate": d_candidate,

        "adam_target_self_term": self_D,
        "adam_target_cross_term": cross_D,
        "adam_target_interference_ratio": adam_interference,

        "adam_target_max_abs_normalized_coupling":
            float(torch.max(torch.abs(row)).item()),

        "target_dot_current": target_dot_current,
        "target_dot_history": target_dot_history,
        "target_dot_candidate": target_dot_candidate,

        "total_dot_current": total_dot_current,
        "total_dot_history": total_dot_history,
        "total_dot_candidate": total_dot_candidate,

        "adam_target_uphill_cosine":
            target_uphill_cosine,

        "adam_candidate_target_uphill":
            bool(target_dot_candidate > 0.0),

        "candidate_norm":
            float(torch.linalg.vector_norm(d_candidate).item()),

        "current_norm":
            float(torch.linalg.vector_norm(d_current).item()),

        "history_norm":
            float(torch.linalg.vector_norm(d_history).item()),
    }


def verify_predicted_adam_step(
    exp,
    predicted: torch.Tensor,
) -> dict:

    snapshot = capture_state(exp)

    params = tuple(
        p for p in exp.model.parameters()
        if p.requires_grad
    )

    before = flatten([
        p.detach().clone()
        for p in params
    ])

    exp.train_step()

    after = flatten([
        p.detach().clone()
        for p in params
    ])

    actual = after - before

    max_abs = float(
        torch.max(
            torch.abs(actual - predicted)
        ).item()
    )

    rel = float(
        (
            torch.linalg.vector_norm(actual - predicted)
            /
            torch.clamp(
                torch.linalg.vector_norm(actual),
                min=1.0e-300,
            )
        ).item()
    )

    restore_state(exp, snapshot)

    return {
        "max_abs_difference": max_abs,
        "relative_difference": rel,
        "pass": bool(max_abs <= ADAM_FORMULA_TOL),
    }


# =============================================================================
# Pareto midpoint and curvature
# =============================================================================

def strict_negative_interval(
    f0: float,
    f1: float,
    tol: float = 1.0e-15,
) -> Tuple[float, float]:

    slope = f1 - f0

    if abs(slope) <= tol * max(1.0, abs(f0), abs(f1)):
        if f0 < 0.0:
            return 0.0, 1.0
        return math.inf, -math.inf

    root = -f0 / slope

    if slope > 0.0:
        return 0.0, min(1.0, root)

    return max(0.0, root), 1.0


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

    return float(
        torch.dot(d, hv).item()
    )


def exact_direction_changes(
    exp,
    d: torch.Tensor,
    target_index: int,
) -> dict:

    params = tuple(
        p for p in exp.model.parameters()
        if p.requires_grad
    )

    base = [
        p.detach().clone()
        for p in params
    ]

    pre_r = exp.weak_residuals().detach()
    M = pre_r.numel()

    pre_L = float(
        torch.mean(pre_r.square()).item()
    )

    pre_T = float(
        (pre_r[target_index].square() / M).item()
    )

    offset = 0

    try:
        with torch.no_grad():
            for p, p0 in zip(params, base):
                n = p.numel()
                p.copy_(
                    p0
                    + d[offset:offset+n].reshape_as(p)
                )
                offset += n

        post_r = exp.weak_residuals().detach()

        post_L = float(
            torch.mean(post_r.square()).item()
        )

        post_T = float(
            (post_r[target_index].square() / M).item()
        )

    finally:
        with torch.no_grad():
            for p, p0 in zip(params, base):
                p.copy_(p0)

    return {
        "pre_total_loss": pre_L,
        "post_total_loss": post_L,
        "exact_total_change": post_L - pre_L,

        "pre_target_loss": pre_T,
        "post_target_loss": post_T,
        "exact_target_change": post_T - pre_T,
    }


def pareto_curvature_audit(
    exp,
    kernel: dict,
    adam: dict,
    target_index: int,
) -> dict:

    candidate = adam["delta_candidate"]
    gL = adam["g_total"]
    gT = adam["g_target"]

    candidate_target_dot = float(
        torch.dot(gT, candidate).item()
    )

    if candidate_target_dot <= 0.0:
        return {
            "pareto_status": "ADAM_ALREADY_TARGET_NONUPHILL",
            "pareto_active": False,
        }

    gt2 = torch.dot(gT, gT)

    reflected = (
        candidate
        - 2.0
        * (
            torch.dot(gT, candidate) / gt2
        )
        * gT
    )

    candidate_total_dot = float(
        torch.dot(gL, candidate).item()
    )

    reflect_total_dot = float(
        torch.dot(gL, reflected).item()
    )

    reflect_target_dot = float(
        torch.dot(gT, reflected).item()
    )

    int_L = strict_negative_interval(
        candidate_total_dot,
        reflect_total_dot,
    )

    int_T = strict_negative_interval(
        candidate_target_dot,
        reflect_target_dot,
    )

    lo = max(int_L[0], int_T[0])
    hi = min(int_L[1], int_T[1])

    if not (
        math.isfinite(lo)
        and math.isfinite(hi)
        and lo < hi
    ):
        return {
            "pareto_status": "NO_STRICT_INTERVAL",
            "pareto_active": True,
            "lambda_lower": lo,
            "lambda_upper": hi,
        }

    lam = 0.5 * (lo + hi)

    d = (
        (1.0 - lam) * candidate
        + lam * reflected
    )

    first_L = float(torch.dot(gL, d).item())
    first_T = float(torch.dot(gT, d).item())

    if not (first_L < 0.0 and first_T < 0.0):
        raise RuntimeError(
            "Pareto midpoint is not strict first-order joint descent."
        )

    params = kernel["params"]
    M = kernel["r"].numel()

    residuals_L = exp.weak_residuals()
    loss_L = residuals_L.square().mean()

    kappa_L = directional_hessian(
        loss_L,
        params,
        d,
    )

    residuals_T = exp.weak_residuals()
    loss_T = residuals_T[target_index].square() / M

    kappa_T = directional_hessian(
        loss_T,
        params,
        d,
    )

    Jd = kernel["J"] @ d

    kappa_L_GN = float(
        ((2.0 / M) * torch.dot(Jd, Jd)).item()
    )

    kappa_L_NL = kappa_L - kappa_L_GN

    jt_d = float(
        torch.dot(
            kernel["J"][target_index, :],
            d,
        ).item()
    )

    kappa_T_GN = (
        (2.0 / M) * jt_d * jt_d
    )

    kappa_T_NL = kappa_T - kappa_T_GN

    second_L = first_L + 0.5 * kappa_L
    second_T = first_T + 0.5 * kappa_T

    exact = exact_direction_changes(
        exp=exp,
        d=d,
        target_index=target_index,
    )

    tol_L = (
        LOSS_TOL_FACTOR
        * max(1.0, abs(exact["pre_total_loss"]))
    )

    tol_T = (
        LOSS_TOL_FACTOR
        * max(1.0, abs(exact["pre_target_loss"]))
    )

    fail_L = bool(
        exact["exact_total_change"] > tol_L
    )

    fail_T = bool(
        exact["exact_target_change"] > tol_T
    )

    triggering = []

    if fail_L:
        triggering.append(
            second_L > 0.0
        )

    if fail_T:
        triggering.append(
            second_T > 0.0
        )

    if not triggering:
        barrier_class = "EXACT_FULL_STEP_SAFE"
    elif all(triggering):
        barrier_class = "SECOND_ORDER_EXPLAINS_ALL_TRIGGERING_OBJECTIVES"
    elif any(triggering):
        barrier_class = "MIXED_SECOND_AND_HIGHER_ORDER_BARRIER"
    else:
        barrier_class = "HIGHER_ORDER_BARRIER"

    return {
        "pareto_status": barrier_class,
        "pareto_active": True,

        "lambda_lower": lo,
        "lambda_upper": hi,
        "lambda_mid": lam,
        "lambda_width": hi - lo,

        "first_order_total": first_L,
        "first_order_target": first_T,

        "kappa_total": kappa_L,
        "kappa_total_GN": kappa_L_GN,
        "kappa_total_NL": kappa_L_NL,

        "kappa_target": kappa_T,
        "kappa_target_GN": kappa_T_GN,
        "kappa_target_NL": kappa_T_NL,

        "second_order_total_change": second_L,
        "second_order_target_change": second_T,

        **exact,

        "total_finite_step_unsafe": fail_L,
        "target_finite_step_unsafe": fail_T,

        "total_curvature_to_descent_ratio": (
            0.5 * kappa_L / max(-first_L, 1.0e-300)
        ),

        "target_curvature_to_descent_ratio": (
            0.5 * kappa_T / max(-first_T, 1.0e-300)
        ),

        "barrier_class": barrier_class,
    }


# =============================================================================
# Phase-state audit
# =============================================================================

def phase_state_audit(
    exp,
    mode: int,
    seed: int,
    epoch: int,
    rel_l2: float,
    residual_metrics: dict,
) -> dict:

    target_index = mode - 1

    kernel = residual_jacobian(exp)

    raw = kernel_summary(
        r=kernel["r"],
        K=kernel["K"],
        Kcorr=kernel["Kcorr"],
        target_index=target_index,
    )

    adam = predict_and_decompose_adam(
        exp=exp,
        J=kernel["J"],
        r=kernel["r"],
        params=kernel["params"],
        target_index=target_index,
    )

    formula_check = verify_predicted_adam_step(
        exp=exp,
        predicted=adam["delta_candidate"],
    )

    if not formula_check["pass"]:
        raise RuntimeError(
            f"Adam formula mismatch seed={seed}, mode={mode}, "
            f"epoch={epoch}: {formula_check['max_abs_difference']:.3e}"
        )

    pareto = pareto_curvature_audit(
        exp=exp,
        kernel=kernel,
        adam=adam,
        target_index=target_index,
    )

    signed_cos = (
        torch.sign(
            kernel["r"][:, None]
            * kernel["r"][None, :]
        )
        * kernel["Kcorr"]
    )

    return {
        "seed": seed,
        "mode": mode,
        "epoch": epoch,

        "amplitude": exp.amplitude,
        "matched_weak_scale_a_times_m":
            exp.amplitude * mode,

        "relative_l2_error": rel_l2,

        **residual_metrics,

        **raw,

        "adam_target_interference_ratio":
            adam["adam_target_interference_ratio"],

        "adam_target_max_abs_normalized_coupling":
            adam["adam_target_max_abs_normalized_coupling"],

        "adam_target_uphill_cosine":
            adam["adam_target_uphill_cosine"],

        "adam_candidate_target_uphill":
            adam["adam_candidate_target_uphill"],

        "target_dot_current":
            adam["target_dot_current"],

        "target_dot_history":
            adam["target_dot_history"],

        "target_dot_candidate":
            adam["target_dot_candidate"],

        "total_dot_current":
            adam["total_dot_current"],

        "total_dot_history":
            adam["total_dot_history"],

        "total_dot_candidate":
            adam["total_dot_candidate"],

        "adam_candidate_norm":
            adam["candidate_norm"],

        "adam_current_norm":
            adam["current_norm"],

        "adam_history_norm":
            adam["history_norm"],

        "adam_formula_max_abs_difference":
            formula_check["max_abs_difference"],

        "adam_formula_relative_difference":
            formula_check["relative_difference"],

        **pareto,

        "_raw_kernel":
            kernel["K"].cpu().numpy(),

        "_raw_kernel_corr":
            kernel["Kcorr"].cpu().numpy(),

        "_signed_squared_residual_gradient_cosine":
            signed_cos.cpu().numpy(),

        "_adam_current_kernel":
            adam["K_D"].cpu().numpy(),

        "_adam_current_kernel_corr":
            adam["K_D_corr"].cpu().numpy(),

        "_residuals":
            kernel["r"].cpu().numpy(),
    }


# =============================================================================
# Stage-3 seed-0 reproduction
# =============================================================================

def verify_stage3_anchor(
    mode: int,
    epoch: int,
    rel_l2: float,
    rm: dict,
    anchor_map: dict,
) -> dict:

    key = (mode, epoch)

    if key not in anchor_map:
        raise RuntimeError(
            f"Stage-3 anchor missing mode={mode}, epoch={epoch}."
        )

    old = anchor_map[key]

    diffs = {
        "relative_l2_error":
            abs(
                rel_l2
                - float(old["relative_l2_error"])
            ),

        "vpinn_loss":
            abs(
                float(rm["vpinn_loss"])
                - float(old["vpinn_loss"])
            ),

        "target_share":
            abs(
                float(
                    rm[
                        "target_mode_residual_energy_share"
                    ]
                )
                - float(
                    old[
                        "target_mode_residual_energy_share"
                    ]
                )
            ),
    }

    max_diff = max(diffs.values())

    return {
        "mode": mode,
        "epoch": epoch,
        "max_abs_difference": max_diff,
        "pass": bool(max_diff <= REPRO_TOL),
    }


# =============================================================================
# Aggregate helpers
# =============================================================================

def median_or_none(values):
    vals = [float(x) for x in values if x is not None]
    return float(np.median(vals)) if vals else None


def paired_endpoint_effects(phase_rows: List[dict]) -> List[dict]:
    by = {
        (int(r["seed"]), int(r["mode"])): r
        for r in phase_rows
    }

    rows = []

    for seed in SEEDS:
        k3 = (seed, 3)
        k9 = (seed, 9)

        if k3 not in by or k9 not in by:
            rows.append(
                {
                    "seed": seed,
                    "paired_available": False,
                }
            )
            continue

        a = by[k3]
        b = by[k9]

        rows.append(
            {
                "seed": seed,
                "paired_available": True,

                "phase_epoch_m3":
                    int(a["epoch"]),
                "phase_epoch_m9":
                    int(b["epoch"]),

                "phase_epoch_difference_m9_minus_m3":
                    int(b["epoch"]) - int(a["epoch"]),

                "phase_epoch_later_at_m9":
                    bool(
                        int(b["epoch"]) > int(a["epoch"])
                    ),

                "raw_target_coupling_m3":
                    float(
                        a[
                            "target_max_abs_normalized_coupling"
                        ]
                    ),

                "raw_target_coupling_m9":
                    float(
                        b[
                            "target_max_abs_normalized_coupling"
                        ]
                    ),

                "raw_target_coupling_difference_m9_minus_m3":
                    float(
                        b[
                            "target_max_abs_normalized_coupling"
                        ]
                    )
                    - float(
                        a[
                            "target_max_abs_normalized_coupling"
                        ]
                    ),

                "stronger_raw_target_coupling_at_m9":
                    bool(
                        float(
                            b[
                                "target_max_abs_normalized_coupling"
                            ]
                        )
                        >
                        float(
                            a[
                                "target_max_abs_normalized_coupling"
                            ]
                        )
                    ),

                "adam_target_uphill_cosine_m3":
                    float(a["adam_target_uphill_cosine"]),

                "adam_target_uphill_cosine_m9":
                    float(b["adam_target_uphill_cosine"]),

                "adam_uphill_cosine_difference_m9_minus_m3":
                    float(b["adam_target_uphill_cosine"])
                    - float(a["adam_target_uphill_cosine"]),

                "adam_more_target_uphill_at_m9":
                    bool(
                        float(b["adam_target_uphill_cosine"])
                        >
                        float(a["adam_target_uphill_cosine"])
                    ),
            }
        )

    return rows


# =============================================================================
# Plotting
# =============================================================================

def line_plot_by_seed(
    rows: List[dict],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    zero_line: bool = False,
) -> None:

    fig, ax = plt.subplots(figsize=(9.4, 5.5))

    for seed in SEEDS:
        rr = sorted(
            [
                r for r in rows
                if int(r["seed"]) == seed
            ],
            key=lambda x: int(x["mode"]),
        )

        if not rr:
            continue

        ax.plot(
            [int(r["mode"]) for r in rr],
            [float(r[metric]) for r in rr],
            marker="o",
            linewidth=1.4,
            label=f"seed {seed}",
        )

    if zero_line:
        ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xticks(list(MODES))
    ax.set_xlabel("Target frequency m")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def prevalence_plot(mode_rows: List[dict], path: Path) -> None:
    modes = [int(r["mode"]) for r in mode_rows]

    uphill = [
        100.0 * float(r["adam_target_uphill_fraction"])
        for r in mode_rows
    ]

    unsafe = [
        100.0 * float(r["pareto_exact_unsafe_fraction_among_active"])
        if r["pareto_exact_unsafe_fraction_among_active"] is not None
        else 0.0
        for r in mode_rows
    ]

    x = np.arange(len(modes))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.0, 5.3))

    ax.bar(
        x - width/2,
        uphill,
        width,
        label="Adam target-uphill",
    )

    ax.bar(
        x + width/2,
        unsafe,
        width,
        label="Pareto full-step unsafe | active",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in modes])
    ax.set_xlabel("Target frequency m")
    ax.set_ylabel("Prevalence (%)")
    ax.set_ylim(0.0, 105.0)
    ax.set_title("Frequency transfer of optimizer opposition and curvature risk")
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

    stage3_dir = Path(args.stage3_dir)
    if not stage3_dir.is_absolute():
        stage3_dir = root / stage3_dir

    stage17r_dir = Path(args.stage17r_dir)
    if not stage17r_dir.is_absolute():
        stage17r_dir = root / stage17r_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage3_dir=stage3_dir,
        stage17r_dir=stage17r_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage18R",
    )

    precommitment = {
        "stage":
            "multi_seed_phase_matched_frequency_transfer",

        "modes":
            list(MODES),

        "seeds":
            list(SEEDS),

        "paired_seed_design":
            True,

        "amplitude_rule":
            "a_m = 0.15 * 7 / m",

        "matched_quantity":
            "a_m * m = 1.05",

        "phase_state_definition": {
            "tracking_grid":
                25,

            "first_epoch_with_target_share_ge":
                PHASE_TARGET_SHARE,

            "and_relative_l2_gt":
                CONVERGENCE_REL_L2,
        },

        "max_epoch":
            MAX_EPOCH,

        "expensive_diagnostics_once_per_phase_state":
            True,

        "ordinary_adam_training_only":
            True,

        "endpoint_trends_m9_vs_m3": [
            "phase epoch later",
            "raw normalized target coupling stronger",
            "Adam target-uphill cosine larger",
        ],

        "sign_consistency_requirement":
            ">=4/5 paired seeds",

        "strong_gate":
            "coverage >=4/5 for every mode AND >=2/3 endpoint trends sign-consistent",

        "no_frequency_law_claim_if_gate_fails":
            True,
    }

    manifest = {
        "python":
            sys.version,

        "platform":
            platform.platform(),

        "torch_version":
            torch.__version__,

        "numpy_version":
            np.__version__,

        "device_resolved":
            str(device),

        "stage3_solver_sha256":
            pf["stage3_sha256"],

        "stage17r_manifest":
            pf["stage17r_manifest"],

        "stage18r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment":
            precommitment,
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 166)
    print(
        "VPINN — STAGE 18R MULTI-SEED PHASE-MATCHED FREQUENCY TRANSFER"
    )
    print("=" * 166)

    print(f"device                    : {device}")
    print(f"modes                     : {list(MODES)}")
    print(f"seeds                     : {list(SEEDS)}")
    print(
        "phase state               : first 25-grid epoch with "
        "target share >=0.98 and relL2 >1e-2"
    )
    print("expensive diagnostics     : once per valid phase state")
    print("training intervention      : NONE")
    print("=" * 166)

    tracking_rows: List[dict] = []
    phase_rows: List[dict] = []
    run_rows: List[dict] = []
    reproduction_rows: List[dict] = []

    global_start = time.perf_counter()

    for seed in SEEDS:
        for mode in MODES:

            run_dir = (
                out_dir
                / f"seed_{seed:03d}"
                / f"mode_{mode:02d}"
            )

            exp = make_experiment(
                stage3=stage3,
                device=device,
                seed=seed,
                mode=mode,
                out_dir=run_dir,
            )

            phase_found = False
            phase_epoch = -1

            print()
            print("-" * 166)
            print(
                f"seed={seed} mode={mode} "
                f"amplitude={exp.amplitude:.12g} "
                f"a*m={exp.amplitude*mode:.12g}"
            )

            for epoch in range(MAX_EPOCH + 1):

                is_track = (
                    epoch % TRACK_INTERVAL == 0
                )

                if is_track:
                    rm = exp.residual_metrics()
                    rel = exp.relative_l2_error()

                    row = {
                        "seed": seed,
                        "mode": mode,
                        "epoch": epoch,
                        "amplitude": exp.amplitude,
                        "matched_weak_scale_a_times_m":
                            exp.amplitude * mode,
                        "relative_l2_error": rel,
                        **rm,
                    }

                    tracking_rows.append(row)

                    if seed == 0:
                        check = verify_stage3_anchor(
                            mode=mode,
                            epoch=epoch,
                            rel_l2=rel,
                            rm=rm,
                            anchor_map=pf["stage3_anchor"],
                        )

                        reproduction_rows.append(check)

                        if not check["pass"]:
                            raise RuntimeError(
                                f"Stage-3 reproduction failed "
                                f"mode={mode}, epoch={epoch}, "
                                f"gap={check['max_abs_difference']:.3e}"
                            )

                    phase_now = bool(
                        rm[
                            "target_mode_residual_energy_share"
                        ] >= PHASE_TARGET_SHARE
                        and
                        rel > CONVERGENCE_REL_L2
                    )

                    if phase_now:
                        phase_found = True
                        phase_epoch = epoch

                        audit = phase_state_audit(
                            exp=exp,
                            mode=mode,
                            seed=seed,
                            epoch=epoch,
                            rel_l2=rel,
                            residual_metrics=rm,
                        )

                        arrays = {
                            key: audit.pop(key)
                            for key in list(audit.keys())
                            if key.startswith("_")
                        }

                        phase_rows.append(audit)

                        np.savez_compressed(
                            run_dir / "phase_state_kernels.npz",
                            raw_kernel=
                                arrays["_raw_kernel"],
                            raw_kernel_corr=
                                arrays["_raw_kernel_corr"],
                            signed_squared_residual_gradient_cosine=
                                arrays[
                                    "_signed_squared_residual_gradient_cosine"
                                ],
                            adam_current_kernel=
                                arrays["_adam_current_kernel"],
                            adam_current_kernel_corr=
                                arrays["_adam_current_kernel_corr"],
                            residuals=
                                arrays["_residuals"],
                        )

                        torch.save(
                            {
                                "seed": seed,
                                "mode": mode,
                                "epoch": epoch,
                                "model_state_dict":
                                    copy.deepcopy(
                                        exp.model.state_dict()
                                    ),
                                "optimizer_state_dict":
                                    copy.deepcopy(
                                        exp.optimizer.state_dict()
                                    ),
                            },
                            run_dir / "phase_state.pt",
                        )

                        print(
                            f"  PHASE STATE @ {epoch}: "
                            f"relL2={rel:.6e}, "
                            f"share={rm['target_mode_residual_energy_share']:.6f}, "
                            f"coupling={audit['target_max_abs_normalized_coupling']:.6f}, "
                            f"Adam uphill cosine={audit['adam_target_uphill_cosine']:+.6f}, "
                            f"Pareto={audit['pareto_status']}"
                        )

                        break

                if epoch < MAX_EPOCH:
                    exp.train_step()

            run_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "status": (
                        "PHASE_STATE_FOUND"
                        if phase_found
                        else "NO_PHASE_STATE_BY_2500"
                    ),
                    "phase_epoch": phase_epoch,
                }
            )

            if not phase_found:
                print("  NO PHASE STATE by epoch 2500")

    # -------------------------------------------------------------------------
    # Aggregation.
    # -------------------------------------------------------------------------
    write_csv(
        out_dir / "tracking_metrics.csv",
        tracking_rows,
    )

    write_csv(
        out_dir / "phase_state_metrics.csv",
        phase_rows,
    )

    write_csv(
        out_dir / "run_status.csv",
        run_rows,
    )

    write_csv(
        out_dir / "stage3_seed0_reproduction.csv",
        reproduction_rows,
    )

    mode_summary = []

    for mode in MODES:
        rr = [
            r for r in phase_rows
            if int(r["mode"]) == mode
        ]

        n = len(rr)

        uphill_count = sum(
            int(bool(r["adam_candidate_target_uphill"]))
            for r in rr
        )

        active = [
            r for r in rr
            if bool(r.get("pareto_active", False))
        ]

        unsafe = [
            r for r in active
            if bool(r.get("total_finite_step_unsafe", False))
            or bool(r.get("target_finite_step_unsafe", False))
        ]

        second_order = [
            r for r in active
            if r.get("barrier_class")
            == "SECOND_ORDER_EXPLAINS_ALL_TRIGGERING_OBJECTIVES"
        ]

        mode_summary.append(
            {
                "mode": mode,

                "phase_state_count":
                    n,

                "phase_coverage_fraction":
                    n / len(SEEDS),

                "median_phase_epoch":
                    median_or_none(
                        [r["epoch"] for r in rr]
                    ),

                "median_target_share":
                    median_or_none(
                        [
                            r[
                                "target_mode_residual_energy_share"
                            ]
                            for r in rr
                        ]
                    ),

                "median_raw_target_coupling":
                    median_or_none(
                        [
                            r[
                                "target_max_abs_normalized_coupling"
                            ]
                            for r in rr
                        ]
                    ),

                "median_adam_target_coupling":
                    median_or_none(
                        [
                            r[
                                "adam_target_max_abs_normalized_coupling"
                            ]
                            for r in rr
                        ]
                    ),

                "median_raw_interference_ratio":
                    median_or_none(
                        [
                            r["target_interference_ratio"]
                            for r in rr
                        ]
                    ),

                "median_adam_interference_ratio":
                    median_or_none(
                        [
                            r["adam_target_interference_ratio"]
                            for r in rr
                        ]
                    ),

                "median_adam_target_uphill_cosine":
                    median_or_none(
                        [
                            r["adam_target_uphill_cosine"]
                            for r in rr
                        ]
                    ),

                "adam_target_uphill_count":
                    uphill_count,

                "adam_target_uphill_fraction":
                    (
                        uphill_count / n
                        if n
                        else 0.0
                    ),

                "pareto_active_count":
                    len(active),

                "pareto_exact_unsafe_count":
                    len(unsafe),

                "pareto_exact_unsafe_fraction_among_active":
                    (
                        len(unsafe) / len(active)
                        if active
                        else None
                    ),

                "second_order_barrier_count":
                    len(second_order),

                "second_order_barrier_fraction_among_active":
                    (
                        len(second_order) / len(active)
                        if active
                        else None
                    ),

                "median_current_target_dot":
                    median_or_none(
                        [
                            r["target_dot_current"]
                            for r in rr
                        ]
                    ),

                "median_history_target_dot":
                    median_or_none(
                        [
                            r["target_dot_history"]
                            for r in rr
                        ]
                    ),
            }
        )

    write_csv(
        out_dir / "mode_summary.csv",
        mode_summary,
    )

    paired = paired_endpoint_effects(
        phase_rows
    )

    write_csv(
        out_dir / "paired_m3_m9_effects.csv",
        paired,
    )

    available_paired = [
        r for r in paired
        if bool(r["paired_available"])
    ]

    later_count = sum(
        int(bool(r["phase_epoch_later_at_m9"]))
        for r in available_paired
    )

    coupling_count = sum(
        int(
            bool(
                r[
                    "stronger_raw_target_coupling_at_m9"
                ]
            )
        )
        for r in available_paired
    )

    uphill_count = sum(
        int(bool(r["adam_more_target_uphill_at_m9"]))
        for r in available_paired
    )

    trend_flags = {
        "phase_epoch_later_m9_vs_m3":
            bool(
                len(available_paired) == 5
                and later_count >= 4
            ),

        "raw_target_coupling_stronger_m9_vs_m3":
            bool(
                len(available_paired) == 5
                and coupling_count >= 4
            ),

        "adam_more_target_uphill_m9_vs_m3":
            bool(
                len(available_paired) == 5
                and uphill_count >= 4
            ),
    }

    trend_pass_count = sum(
        int(v) for v in trend_flags.values()
    )

    coverage_by_mode = {
        mode: sum(
            int(
                int(r["mode"]) == mode
                and r["status"] == "PHASE_STATE_FOUND"
            )
            for r in run_rows
        )
        for mode in MODES
    }

    coverage_pass = all(
        coverage_by_mode[m] >= 4
        for m in MODES
    )

    m9_summary = next(
        r for r in mode_summary
        if int(r["mode"]) == 9
    )

    strong_gate = bool(
        coverage_pass
        and trend_pass_count >= 2
    )

    if strong_gate:
        route_class = (
            "frequency_ordered_interaction_mechanism_supported"
        )

        next_route = (
            "stage19R_dense_frequency_transition_localization"
        )

    elif (
        coverage_pass
        and int(m9_summary["adam_target_uphill_count"]) >= 4
    ):
        route_class = (
            "high_frequency_specific_mechanism_without_ordered_law"
        )

        next_route = (
            "stage19R_high_frequency_local_band_mechanism_audit"
        )

    else:
        route_class = (
            "frequency_transfer_not_cleanly_supported"
        )

        next_route = (
            "stage19R_architecture_or_phase_heterogeneity_audit"
        )

    decision = {
        "n_requested_runs":
            len(MODES) * len(SEEDS),

        "n_phase_states":
            len(phase_rows),

        "coverage_by_mode":
            coverage_by_mode,

        "coverage_gate_pass":
            coverage_pass,

        "stage3_seed0_reproduction_all_pass":
            all(
                bool(r["pass"])
                for r in reproduction_rows
            ),

        "paired_m3_m9_available_count":
            len(available_paired),

        "paired_endpoint_support_counts": {
            "phase_epoch_later":
                later_count,

            "raw_target_coupling_stronger":
                coupling_count,

            "adam_more_target_uphill":
                uphill_count,
        },

        "paired_endpoint_trend_flags":
            trend_flags,

        "paired_endpoint_trend_pass_count":
            trend_pass_count,

        "strong_frequency_ordered_support":
            strong_gate,

        "mode9_adam_target_uphill_count":
            int(m9_summary["adam_target_uphill_count"]),

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A PASS supports frequency ordering of selected finite-width "
            "optimization-geometry diagnostics under this matched-amplitude "
            "1D VPINN family. It does not establish a universal VPINN law, "
            "causality across architectures, or novelty without a separate "
            "literature and robustness audit."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    # -------------------------------------------------------------------------
    # Figures.
    # -------------------------------------------------------------------------
    if phase_rows:
        line_plot_by_seed(
            phase_rows,
            metric="epoch",
            ylabel="Phase-matched epoch",
            title="When does the unresolved mode capture >=98% residual energy?",
            path=out_dir / "phase_epoch_vs_frequency.png",
        )

        line_plot_by_seed(
            phase_rows,
            metric="target_max_abs_normalized_coupling",
            ylabel="max |normalized target coupling|",
            title="Finite-width target weak-test coupling vs frequency",
            path=out_dir / "target_coupling_vs_frequency.png",
        )

        line_plot_by_seed(
            phase_rows,
            metric="adam_target_uphill_cosine",
            ylabel="Adam target-uphill cosine",
            title="Does Adam become more opposed to the unresolved mode?",
            path=out_dir / "adam_target_uphill_vs_frequency.png",
            zero_line=True,
        )

        line_plot_by_seed(
            phase_rows,
            metric="adam_target_interference_ratio",
            ylabel="Adam-metric target interference ratio",
            title="State-dependent Adam weak-test interference vs frequency",
            path=out_dir / "adam_interference_vs_frequency.png",
        )

    prevalence_plot(
        mode_summary,
        out_dir / "mechanism_prevalence_vs_frequency.png",
    )

    elapsed = time.perf_counter() - global_start

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------
    lines = []

    lines.append("=" * 168)
    lines.append(
        "VPINN — STAGE 18R PHASE-MATCHED FREQUENCY-TRANSFER SUMMARY"
    )
    lines.append("=" * 168)

    lines.append(
        "mode | phase states | median epoch | median raw coupling | "
        "median Adam uphill cos | Adam uphill | Pareto active | Pareto unsafe"
    )

    lines.append("-" * 168)

    for r in mode_summary:
        lines.append(
            f"{int(r['mode']):4d} | "
            f"{int(r['phase_state_count']):12d} | "
            f"{str(r['median_phase_epoch']):12s} | "
            f"{str(r['median_raw_target_coupling']):19s} | "
            f"{str(r['median_adam_target_uphill_cosine']):22s} | "
            f"{int(r['adam_target_uphill_count']):5d}/"
            f"{int(r['phase_state_count']):<5d} | "
            f"{int(r['pareto_active_count']):6d} | "
            f"{int(r['pareto_exact_unsafe_count']):6d}"
        )

    lines.append("-" * 168)

    lines.append(
        f"coverage by mode                      : "
        f"{coverage_by_mode}"
    )

    lines.append(
        f"coverage gate                         : "
        f"{'PASS' if coverage_pass else 'FAIL'}"
    )

    lines.append(
        f"Stage-3 seed0 reproduction            : "
        f"{sum(int(r['pass']) for r in reproduction_rows)}/"
        f"{len(reproduction_rows)} PASS"
    )

    lines.append(
        f"paired m9>m3 phase epoch support      : "
        f"{later_count}/5"
    )

    lines.append(
        f"paired m9>m3 raw coupling support     : "
        f"{coupling_count}/5"
    )

    lines.append(
        f"paired m9 more target-uphill support  : "
        f"{uphill_count}/5"
    )

    lines.append(
        f"endpoint trend gate count             : "
        f"{trend_pass_count}/3"
    )

    lines.append(
        f"strong frequency-ordered support      : "
        f"{strong_gate}"
    )

    lines.append(
        f"route class                            : "
        f"{route_class}"
    )

    lines.append(
        f"next route                             : "
        f"{next_route}"
    )

    lines.append(
        f"elapsed seconds                        : "
        f"{elapsed:.2f}"
    )

    lines.append("=" * 168)

    lines.append(
        "Guardrail: this stage may support a frequency-ordered mechanism in the "
        "matched 1D family; it cannot by itself justify a universal VPINN law."
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
