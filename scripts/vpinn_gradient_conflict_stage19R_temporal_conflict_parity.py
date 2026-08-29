#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 19R
Temporal Conflict-Onset + Parity-Ladder Weak-Test Audit
=======================================================

Motivation
----------
Stage 18R falsified the initial "first 98%-share state" as a useful lock-phase
comparator: across all 20 (seed,frequency) runs, that state appeared very
early (typically epoch 100-150) and Adam was still target-downhill in 20/20.

However Stage 18R exposed a striking structural pattern at those early states:

    in 20/20 runs, the strongest target-row parameter-gradient coupling
    occurred at test mode m+2,

with near-perfect signed anti-alignment of squared weak-residual gradients.

Stage 19R therefore does NOT change architecture and does NOT invent another
optimizer. It asks two lower-cost mechanistic questions along ordinary-Adam
trajectories:

  Q1. When does persistent ACTUAL optimizer opposition to the unresolved
      target mode first appear as frequency changes?

  Q2. Does the same-parity m <-> m+2 weak-test gradient anti-alignment persist
      from the early localized phase toward genuine optimizer conflict?

Family
------
Exactly the Stage-3 matched-amplitude family:

    m in {3,5,7,9}
    seeds in {0,1,2,3,4}

    u*_m = sin(pi x) + a_m sin(m pi x)
    a_m  = 0.15 * 7 / m
    a_m m = 1.05

Ordinary Adam only. No trained intervention.

Cheap temporal probe
--------------------
On the global 25-epoch grid, while the run remains unresolved:

    relative L2 error > 1e-2,

compute:

  * target residual-energy share;
  * exact next-Adam target alignment cosine

        U_A = <g_T, Delta_A> / (||g_T|| ||Delta_A||),

    positive means the ACTUAL inherited Adam candidate is target-uphill;

  * signed cosine between squared-residual gradients for

        (m, m+2)

    and, secondarily, (m, m-2).

These probes require only a few gradient vectors, not the full residual
Jacobian.

Conflict-active condition
-------------------------
A probe is mechanism-active when

    target residual-energy share >= 0.80
    AND relative L2 error > 1e-2.

Certified optimizer conflict
----------------------------
A run has CERTIFIED_CONFLICT only if

    U_A > 0

for THREE consecutive 25-epoch mechanism-active probes.

The certified conflict onset is the first point of that 3-point run.

This rejects one-point sign noise and mirrors the project's existing
three-observation certification logic.

Certified parity-ladder anti-alignment
--------------------------------------
The m <-> m+2 ladder is certified when

    C_sq(m,m+2) <= -0.95

for THREE consecutive mechanism-active probes.

Trajectory stopping
-------------------
Each run stops at the earliest of:

  * certified optimizer conflict;
  * certified numerical convergence;
  * epoch 2500.

Convergence is inherited:

    relative L2 <= 1e-2
    AND target share <= 0.20

for THREE consecutive 25-epoch observations.

Expensive audit
---------------
Full J J^T / Adam-metric kernel / Pareto-curvature audit is computed ONCE:

  * at certified conflict onset, if conflict occurs;
  * otherwise at the final mechanism-active state before convergence/horizon.

Thus Stage 19R localizes the true temporal mechanism without paying Hessian
cost at every epoch.

Seed-0 reproduction
-------------------
Every Stage-19R seed-0 tracking point is checked against the existing full
Stage-3 frequency-transfer trajectory. Tolerance = 1e-10.

Primary frequency-selective conflict gate
-----------------------------------------
Let N_m be the number of seeds with CERTIFIED_CONFLICT for frequency m.

STRONG_FREQUENCY_SELECTIVE_CONFLICT requires all of:

    N_3 <= N_5 <= N_7 <= N_9
    N_9 >= 4
    N_9 - N_3 >= 3.

This is stronger than merely observing one difficult m=9 seed, but does not
require all five seeds or a fabricated p-value from n=5.

Parity-ladder persistence gate
------------------------------
For each frequency, among all mechanism-active probes, compute

    P_m = fraction with C_sq(m,m+2) <= -0.95.

PERSISTENT_PARITY_LADDER requires

    P_m >= 0.80 for all four frequencies.

Decision routes
---------------
A) Strong frequency-selective conflict PASS:
       Stage 20R = dense frequency transition localization.

B) Frequency-selective gate FAIL but parity-ladder persistence PASS:
       Stage 20R = architecture control of parity-ladder coupling
                   (tanh vs a spectrally enriched control).

C) Both FAIL:
       Stage 20R = phase/architecture heterogeneity audit; do not claim a
                   frequency or parity law.

No novelty claim is authorized by Stage 19R alone.
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


MODES = (3, 5, 7, 9)
SEEDS = (0, 1, 2, 3, 4)

MAX_EPOCH = 2500
TRACK_INTERVAL = 25

ACTIVE_TARGET_SHARE = 0.80
CONVERGENCE_REL_L2 = 1.0e-2
CONVERGENCE_TARGET_SHARE = 0.20
CERTIFY_POINTS = 3

PARITY_ANTI_THRESHOLD = -0.95
PARITY_PERSISTENCE_FRACTION = 0.80

REPRO_TOL = 1.0e-10
ADAM_FORMULA_TOL = 5.0e-12


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-19R temporal conflict onset and parity ladder audit."
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
        "--stage18-script",
        default="vpinn_gradient_conflict_stage18R_frequency_transfer.py",
    )

    p.add_argument(
        "--stage18-dir",
        default="vpinn_gradient_conflict_stage18R_frequency_transfer",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage19R_temporal_conflict_parity",
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(normalized)


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


def flatten(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([x.reshape(-1) for x in parts], dim=0)


# =============================================================================
# Preflight
# =============================================================================

def preflight(
    stage3_script: Path,
    stage3_dir: Path,
    stage18_script: Path,
    stage18_dir: Path,
) -> dict:

    paths = {
        "s3_manifest": stage3_dir / "manifest.json",
        "s3_tracking": stage3_dir / "aggregate_tracking_metrics.csv",
        "s18_manifest": stage18_dir / "manifest.json",
        "s18_decision": stage18_dir / "decision.json",
        "s18_phase": stage18_dir / "phase_state_metrics.csv",
    }

    missing = [str(p) for p in paths.values() if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing prerequisites:\n  " + "\n  ".join(missing)
        )

    s3m = read_json(paths["s3_manifest"])
    s18m = read_json(paths["s18_manifest"])
    s18d = read_json(paths["s18_decision"])

    s3_sha = sha256_file(stage3_script)
    s18_sha = sha256_file(stage18_script)

    if s3m.get("script_sha256") != s3_sha:
        raise RuntimeError("Stage-3 source SHA mismatch.")

    if s18m.get("stage3_solver_sha256") != s3_sha:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 18.")

    if s18m.get("stage18r_script_sha256") != s18_sha:
        raise RuntimeError(
            "Stage-18 source SHA mismatch against executed manifest."
        )

    if s18d.get("next_route") != (
        "stage19R_architecture_or_phase_heterogeneity_audit"
    ):
        raise RuntimeError(
            "Stage-18 did not authorize the phase/architecture audit route."
        )

    if not bool(s18d.get("stage3_seed0_reproduction_all_pass", False)):
        raise RuntimeError("Stage-18 seed-0 reproduction did not all PASS.")

    # Verify the Stage-18 newly observed m -> m+2 argmax pattern directly from
    # saved kernels. This is a hypothesis-generating prerequisite, not a
    # confirmatory Stage-19 result.
    argmax_count = 0
    phase_rows = read_csv(paths["s18_phase"])

    for row in phase_rows:
        seed = int(row["seed"])
        mode = int(row["mode"])

        npz = (
            stage18_dir
            / f"seed_{seed:03d}"
            / f"mode_{mode:02d}"
            / "phase_state_kernels.npz"
        )

        if not npz.is_file():
            raise FileNotFoundError(npz)

        data = np.load(npz)
        C = data["raw_kernel_corr"]

        t = mode - 1
        rr = C[t, :].copy()
        rr[t] = 0.0

        argmax_mode = int(np.argmax(np.abs(rr))) + 1

        if argmax_mode == mode + 2:
            argmax_count += 1

    if argmax_count != 20:
        raise RuntimeError(
            f"Expected Stage-18 m->m+2 argmax pattern in 20/20, got "
            f"{argmax_count}/20."
        )

    stage3_tracking = read_csv(paths["s3_tracking"])

    anchor = {
        (int(r["mode"]), int(r["epoch"])): r
        for r in stage3_tracking
    }

    return {
        "stage3_sha256": s3_sha,
        "stage18_sha256": s18_sha,
        "stage3_anchor": anchor,
        "stage18_argmax_m_plus_2_count": argmax_count,
    }


# =============================================================================
# Cheap temporal geometry
# =============================================================================

def predict_adam_candidate_from_raw_gradient(
    exp,
    params,
    g_total: torch.Tensor,
) -> torch.Tensor:

    group = exp.optimizer.param_groups[0]

    if float(group.get("weight_decay", 0.0)) != 0.0:
        raise RuntimeError("Expected weight_decay=0.")
    if bool(group.get("amsgrad", False)):
        raise RuntimeError("Expected amsgrad=False.")
    if bool(group.get("maximize", False)):
        raise RuntimeError("Expected maximize=False.")

    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    lr = float(group["lr"])

    parts = []
    offset = 0

    for p in params:
        n = p.numel()
        gp = g_total[offset:offset+n].reshape_as(p)
        offset += n

        state = exp.optimizer.state.get(p, {})

        if "exp_avg" in state:
            m_old = state["exp_avg"]
            v_old = state["exp_avg_sq"]
            raw_step = state["step"]
            step_old = (
                int(raw_step.item())
                if torch.is_tensor(raw_step)
                else int(raw_step)
            )
        else:
            m_old = torch.zeros_like(p)
            v_old = torch.zeros_like(p)
            step_old = 0

        step_new = step_old + 1

        m_new = beta1 * m_old + (1.0 - beta1) * gp
        v_new = beta2 * v_old + (1.0 - beta2) * gp.square()

        bc1 = 1.0 - beta1 ** step_new
        bc2 = 1.0 - beta2 ** step_new

        denom = (
            v_new.sqrt() / math.sqrt(bc2)
        ).add(eps)

        delta = -lr / bc1 * m_new / denom
        parts.append(delta.reshape(-1))

    return flatten(parts)


def cheap_probe(exp, mode: int) -> dict:
    residuals = exp.weak_residuals()

    params = tuple(
        p for p in exp.model.parameters()
        if p.requires_grad
    )

    M = residuals.numel()
    t = mode - 1

    partner_plus = mode + 2
    partner_minus = mode - 2

    grad_t_parts = torch.autograd.grad(
        residuals[t],
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )
    grad_t = flatten(grad_t_parts).detach()

    def residual_grad(index_zero: int):
        parts = torch.autograd.grad(
            residuals[index_zero],
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        return flatten(parts).detach()

    grad_plus = residual_grad(partner_plus - 1)

    grad_minus = (
        residual_grad(partner_minus - 1)
        if partner_minus >= 1
        else None
    )

    total_loss = residuals.square().mean()

    gL_parts = torch.autograd.grad(
        total_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    g_total = flatten(gL_parts).detach()

    g_target = (
        (2.0 / M)
        * residuals[t].detach()
        * grad_t
    )

    candidate = predict_adam_candidate_from_raw_gradient(
        exp=exp,
        params=params,
        g_total=g_total,
    )

    denom = (
        torch.linalg.vector_norm(g_target)
        *
        torch.linalg.vector_norm(candidate)
    ).clamp_min(1.0e-300)

    uphill_cosine = float(
        (
            torch.dot(g_target, candidate)
            / denom
        ).item()
    )

    target_dot = float(
        torch.dot(g_target, candidate).item()
    )

    def signed_squared_residual_grad_cos(
        j_index_zero: int,
        grad_j: torch.Tensor,
    ) -> float:

        denom_j = (
            torch.linalg.vector_norm(grad_t)
            *
            torch.linalg.vector_norm(grad_j)
        ).clamp_min(1.0e-300)

        raw_cos = torch.dot(grad_t, grad_j) / denom_j

        sign = torch.sign(
            residuals[t].detach()
            * residuals[j_index_zero].detach()
        )

        return float((sign * raw_cos).item())

    plus_cos = signed_squared_residual_grad_cos(
        partner_plus - 1,
        grad_plus,
    )

    minus_cos = (
        signed_squared_residual_grad_cos(
            partner_minus - 1,
            grad_minus,
        )
        if grad_minus is not None
        else None
    )

    return {
        "adam_target_uphill_cosine": uphill_cosine,
        "adam_target_dot": target_dot,
        "adam_candidate_target_uphill": bool(target_dot > 0.0),

        "partner_plus_mode": partner_plus,
        "signed_sqgrad_cos_m_plus_2": plus_cos,

        "partner_minus_mode": (
            partner_minus if partner_minus >= 1 else None
        ),
        "signed_sqgrad_cos_m_minus_2": minus_cos,
    }


# =============================================================================
# Reproduction
# =============================================================================

def verify_stage3(
    mode: int,
    epoch: int,
    rel_l2: float,
    rm: dict,
    anchor: dict,
) -> dict:

    key = (mode, epoch)

    if key not in anchor:
        raise RuntimeError(
            f"Stage-3 tracking anchor missing mode={mode}, epoch={epoch}."
        )

    old = anchor[key]

    diffs = {
        "relative_l2_error":
            abs(rel_l2 - float(old["relative_l2_error"])),

        "vpinn_loss":
            abs(
                float(rm["vpinn_loss"])
                - float(old["vpinn_loss"])
            ),

        "target_share":
            abs(
                float(
                    rm["target_mode_residual_energy_share"]
                )
                - float(
                    old["target_mode_residual_energy_share"]
                )
            ),
    }

    gap = max(diffs.values())

    return {
        "mode": mode,
        "epoch": epoch,
        "max_abs_difference": gap,
        "pass": bool(gap <= REPRO_TOL),
    }


# =============================================================================
# Full audit from saved state
# =============================================================================

def audit_saved_state(
    stage18,
    stage3,
    device,
    seed: int,
    mode: int,
    epoch: int,
    state: dict,
    out_dir: Path,
    audit_kind: str,
) -> dict:

    exp = stage18.make_experiment(
        stage3=stage3,
        device=device,
        seed=seed,
        mode=mode,
        out_dir=out_dir,
    )

    stage18.restore_state(exp, state)

    rm = exp.residual_metrics()
    rel = exp.relative_l2_error()

    audit = stage18.phase_state_audit(
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

    np.savez_compressed(
        out_dir / f"{audit_kind.lower()}_kernels.npz",
        raw_kernel=
            arrays["_raw_kernel"],
        raw_kernel_corr=
            arrays["_raw_kernel_corr"],
        signed_squared_residual_gradient_cosine=
            arrays["_signed_squared_residual_gradient_cosine"],
        adam_current_kernel=
            arrays["_adam_current_kernel"],
        adam_current_kernel_corr=
            arrays["_adam_current_kernel_corr"],
        residuals=
            arrays["_residuals"],
    )

    audit["audit_kind"] = audit_kind

    return audit


# =============================================================================
# Plots
# =============================================================================

def plot_temporal(
    rows: List[dict],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    zero: bool = False,
) -> None:

    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    for seed in SEEDS:
        for mode in MODES:
            rr = [
                r for r in rows
                if int(r["seed"]) == seed
                and int(r["mode"]) == mode
                and bool(r.get("probe_active", False))
                and r.get(metric) not in (None, "")
            ]

            rr.sort(key=lambda x: int(x["epoch"]))

            if not rr:
                continue

            ax.plot(
                [int(r["epoch"]) for r in rr],
                [float(r[metric]) for r in rr],
                linewidth=0.9,
                alpha=0.65,
                label=(
                    f"m={mode}, s={seed}"
                    if seed == 0
                    else None
                ),
            )

    if zero:
        ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_mode_counts(mode_summary: List[dict], path: Path) -> None:
    modes = [int(r["mode"]) for r in mode_summary]

    conflicts = [
        int(r["certified_conflict_count"])
        for r in mode_summary
    ]

    no_conflict_conv = [
        int(r["converged_without_certified_conflict_count"])
        for r in mode_summary
    ]

    x = np.arange(len(modes))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    ax.bar(
        x - width/2,
        conflicts,
        width,
        label="Certified optimizer conflict",
    )

    ax.bar(
        x + width/2,
        no_conflict_conv,
        width,
        label="Converged without certified conflict",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in modes])
    ax.set_xlabel("Target frequency m")
    ax.set_ylabel("Seeds out of 5")
    ax.set_ylim(0, 5.5)
    ax.set_title("Does persistent optimizer opposition emerge selectively with frequency?")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=230, bbox_inches="tight")
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

    stage18_script = Path(args.stage18_script)
    if not stage18_script.is_absolute():
        stage18_script = root / stage18_script

    stage18_dir = Path(args.stage18_dir)
    if not stage18_dir.is_absolute():
        stage18_dir = root / stage18_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage3_dir=stage3_dir,
        stage18_script=stage18_script,
        stage18_dir=stage18_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage19R",
    )

    stage18 = load_module(
        stage18_script,
        "vpinn_stage18_stage19R",
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

        "stage19r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "temporal_conflict_onset_and_parity_ladder_audit",

            "modes":
                list(MODES),

            "seeds":
                list(SEEDS),

            "tracking_interval":
                TRACK_INTERVAL,

            "mechanism_active":
                "target share >=0.80 and relL2 >1e-2",

            "certified_conflict":
                "Adam target dot >0 for 3 consecutive active probes",

            "certified_parity_anti_alignment":
                "signed sqgrad cosine(m,m+2) <= -0.95 for 3 consecutive active probes",

            "strong_frequency_selective_gate":
                "N3<=N5<=N7<=N9 AND N9>=4 AND N9-N3>=3",

            "persistent_parity_gate":
                "fraction C(m,m+2)<=-0.95 >=0.80 for every mode",

            "expensive_audit":
                "once per run at conflict onset or final active no-conflict state",

            "new_optimizer_intervention":
                False,
        },
    }

    write_json(out_dir / "manifest.json", manifest)

    print("=" * 168)
    print("VPINN — STAGE 19R TEMPORAL CONFLICT-ONSET + PARITY-LADDER AUDIT")
    print("=" * 168)
    print(f"device                    : {device}")
    print(f"modes                     : {list(MODES)}")
    print(f"seeds                     : {list(SEEDS)}")
    print("active probe              : target share>=0.80 and relL2>1e-2")
    print("conflict certification    : 3 consecutive target-uphill probes")
    print("full J/Hessian audit      : once per run")
    print("trained intervention      : NONE")
    print("=" * 168)

    tracking_rows: List[dict] = []
    run_rows: List[dict] = []
    full_audit_rows: List[dict] = []
    repro_rows: List[dict] = []

    global_start = time.perf_counter()

    for seed in SEEDS:
        for mode in MODES:

            run_dir = (
                out_dir
                / f"seed_{seed:03d}"
                / f"mode_{mode:02d}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            exp = stage18.make_experiment(
                stage3=stage3,
                device=device,
                seed=seed,
                mode=mode,
                out_dir=run_dir,
            )

            conflict_streak = 0
            conflict_candidate_epoch = None
            conflict_candidate_state = None

            parity_streak = 0
            parity_onset = -1

            convergence_streak = 0
            convergence_candidate_epoch = None
            convergence_onset = -1
            convergence_confirmation = -1

            certified_conflict = False
            conflict_onset = -1
            conflict_confirmation = -1

            last_active_epoch = -1
            last_active_state = None

            active_probe_count = 0
            parity_anti_count = 0
            positive_adam_probe_count = 0

            stop_reason = "MAX_EPOCH"

            print()
            print("-" * 168)
            print(
                f"seed={seed} mode={mode} "
                f"a={exp.amplitude:.12g} a*m={exp.amplitude*mode:.12g}"
            )

            for epoch in range(MAX_EPOCH + 1):

                if epoch % TRACK_INTERVAL == 0:

                    rm = exp.residual_metrics()
                    rel = exp.relative_l2_error()

                    if seed == 0:
                        check = verify_stage3(
                            mode=mode,
                            epoch=epoch,
                            rel_l2=rel,
                            rm=rm,
                            anchor=pf["stage3_anchor"],
                        )

                        repro_rows.append(check)

                        if not check["pass"]:
                            raise RuntimeError(
                                f"Stage-3 reproduction failed m={mode}, "
                                f"epoch={epoch}, gap={check['max_abs_difference']:.3e}"
                            )

                    active = bool(
                        rm["target_mode_residual_energy_share"]
                        >= ACTIVE_TARGET_SHARE
                        and
                        rel > CONVERGENCE_REL_L2
                    )

                    probe = {}

                    if active:
                        active_probe_count += 1

                        probe = cheap_probe(
                            exp=exp,
                            mode=mode,
                        )

                        last_active_epoch = epoch
                        last_active_state = stage18.capture_state(exp)

                        if probe["adam_candidate_target_uphill"]:
                            positive_adam_probe_count += 1

                            if conflict_streak == 0:
                                conflict_candidate_epoch = epoch
                                conflict_candidate_state = stage18.capture_state(exp)

                            conflict_streak += 1

                        else:
                            conflict_streak = 0
                            conflict_candidate_epoch = None
                            conflict_candidate_state = None

                        if (
                            probe["signed_sqgrad_cos_m_plus_2"]
                            <= PARITY_ANTI_THRESHOLD
                        ):
                            parity_anti_count += 1

                            if parity_streak == 0:
                                parity_candidate_epoch = epoch

                            parity_streak += 1

                            if (
                                parity_onset < 0
                                and parity_streak >= CERTIFY_POINTS
                            ):
                                parity_onset = parity_candidate_epoch

                        else:
                            parity_streak = 0

                        if (
                            not certified_conflict
                            and conflict_streak >= CERTIFY_POINTS
                        ):
                            certified_conflict = True
                            conflict_onset = int(conflict_candidate_epoch)
                            conflict_confirmation = epoch

                    qualifies_convergence = bool(
                        rel <= CONVERGENCE_REL_L2
                        and
                        rm["target_mode_residual_energy_share"]
                        <= CONVERGENCE_TARGET_SHARE
                    )

                    if qualifies_convergence:
                        if convergence_streak == 0:
                            convergence_candidate_epoch = epoch

                        convergence_streak += 1

                        if (
                            convergence_onset < 0
                            and convergence_streak >= CERTIFY_POINTS
                        ):
                            convergence_onset = int(
                                convergence_candidate_epoch
                            )
                            convergence_confirmation = epoch
                    else:
                        convergence_streak = 0
                        convergence_candidate_epoch = None

                    tracking_rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "epoch": epoch,

                            "relative_l2_error": rel,
                            **rm,

                            "probe_active": active,

                            "adam_target_uphill_cosine":
                                probe.get("adam_target_uphill_cosine"),

                            "adam_target_dot":
                                probe.get("adam_target_dot"),

                            "adam_candidate_target_uphill":
                                probe.get("adam_candidate_target_uphill"),

                            "signed_sqgrad_cos_m_plus_2":
                                probe.get("signed_sqgrad_cos_m_plus_2"),

                            "signed_sqgrad_cos_m_minus_2":
                                probe.get("signed_sqgrad_cos_m_minus_2"),

                            "conflict_streak":
                                conflict_streak,

                            "parity_streak":
                                parity_streak,
                        }
                    )

                    # Once certified, audit the onset state saved at the first
                    # point in the certified 3-point run, then stop. This keeps
                    # cost minimal and preserves the causal onset state.
                    if certified_conflict:
                        audit = audit_saved_state(
                            stage18=stage18,
                            stage3=stage3,
                            device=device,
                            seed=seed,
                            mode=mode,
                            epoch=conflict_onset,
                            state=conflict_candidate_state,
                            out_dir=run_dir,
                            audit_kind="CERTIFIED_CONFLICT_ONSET",
                        )

                        full_audit_rows.append(audit)

                        stop_reason = "CERTIFIED_CONFLICT"

                        print(
                            f"  CERTIFIED CONFLICT: onset={conflict_onset}, "
                            f"confirm={conflict_confirmation}, "
                            f"U={audit['adam_target_uphill_cosine']:+.6f}, "
                            f"C(m,m+2)="
                            f"{next(r['signed_sqgrad_cos_m_plus_2'] for r in reversed(tracking_rows) if int(r['seed'])==seed and int(r['mode'])==mode and int(r['epoch'])==conflict_onset):+.6f}"
                        )
                        break

                    if convergence_onset >= 0:
                        stop_reason = "CERTIFIED_CONVERGENCE"
                        break

                if epoch < MAX_EPOCH:
                    exp.train_step()

            if not certified_conflict:

                if last_active_state is not None:
                    audit = audit_saved_state(
                        stage18=stage18,
                        stage3=stage3,
                        device=device,
                        seed=seed,
                        mode=mode,
                        epoch=last_active_epoch,
                        state=last_active_state,
                        out_dir=run_dir,
                        audit_kind="FINAL_ACTIVE_NO_CONFLICT",
                    )

                    full_audit_rows.append(audit)

                if stop_reason == "MAX_EPOCH":
                    stop_reason = (
                        "CENSORED_UNRESOLVED"
                        if convergence_onset < 0
                        else "CERTIFIED_CONVERGENCE"
                    )

            run_rows.append(
                {
                    "seed": seed,
                    "mode": mode,

                    "stop_reason": stop_reason,

                    "certified_conflict":
                        certified_conflict,

                    "conflict_onset_epoch":
                        conflict_onset,

                    "conflict_confirmation_epoch":
                        conflict_confirmation,

                    "certified_parity_ladder":
                        bool(parity_onset >= 0),

                    "parity_ladder_onset_epoch":
                        parity_onset,

                    "active_probe_count":
                        active_probe_count,

                    "positive_adam_probe_count":
                        positive_adam_probe_count,

                    "positive_adam_probe_fraction":
                        (
                            positive_adam_probe_count
                            / active_probe_count
                            if active_probe_count
                            else None
                        ),

                    "parity_anti_probe_count":
                        parity_anti_count,

                    "parity_anti_probe_fraction":
                        (
                            parity_anti_count
                            / active_probe_count
                            if active_probe_count
                            else None
                        ),

                    "convergence_onset_epoch":
                        convergence_onset,

                    "convergence_confirmation_epoch":
                        convergence_confirmation,

                    "last_active_epoch":
                        last_active_epoch,
                }
            )

            print(
                f"  result={stop_reason} | "
                f"conflict={certified_conflict} onset={conflict_onset} | "
                f"active probes={active_probe_count} | "
                f"parity fraction="
                f"{(parity_anti_count/max(active_probe_count,1)):.3f}"
            )

    # -------------------------------------------------------------------------
    # Persist raw results.
    # -------------------------------------------------------------------------
    write_csv(out_dir / "tracking_metrics.csv", tracking_rows)
    write_csv(out_dir / "run_summary.csv", run_rows)
    write_csv(out_dir / "full_state_audits.csv", full_audit_rows)
    write_csv(out_dir / "stage3_seed0_reproduction.csv", repro_rows)

    # -------------------------------------------------------------------------
    # Aggregate.
    # -------------------------------------------------------------------------
    mode_summary = []

    for mode in MODES:
        rr = [
            r for r in run_rows
            if int(r["mode"]) == mode
        ]

        conflicts = [
            r for r in rr
            if bool(r["certified_conflict"])
        ]

        converged_no_conflict = [
            r for r in rr
            if (
                not bool(r["certified_conflict"])
                and int(r["convergence_onset_epoch"]) >= 0
            )
        ]

        active_probe_total = sum(
            int(r["active_probe_count"])
            for r in rr
        )

        parity_total = sum(
            int(r["parity_anti_probe_count"])
            for r in rr
        )

        mode_summary.append(
            {
                "mode": mode,

                "certified_conflict_count":
                    len(conflicts),

                "certified_conflict_fraction":
                    len(conflicts) / len(SEEDS),

                "median_conflict_onset_epoch":
                    (
                        float(
                            np.median([
                                int(r["conflict_onset_epoch"])
                                for r in conflicts
                            ])
                        )
                        if conflicts
                        else None
                    ),

                "converged_without_certified_conflict_count":
                    len(converged_no_conflict),

                "censored_unresolved_without_conflict_count":
                    sum(
                        int(
                            r["stop_reason"]
                            == "CENSORED_UNRESOLVED"
                        )
                        for r in rr
                    ),

                "total_active_probe_count":
                    active_probe_total,

                "parity_anti_probe_count":
                    parity_total,

                "parity_anti_probe_fraction":
                    (
                        parity_total / active_probe_total
                        if active_probe_total
                        else None
                    ),

                "certified_parity_ladder_seed_count":
                    sum(
                        int(bool(r["certified_parity_ladder"]))
                        for r in rr
                    ),

                "median_positive_adam_probe_fraction":
                    (
                        float(
                            np.median([
                                float(r["positive_adam_probe_fraction"])
                                for r in rr
                                if r["positive_adam_probe_fraction"]
                                is not None
                            ])
                        )
                        if any(
                            r["positive_adam_probe_fraction"] is not None
                            for r in rr
                        )
                        else None
                    ),
            }
        )

    write_csv(out_dir / "mode_summary.csv", mode_summary)

    counts = {
        int(r["mode"]): int(r["certified_conflict_count"])
        for r in mode_summary
    }

    strong_frequency = bool(
        counts[3] <= counts[5] <= counts[7] <= counts[9]
        and counts[9] >= 4
        and counts[9] - counts[3] >= 3
    )

    parity_by_mode = {
        int(r["mode"]): float(r["parity_anti_probe_fraction"])
        if r["parity_anti_probe_fraction"] is not None
        else 0.0
        for r in mode_summary
    }

    persistent_parity = all(
        parity_by_mode[m] >= PARITY_PERSISTENCE_FRACTION
        for m in MODES
    )

    if strong_frequency:
        route_class = (
            "frequency_selective_certified_optimizer_conflict"
        )

        next_route = (
            "stage20R_dense_frequency_transition_localization"
        )

    elif persistent_parity:
        route_class = (
            "persistent_parity_ladder_without_clean_frequency_conflict_law"
        )

        next_route = (
            "stage20R_architecture_control_parity_ladder"
        )

    else:
        route_class = (
            "phase_and_architecture_heterogeneous_mechanism"
        )

        next_route = (
            "stage20R_phase_architecture_heterogeneity_audit"
        )

    decision = {
        "n_runs":
            len(run_rows),

        "certified_conflict_counts_by_mode":
            counts,

        "strong_frequency_selective_conflict":
            strong_frequency,

        "parity_anti_probe_fraction_by_mode":
            parity_by_mode,

        "persistent_parity_ladder":
            persistent_parity,

        "stage3_seed0_reproduction_all_pass":
            all(bool(r["pass"]) for r in repro_rows),

        "stage18_hypothesis_generating_m_plus_2_argmax":
            f"{pf['stage18_argmax_m_plus_2_count']}/20",

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "Certified conflict is a trajectory diagnostic in this matched 1D "
            "family. Parity-ladder persistence, if present, is an empirical "
            "finite-width structure and is not yet a theorem or universal "
            "VPINN property."
        ),
    }

    write_json(out_dir / "decision.json", decision)

    # -------------------------------------------------------------------------
    # Plots.
    # -------------------------------------------------------------------------
    plot_temporal(
        rows=tracking_rows,
        metric="adam_target_uphill_cosine",
        ylabel="Adam target-uphill cosine",
        title="Temporal emergence of actual optimizer opposition",
        path=out_dir / "adam_target_alignment_over_time.png",
        zero=True,
    )

    plot_temporal(
        rows=tracking_rows,
        metric="signed_sqgrad_cos_m_plus_2",
        ylabel="signed cosine of squared-residual gradients",
        title="Does the m↔m+2 anti-alignment persist toward conflict?",
        path=out_dir / "parity_ladder_alignment_over_time.png",
        zero=True,
    )

    plot_mode_counts(
        mode_summary,
        out_dir / "certified_conflict_counts_vs_frequency.png",
    )

    # -------------------------------------------------------------------------
    # Console.
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - global_start

    lines = []
    lines.append("=" * 170)
    lines.append(
        "VPINN — STAGE 19R TEMPORAL CONFLICT-ONSET + PARITY-LADDER SUMMARY"
    )
    lines.append("=" * 170)

    lines.append(
        "mode | certified conflict | median onset | converge without conflict | "
        "parity anti fraction | parity certified seeds"
    )
    lines.append("-" * 170)

    for r in mode_summary:
        lines.append(
            f"{int(r['mode']):4d} | "
            f"{int(r['certified_conflict_count']):5d}/5           | "
            f"{str(r['median_conflict_onset_epoch']):12s} | "
            f"{int(r['converged_without_certified_conflict_count']):5d}/5                   | "
            f"{str(r['parity_anti_probe_fraction']):20s} | "
            f"{int(r['certified_parity_ladder_seed_count']):5d}/5"
        )

    lines.append("-" * 170)

    lines.append(
        f"conflict counts N3,N5,N7,N9          : "
        f"{counts}"
    )

    lines.append(
        f"strong frequency-selective conflict  : "
        f"{strong_frequency}"
    )

    lines.append(
        f"parity anti fractions                : "
        f"{parity_by_mode}"
    )

    lines.append(
        f"persistent parity ladder             : "
        f"{persistent_parity}"
    )

    lines.append(
        f"Stage-3 seed0 reproduction           : "
        f"{sum(int(r['pass']) for r in repro_rows)}/"
        f"{len(repro_rows)} PASS"
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

    lines.append("=" * 170)
    lines.append(
        "Guardrail: Stage 19R may identify a frequency-selective conflict or a "
        "persistent parity ladder. Neither is a universal VPINN law without "
        "architecture and problem-family controls."
    )
    lines.append("=" * 170)

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
