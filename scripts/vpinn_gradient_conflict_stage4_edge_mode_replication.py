#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 4
Fixed-Budget Edge-Mode Replication + Target-vs-Rest Cancellation Audit
======================================================================

Why this stage exists
---------------------
Stage 3 established matched-weak-scale frequency transfer for m={3,5,7,9}.
For m=9, seed 0 remained unresolved at epoch 2500:

  * relative L2 error stayed near the low-mode-only plateau,
  * the target residual mode v_9 remained dominant,
  * the global gradient coherence became extremely small.

Stage 4 tests whether that edge-mode lock is reproducible across seeds and
adds a more direct diagnostic than pairwise conflict:

    g_target = gradient of R_m^2
    g_rest   = sum_{k != m} gradient of R_k^2

We measure

    cos(target, rest)
        = <g_target, g_rest> / (||g_target|| ||g_rest||)

and

    cancellation_ratio
        = ||g_target + g_rest|| / (||g_target|| + ||g_rest||).

A strong target-vs-rest cancellation state has:
  * cosine close to -1,
  * comparable target/rest gradient norms,
  * cancellation_ratio close to 0.

Scientific design
-----------------
Target mode : m = 9
Amplitude   : a_9 = 0.15 * 7 / 9
Seeds       : {0,1,2,3,4}
Epoch budget: 2500, fixed for every seed
Solver      : imported directly from Stage 3 to prevent implementation drift
BC          : exact hard homogeneous Dirichlet
Test basis  : H_0^1-orthonormal sine basis
dtype       : float64
Quadrature  : 256-point Gauss-Legendre
No early stopping.
All five seeds run unconditionally.

Primary precommitted fixed-budget "edge lock" for a seed
--------------------------------------------------------
At epoch 2500:
  1) relative L2 error > 1e-2;
  2) target residual-energy share >= 0.80;
  3) target mode remains the dominant residual mode;
  4) cos(g_target, g_rest) <= -0.95;
  5) ||g_rest|| / ||g_target|| in [0.5, 2.0];
  6) target-rest cancellation_ratio <= 0.20.

Group replication:
  at least 4/5 seeds satisfy all six conditions.

If the group gate passes, the next scientifically justified stage is a
long-horizon escape-time experiment. If it fails, the m=9 boundary is
initialization-sensitive and should be studied as such rather than advertised
as a universal failure.

This is deliberately a bounded, low-cost replication stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict
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
        description=(
            "Stage-4 fixed-budget m=9 replication and "
            "target-vs-rest gradient cancellation audit."
        )
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]
    )
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n-test", type=int, default=24)
    p.add_argument("--n-quad", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=4001)
    p.add_argument("--track-interval", type=int, default=25)
    p.add_argument(
        "--diagnostic-epochs",
        type=int,
        nargs="+",
        default=[0, 50, 100, 250, 500, 1000, 1500, 2000, 2500],
    )
    p.add_argument(
        "--stage3-script",
        type=str,
        default="vpinn_gradient_conflict_stage3_frequency_transfer.py",
    )
    p.add_argument(
        "--stage3-anchor-dir",
        type=str,
        default="vpinn_gradient_conflict_stage3_frequency_transfer",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="vpinn_gradient_conflict_stage4_edge_mode_replication",
    )
    args = p.parse_args()

    if len(args.seeds) < 2:
        raise ValueError("Use at least two seeds.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.n_test < 11:
        raise ValueError("n_test must be >= 11 for the m=9 audit.")
    if args.n_quad < max(32, 2 * args.n_test):
        raise ValueError("Use n_quad >= max(32, 2*n_test).")
    if args.track_interval < 1:
        raise ValueError("--track-interval must be >= 1.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")

    args.diagnostic_epochs = sorted(
        set(
            e for e in args.diagnostic_epochs
            if 0 <= e <= args.epochs
        )
        | {0, args.epochs}
    )

    return args


# =============================================================================
# Utilities
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stage3_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage-3 solver not found: {path}\n"
            "Place this Stage-4 script next to the Stage-3 Python file."
        )
    spec = importlib.util.spec_from_file_location(
        "vpinn_stage3_solver", str(path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Stage-3 solver from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Direct target-vs-rest geometry
# =============================================================================

def target_vs_rest_geometry(
    gradient_norms: np.ndarray,
    cosine_matrix: np.ndarray,
    target_mode: int,
) -> Dict[str, float]:
    """
    Reconstruct exact aggregate geometry from per-test norms and cosine matrix.

    Let:
        g_t = target gradient
        g_r = sum of all active non-target gradients

    Since <g_i,g_j> = ||g_i|| ||g_j|| C_ij, we can recover:
        ||g_r||^2,
        <g_t,g_r>,
        cos(g_t,g_r),
        ||g_t+g_r||.

    No parameter-gradient vectors need to be stored.
    """
    norms = np.asarray(gradient_norms, dtype=np.float64)
    C = np.asarray(cosine_matrix, dtype=np.float64)

    t = target_mode - 1
    if t < 0 or t >= norms.size:
        raise IndexError("target_mode outside gradient vector.")

    active = np.isfinite(np.diag(C)) & np.isfinite(norms) & (norms > 0.0)
    if not active[t]:
        return {
            "target_gradient_norm": float(norms[t]),
            "rest_gradient_norm": float("nan"),
            "target_rest_norm_ratio": float("nan"),
            "target_vs_rest_cosine": float("nan"),
            "target_rest_cancellation_ratio": float("nan"),
            "target_rest_net_gradient_norm": float("nan"),
            "target_rest_balance": float("nan"),
        }

    others = np.where(active)[0]
    others = others[others != t]

    nt = float(norms[t])

    if others.size == 0:
        return {
            "target_gradient_norm": nt,
            "rest_gradient_norm": 0.0,
            "target_rest_norm_ratio": 0.0,
            "target_vs_rest_cosine": float("nan"),
            "target_rest_cancellation_ratio": 1.0,
            "target_rest_net_gradient_norm": nt,
            "target_rest_balance": 0.0,
        }

    n_o = norms[others]
    C_oo = C[np.ix_(others, others)]

    # ||sum_{j != t} g_j||^2 = n^T C n
    rest_sq = float(n_o @ C_oo @ n_o)
    # Roundoff can make a theoretically nonnegative number tiny-negative.
    rest_sq = max(rest_sq, 0.0)
    nr = math.sqrt(rest_sq)

    # <g_t, sum_{j != t} g_j>
    dot = float(nt * np.sum(n_o * C[t, others]))

    if nt > 0.0 and nr > 0.0:
        cos_tr = dot / (nt * nr)
        cos_tr = max(-1.0, min(1.0, cos_tr))
    else:
        cos_tr = float("nan")

    net_sq = nt * nt + nr * nr + 2.0 * dot
    net_sq = max(net_sq, 0.0)
    net = math.sqrt(net_sq)

    denom = nt + nr
    cancellation = net / denom if denom > 0.0 else float("nan")
    ratio = nr / nt if nt > 0.0 else float("inf")

    max_norm = max(nt, nr)
    balance = min(nt, nr) / max_norm if max_norm > 0.0 else float("nan")

    return {
        "target_gradient_norm": nt,
        "rest_gradient_norm": nr,
        "target_rest_norm_ratio": ratio,
        "target_vs_rest_cosine": cos_tr,
        "target_rest_cancellation_ratio": cancellation,
        "target_rest_net_gradient_norm": net,
        "target_rest_balance": balance,
    }


# =============================================================================
# Stage-3 anchor check
# =============================================================================

def check_seed0_anchor(
    stage3_anchor_dir: Path,
    new_diag_rows: List[dict],
    new_track_rows: List[dict],
    tol: float = 1.0e-10,
) -> dict:
    old_diag_path = stage3_anchor_dir / "mode_09" / "diagnostic_metrics.csv"
    old_track_path = stage3_anchor_dir / "mode_09" / "tracking_metrics.csv"

    result = {
        "anchor_found": old_diag_path.is_file() and old_track_path.is_file(),
        "diagnostic_anchor": str(old_diag_path),
        "tracking_anchor": str(old_track_path),
        "tolerance": tol,
        "shared_diagnostic_epochs": 0,
        "shared_tracking_epochs": 0,
        "max_abs_diagnostic_difference": None,
        "max_abs_tracking_difference": None,
        "pass": None,
    }

    if not result["anchor_found"]:
        return result

    old_diag = {
        int(r["epoch"]): r for r in read_csv(old_diag_path)
    }
    new_diag = {
        int(r["epoch"]): r
        for r in new_diag_rows
        if int(r["seed"]) == 0
    }

    diag_shared = sorted(set(old_diag) & set(new_diag))
    result["shared_diagnostic_epochs"] = len(diag_shared)

    diag_fields = [
        "relative_l2_error",
        "vpinn_loss",
        "target_mode_residual_energy_share",
        "gradient_coherence",
        "weighted_negative_cosine",
    ]

    diag_diffs = []
    for e in diag_shared:
        for field in diag_fields:
            diag_diffs.append(
                abs(float(old_diag[e][field]) - float(new_diag[e][field]))
            )

    old_track = {
        int(r["epoch"]): r for r in read_csv(old_track_path)
    }
    new_track = {
        int(r["epoch"]): r
        for r in new_track_rows
        if int(r["seed"]) == 0
    }

    track_shared = sorted(set(old_track) & set(new_track))
    result["shared_tracking_epochs"] = len(track_shared)

    track_fields = [
        "relative_l2_error",
        "vpinn_loss",
        "target_mode_residual_energy_share",
        "target_mode_abs_residual",
    ]

    track_diffs = []
    for e in track_shared:
        for field in track_fields:
            track_diffs.append(
                abs(float(old_track[e][field]) - float(new_track[e][field]))
            )

    max_diag = max(diag_diffs) if diag_diffs else float("inf")
    max_track = max(track_diffs) if track_diffs else float("inf")

    result["max_abs_diagnostic_difference"] = max_diag
    result["max_abs_tracking_difference"] = max_track
    result["pass"] = bool(max_diag <= tol and max_track <= tol)

    return result


# =============================================================================
# Plotting
# =============================================================================

def plot_by_seed(
    rows: List[dict],
    key: str,
    ylabel: str,
    title: str,
    path: Path,
    log_y: bool = False,
    hline: float | None = None,
) -> None:
    seeds = sorted(set(int(r["seed"]) for r in rows))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for seed in seeds:
        rr = [r for r in rows if int(r["seed"]) == seed]
        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r[key]) for r in rr],
            label=f"seed {seed}",
            marker=None,
        )

    if log_y:
        vals = [
            float(r[key])
            for r in rows
            if np.isfinite(float(r[key])) and float(r[key]) > 0
        ]
        if vals:
            ax.set_yscale("log")

    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1.2)

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

    script_dir = Path(__file__).resolve().parent
    stage3_script = Path(args.stage3_script)
    if not stage3_script.is_absolute():
        stage3_script = script_dir / stage3_script

    stage3_anchor_dir = Path(args.stage3_anchor_dir)
    if not stage3_anchor_dir.is_absolute():
        stage3_anchor_dir = script_dir / stage3_anchor_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    s3 = load_stage3_module(stage3_script)
    device = resolve_device(args.device)

    target_mode = 9
    reference_mode = 7
    reference_amplitude = 0.15
    target_amplitude = reference_amplitude * reference_mode / target_mode
    theoretical_low_only_error = (
        abs(target_amplitude)
        / math.sqrt(1.0 + target_amplitude * target_amplitude)
    )

    # Precommitted criteria.
    convergence_threshold = 1.0e-2
    final_target_share_threshold = 0.80
    target_rest_cosine_threshold = -0.95
    target_rest_ratio_lower = 0.50
    target_rest_ratio_upper = 2.00
    target_rest_cancellation_threshold = 0.20

    min_group_pass = math.ceil(0.80 * len(args.seeds))

    precommitment = {
        "stage": "fixed_budget_edge_mode_replication",
        "target_mode": target_mode,
        "target_amplitude": target_amplitude,
        "matched_weak_scale_a_times_m": target_amplitude * target_mode,
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "all_seeds_unconditional": True,
        "no_early_stop": True,
        "solver_reused_from_stage3": True,
        "per_seed_final_edge_lock": {
            "relative_l2_error_gt": convergence_threshold,
            "target_residual_energy_share_ge":
                final_target_share_threshold,
            "target_mode_is_dominant": True,
            "target_vs_rest_cosine_le":
                target_rest_cosine_threshold,
            "rest_over_target_gradient_norm_in": [
                target_rest_ratio_lower,
                target_rest_ratio_upper,
            ],
            "target_rest_cancellation_ratio_le":
                target_rest_cancellation_threshold,
        },
        "group_gate": f"at least {min_group_pass}/{len(args.seeds)} seeds",
        "next_route_if_pass":
            "long_horizon_escape_time_audit",
        "next_route_if_fail":
            "initialization_sensitive_edge_transition_audit",
    }

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),
        "stage3_solver_path": str(stage3_script),
        "stage3_solver_sha256": sha256_file(stage3_script),
        "stage4_script_sha256": sha256_file(Path(__file__).resolve()),
        "precommitment": precommitment,
    }
    write_json(out_dir / "manifest.json", manifest)

    print("=" * 104)
    print("VPINN GRADIENT GEOMETRY — STAGE 4 FIXED-BUDGET EDGE-MODE REPLICATION")
    print("=" * 104)
    print(f"device                       : {device}")
    print(f"target mode                  : {target_mode}")
    print(f"target amplitude             : {target_amplitude:.12g}")
    print(f"matched a_m*m                : {target_amplitude*target_mode:.12g}")
    print(f"theoretical low-only relL2   : {theoretical_low_only_error:.9e}")
    print(f"seeds                        : {args.seeds}")
    print(f"fixed epoch budget           : {args.epochs}")
    print(f"required group replication   : {min_group_pass}/{len(args.seeds)}")
    print(f"Stage-3 solver SHA256        : {manifest['stage3_solver_sha256']}")
    print("=" * 104)

    track_epochs = set(range(0, args.epochs + 1, args.track_interval))
    track_epochs.add(args.epochs)
    diag_epochs = set(args.diagnostic_epochs)

    all_track: List[dict] = []
    all_diag: List[dict] = []
    summaries: List[dict] = []

    global_start = time.perf_counter()

    for seed in args.seeds:
        seed_dir = out_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        cfg = s3.Config(
            seed=seed,
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.lr,
            width=args.width,
            depth=args.depth,
            n_test=args.n_test,
            n_quad=args.n_quad,
            n_eval=args.n_eval,
            modes=(target_mode,),
            reference_mode=reference_mode,
            reference_amplitude=reference_amplitude,
            track_interval=args.track_interval,
            diagnostic_epochs=tuple(args.diagnostic_epochs),
            output_dir=str(seed_dir),
        )

        exp = s3.ModeExperiment(
            cfg=cfg,
            device=device,
            mode=target_mode,
            out_dir=seed_dir,
        )

        print()
        print("-" * 104)
        print(f"SEED {seed}")

        seed_track: List[dict] = []
        seed_diag: List[dict] = []

        for epoch in range(args.epochs + 1):
            if epoch in track_epochs:
                rm = exp.residual_metrics()
                rel = exp.relative_l2_error()

                row = {
                    "seed": seed,
                    "epoch": epoch,
                    "relative_l2_error": rel,
                    **rm,
                    "preconvergence":
                        int(rel > convergence_threshold),
                }
                seed_track.append(row)
                all_track.append(row)

            if epoch in diag_epochs:
                rm = exp.residual_metrics()
                rel = exp.relative_l2_error()
                gd = exp.gradient_diagnostics()

                cosine = gd.pop("_cosine_matrix")
                gradient_norms = gd.pop("_gradient_norms")
                residuals = gd.pop("_residuals")

                tvr = target_vs_rest_geometry(
                    gradient_norms=gradient_norms,
                    cosine_matrix=cosine,
                    target_mode=target_mode,
                )

                row = {
                    "seed": seed,
                    "epoch": epoch,
                    "relative_l2_error": rel,
                    **rm,
                    **gd,
                    **tvr,
                    "preconvergence":
                        int(rel > convergence_threshold),
                }
                seed_diag.append(row)
                all_diag.append(row)

                np.savez_compressed(
                    seed_dir / f"diagnostic_{epoch:04d}.npz",
                    residuals=residuals,
                    gradient_norms=gradient_norms,
                    cosine_matrix=cosine,
                )

                print(
                    f"epoch={epoch:4d} "
                    f"relL2={rel:.6e} "
                    f"target_share={rm['target_mode_residual_energy_share']:.6f} "
                    f"cos(target,rest)={tvr['target_vs_rest_cosine']:.6f} "
                    f"rest/target={tvr['target_rest_norm_ratio']:.6f} "
                    f"cancel={tvr['target_rest_cancellation_ratio']:.6e}"
                )

            if epoch == args.epochs:
                break

            exp.train_step()

        write_csv(seed_dir / "tracking_metrics.csv", seed_track)
        write_csv(seed_dir / "diagnostic_metrics.csv", seed_diag)

        final_track = next(
            r for r in seed_track if int(r["epoch"]) == args.epochs
        )
        final_diag = next(
            r for r in seed_diag if int(r["epoch"]) == args.epochs
        )

        edge_lock_pass = bool(
            final_track["relative_l2_error"] > convergence_threshold
            and final_track["target_mode_residual_energy_share"]
                >= final_target_share_threshold
            and int(final_track["dominant_residual_mode"]) == target_mode
            and final_diag["target_vs_rest_cosine"]
                <= target_rest_cosine_threshold
            and target_rest_ratio_lower
                <= final_diag["target_rest_norm_ratio"]
                <= target_rest_ratio_upper
            and final_diag["target_rest_cancellation_ratio"]
                <= target_rest_cancellation_threshold
        )

        # Descriptive best locked-state geometry before/at the budget.
        locked_diag = [
            r for r in seed_diag
            if (
                r["relative_l2_error"] > convergence_threshold
                and r["target_mode_residual_energy_share"]
                    >= final_target_share_threshold
            )
        ]

        if locked_diag:
            min_cos_row = min(
                locked_diag,
                key=lambda r: r["target_vs_rest_cosine"],
            )
            min_cancel_row = min(
                locked_diag,
                key=lambda r: r["target_rest_cancellation_ratio"],
            )
        else:
            min_cos_row = seed_diag[-1]
            min_cancel_row = seed_diag[-1]

        first_converged = next(
            (
                int(r["epoch"]) for r in seed_track
                if r["relative_l2_error"] <= convergence_threshold
            ),
            -1,
        )

        summary = {
            "seed": seed,
            "final_relative_l2_error":
                final_track["relative_l2_error"],
            "final_error_over_low_only_theory":
                final_track["relative_l2_error"]
                / theoretical_low_only_error,
            "final_vpinn_loss":
                final_track["vpinn_loss"],
            "final_target_mode_energy_share":
                final_track["target_mode_residual_energy_share"],
            "final_dominant_residual_mode":
                int(final_track["dominant_residual_mode"]),
            "final_gradient_coherence":
                final_diag["gradient_coherence"],
            "final_weighted_negative_cosine":
                final_diag["weighted_negative_cosine"],
            "final_target_gradient_norm":
                final_diag["target_gradient_norm"],
            "final_rest_gradient_norm":
                final_diag["rest_gradient_norm"],
            "final_rest_over_target_gradient_norm":
                final_diag["target_rest_norm_ratio"],
            "final_target_vs_rest_cosine":
                final_diag["target_vs_rest_cosine"],
            "final_target_rest_cancellation_ratio":
                final_diag["target_rest_cancellation_ratio"],
            "min_locked_target_vs_rest_cosine":
                min_cos_row["target_vs_rest_cosine"],
            "epoch_min_locked_target_vs_rest_cosine":
                int(min_cos_row["epoch"]),
            "min_locked_target_rest_cancellation_ratio":
                min_cancel_row["target_rest_cancellation_ratio"],
            "epoch_min_locked_target_rest_cancellation_ratio":
                int(min_cancel_row["epoch"]),
            "first_relative_l2_le_1e_2_epoch":
                first_converged,
            "edge_lock_pass":
                edge_lock_pass,
            "gram_max_abs_error":
                exp.gram_error,
        }
        summaries.append(summary)

    write_csv(out_dir / "aggregate_tracking_metrics.csv", all_track)
    write_csv(out_dir / "aggregate_diagnostic_metrics.csv", all_diag)
    write_csv(out_dir / "seed_summary.csv", summaries)

    # Exact Stage-3 seed-0 reproduction audit when the anchor is available.
    anchor = check_seed0_anchor(
        stage3_anchor_dir=stage3_anchor_dir,
        new_diag_rows=all_diag,
        new_track_rows=all_track,
        tol=1.0e-10,
    )

    edge_lock_count = sum(bool(r["edge_lock_pass"]) for r in summaries)
    group_pass = bool(edge_lock_count >= min_group_pass)
    anchor_gate = bool(anchor["pass"] is not False)

    route_long_horizon = bool(group_pass and anchor_gate)

    decision = {
        "target_mode": target_mode,
        "n_seeds": len(args.seeds),
        "seeds": list(args.seeds),
        "fixed_epoch_budget": args.epochs,
        "edge_lock_pass_count": edge_lock_count,
        "required_group_pass_count": min_group_pass,
        "edge_lock_group_pass": group_pass,
        "stage3_seed0_anchor_reproduction": anchor,
        "route_long_horizon_escape_time_audit":
            route_long_horizon,
        "alternative_route_if_not_authorized":
            "initialization_sensitive_edge_transition_audit",
        "interpretation": (
            "A group PASS establishes reproducible m=9 fixed-budget lock "
            "under the precommitted architecture, optimizer, matched weak "
            "scale, and 2500-epoch budget. It does NOT establish permanent "
            "non-convergence. The next authorized question is whether/when "
            "the locked runs escape under a longer fixed horizon."
        ),
    }
    write_json(out_dir / "decision.json", decision)

    # Aggregate figures.
    plot_by_seed(
        all_track,
        key="relative_l2_error",
        ylabel="Relative L2 error",
        title="m=9 fixed-budget error trajectories",
        path=out_dir / "error_trajectories.png",
        log_y=True,
        hline=convergence_threshold,
    )

    plot_by_seed(
        all_track,
        key="target_mode_residual_energy_share",
        ylabel="Target-mode residual energy share",
        title="m=9 residual localization across seeds",
        path=out_dir / "target_mode_share_trajectories.png",
        hline=final_target_share_threshold,
    )

    plot_by_seed(
        all_diag,
        key="target_vs_rest_cosine",
        ylabel="cos(g_target, g_rest)",
        title="Direct opposition: target gradient vs all other tests",
        path=out_dir / "target_vs_rest_cosine.png",
        hline=target_rest_cosine_threshold,
    )

    plot_by_seed(
        all_diag,
        key="target_rest_norm_ratio",
        ylabel="||g_rest|| / ||g_target||",
        title="Target/rest gradient magnitude balance",
        path=out_dir / "target_rest_norm_ratio.png",
        hline=1.0,
    )

    plot_by_seed(
        all_diag,
        key="target_rest_cancellation_ratio",
        ylabel="||g_target+g_rest|| / (||g_target||+||g_rest||)",
        title="Direct target-vs-rest cancellation",
        path=out_dir / "target_rest_cancellation_ratio.png",
        log_y=True,
        hline=target_rest_cancellation_threshold,
    )

    elapsed = time.perf_counter() - global_start

    lines: List[str] = []
    lines.append("=" * 154)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 4 FIXED-BUDGET EDGE-MODE REPLICATION SUMMARY"
    )
    lines.append("=" * 154)
    lines.append(
        "seed | final relL2 | err/theory | target share | "
        "cos(target,rest) | rest/target | cancel ratio | edge lock"
    )
    lines.append("-" * 154)

    for r in summaries:
        lines.append(
            f"{int(r['seed']):4d} | "
            f"{r['final_relative_l2_error']:11.4e} | "
            f"{r['final_error_over_low_only_theory']:10.4f} | "
            f"{r['final_target_mode_energy_share']:12.6f} | "
            f"{r['final_target_vs_rest_cosine']:16.8f} | "
            f"{r['final_rest_over_target_gradient_norm']:11.6f} | "
            f"{r['final_target_rest_cancellation_ratio']:12.4e} | "
            f"{'PASS' if r['edge_lock_pass'] else 'FAIL'}"
        )

    lines.append("-" * 154)
    lines.append(
        f"edge-lock replication              : "
        f"{edge_lock_count}/{len(args.seeds)} "
        f"(required {min_group_pass}) -> "
        f"{'PASS' if group_pass else 'FAIL'}"
    )

    if anchor["anchor_found"]:
        lines.append(
            f"Stage-3 seed-0 reproduction        : "
            f"{'PASS' if anchor['pass'] else 'FAIL'}"
        )
        lines.append(
            f"  max diagnostic difference        : "
            f"{anchor['max_abs_diagnostic_difference']:.3e}"
        )
        lines.append(
            f"  max tracking difference          : "
            f"{anchor['max_abs_tracking_difference']:.3e}"
        )
    else:
        lines.append(
            "Stage-3 seed-0 reproduction        : NOT CHECKED "
            "(anchor directory not found)"
        )

    lines.append(
        f"long-horizon escape-time route     : "
        f"{'AUTHORIZED' if route_long_horizon else 'NOT AUTHORIZED'}"
    )
    lines.append(f"elapsed seconds                    : {elapsed:.2f}")
    lines.append("=" * 154)
    lines.append(
        "Guardrail: fixed-budget lock is not permanent non-convergence. "
        "Only a long-horizon experiment can address escape time."
    )
    lines.append("=" * 154)

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    (out_dir / "console_summary.txt").write_text(
        summary_text, encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
