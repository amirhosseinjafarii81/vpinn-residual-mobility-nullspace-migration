#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 5
Bounded Long-Horizon Escape-Time Audit
======================================

Purpose
-------
Stage 4 established a reproducible fixed-budget m=9 lock across five seeds.
This stage asks the next logically necessary question:

    Does that locked state eventually escape under continued training,
    and when does the escape occur?

The design is intentionally bounded and efficient:
  * same m=9 problem and solver as Stages 3-4,
  * same five seeds {0,1,2,3,4},
  * fixed maximum horizon = 4000 epochs,
  * exact Stage-4 reproduction gate at epoch 2500,
  * post-2500 measurements every 25 epochs,
  * direct target-vs-rest geometry using only TWO autograd gradient calls,
    rather than reconstructing all 24 pairwise gradients,
  * each seed stops only after a precommitted escape event is confirmed,
  * all seeds are unconditional,
  * locked state at epoch 2500 is saved for later causal-intervention tests.

Important
---------
Event-based stopping here is NOT an optimization early-stop criterion.
The scientific estimand is an escape time. Once a precommitted event has
been observed and confirmed, continuing that seed adds no information to
that estimand.

Model problem
-------------
    -u'' = f on (0,1), u(0)=u(1)=0

Target mode
-----------
    m = 9
    a_9 = 0.15 * 7 / 9

so the matched weak scale remains
    a_9 * 9 = 1.05.

Certified escape event
----------------------
At tracking interval Δ=25, define Q(t) as:

    relative_L2(t) <= 1e-2
    AND
    target_residual_energy_share(t) <= 0.20.

A certified escape onset is the earliest t for which Q(t), Q(t+25),
and Q(t+50) are all true.

Thus the event requires THREE consecutive qualifying observations.
The run for that seed stops at the confirmation epoch t+50.

Additional temporal markers
---------------------------
Residual-release onset:
    first post-2500 tracked epoch with target share < 0.80.

Cancellation-break onset:
    first post-2500 diagnostic epoch for which the Stage-4 lock geometry
    ceases to hold, i.e. NOT all of:
        cos(g_target, g_rest) <= -0.95
        0.5 <= ||g_rest||/||g_target|| <= 2.0
        cancellation_ratio <= 0.20

where
    cancellation_ratio =
        ||g_target + g_rest|| / (||g_target|| + ||g_rest||).

Decision routes
---------------
A) 5/5 certified escapes by epoch 4000:
      authorize causal intervention from the saved epoch-2500 locked states.

B) 4/5 certified escapes:
      authorize a bounded tail extension ONLY for censored seeds.

C) <=3/5 certified escapes:
      route to initialization-heterogeneity analysis.

This stage does not claim permanent convergence or non-convergence.
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
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-5 bounded long-horizon VPINN escape-time audit."
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--max-epoch", type=int, default=4000)
    p.add_argument("--track-interval", type=int, default=25)

    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n-test", type=int, default=24)
    p.add_argument("--n-quad", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=4001)

    p.add_argument(
        "--stage3-script",
        type=str,
        default="vpinn_gradient_conflict_stage3_frequency_transfer.py",
    )
    p.add_argument(
        "--stage4-dir",
        type=str,
        default="vpinn_gradient_conflict_stage4_edge_mode_replication",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="vpinn_gradient_conflict_stage5_escape_time",
    )

    args = p.parse_args()

    if len(args.seeds) < 2:
        raise ValueError("At least two seeds are required.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")
    if args.max_epoch <= 2500:
        raise ValueError("--max-epoch must be > 2500.")
    if args.track_interval <= 0:
        raise ValueError("--track-interval must be positive.")
    if 2500 % args.track_interval != 0:
        raise ValueError(
            "--track-interval must divide 2500 so the anchor epoch is tracked."
        )
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.n_test < 9:
        raise ValueError("--n-test must be >= 9.")
    if args.n_quad < max(32, 2 * args.n_test):
        raise ValueError("--n-quad is too small.")

    return args


# =============================================================================
# Generic utilities
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

    spec = importlib.util.spec_from_file_location("vpinn_stage3_solver", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Stage-3 solver: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def flatten_gradients(grads) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


# =============================================================================
# Preflight: Stage-4 authorization + solver identity
# =============================================================================

def stage4_preflight(stage4_dir: Path, stage3_script: Path, seeds: List[int]) -> dict:
    decision_path = stage4_dir / "decision.json"
    manifest_path = stage4_dir / "manifest.json"
    track_path = stage4_dir / "aggregate_tracking_metrics.csv"
    diag_path = stage4_dir / "aggregate_diagnostic_metrics.csv"

    required = [decision_path, manifest_path, track_path, diag_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Stage-4 prerequisite files are missing:\n  " + "\n  ".join(missing)
        )

    decision = read_json(decision_path)
    manifest = read_json(manifest_path)

    if not bool(decision.get("route_long_horizon_escape_time_audit", False)):
        raise RuntimeError(
            "Stage 4 did not authorize the long-horizon escape-time route."
        )

    if int(decision.get("target_mode", -1)) != 9:
        raise RuntimeError("Stage-4 target mode is not m=9.")

    stage4_seeds = [int(s) for s in decision.get("seeds", [])]
    if stage4_seeds != list(seeds):
        raise RuntimeError(
            f"Seed mismatch: Stage 4 used {stage4_seeds}, Stage 5 requested {seeds}."
        )

    expected_sha = manifest.get("stage3_solver_sha256")
    actual_sha = sha256_file(stage3_script)

    if expected_sha != actual_sha:
        raise RuntimeError(
            "Stage-3 solver SHA256 mismatch.\n"
            f"Stage-4 solver: {expected_sha}\n"
            f"Current solver : {actual_sha}\n"
            "Refusing to continue with implementation drift."
        )

    return {
        "decision": decision,
        "manifest": manifest,
        "stage4_tracking": read_csv(track_path),
        "stage4_diagnostics": read_csv(diag_path),
        "stage3_sha256": actual_sha,
    }


# =============================================================================
# Efficient exact measurement
# =============================================================================

def measure(
    exp,
    target_mode: int,
    need_geometry: bool,
) -> Tuple[dict, torch.Tensor]:
    """
    One weak-residual forward/autograd construction per measured epoch.

    If need_geometry=True, compute exactly two parameter gradients:
      g_target = grad(R_target^2)
      g_rest   = grad(sum_{k != target} R_k^2)

    This is substantially cheaper than the 24-gradient pairwise diagnostic
    and directly measures the geometry needed in Stage 5.
    """
    residuals = exp.weak_residuals()
    residuals_det = residuals.detach()

    energy = residuals_det.square()
    total_energy = energy.sum().clamp_min(1.0e-300)

    t = target_mode - 1
    dominant_idx = int(torch.argmax(energy).item())

    metrics = {
        "vpinn_loss": float(torch.mean(energy).item()),
        "residual_l2_norm": float(torch.linalg.vector_norm(residuals_det).item()),
        "dominant_residual_mode": dominant_idx + 1,
        "dominant_residual_energy_share": float(
            (energy[dominant_idx] / total_energy).item()
        ),
        "target_mode_residual_energy_share": float(
            (energy[t] / total_energy).item()
        ),
        "target_mode_abs_residual": float(torch.abs(residuals_det[t]).item()),
    }

    if need_geometry:
        params = tuple(p for p in exp.model.parameters() if p.requires_grad)

        target_loss = residuals[t].square()
        rest_loss = residuals.square().sum() - target_loss

        g_target_parts = torch.autograd.grad(
            target_loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        g_rest_parts = torch.autograd.grad(
            rest_loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

        g_target = flatten_gradients(g_target_parts).detach()
        g_rest = flatten_gradients(g_rest_parts).detach()

        nt = torch.linalg.vector_norm(g_target)
        nr = torch.linalg.vector_norm(g_rest)

        eps = torch.tensor(
            1.0e-300,
            dtype=nt.dtype,
            device=nt.device,
        )

        dot = torch.dot(g_target, g_rest)

        cosine = dot / torch.clamp(nt * nr, min=eps)
        cosine = torch.clamp(cosine, -1.0, 1.0)

        net = torch.linalg.vector_norm(g_target + g_rest)
        cancellation = net / torch.clamp(nt + nr, min=eps)
        norm_ratio = nr / torch.clamp(nt, min=eps)

        metrics.update(
            {
                "target_gradient_norm": float(nt.item()),
                "rest_gradient_norm": float(nr.item()),
                "target_rest_norm_ratio": float(norm_ratio.item()),
                "target_vs_rest_cosine": float(cosine.item()),
                "target_rest_net_gradient_norm": float(net.item()),
                "target_rest_cancellation_ratio": float(cancellation.item()),
            }
        )
    else:
        metrics.update(
            {
                "target_gradient_norm": float("nan"),
                "rest_gradient_norm": float("nan"),
                "target_rest_norm_ratio": float("nan"),
                "target_vs_rest_cosine": float("nan"),
                "target_rest_net_gradient_norm": float("nan"),
                "target_rest_cancellation_ratio": float("nan"),
            }
        )

    return metrics, residuals


def optimizer_step_from_residuals(exp, residuals: torch.Tensor) -> float:
    """
    Reuse the already-constructed graph at measured epochs.
    This avoids a second weak-residual evaluation at those epochs.
    """
    exp.optimizer.zero_grad(set_to_none=True)
    loss = torch.mean(residuals.square())

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {loss.item()}")

    loss.backward()
    exp.optimizer.step()

    return float(loss.detach().item())


# =============================================================================
# Exact Stage-4 anchor verification at epoch 2500
# =============================================================================

def build_anchor_maps(preflight: dict):
    track_map = {}
    for row in preflight["stage4_tracking"]:
        key = (int(row["seed"]), int(row["epoch"]))
        track_map[key] = row

    diag_map = {}
    for row in preflight["stage4_diagnostics"]:
        key = (int(row["seed"]), int(row["epoch"]))
        diag_map[key] = row

    return track_map, diag_map


def verify_anchor(
    seed: int,
    current: dict,
    track_map: dict,
    diag_map: dict,
    tolerance: float = 1.0e-10,
) -> dict:
    key = (seed, 2500)

    if key not in track_map or key not in diag_map:
        raise RuntimeError(f"Stage-4 epoch-2500 anchor missing for seed {seed}.")

    old_t = track_map[key]
    old_d = diag_map[key]

    comparisons = {
        "relative_l2_error": (
            float(old_t["relative_l2_error"]),
            float(current["relative_l2_error"]),
        ),
        "vpinn_loss": (
            float(old_t["vpinn_loss"]),
            float(current["vpinn_loss"]),
        ),
        "target_mode_residual_energy_share": (
            float(old_t["target_mode_residual_energy_share"]),
            float(current["target_mode_residual_energy_share"]),
        ),
        "target_mode_abs_residual": (
            float(old_t["target_mode_abs_residual"]),
            float(current["target_mode_abs_residual"]),
        ),
        "target_vs_rest_cosine": (
            float(old_d["target_vs_rest_cosine"]),
            float(current["target_vs_rest_cosine"]),
        ),
        "target_rest_norm_ratio": (
            float(old_d["target_rest_norm_ratio"]),
            float(current["target_rest_norm_ratio"]),
        ),
        "target_rest_cancellation_ratio": (
            float(old_d["target_rest_cancellation_ratio"]),
            float(current["target_rest_cancellation_ratio"]),
        ),
    }

    diffs = {
        field: abs(old - new)
        for field, (old, new) in comparisons.items()
    }

    max_diff = max(diffs.values())

    return {
        "seed": seed,
        "tolerance": tolerance,
        "max_abs_difference": max_diff,
        "field_abs_differences": diffs,
        "pass": bool(max_diff <= tolerance),
    }


# =============================================================================
# State checkpoint
# =============================================================================

def save_locked_state(exp, path: Path, seed: int, epoch: int) -> None:
    torch.save(
        {
            "seed": seed,
            "epoch": epoch,
            "model_state_dict": exp.model.state_dict(),
            "optimizer_state_dict": exp.optimizer.state_dict(),
            "mode": exp.mode,
            "amplitude": exp.amplitude,
            "gram_error": exp.gram_error,
        },
        path,
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_seed_trajectories(
    rows: List[dict],
    key: str,
    ylabel: str,
    title: str,
    output: Path,
    log_y: bool = False,
    hline: float | None = None,
) -> None:
    seeds = sorted(set(int(r["seed"]) for r in rows))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for seed in seeds:
        rr = [r for r in rows if int(r["seed"]) == seed]
        rr.sort(key=lambda r: int(r["epoch"]))

        ax.plot(
            [int(r["epoch"]) for r in rr],
            [float(r[key]) for r in rr],
            label=f"seed {seed}",
        )

    if log_y:
        positive = [
            float(r[key])
            for r in rows
            if np.isfinite(float(r[key])) and float(r[key]) > 0
        ]
        if positive:
            ax.set_yscale("log")

    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
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

    stage4_dir = Path(args.stage4_dir)
    if not stage4_dir.is_absolute():
        stage4_dir = script_dir / stage4_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    preflight = stage4_preflight(
        stage4_dir=stage4_dir,
        stage3_script=stage3_script,
        seeds=args.seeds,
    )

    stage3 = load_stage3_module(stage3_script)
    track_anchor_map, diag_anchor_map = build_anchor_maps(preflight)

    # Fixed scientific design.
    target_mode = 9
    reference_mode = 7
    reference_amplitude = 0.15
    target_amplitude = reference_amplitude * reference_mode / target_mode

    locked_epoch = 2500

    escape_error_threshold = 1.0e-2
    escape_target_share_threshold = 0.20
    required_consecutive = 3

    release_target_share_threshold = 0.80

    lock_cosine_threshold = -0.95
    lock_norm_ratio_lower = 0.50
    lock_norm_ratio_upper = 2.00
    lock_cancellation_threshold = 0.20

    precommitment = {
        "stage": "bounded_long_horizon_escape_time_audit",
        "target_mode": target_mode,
        "target_amplitude": target_amplitude,
        "matched_weak_scale_a_times_m": target_amplitude * target_mode,
        "seeds": list(args.seeds),
        "locked_anchor_epoch": locked_epoch,
        "maximum_epoch": args.max_epoch,
        "tracking_interval": args.track_interval,
        "all_seeds_unconditional": True,
        "event_based_stop_only_after_certified_escape": True,
        "certified_escape": {
            "relative_l2_error_le": escape_error_threshold,
            "target_residual_energy_share_le":
                escape_target_share_threshold,
            "required_consecutive_tracking_observations":
                required_consecutive,
        },
        "residual_release_onset": {
            "target_residual_energy_share_lt":
                release_target_share_threshold,
        },
        "cancellation_lock_geometry": {
            "target_vs_rest_cosine_le": lock_cosine_threshold,
            "rest_over_target_norm_in": [
                lock_norm_ratio_lower,
                lock_norm_ratio_upper,
            ],
            "cancellation_ratio_le": lock_cancellation_threshold,
        },
        "decision_routes": {
            "5_of_5_escape":
                "causal_intervention_from_epoch2500_locked_states",
            "4_of_5_escape":
                "bounded_tail_extension_for_censored_seeds_only",
            "0_to_3_of_5_escape":
                "initialization_heterogeneity_audit",
        },
    }

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_resolved": str(device),
        "stage3_solver_path": str(stage3_script),
        "stage3_solver_sha256": preflight["stage3_sha256"],
        "stage4_dir": str(stage4_dir),
        "stage4_decision_authorized": True,
        "stage5_script_sha256": sha256_file(Path(__file__).resolve()),
        "precommitment": precommitment,
    }
    write_json(output_dir / "manifest.json", manifest)

    print("=" * 110)
    print("VPINN GRADIENT GEOMETRY — STAGE 5 BOUNDED LONG-HORIZON ESCAPE-TIME AUDIT")
    print("=" * 110)
    print(f"device                         : {device}")
    print(f"target mode                    : {target_mode}")
    print(f"target amplitude               : {target_amplitude:.12g}")
    print(f"matched a_m*m                  : {target_amplitude*target_mode:.12g}")
    print(f"seeds                          : {args.seeds}")
    print(f"Stage-4 locked anchor          : epoch {locked_epoch}")
    print(f"maximum epoch                  : {args.max_epoch}")
    print(f"tracking interval              : {args.track_interval}")
    print(
        "certified escape              : "
        f"relL2 <= {escape_error_threshold:g} AND "
        f"target share <= {escape_target_share_threshold:g}, "
        f"{required_consecutive} consecutive observations"
    )
    print(f"Stage-3 solver SHA256          : {preflight['stage3_sha256']}")
    print("=" * 110)

    all_postlock_rows: List[dict] = []
    seed_summaries: List[dict] = []
    anchor_results: List[dict] = []

    global_start = time.perf_counter()

    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        cfg = stage3.Config(
            seed=seed,
            device=args.device,
            epochs=args.max_epoch,
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
            diagnostic_epochs=(locked_epoch,),
            output_dir=str(seed_dir),
        )

        exp = stage3.ModeExperiment(
            cfg=cfg,
            device=device,
            mode=target_mode,
            out_dir=seed_dir,
        )

        print()
        print("-" * 110)
        print(f"SEED {seed}: reproducing Stage-4 trajectory to epoch {locked_epoch} ...")

        # -------------------------------------------------------------
        # Reproduce exactly to locked epoch with minimal measurement
        # overhead. No diagnostic sweep is repeated.
        # -------------------------------------------------------------
        for epoch in range(locked_epoch):
            exp.train_step()

        # Exact anchor measurement at epoch 2500.
        anchor_metrics, anchor_residuals = measure(
            exp=exp,
            target_mode=target_mode,
            need_geometry=True,
        )
        anchor_metrics["relative_l2_error"] = exp.relative_l2_error()

        anchor_check = verify_anchor(
            seed=seed,
            current=anchor_metrics,
            track_map=track_anchor_map,
            diag_map=diag_anchor_map,
            tolerance=1.0e-10,
        )

        anchor_results.append(anchor_check)

        if not anchor_check["pass"]:
            write_json(
                seed_dir / "anchor_failure.json",
                anchor_check,
            )
            raise RuntimeError(
                f"Stage-4 reproduction FAILED for seed {seed}: "
                f"max abs difference={anchor_check['max_abs_difference']:.3e}"
            )

        print(
            f"  anchor reproduction PASS | "
            f"max abs difference={anchor_check['max_abs_difference']:.3e}"
        )

        # Save the exact locked state. This is intentionally before any
        # post-2500 update and is the branching point for a future causal test.
        save_locked_state(
            exp,
            seed_dir / "locked_state_epoch_2500.pt",
            seed=seed,
            epoch=locked_epoch,
        )

        # Record epoch-2500 state as the first long-horizon row.
        current = {
            "seed": seed,
            "epoch": locked_epoch,
            "relative_l2_error": anchor_metrics["relative_l2_error"],
            **{
                k: v
                for k, v in anchor_metrics.items()
                if k != "relative_l2_error"
            },
        }
        all_postlock_rows.append(current)
        seed_rows = [current]

        # Stage-4 lock is guaranteed by preflight, but compute the same
        # geometry predicate explicitly from the reproduced anchor.
        def geometry_is_locked(row: dict) -> bool:
            return bool(
                row["target_vs_rest_cosine"] <= lock_cosine_threshold
                and lock_norm_ratio_lower
                    <= row["target_rest_norm_ratio"]
                    <= lock_norm_ratio_upper
                and row["target_rest_cancellation_ratio"]
                    <= lock_cancellation_threshold
            )

        if not geometry_is_locked(current):
            raise RuntimeError(
                f"Seed {seed} reproduced numerically but no longer satisfies "
                "the precommitted Stage-4 lock geometry."
            )

        release_onset_epoch = -1
        cancellation_break_epoch = -1

        consecutive_escape = 0
        candidate_escape_onset = -1
        certified_escape_onset = -1
        certified_escape_confirmation = -1

        # Continue training from epoch 2500.
        # anchor_residuals is the graph at the current epoch; reuse it for
        # the first update to avoid an extra residual construction.
        optimizer_step_from_residuals(exp, anchor_residuals)

        for epoch in range(locked_epoch + 1, args.max_epoch + 1):
            is_track = (epoch % args.track_interval == 0)

            if is_track:
                # Direct target-vs-rest geometry at every post-lock tracking
                # point. Only two gradient calls are needed.
                metrics, residuals = measure(
                    exp=exp,
                    target_mode=target_mode,
                    need_geometry=True,
                )
                rel_error = exp.relative_l2_error()

                row = {
                    "seed": seed,
                    "epoch": epoch,
                    "relative_l2_error": rel_error,
                    **metrics,
                }

                seed_rows.append(row)
                all_postlock_rows.append(row)

                # Residual-release onset.
                if (
                    release_onset_epoch < 0
                    and metrics["target_mode_residual_energy_share"]
                        < release_target_share_threshold
                ):
                    release_onset_epoch = epoch

                # Cancellation-lock break.
                if (
                    cancellation_break_epoch < 0
                    and not geometry_is_locked(row)
                ):
                    cancellation_break_epoch = epoch

                # Three-consecutive-observation certified escape.
                qualifies = bool(
                    rel_error <= escape_error_threshold
                    and metrics["target_mode_residual_energy_share"]
                        <= escape_target_share_threshold
                )

                if qualifies:
                    if consecutive_escape == 0:
                        candidate_escape_onset = epoch
                    consecutive_escape += 1
                else:
                    consecutive_escape = 0
                    candidate_escape_onset = -1

                if consecutive_escape >= required_consecutive:
                    certified_escape_onset = candidate_escape_onset
                    certified_escape_confirmation = epoch

                    print(
                        f"  CERTIFIED ESCAPE | onset={certified_escape_onset} "
                        f"| confirmed={certified_escape_confirmation} "
                        f"| relL2={rel_error:.6e} "
                        f"| target_share="
                        f"{metrics['target_mode_residual_energy_share']:.6f}"
                    )
                    break

                # Reuse graph for the update at tracked epochs.
                if epoch < args.max_epoch:
                    optimizer_step_from_residuals(exp, residuals)

            else:
                exp.train_step()

        write_csv(seed_dir / "postlock_metrics.csv", seed_rows)

        final_row = seed_rows[-1]
        escaped = certified_escape_onset >= 0
        censored = not escaped

        summary = {
            "seed": seed,
            "anchor_reproduction_pass": True,
            "anchor_max_abs_difference":
                anchor_check["max_abs_difference"],
            "release_onset_epoch":
                release_onset_epoch,
            "cancellation_break_epoch":
                cancellation_break_epoch,
            "certified_escape_onset_epoch":
                certified_escape_onset,
            "certified_escape_confirmation_epoch":
                certified_escape_confirmation,
            "censored_at_max_epoch":
                censored,
            "observed_final_epoch":
                int(final_row["epoch"]),
            "final_relative_l2_error":
                float(final_row["relative_l2_error"]),
            "final_target_mode_energy_share":
                float(final_row["target_mode_residual_energy_share"]),
            "final_target_vs_rest_cosine":
                float(final_row["target_vs_rest_cosine"]),
            "final_rest_over_target_norm":
                float(final_row["target_rest_norm_ratio"]),
            "final_cancellation_ratio":
                float(final_row["target_rest_cancellation_ratio"]),
            "gap_cancellation_break_to_release":
                (
                    release_onset_epoch - cancellation_break_epoch
                    if (
                        release_onset_epoch >= 0
                        and cancellation_break_epoch >= 0
                    )
                    else -1
                ),
            "gap_release_to_escape_onset":
                (
                    certified_escape_onset - release_onset_epoch
                    if (
                        certified_escape_onset >= 0
                        and release_onset_epoch >= 0
                    )
                    else -1
                ),
        }
        seed_summaries.append(summary)

        write_json(seed_dir / "summary.json", summary)

        if censored:
            print(
                f"  CENSORED at epoch {args.max_epoch} | "
                f"relL2={final_row['relative_l2_error']:.6e} | "
                f"target_share="
                f"{final_row['target_mode_residual_energy_share']:.6f}"
            )

    write_csv(output_dir / "aggregate_postlock_metrics.csv", all_postlock_rows)
    write_csv(output_dir / "escape_time_summary.csv", seed_summaries)
    write_json(output_dir / "anchor_reproduction.json", {
        "all_pass": all(r["pass"] for r in anchor_results),
        "results": anchor_results,
    })

    escaped_count = sum(
        int(r["certified_escape_onset_epoch"] >= 0)
        for r in seed_summaries
    )
    censored_seeds = [
        int(r["seed"])
        for r in seed_summaries
        if bool(r["censored_at_max_epoch"])
    ]

    cancellation_precedes_release_count = sum(
        int(
            r["cancellation_break_epoch"] >= 0
            and r["release_onset_epoch"] >= 0
            and r["cancellation_break_epoch"]
                <= r["release_onset_epoch"]
        )
        for r in seed_summaries
    )

    if escaped_count == len(args.seeds):
        route = "causal_intervention_from_epoch2500_locked_states"
        route_authorized = True
    elif escaped_count >= len(args.seeds) - 1:
        route = "bounded_tail_extension_for_censored_seeds_only"
        route_authorized = True
    else:
        route = "initialization_heterogeneity_audit"
        route_authorized = True

    escape_epochs = [
        int(r["certified_escape_onset_epoch"])
        for r in seed_summaries
        if int(r["certified_escape_onset_epoch"]) >= 0
    ]

    decision = {
        "n_seeds": len(args.seeds),
        "seeds": list(args.seeds),
        "maximum_epoch": args.max_epoch,
        "certified_escape_count": escaped_count,
        "censored_seeds": censored_seeds,
        "all_stage4_anchors_reproduced":
            all(r["pass"] for r in anchor_results),
        "escape_onset_epochs": escape_epochs,
        "escape_onset_min":
            min(escape_epochs) if escape_epochs else None,
        "escape_onset_median":
            float(np.median(escape_epochs)) if escape_epochs else None,
        "escape_onset_max":
            max(escape_epochs) if escape_epochs else None,
        "cancellation_break_precedes_or_equals_release_count":
            cancellation_precedes_release_count,
        "next_route": route,
        "next_route_authorized": route_authorized,
        "interpretation_guardrail": (
            "Certified escape demonstrates finite-time departure from the "
            "fixed-budget locked state under continued Adam training. "
            "Temporal ordering is descriptive and does not by itself prove "
            "that gradient cancellation causes the delay. A causal "
            "intervention must branch from identical saved locked states."
        ),
    }
    write_json(output_dir / "decision.json", decision)

    # -------------------------------------------------------------------------
    # Aggregate plots: post-lock only, because Stage 4 already certified the
    # pre-2500 regime. This keeps the visual question focused.
    # -------------------------------------------------------------------------
    plot_seed_trajectories(
        all_postlock_rows,
        key="relative_l2_error",
        ylabel="Relative L2 error",
        title="m=9 escape from the fixed-budget VPINN plateau",
        output=output_dir / "postlock_error_trajectories.png",
        log_y=True,
        hline=escape_error_threshold,
    )

    plot_seed_trajectories(
        all_postlock_rows,
        key="target_mode_residual_energy_share",
        ylabel="Target-mode residual energy share",
        title="Release of the unresolved m=9 weak residual",
        output=output_dir / "postlock_target_share.png",
        hline=release_target_share_threshold,
    )

    plot_seed_trajectories(
        all_postlock_rows,
        key="target_vs_rest_cosine",
        ylabel="cos(g_target, g_rest)",
        title="Target-vs-rest gradient opposition through escape",
        output=output_dir / "postlock_target_vs_rest_cosine.png",
        hline=lock_cosine_threshold,
    )

    plot_seed_trajectories(
        all_postlock_rows,
        key="target_rest_cancellation_ratio",
        ylabel="Cancellation ratio",
        title="Breakdown of target-vs-rest gradient cancellation",
        output=output_dir / "postlock_cancellation_ratio.png",
        log_y=True,
        hline=lock_cancellation_threshold,
    )

    elapsed = time.perf_counter() - global_start

    lines: List[str] = []
    lines.append("=" * 150)
    lines.append(
        "VPINN GRADIENT GEOMETRY — STAGE 5 BOUNDED LONG-HORIZON ESCAPE-TIME SUMMARY"
    )
    lines.append("=" * 150)
    lines.append(
        "seed | release | cancel break | escape onset | confirm | "
        "final epoch | final relL2 | final target share | censored"
    )
    lines.append("-" * 150)

    for r in seed_summaries:
        lines.append(
            f"{int(r['seed']):4d} | "
            f"{int(r['release_onset_epoch']):7d} | "
            f"{int(r['cancellation_break_epoch']):12d} | "
            f"{int(r['certified_escape_onset_epoch']):12d} | "
            f"{int(r['certified_escape_confirmation_epoch']):7d} | "
            f"{int(r['observed_final_epoch']):11d} | "
            f"{r['final_relative_l2_error']:11.4e} | "
            f"{r['final_target_mode_energy_share']:18.6f} | "
            f"{str(bool(r['censored_at_max_epoch'])):8s}"
        )

    lines.append("-" * 150)
    lines.append(
        f"Stage-4 anchor reproduction       : "
        f"{'PASS' if decision['all_stage4_anchors_reproduced'] else 'FAIL'}"
    )
    lines.append(
        f"certified escapes                 : "
        f"{escaped_count}/{len(args.seeds)}"
    )

    if escape_epochs:
        lines.append(
            f"escape onset range                : "
            f"{min(escape_epochs)} .. {max(escape_epochs)}"
        )
        lines.append(
            f"escape onset median               : "
            f"{np.median(escape_epochs):.1f}"
        )

    lines.append(
        f"cancel-break <= residual-release  : "
        f"{cancellation_precedes_release_count}/{len(args.seeds)}"
    )
    lines.append(
        f"next route                        : {route}"
    )
    lines.append(
        f"elapsed seconds                   : {elapsed:.2f}"
    )
    lines.append("=" * 150)
    lines.append(
        "Guardrail: temporal precedence is not causality. "
        "The next causal stage must branch from identical epoch-2500 "
        "locked checkpoints."
    )
    lines.append("=" * 150)

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    (output_dir / "console_summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
