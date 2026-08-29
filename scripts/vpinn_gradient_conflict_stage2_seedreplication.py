#!/usr/bin/env python3
"""
VPINN Test-Function Gradient Conflict — Stage 2
================================================

Purpose
-------
Replicate the Stage-1 VPINN gradient-conflict observation across a small,
precommitted set of independent random seeds while preserving the Stage-1
scientific setup exactly.

This script is intentionally a *driver + auditor*, not a rewritten solver.
It executes the existing Stage-1 script as a subprocess for each seed using
this same Python interpreter. That design avoids silent divergence between
Stage 1 and Stage 2 and gives every seed the exact same model, PDE, test
space, quadrature, diagnostics, and checkpoint logic.

Default replication set
-----------------------
    seeds = {0, 1, 2, 3, 4}

Precommitted Stage-2 gates
--------------------------
The Stage-2 output reports three separate reproducibility questions:

1) Conflict signature
   For a seed, before numerical convergence:
       min Gamma <= 0.20
       max weighted_negative_cosine >= 0.50

2) Spectral localization
   The known high-frequency target mode is k = 7 because
       u*(x) = sin(pi x) + 0.15 sin(7 pi x).
   A seed passes localization if, before convergence, the maximum fraction
   of weak-residual energy carried by R_7 is at least 0.80:
       max R_7^2 / sum_k R_k^2 >= 0.80

3) Eventual convergence
   Final relative L2 error <= 1e-2.

A group-level gate passes when at least 4 out of 5 seeds satisfy it.
These thresholds are fixed in this script *before* the Stage-2 runs.

Why this stage is cheap
-----------------------
Stage 1 took roughly tens of seconds on CPU. Stage 2 repeats the exact run
only five times and performs lightweight aggregation. No architecture sweep,
no 2D PDE, no adaptive test-space search, and no brute-force hyperparameter
scan are introduced here.

Outputs
-------
vpinn_gradient_conflict_stage2_seedreplication/
    manifest.json
    aggregate_checkpoint_metrics.csv
    seed_summary.csv
    decision.json
    console_summary.txt
    error_trajectories.png
    gradient_coherence_trajectories.png
    r7_energy_share_trajectories.png
    weighted_negative_cosine_trajectories.png
    runs/
        seed_000/
        seed_001/
        ...
    logs/
        seed_000.log
        ...

Typical execution
-----------------
From the directory containing both Stage-1 and Stage-2 scripts:

    ../.venv/bin/python vpinn_gradient_conflict_stage2_seedreplication.py \
        --device cpu

For a completely fresh rerun, this is the recommended command. Existing
Stage-2 seed folders are rejected by default rather than silently reused.
Use --reuse-existing only when you explicitly want validated reuse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class Stage2Config:
    stage1_script: str = "vpinn_gradient_conflict_stage1.py"
    output_dir: str = "vpinn_gradient_conflict_stage2_seedreplication"

    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    device: str = "cpu"

    # Stage-1 scientific configuration: intentionally fixed for replication.
    epochs: int = 2500
    learning_rate: float = 1.0e-3
    width: int = 32
    depth: int = 3
    n_test: int = 24
    n_quad: int = 256
    n_eval: int = 4001

    # Exact-solution high-frequency mode.
    target_mode: int = 7

    # Precommitted scientific gates.
    convergence_error_threshold: float = 1.0e-2
    conflict_gamma_threshold: float = 0.20
    conflict_weighted_negative_threshold: float = 0.50
    target_mode_energy_share_threshold: float = 0.80
    required_seed_fraction: float = 0.80  # 4 / 5 by default.

    # Operational controls.
    reuse_existing: bool = False
    dpi: int = 220


def parse_args() -> Stage2Config:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 exact seed replication for the VPINN per-test "
            "gradient-conflict experiment."
        )
    )

    parser.add_argument(
        "--stage1-script",
        default="vpinn_gradient_conflict_stage1.py",
        help="Path to the Stage-1 Python script.",
    )
    parser.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage2_seedreplication",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Independent seeds. Default: 0 1 2 3 4",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Reuse a completed seed folder only after validating its metadata "
            "against the precommitted Stage-2 configuration."
        ),
    )

    args = parser.parse_args()

    seeds = tuple(args.seeds)
    if len(seeds) < 2:
        raise ValueError("Stage 2 requires at least two independent seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not allowed.")

    return Stage2Config(
        stage1_script=args.stage1_script,
        output_dir=args.output_dir,
        seeds=seeds,
        device=args.device,
        reuse_existing=args.reuse_existing,
    )


# =============================================================================
# Small I/O helpers
# =============================================================================


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Mapping) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return

    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value) -> float:
    return float(value)


def as_int(value) -> int:
    return int(float(value))


def finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


# =============================================================================
# Stage-1 contract validation
# =============================================================================


def expected_stage1_config(cfg: Stage2Config, seed: int) -> dict:
    # Match the Stage-1 metadata field names exactly.  The dtype is stored
    # as an execution invariant outside the config block in Stage 1, while
    # the learning-rate key is named ``lr``.
    return {
        "seed": seed,
        "device": cfg.device,
        "epochs": cfg.epochs,
        "lr": cfg.learning_rate,
        "width": cfg.width,
        "depth": cfg.depth,
        "n_test": cfg.n_test,
        "n_quad": cfg.n_quad,
        "n_eval": cfg.n_eval,
    }


def validate_stage1_run(
    run_dir: Path,
    cfg: Stage2Config,
    seed: int,
) -> Tuple[dict, List[dict]]:
    metadata_path = run_dir / "run_metadata.json"
    summary_path = run_dir / "run_summary.json"
    metrics_path = run_dir / "checkpoint_metrics.csv"

    for path in (metadata_path, summary_path, metrics_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required Stage-1 output: {path}")

    metadata = load_json(metadata_path)
    _ = load_json(summary_path)
    rows = read_csv_rows(metrics_path)

    if not rows:
        raise RuntimeError(f"No checkpoint metrics in {metrics_path}")

    actual = metadata.get("config", {})
    expected = expected_stage1_config(cfg, seed)

    mismatches = []
    for key, expected_value in expected.items():
        if key not in actual:
            mismatches.append(f"missing config field {key!r}")
            continue

        actual_value = actual[key]

        if isinstance(expected_value, float):
            if not math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                mismatches.append(
                    f"{key}: expected {expected_value}, got {actual_value}"
                )
        else:
            if actual_value != expected_value:
                mismatches.append(
                    f"{key}: expected {expected_value!r}, got {actual_value!r}"
                )

    gram_error = float(metadata.get("gram_max_abs_error", float("inf")))
    if not math.isfinite(gram_error) or gram_error > 1.0e-10:
        mismatches.append(f"Gram error too large: {gram_error:.3e}")

    final_epoch = max(as_int(row["epoch"]) for row in rows)
    if final_epoch != cfg.epochs:
        mismatches.append(
            f"final checkpoint epoch: expected {cfg.epochs}, got {final_epoch}"
        )

    required_npz = run_dir / f"checkpoint_{cfg.epochs:04d}.npz"
    if not required_npz.is_file():
        mismatches.append(f"missing final checkpoint array: {required_npz.name}")

    if mismatches:
        joined = "\n  - ".join(mismatches)
        raise RuntimeError(
            f"Stage-1 run validation failed for seed {seed}:\n  - {joined}"
        )

    return metadata, rows


# =============================================================================
# Stage-1 subprocess execution
# =============================================================================


def build_stage1_command(
    stage1_script: Path,
    run_dir: Path,
    cfg: Stage2Config,
    seed: int,
) -> List[str]:
    return [
        sys.executable,
        str(stage1_script),
        "--seed",
        str(seed),
        "--device",
        cfg.device,
        "--epochs",
        str(cfg.epochs),
        "--lr",
        repr(cfg.learning_rate),
        "--width",
        str(cfg.width),
        "--depth",
        str(cfg.depth),
        "--n-test",
        str(cfg.n_test),
        "--n-quad",
        str(cfg.n_quad),
        "--n-eval",
        str(cfg.n_eval),
        "--output-dir",
        str(run_dir),
    ]


def run_and_tee(command: Sequence[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Stage-1 subprocess failed with exit code {return_code}. "
            f"See log: {log_path}"
        )


# =============================================================================
# Per-seed scientific analysis
# =============================================================================


def checkpoint_residual_energy_share(
    run_dir: Path,
    epoch: int,
    target_mode: int,
) -> Tuple[int, float, float, float]:
    """
    Returns
    -------
    dominant_mode, dominant_share, target_share, residual_l2_norm
    """
    path = run_dir / f"checkpoint_{epoch:04d}.npz"
    if not path.is_file():
        raise RuntimeError(f"Missing checkpoint file: {path}")

    with np.load(path) as data:
        residuals = np.asarray(data["residuals"], dtype=float).reshape(-1)

    energy = residuals ** 2
    total = float(np.sum(energy))

    if not math.isfinite(total) or total <= 0.0:
        return -1, float("nan"), float("nan"), float("nan")

    dominant_zero = int(np.argmax(energy))
    dominant_mode = dominant_zero + 1
    dominant_share = float(energy[dominant_zero] / total)

    target_zero = target_mode - 1
    if target_zero < 0 or target_zero >= residuals.size:
        raise ValueError(
            f"target_mode={target_mode} is outside 1..{residuals.size}"
        )

    target_share = float(energy[target_zero] / total)
    residual_l2 = float(np.linalg.norm(residuals))

    return dominant_mode, dominant_share, target_share, residual_l2


def analyze_seed(
    run_dir: Path,
    cfg: Stage2Config,
    seed: int,
) -> Tuple[List[dict], dict]:
    _, raw_rows = validate_stage1_run(run_dir, cfg, seed)

    enriched: List[dict] = []

    for raw in raw_rows:
        epoch = as_int(raw["epoch"])
        rel_error = as_float(raw["relative_l2_error"])
        gamma = as_float(raw["gradient_coherence"])
        weighted_neg = as_float(raw["weighted_negative_cosine"])

        dominant_mode, dominant_share, target_share, residual_l2 = (
            checkpoint_residual_energy_share(
                run_dir,
                epoch,
                cfg.target_mode,
            )
        )

        row = {
            "seed": seed,
            "epoch": epoch,
            "vpinn_loss": as_float(raw["vpinn_loss"]),
            "relative_l2_error": rel_error,
            "active_tests": as_int(raw["active_tests"]),
            "conflict_fraction": as_float(raw["conflict_fraction"]),
            "mean_pairwise_cosine": as_float(raw["mean_pairwise_cosine"]),
            "minimum_pairwise_cosine": as_float(raw["minimum_pairwise_cosine"]),
            "gradient_coherence": gamma,
            "weighted_negative_cosine": weighted_neg,
            "worst_test_i": as_int(raw["worst_test_i"]),
            "worst_test_j": as_int(raw["worst_test_j"]),
            "residual_l2_norm": residual_l2,
            "dominant_residual_mode": dominant_mode,
            "dominant_residual_energy_share": dominant_share,
            "target_mode": cfg.target_mode,
            "target_mode_residual_energy_share": target_share,
            "preconvergence": int(
                rel_error > cfg.convergence_error_threshold
            ),
        }
        enriched.append(row)

    pre = [r for r in enriched if r["preconvergence"] == 1]
    if not pre:
        # If a seed is already converged by the first checkpoint, treating all
        # points as preconvergence would be scientifically wrong. Keep NaNs.
        min_gamma_pre = float("nan")
        max_weighted_neg_pre = float("nan")
        max_target_share_pre = float("nan")
        target_dominant_any_pre = False
        target_in_worst_pair_any_pre = False
        epoch_min_gamma_pre = -1
        epoch_max_target_share_pre = -1
    else:
        min_gamma_row = min(pre, key=lambda r: r["gradient_coherence"])
        target_share_row = max(
            pre,
            key=lambda r: r["target_mode_residual_energy_share"],
        )

        min_gamma_pre = float(min_gamma_row["gradient_coherence"])
        max_weighted_neg_pre = max(
            float(r["weighted_negative_cosine"]) for r in pre
        )
        max_target_share_pre = float(
            target_share_row["target_mode_residual_energy_share"]
        )
        target_dominant_any_pre = any(
            int(r["dominant_residual_mode"]) == cfg.target_mode for r in pre
        )
        target_in_worst_pair_any_pre = any(
            cfg.target_mode in (
                int(r["worst_test_i"]),
                int(r["worst_test_j"]),
            )
            for r in pre
        )
        epoch_min_gamma_pre = int(min_gamma_row["epoch"])
        epoch_max_target_share_pre = int(target_share_row["epoch"])

    final = max(enriched, key=lambda r: r["epoch"])

    conflict_signature = (
        math.isfinite(min_gamma_pre)
        and min_gamma_pre <= cfg.conflict_gamma_threshold
        and math.isfinite(max_weighted_neg_pre)
        and max_weighted_neg_pre >= cfg.conflict_weighted_negative_threshold
    )

    spectral_localization = (
        math.isfinite(max_target_share_pre)
        and max_target_share_pre >= cfg.target_mode_energy_share_threshold
    )

    eventual_convergence = (
        float(final["relative_l2_error"])
        <= cfg.convergence_error_threshold
    )

    summary = {
        "seed": seed,
        "final_relative_l2_error": float(final["relative_l2_error"]),
        "final_vpinn_loss": float(final["vpinn_loss"]),
        "final_gradient_coherence": float(final["gradient_coherence"]),
        "min_preconvergence_gamma": finite_or_none(min_gamma_pre),
        "epoch_min_preconvergence_gamma": epoch_min_gamma_pre,
        "max_preconvergence_weighted_negative_cosine": finite_or_none(
            max_weighted_neg_pre
        ),
        "max_preconvergence_target_mode_energy_share": finite_or_none(
            max_target_share_pre
        ),
        "epoch_max_target_mode_energy_share": epoch_max_target_share_pre,
        "target_mode_dominant_at_any_preconvergence_checkpoint": bool(
            target_dominant_any_pre
        ),
        "target_mode_in_worst_pair_at_any_preconvergence_checkpoint": bool(
            target_in_worst_pair_any_pre
        ),
        "conflict_signature_pass": bool(conflict_signature),
        "spectral_localization_pass": bool(spectral_localization),
        "eventual_convergence_pass": bool(eventual_convergence),
        "full_seed_story_pass": bool(
            conflict_signature and spectral_localization and eventual_convergence
        ),
    }

    return enriched, summary


# =============================================================================
# Aggregate decision logic
# =============================================================================


def count_pass(rows: Sequence[Mapping], key: str) -> int:
    return sum(bool(r[key]) for r in rows)


def aggregate_decision(
    cfg: Stage2Config,
    seed_summaries: Sequence[Mapping],
) -> dict:
    n = len(seed_summaries)
    required = math.ceil(cfg.required_seed_fraction * n - 1.0e-12)

    conflict_count = count_pass(seed_summaries, "conflict_signature_pass")
    spectral_count = count_pass(seed_summaries, "spectral_localization_pass")
    convergence_count = count_pass(seed_summaries, "eventual_convergence_pass")
    full_count = count_pass(seed_summaries, "full_seed_story_pass")

    finite_final_errors = [
        float(r["final_relative_l2_error"]) for r in seed_summaries
    ]

    finite_gammas = [
        float(r["min_preconvergence_gamma"])
        for r in seed_summaries
        if r["min_preconvergence_gamma"] is not None
    ]

    finite_target_shares = [
        float(r["max_preconvergence_target_mode_energy_share"])
        for r in seed_summaries
        if r["max_preconvergence_target_mode_energy_share"] is not None
    ]

    return {
        "n_seeds": n,
        "required_pass_count": required,
        "precommitted_required_fraction": cfg.required_seed_fraction,
        "conflict_signature_pass_count": conflict_count,
        "spectral_localization_pass_count": spectral_count,
        "eventual_convergence_pass_count": convergence_count,
        "full_seed_story_pass_count": full_count,
        "conflict_reproduction_group_pass": conflict_count >= required,
        "spectral_localization_group_pass": spectral_count >= required,
        "eventual_convergence_group_pass": convergence_count >= required,
        "full_story_group_pass": full_count >= required,
        "median_final_relative_l2_error": statistics.median(finite_final_errors),
        "median_min_preconvergence_gamma": (
            statistics.median(finite_gammas) if finite_gammas else None
        ),
        "median_max_preconvergence_target_mode_energy_share": (
            statistics.median(finite_target_shares)
            if finite_target_shares
            else None
        ),
        "scientific_interpretation": (
            "Stage 2 is a seed-replication gate only. Passing it supports "
            "reproducibility of the observed optimization geometry for this "
            "fixed 1D problem and architecture. It does not establish a "
            "general theorem about VPINNs."
        ),
    }


# =============================================================================
# Plotting
# =============================================================================


def grouped_by_seed(rows: Sequence[Mapping]) -> Dict[int, List[Mapping]]:
    groups: Dict[int, List[Mapping]] = {}
    for row in rows:
        groups.setdefault(int(row["seed"]), []).append(row)
    for seed in groups:
        groups[seed] = sorted(groups[seed], key=lambda r: int(r["epoch"]))
    return groups


def plot_trajectories(
    rows: Sequence[Mapping],
    field: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
    log_y: bool,
) -> None:
    groups = grouped_by_seed(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for seed, seed_rows in sorted(groups.items()):
        epochs = [int(r["epoch"]) for r in seed_rows]
        values = [float(r[field]) for r in seed_rows]
        ax.plot(epochs, values, marker="o", label=f"seed {seed}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if log_y:
        positive = [
            float(r[field])
            for r in rows
            if math.isfinite(float(r[field])) and float(r[field]) > 0.0
        ]
        if positive:
            ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Human-readable summary
# =============================================================================


def format_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def build_console_summary(
    cfg: Stage2Config,
    seed_summaries: Sequence[Mapping],
    decision: Mapping,
    elapsed: float,
) -> str:
    lines: List[str] = []
    lines.append("=" * 94)
    lines.append("VPINN GRADIENT-CONFLICT — STAGE 2 SEED-REPLICATION SUMMARY")
    lines.append("=" * 94)
    lines.append(
        "seed | final relL2 | min preconv Gamma | max preconv WNC | "
        "max R7 share | conflict | R7-local | converged | full"
    )
    lines.append("-" * 94)

    for row in seed_summaries:
        gamma = row["min_preconvergence_gamma"]
        wnc = row["max_preconvergence_weighted_negative_cosine"]
        share = row["max_preconvergence_target_mode_energy_share"]

        lines.append(
            f"{int(row['seed']):4d} | "
            f"{float(row['final_relative_l2_error']):11.4e} | "
            f"{(gamma if gamma is not None else float('nan')):17.4e} | "
            f"{(wnc if wnc is not None else float('nan')):15.4f} | "
            f"{(share if share is not None else float('nan')):12.4f} | "
            f"{format_bool(bool(row['conflict_signature_pass'])):8s} | "
            f"{format_bool(bool(row['spectral_localization_pass'])):8s} | "
            f"{format_bool(bool(row['eventual_convergence_pass'])):9s} | "
            f"{format_bool(bool(row['full_seed_story_pass'])):4s}"
        )

    lines.append("-" * 94)
    lines.append(
        f"required pass count                 : "
        f"{decision['required_pass_count']} / {decision['n_seeds']}"
    )
    lines.append(
        f"conflict reproduction              : "
        f"{decision['conflict_signature_pass_count']} / {decision['n_seeds']} "
        f"-> {format_bool(bool(decision['conflict_reproduction_group_pass']))}"
    )
    lines.append(
        f"spectral localization to v_{cfg.target_mode:<2d}        : "
        f"{decision['spectral_localization_pass_count']} / {decision['n_seeds']} "
        f"-> {format_bool(bool(decision['spectral_localization_group_pass']))}"
    )
    lines.append(
        f"eventual convergence               : "
        f"{decision['eventual_convergence_pass_count']} / {decision['n_seeds']} "
        f"-> {format_bool(bool(decision['eventual_convergence_group_pass']))}"
    )
    lines.append(
        f"full story                         : "
        f"{decision['full_seed_story_pass_count']} / {decision['n_seeds']} "
        f"-> {format_bool(bool(decision['full_story_group_pass']))}"
    )
    lines.append(
        f"median final relative L2 error     : "
        f"{decision['median_final_relative_l2_error']:.6e}"
    )

    med_gamma = decision["median_min_preconvergence_gamma"]
    if med_gamma is not None:
        lines.append(
            f"median min preconvergence Gamma    : {float(med_gamma):.6e}"
        )

    med_share = decision["median_max_preconvergence_target_mode_energy_share"]
    if med_share is not None:
        lines.append(
            f"median max preconvergence R{cfg.target_mode} share : "
            f"{float(med_share):.6f}"
        )

    lines.append(f"elapsed seconds                    : {elapsed:.2f}")
    lines.append("=" * 94)
    lines.append(
        "Guardrail: a Stage-2 PASS means reproducible evidence for this fixed "
        "experiment, not a universal VPINN claim."
    )
    lines.append("=" * 94)

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    cfg = parse_args()

    stage1_script = Path(cfg.stage1_script).expanduser().resolve()
    if not stage1_script.is_file():
        raise FileNotFoundError(
            f"Stage-1 script not found: {stage1_script}\n"
            "Place this Stage-2 script in the same directory as Stage 1, or "
            "pass --stage1-script /path/to/vpinn_gradient_conflict_stage1.py"
        )

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    runs_dir = output_dir / "runs"
    logs_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    stage1_hash = sha256_file(stage1_script)

    manifest = {
        "stage2_config": asdict(cfg),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "stage1_script": str(stage1_script),
        "stage1_sha256": stage1_hash,
        "precommitment": {
            "seeds": list(cfg.seeds),
            "conflict_signature": {
                "min_preconvergence_gamma_le": cfg.conflict_gamma_threshold,
                "max_preconvergence_weighted_negative_cosine_ge": (
                    cfg.conflict_weighted_negative_threshold
                ),
            },
            "spectral_localization": {
                "target_mode": cfg.target_mode,
                "max_preconvergence_target_mode_energy_share_ge": (
                    cfg.target_mode_energy_share_threshold
                ),
            },
            "eventual_convergence": {
                "final_relative_l2_error_le": cfg.convergence_error_threshold,
            },
            "group_pass_fraction_ge": cfg.required_seed_fraction,
        },
    }
    save_json(output_dir / "manifest.json", manifest)

    print("=" * 88)
    print("VPINN GRADIENT-CONFLICT — STAGE 2 SEED REPLICATION")
    print("=" * 88)
    print(f"Stage-1 script       : {stage1_script}")
    print(f"Stage-1 SHA256       : {stage1_hash}")
    print(f"Python executable    : {sys.executable}")
    print(f"Seeds                : {list(cfg.seeds)}")
    print(f"Device               : {cfg.device}")
    print(f"Output               : {output_dir}")
    print("-" * 88)
    print("PRECOMMITTED GATES")
    print(
        f"Conflict per seed    : min preconv Gamma <= {cfg.conflict_gamma_threshold:.2f} "
        f"AND max preconv weighted-neg-cos >= "
        f"{cfg.conflict_weighted_negative_threshold:.2f}"
    )
    print(
        f"R{cfg.target_mode} localization      : max preconv R{cfg.target_mode} energy share >= "
        f"{cfg.target_mode_energy_share_threshold:.2f}"
    )
    print(
        f"Convergence per seed : final relL2 <= {cfg.convergence_error_threshold:.1e}"
    )
    print(
        f"Group pass           : >= {cfg.required_seed_fraction:.0%} of seeds"
    )
    print("=" * 88)

    started = time.perf_counter()

    all_checkpoint_rows: List[dict] = []
    seed_summaries: List[dict] = []

    for index, seed in enumerate(cfg.seeds, start=1):
        run_dir = runs_dir / f"seed_{seed:03d}"
        log_path = logs_dir / f"seed_{seed:03d}.log"

        print()
        print("#" * 88)
        print(f"SEED {seed}  [{index}/{len(cfg.seeds)}]")
        print("#" * 88)

        if run_dir.exists():
            if cfg.reuse_existing:
                print(f"Validating existing run: {run_dir}")
                validate_stage1_run(run_dir, cfg, seed)
                print("Validated existing run; reusing it.")
            else:
                print(f"Removing stale/existing seed directory: {run_dir}")
                shutil.rmtree(run_dir)

        if not run_dir.exists():
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            command = build_stage1_command(
                stage1_script,
                run_dir,
                cfg,
                seed,
            )

            print("Command:")
            print("  " + " ".join(command))
            print()
            run_and_tee(command, log_path)

        rows, summary = analyze_seed(run_dir, cfg, seed)
        all_checkpoint_rows.extend(rows)
        seed_summaries.append(summary)

        print(
            f"Stage-2 seed audit: conflict={format_bool(summary['conflict_signature_pass'])}, "
            f"R{cfg.target_mode}-local={format_bool(summary['spectral_localization_pass'])}, "
            f"converged={format_bool(summary['eventual_convergence_pass'])}"
        )

    # Stable ordering for reproducible CSV output.
    all_checkpoint_rows.sort(key=lambda r: (int(r["seed"]), int(r["epoch"])))
    seed_summaries.sort(key=lambda r: int(r["seed"]))

    write_csv_rows(
        output_dir / "aggregate_checkpoint_metrics.csv",
        all_checkpoint_rows,
    )
    write_csv_rows(output_dir / "seed_summary.csv", seed_summaries)

    decision = aggregate_decision(cfg, seed_summaries)
    save_json(output_dir / "decision.json", decision)

    plot_trajectories(
        all_checkpoint_rows,
        field="relative_l2_error",
        ylabel="Relative L2 error",
        title="VPINN solution error across independent seeds",
        output_path=output_dir / "error_trajectories.png",
        dpi=cfg.dpi,
        log_y=True,
    )

    plot_trajectories(
        all_checkpoint_rows,
        field="gradient_coherence",
        ylabel="Gradient coherence Γ",
        title="Per-test gradient coherence across independent seeds",
        output_path=output_dir / "gradient_coherence_trajectories.png",
        dpi=cfg.dpi,
        log_y=True,
    )

    plot_trajectories(
        all_checkpoint_rows,
        field="target_mode_residual_energy_share",
        ylabel=f"R{cfg.target_mode} residual-energy share",
        title=(
            f"Localization of unresolved residual energy to test mode v_{cfg.target_mode}"
        ),
        output_path=output_dir / "r7_energy_share_trajectories.png",
        dpi=cfg.dpi,
        log_y=False,
    )

    plot_trajectories(
        all_checkpoint_rows,
        field="weighted_negative_cosine",
        ylabel="Weighted negative cosine",
        title="Strength of consequential per-test gradient conflict",
        output_path=output_dir / "weighted_negative_cosine_trajectories.png",
        dpi=cfg.dpi,
        log_y=False,
    )

    elapsed = time.perf_counter() - started
    summary_text = build_console_summary(
        cfg,
        seed_summaries,
        decision,
        elapsed,
    )

    with (output_dir / "console_summary.txt").open("w", encoding="utf-8") as f:
        f.write(summary_text)
        f.write("\n")

    print()
    print(summary_text)
    print()
    print(f"Detailed outputs: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
