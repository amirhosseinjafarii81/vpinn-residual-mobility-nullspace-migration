#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 30R
Non-Fourier Control Nonescape: Weak-Test Nullspace Localization
================================================================

Scientific status
-----------------
Stage 29R changed the test span from Fourier sine tests to an
energy-orthonormalized 24D continuous P1 H_0^1 space on 25 uniform elements.

Results:

    b=1 persistent deep lock = 5/5
    b=2 persistent deep lock = 0/5
    b=2 certified escape     = 3/5

Thus the DEEP-LOCK phenotype itself transferred strongly, but the precommitted
b=2 escape-control gate failed because seeds 25 and 27 reached epoch 3000
without satisfying relL2<=1e-2.

The two nonescaping controls are not deep locked:

    seed 25 at 3000:
        relL2 ~ 1.07e-2
        VPINN loss ~ 7.97e-8
        target share ~ 9.27e-3

    seed 27 at 3000:
        relL2 ~ 1.67e-2
        VPINN loss ~ 4.16e-13
        target share ~ 9.59e-4

This suggests a different failure mode:
the finite P1 weak-test space may be nearly or exactly blind to part of the
remaining trial-function error.

Stage 30R localizes that possibility before ANY further training experiment.

Scope
-----
Only the two Stage-29 nonescaping b=2 controls:

    seeds {25,27}
    base b=2
    target m=9.

Each is deterministically replayed to epoch 3000.

No continuation beyond 3000.
No new seed.
No optimizer intervention.
No changed test space.

Exact endpoint replay
---------------------
At epoch 3000 reproduce Stage-29:

    relL2
    VPINN loss
    residual L2 norm
    target-template share
    target-template absolute residual

to <=1e-10.

Energy visibility of the actual endpoint error
----------------------------------------------
Let

    e = u_theta - u_exact.

Because the P1 tests are energy-orthonormal,

    r_j = a(e,v_j)

and therefore

    ||P_V e||_a^2 = sum_j r_j^2.

Compute

    chi_seen =
      sum_j r_j^2 / ||e||_a^2,

where

    ||e||_a^2 = int (e')^2 + sigma e^2.

Thus

    1-chi_seen

is the fraction of endpoint error energy invisible to the 24D weak-test
space, up to the same piecewise quadrature used by the experiment.

This is basis invariant.

Analytic blind-spectrum audit
-----------------------------
For sine modes k=1,...,80 compute

    capture_k =
      ||q_k|| / sqrt(lambda_k/2),

where q_k is the P1 weak-response vector.

A STRUCTURALLY BLIND mode is precommitted as

    capture_k <= 1e-10.

On a uniform 25-element P1 mesh, modes at multiples of 25 are expected from
the mesh structure to be candidates; Stage 30 verifies this numerically
rather than assuming it.

Endpoint sine-spectrum audit
----------------------------
On a dense 16001-point grid, compute the first 80 sine coefficients of e:

    c_k = 2 int e(x) sin(k*pi*x) dx.

Record:

    * total L2 error captured by modes 1..80;
    * top ten error modes;
    * fraction of total endpoint L2-error energy in structurally blind modes;
    * capture-weighted mean visibility of the error spectrum.

The spectral decomposition is descriptive. The primary nullspace certificate
remains the exact energy-projection ratio chi_seen.

Precommitted gates
------------------

N1 — EXACT STAGE-29 ENDPOINT REPLAY
    2/2 endpoints reproduce to <=1e-10.

N2 — STRUCTURAL BLIND DIRECTION EXISTS
    capture_25 <=1e-10.

N3 — WEAKLY RESOLVED BUT L2-UNRESOLVED ENDPOINTS
    for both seeds:
        relL2 > 1e-2
        AND VPINN loss <=1e-6
        AND target-template share <=0.20.

N4 — ENDPOINT ERROR IS OVERWHELMINGLY WEAK-TEST INVISIBLE
    chi_seen <=1e-2 in 2/2 seeds.

N5 — STRUCTURAL BLIND-MODE DOMINANCE
    at least 50% of endpoint L2-error energy lies in modes with
    capture_k<=1e-10, in 2/2 seeds.

Interpretation
--------------
If N1-N4 PASS:
    Stage-29 S4 failed because the b=2 controls can become weakly solved while
    retaining error outside the chosen P1 test space. This is a distinct
    finite-test-space underdetermination phenomenon, not deep mobility lock.

If N5 also PASS:
    the error floor is sharply localized to the exact mesh-blind sine family
    and the next causal control is a ONE-DIRECTION blind-mode enrichment.

If N1-N4 PASS but N5 FAIL:
    the invisible error is distributed over a broader near-null complement;
    next use a minimal P1 mesh-refinement control rather than cherry-picking a
    single sine direction.

If N4 FAIL:
    do not blame the P1 nullspace. Route to targeted horizon/optimizer-state
    analysis of seeds 25 and 27.

Guardrail
---------
This stage does not retroactively change Stage-29 S4 from FAIL to PASS.
It determines WHY the escape-control gate failed.
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
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


SEEDS = (25, 27)
BASE_MODE = 2
TARGET_MODE = 9
END_EPOCH = 3000

SPECTRAL_K_MAX = 80
DENSE_N = 16001

REPLAY_TOL = 1.0e-10
STRUCTURAL_BLIND_TOL = 1.0e-10

WEAK_LOSS_TOL = 1.0e-6
TARGET_RESOLVED_SHARE = 0.20
REL_L2_UNRESOLVED = 1.0e-2

SEEN_ENERGY_FRACTION_MAX = 1.0e-2
BLIND_L2_FRACTION_MIN = 0.50


# =============================================================================
# CLI / helpers
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-30R weak-test nullspace localization of Stage-29 b2 nonescapes."
    )

    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")

    p.add_argument(
        "--stage3-script",
        default="vpinn_gradient_conflict_stage3_frequency_transfer.py",
    )

    p.add_argument(
        "--stage29-script",
        default="vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness.py",
    )

    p.add_argument(
        "--stage29-dir",
        default="vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage30R_p1_nullspace_localization",
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


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{key: row.get(key, None) for key in fields} for row in rows]
        )


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    return mod


# =============================================================================
# Provenance / expected Stage-29 endpoints
# =============================================================================

def preflight(
    stage3_script: Path,
    stage29_script: Path,
    stage29_dir: Path,
) -> dict:

    manifest_path = stage29_dir / "manifest.json"
    decision_path = stage29_dir / "decision.json"
    run_path = stage29_dir / "run_summary.csv"
    tracking_path = stage29_dir / "tracking_metrics.csv"

    for path in (
        manifest_path,
        decision_path,
        run_path,
        tracking_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    decision = read_json(decision_path)

    s3 = sha256_file(stage3_script)
    s29 = sha256_file(stage29_script)

    if manifest.get("stage3_solver_sha256") != s3:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 29.")

    if manifest.get("stage29r_script_sha256") != s29:
        raise RuntimeError("Stage-29 source SHA mismatch.")

    expected_decision = {
        "b1_persistent_deep_lock_count": 5,
        "b2_persistent_deep_lock_count": 0,
        "b2_certified_escape_count": 3,
        "S4_paired_b2_nonlock_control": False,
        "next_route": "stage30R_testspace_specific_control_localization",
    }

    for key, expected in expected_decision.items():
        if decision.get(key) != expected:
            raise RuntimeError(
                f"Unexpected Stage-29 decision field {key}: "
                f"{decision.get(key)!r} != {expected!r}"
            )

    runs = read_csv(run_path)
    tracking = read_csv(tracking_path)

    nonescape = []

    for row in runs:
        if int(row["base_mode"]) != BASE_MODE:
            continue

        if str(row["persistent_deep_lock"]).lower() == "true":
            continue

        if str(row["certified_escape"]).lower() == "true":
            continue

        if row["stop_reason"] == "HORIZON":
            nonescape.append(int(row["seed"]))

    nonescape = sorted(nonescape)

    if nonescape != list(SEEDS):
        raise RuntimeError(
            f"Expected Stage-29 b2 horizon seeds {list(SEEDS)}, got {nonescape}."
        )

    expected_endpoint = {}

    for row in tracking:
        seed = int(row["seed"])
        b = int(row["base_mode"])
        epoch = int(row["epoch"])

        if (
            seed in SEEDS
            and b == BASE_MODE
            and epoch == END_EPOCH
        ):
            expected_endpoint[seed] = row

    if set(expected_endpoint) != set(SEEDS):
        raise RuntimeError("Missing Stage-29 epoch-3000 endpoint rows.")

    return {
        "stage3_sha256": s3,
        "stage29_sha256": s29,
        "decision": decision,
        "expected_endpoint": expected_endpoint,
    }


# =============================================================================
# Endpoint diagnostics
# =============================================================================

def endpoint_error_energy(exp):
    """
    Energy norm of e=u_theta-u_exact and the part seen by the orthonormal
    weak-test space.
    """

    x = exp.x_quad

    u = exp.model(x)

    du = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=False,
        retain_graph=True,
    )[0]

    x_det = x.detach()

    u_exact = exp.exact_solution(x_det)
    du_exact = exp.exact_derivative(x_det)

    e = u.detach() - u_exact
    de = du.detach() - du_exact

    energy_sq = float(
        torch.sum(
            exp.w_quad
            * (
                de.square()
                +
                exp.sigma * e.square()
            )
        ).item()
    )

    residual = exp.weak_residuals().detach()

    seen_energy_sq = float(
        torch.sum(residual.square()).item()
    )

    seen_fraction = (
        seen_energy_sq / energy_sq
        if energy_sq > 0.0
        else float("nan")
    )

    return {
        "error_energy_sq":
            energy_sq,

        "seen_projection_energy_sq":
            seen_energy_sq,

        "seen_energy_fraction":
            seen_fraction,

        "invisible_energy_fraction":
            1.0 - seen_fraction,
    }


def analytic_capture_spectrum(exp, kmax: int):
    rows = []

    for k in range(1, kmax + 1):
        q = exp.unit_mode_response(k)

        qnorm = float(
            torch.linalg.vector_norm(q).item()
        )

        lam = (
            (k * math.pi) ** 2
            + exp.sigma
        )

        full_energy_norm = math.sqrt(
            lam / 2.0
        )

        capture = (
            qnorm / full_energy_norm
        )

        rows.append(
            {
                "mode": k,
                "response_norm": qnorm,
                "full_energy_norm": full_energy_norm,
                "capture_ratio": capture,
                "structurally_blind":
                    bool(capture <= STRUCTURAL_BLIND_TOL),
            }
        )

    return rows


@torch.no_grad()
def dense_error_spectrum(exp, capture_map: dict):
    x = torch.linspace(
        0.0,
        1.0,
        DENSE_N,
        dtype=exp.dtype,
        device=exp.device,
    ).reshape(-1, 1)

    pred = exp.model(x)
    truth = exp.exact_solution(x)

    e = (
        pred - truth
    ).reshape(-1).detach().cpu().numpy()

    x_np = (
        x.reshape(-1)
        .detach()
        .cpu()
        .numpy()
    )

    total_l2_sq = float(
        np.trapezoid(
            e * e,
            x_np,
        )
    )

    rows = []

    for k in range(1, SPECTRAL_K_MAX + 1):

        basis = np.sin(
            k * math.pi * x_np
        )

        coeff = 2.0 * float(
            np.trapezoid(
                e * basis,
                x_np,
            )
        )

        mode_l2_energy = (
            0.5 * coeff * coeff
        )

        rows.append(
            {
                "mode":
                    k,

                "sine_coefficient":
                    coeff,

                "mode_l2_energy":
                    mode_l2_energy,

                "fraction_total_l2_error":
                    (
                        mode_l2_energy / total_l2_sq
                        if total_l2_sq > 0.0
                        else float("nan")
                    ),

                "testspace_capture_ratio":
                    capture_map[k],

                "structurally_blind":
                    bool(
                        capture_map[k]
                        <= STRUCTURAL_BLIND_TOL
                    ),
            }
        )

    captured_1_80 = sum(
        float(r["mode_l2_energy"])
        for r in rows
    )

    blind_energy = sum(
        float(r["mode_l2_energy"])
        for r in rows
        if bool(r["structurally_blind"])
    )

    weighted_capture_numer = sum(
        float(r["mode_l2_energy"])
        * float(r["testspace_capture_ratio"])
        for r in rows
    )

    return rows, {
        "dense_l2_error_sq":
            total_l2_sq,

        "spectral_1_80_l2_energy":
            captured_1_80,

        "spectral_1_80_capture_fraction":
            (
                captured_1_80 / total_l2_sq
                if total_l2_sq > 0.0
                else float("nan")
            ),

        "blind_modes_l2_energy":
            blind_energy,

        "blind_modes_fraction_total_l2_error":
            (
                blind_energy / total_l2_sq
                if total_l2_sq > 0.0
                else float("nan")
            ),

        "capture_weighted_mean_visibility_1_80":
            (
                weighted_capture_numer / captured_1_80
                if captured_1_80 > 0.0
                else float("nan")
            ),
    }


# =============================================================================
# Plots
# =============================================================================

def plot_capture(rows: List[dict], path: Path):
    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    ax.plot(
        [int(r["mode"]) for r in rows],
        [float(r["capture_ratio"]) for r in rows],
        marker="o",
        markersize=2.5,
        linewidth=1.0,
    )

    ax.axhline(
        STRUCTURAL_BLIND_TOL,
        linestyle="--",
        linewidth=1.0,
        label="structural blind threshold",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Sine mode k")
    ax.set_ylabel("P1 weak-response capture ratio")
    ax.set_title("Analytic visibility spectrum of the 24D P1 test space")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_error_spectra(rows: List[dict], path: Path):
    fig, ax = plt.subplots(figsize=(10.0, 5.4))

    for seed in SEEDS:
        rr = [
            r for r in rows
            if int(r["seed"]) == seed
        ]

        rr.sort(
            key=lambda r: int(r["mode"])
        )

        ax.plot(
            [int(r["mode"]) for r in rr],
            [
                max(
                    float(r["fraction_total_l2_error"]),
                    1.0e-18,
                )
                for r in rr
            ],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            label=f"seed {seed}",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Sine mode k")
    ax.set_ylabel("Fraction of endpoint L2-error energy")
    ax.set_title("Where does the unresolved b=2 endpoint error live?")
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

    def resolve(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else root / p

    stage3_script = resolve(args.stage3_script)
    stage29_script = resolve(args.stage29_script)
    stage29_dir = resolve(args.stage29_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    pf = preflight(
        stage3_script=stage3_script,
        stage29_script=stage29_script,
        stage29_dir=stage29_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage30R",
    )

    stage29 = load_module(
        stage29_script,
        "vpinn_stage29_stage30R",
    )

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

        "stage29_script_sha256":
            pf["stage29_sha256"],

        "stage30r_script_sha256":
            sha256_file(
                Path(__file__).resolve()
            ),

        "precommitment": {
            "stage":
                "P1_control_nonescape_weak_test_nullspace_localization",

            "seeds":
                list(SEEDS),

            "base_mode":
                BASE_MODE,

            "target_mode":
                TARGET_MODE,

            "reconstruct_epoch":
                END_EPOCH,

            "spectral_k_max":
                SPECTRAL_K_MAX,

            "structural_blind_capture_tol":
                STRUCTURAL_BLIND_TOL,

            "weak_loss_tol":
                WEAK_LOSS_TOL,

            "seen_energy_fraction_max":
                SEEN_ENERGY_FRACTION_MAX,

            "blind_l2_fraction_min":
                BLIND_L2_FRACTION_MIN,

            "no_continuation":
                True,

            "no_optimizer_intervention":
                True,
        },
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 180)
    print(
        "VPINN — STAGE 30R P1 CONTROL NONESCAPE / WEAK-TEST NULLSPACE LOCALIZATION"
    )
    print("=" * 180)
    print(f"device                    : {device}")
    print(f"replayed seeds            : {list(SEEDS)}")
    print(f"cell                      : b=2,m=9")
    print(f"endpoint                  : epoch {END_EPOCH}")
    print(f"sine visibility audit     : k=1..{SPECTRAL_K_MAX}")
    print("continuation              : NONE")
    print("=" * 180)

    replay_rows = []
    endpoint_rows = []
    spectral_rows = []

    analytic_capture_rows = None
    capture_map = None

    for seed in SEEDS:

        run_dir = (
            out_dir
            / f"seed_{seed:03d}"
            / "base_02_target_09"
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cfg = stage29.make_config(
            stage3=stage3,
            seed=seed,
            device=device,
            out_dir=run_dir,
        )

        exp = stage29.P1ReactionDiffusionExperiment(
            stage3=stage3,
            cfg=cfg,
            device=device,
            base_mode=BASE_MODE,
            target_mode=TARGET_MODE,
            out_dir=run_dir,
        )

        if analytic_capture_rows is None:
            analytic_capture_rows = analytic_capture_spectrum(
                exp,
                SPECTRAL_K_MAX,
            )

            capture_map = {
                int(r["mode"]):
                    float(r["capture_ratio"])
                for r in analytic_capture_rows
            }

        for _ in range(END_EPOCH):
            exp.train_step()

        rm = exp.residual_metrics()
        rel = exp.relative_l2_error()

        expected = pf[
            "expected_endpoint"
        ][seed]

        diffs = {
            "relative_l2_error":
                abs(
                    rel
                    - float(
                        expected[
                            "relative_l2_error"
                        ]
                    )
                ),

            "vpinn_loss":
                abs(
                    float(rm["vpinn_loss"])
                    - float(expected["vpinn_loss"])
                ),

            "residual_l2_norm":
                abs(
                    float(rm["residual_l2_norm"])
                    - float(
                        expected[
                            "residual_l2_norm"
                        ]
                    )
                ),

            "target_share":
                abs(
                    float(
                        rm[
                            "target_mode_residual_energy_share"
                        ]
                    )
                    - float(
                        expected[
                            "target_mode_residual_energy_share"
                        ]
                    )
                ),

            "target_abs_residual":
                abs(
                    float(
                        rm[
                            "target_template_abs_residual"
                        ]
                    )
                    - float(
                        expected[
                            "target_template_abs_residual"
                        ]
                    )
                ),
        }

        max_gap = max(
            diffs.values()
        )

        if max_gap > REPLAY_TOL:
            raise RuntimeError(
                f"Stage-29 endpoint replay failed seed={seed}: "
                f"gap={max_gap:.3e}, diffs={diffs}"
            )

        replay_rows.append(
            {
                "seed":
                    seed,

                "max_abs_difference":
                    max_gap,

                "pass":
                    True,

                **{
                    f"gap_{key}": value
                    for key, value in diffs.items()
                },
            }
        )

        energy = endpoint_error_energy(
            exp
        )

        spec_rows, spec_summary = dense_error_spectrum(
            exp,
            capture_map,
        )

        for row in spec_rows:
            spectral_rows.append(
                {
                    "seed":
                        seed,

                    **row,
                }
            )

        top_modes = sorted(
            spec_rows,
            key=lambda r: float(
                r["mode_l2_energy"]
            ),
            reverse=True,
        )[:10]

        top_modes_text = ",".join(
            f"{int(r['mode'])}:"
            f"{float(r['fraction_total_l2_error']):.6g}"
            for r in top_modes
        )

        weakly_resolved_l2_unresolved = bool(
            rel > REL_L2_UNRESOLVED
            and
            float(rm["vpinn_loss"])
            <= WEAK_LOSS_TOL
            and
            float(
                rm[
                    "target_mode_residual_energy_share"
                ]
            )
            <= TARGET_RESOLVED_SHARE
        )

        invisible_endpoint = bool(
            float(
                energy[
                    "seen_energy_fraction"
                ]
            )
            <= SEEN_ENERGY_FRACTION_MAX
        )

        blind_dominated = bool(
            float(
                spec_summary[
                    "blind_modes_fraction_total_l2_error"
                ]
            )
            >= BLIND_L2_FRACTION_MIN
        )

        endpoint_rows.append(
            {
                "seed":
                    seed,

                "relative_l2_error":
                    rel,

                **rm,
                **energy,
                **spec_summary,

                "weakly_resolved_but_l2_unresolved":
                    weakly_resolved_l2_unresolved,

                "weak_test_invisible_endpoint":
                    invisible_endpoint,

                "blind_mode_l2_dominated":
                    blind_dominated,

                "top10_error_modes_fraction":
                    top_modes_text,
            }
        )

        print()
        print(
            f"seed={seed}: relL2={rel:.6e}, "
            f"loss={float(rm['vpinn_loss']):.6e}, "
            f"seen_energy={float(energy['seen_energy_fraction']):.6e}, "
            f"blind_L2={float(spec_summary['blind_modes_fraction_total_l2_error']):.6f}, "
            f"replay={max_gap:.3e}"
        )

        print(
            f"  top modes: {top_modes_text}"
        )

    # =========================================================================
    # Persist
    # =========================================================================
    write_csv(
        out_dir / "endpoint_replay_checks.csv",
        replay_rows,
    )

    write_csv(
        out_dir / "analytic_sine_capture_spectrum.csv",
        analytic_capture_rows,
    )

    write_csv(
        out_dir / "endpoint_nullspace_summary.csv",
        endpoint_rows,
    )

    write_csv(
        out_dir / "endpoint_error_sine_spectrum.csv",
        spectral_rows,
    )

    # =========================================================================
    # Gates
    # =========================================================================
    N1 = bool(
        len(replay_rows) == 2
        and
        all(
            bool(r["pass"])
            for r in replay_rows
        )
    )

    cap25 = next(
        float(r["capture_ratio"])
        for r in analytic_capture_rows
        if int(r["mode"]) == 25
    )

    blind_modes = [
        int(r["mode"])
        for r in analytic_capture_rows
        if bool(r["structurally_blind"])
    ]

    N2 = bool(
        cap25 <= STRUCTURAL_BLIND_TOL
    )

    N3 = bool(
        len(endpoint_rows) == 2
        and
        all(
            bool(
                r[
                    "weakly_resolved_but_l2_unresolved"
                ]
            )
            for r in endpoint_rows
        )
    )

    N4 = bool(
        len(endpoint_rows) == 2
        and
        all(
            bool(
                r[
                    "weak_test_invisible_endpoint"
                ]
            )
            for r in endpoint_rows
        )
    )

    N5 = bool(
        len(endpoint_rows) == 2
        and
        all(
            bool(
                r[
                    "blind_mode_l2_dominated"
                ]
            )
            for r in endpoint_rows
        )
    )

    nullspace_localized = bool(
        N1 and N2 and N3 and N4
    )

    sharp_blind_mode = bool(
        nullspace_localized and N5
    )

    if sharp_blind_mode:

        route_class = (
            "P1_b2_nonescape_is_weak_test_nullspace_floor_sharply_localized_to_structural_blind_modes"
        )

        next_route = (
            "stage31R_one_direction_blind_mode_enrichment_control"
        )

    elif nullspace_localized:

        route_class = (
            "P1_b2_nonescape_is_distributed_weak_test_nullspace_floor"
        )

        next_route = (
            "stage31R_minimal_P1_mesh_refinement_control"
        )

    else:

        route_class = (
            "P1_b2_nonescape_not_explained_by_weak_test_nullspace"
        )

        next_route = (
            "stage31R_targeted_horizon_optimizer_state_audit"
        )

    decision = {
        "replayed_seeds":
            list(SEEDS),

        "N1_exact_stage29_endpoint_replay":
            N1,

        "structurally_blind_modes_1_80":
            blind_modes,

        "capture_mode25":
            cap25,

        "N2_structural_blind_direction_exists":
            N2,

        "N3_weakly_resolved_but_l2_unresolved":
            N3,

        "N4_endpoint_error_weak_test_invisible":
            N4,

        "N5_structural_blind_mode_l2_dominance":
            N5,

        "weak_test_nullspace_localization_supported":
            nullspace_localized,

        "sharp_structural_blind_mode_localization_supported":
            sharp_blind_mode,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A PASS explains why Stage-29 b2 controls can miss the strict L2 "
            "escape gate despite no deep mobility lock. It does not "
            "retroactively change the precommitted Stage-29 S4 result."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    plot_capture(
        analytic_capture_rows,
        out_dir / "P1_sine_visibility_spectrum.png",
    )

    plot_error_spectra(
        spectral_rows,
        out_dir / "nonescape_endpoint_error_spectra.png",
    )

    # =========================================================================
    # Console
    # =========================================================================
    lines = []

    lines.append("=" * 182)
    lines.append(
        "VPINN — STAGE 30R P1 WEAK-TEST NULLSPACE LOCALIZATION SUMMARY"
    )
    lines.append("=" * 182)

    lines.append(
        f"structurally blind modes k<=80 : {blind_modes}"
    )

    lines.append(
        f"capture(mode 25)               : {cap25:.6e}"
    )

    lines.append(
        "seed | relL2 | VPINN loss | seen energy fraction | invisible energy | "
        "blind L2 fraction | top error modes"
    )

    lines.append("-" * 182)

    for r in endpoint_rows:
        lines.append(
            f"{int(r['seed']):4d} | "
            f"{float(r['relative_l2_error']):.6e} | "
            f"{float(r['vpinn_loss']):.6e} | "
            f"{float(r['seen_energy_fraction']):.6e} | "
            f"{float(r['invisible_energy_fraction']):.6f} | "
            f"{float(r['blind_modes_fraction_total_l2_error']):.6f} | "
            f"{r['top10_error_modes_fraction']}"
        )

    lines.append("-" * 182)

    lines.append(
        f"N1 exact endpoint replay             : "
        f"{sum(int(r['pass']) for r in replay_rows)}/2 -> {N1}"
    )

    lines.append(
        f"N2 structural blind direction        : {N2}"
    )

    lines.append(
        f"N3 weakly solved / L2 unresolved     : "
        f"{sum(int(r['weakly_resolved_but_l2_unresolved']) for r in endpoint_rows)}/2 -> {N3}"
    )

    lines.append(
        f"N4 weak-test invisible endpoint      : "
        f"{sum(int(r['weak_test_invisible_endpoint']) for r in endpoint_rows)}/2 -> {N4}"
    )

    lines.append(
        f"N5 exact blind-mode L2 dominance     : "
        f"{sum(int(r['blind_mode_l2_dominated']) for r in endpoint_rows)}/2 -> {N5}"
    )

    lines.append(
        f"WEAK-TEST NULLSPACE LOCALIZATION     : {nullspace_localized}"
    )

    lines.append(
        f"SHARP BLIND-MODE LOCALIZATION        : {sharp_blind_mode}"
    )

    lines.append(
        f"route class                           : {route_class}"
    )

    lines.append(
        f"next route                            : {next_route}"
    )

    lines.append("=" * 182)

    lines.append(
        "Guardrail: Stage-29 S4 remains FAIL. Stage 30 only localizes the reason."
    )

    lines.append("=" * 182)

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
