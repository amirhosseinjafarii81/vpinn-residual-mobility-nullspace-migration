#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 34R
Read-Only Union-Visible Residual / Adam Response Audit
======================================================

Why this stage exists
---------------------
Stage 33R established:

    U1 exact Stage-31 endpoint replay              PASS
    U2 common V26 equivalence                      PASS
    U3 V26+V25 union visibility rescue             PASS
    U4 union certified escape                      PASS
    U5 matched trajectory benefit                  FAIL

The union branch was far better for most of the 4000..4500 horizon:
    * certified escape onset = 4025
    * 18/21 tracked points had relL2 <= 1e-2
    * minimum relL2 ~ 2.94e-3 at epoch 4475
    * AUC was much lower than CONTROL26

but two late instability bursts returned relL2 above the strict threshold,
including the final 4475->4500 burst.

Importantly, the union endpoint remains strongly visible to the union space.
Therefore this is not another nullspace-migration floor.

Stage 34 asks the narrow next question:

    When the persistent union trajectory spikes, is the actual Adam step
    already first-order uphill because optimizer history reverses the current
    gradient contribution, or is the candidate direction first-order descent
    but the finite step overshoots because of curvature?

This is a deterministic replay / diagnostic audit only.

No new training condition
-------------------------
One existing trajectory only:

    seed 27, UNION26_25, epochs 4000..4500.

No new seed.
No new branch.
No changed optimizer.
No changed learning rate.
No test-space change.

Exact replay
------------
Reconstruct the Stage-33 UNION26_25 branch and reproduce every 25-grid row
from epoch 4000 through 4500.

At every single training step record:

    L_before, L_after
    relL2_before, relL2_after
    raw gradient norm
    exact Adam candidate displacement
    exact current-gradient Adam component
    exact history Adam component
    g^T Delta_current
    g^T Delta_history
    g^T Delta_candidate
    parameter-delta prediction gap

Two precommitted instability windows
------------------------------------
Stage-33 coarse tracking contains two clear upward bursts:

    W1: 4350 -> 4375
    W2: 4475 -> 4500

Within each fixed window, select the ONE one-step transition with the largest
positive exact weak-loss change. This is diagnostic localization inside a
predeclared window, not an adaptive training choice.

At each selected event compute:

    * exact line scan along the Adam candidate for
          alpha in {1/64,1/32,1/16,1/8,1/4,1/2,1}
    * directional second derivative kappa = d^T H_L d
    * Gauss-Newton curvature
          kappa_GN = (2/24)||J d||^2
    * nonlinear residual curvature
          kappa_NL = kappa - kappa_GN
    * second-order prediction
          Delta L_(2) = g^T d + 0.5 kappa

Mechanism classes
-----------------
HISTORY_FLIP:
    g^T Delta_candidate >= 0
    AND g^T Delta_current < 0
    AND g^T Delta_history > |g^T Delta_current|.

CURVATURE_OVERSHOOT:
    g^T Delta_candidate < 0
    BUT exact Delta L > 0,
    AND at least one alpha < 1 gives exact Delta L < 0.

MIXED_OR_OTHER:
    neither of the above.

Primary gates
-------------
A1 — EXACT STAGE-33 UNION REPLAY
    all 21 tracked rows reproduce to <=1e-10
    and every predicted Adam displacement matches the actual parameter update
    to relative error <=1e-10.

A2 — BOTH COARSE SPIKE WINDOWS CONTAIN A TRUE ONE-STEP LOSS SPIKE
    one selected positive-Delta-L event in W1 and one in W2.

A3 — CONSISTENT INSTABILITY MECHANISM
    the two selected events have the same mechanism class and that class is
    either HISTORY_FLIP or CURVATURE_OVERSHOOT.

Decision
--------
If A1&A2&A3 and class=HISTORY_FLIP:
    conclude the union made the residual visible, but retained Adam history
    can transiently push the actual step against the current union objective.

If A1&A2&A3 and class=CURVATURE_OVERSHOOT:
    conclude the union made the residual visible, but the fixed Adam step can
    overshoot the local union-loss basin.

Otherwise:
    classify the late instability as mixed and do NOT launch another rescue
    sweep before the literature/claim audit.

Next in every case
------------------
Stage 35R = literature / novelty / claim audit.

No additional optimizer sweep is automatically authorized by Stage 34.

Guardrail
---------
This stage diagnoses why Stage-33 U5 failed. It does not change Stage-33 U5
to PASS and does not promote a universal Adam claim from one trajectory.
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
from pathlib import Path
from typing import List

import numpy as np
import torch


SEED = 27
START_EPOCH = 4000
FINAL_EPOCH = 4500
TRACK_INTERVAL = 25
LOSS_DENOMINATOR = 24.0

WINDOWS = (
    ("W1", 4350, 4375),
    ("W2", 4475, 4500),
)

ALPHAS = (
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
)

REPLAY_TOL = 1.0e-10
ADAM_PRED_REL_TOL = 1.0e-10


# =============================================================================
# Generic
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-34R read-only union-visible residual Adam-response audit."
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
        "--stage31-script",
        default="vpinn_gradient_conflict_stage31R_minimal_P1_refinement_rescue.py",
    )

    p.add_argument(
        "--stage33-script",
        default="vpinn_gradient_conflict_stage33R_persistent_union_rescue.py",
    )

    p.add_argument(
        "--stage33-dir",
        default="vpinn_gradient_conflict_stage33R_persistent_union_rescue",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage34R_union_adam_response_audit",
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(
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


def flatten_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [p.detach().reshape(-1) for p in model.parameters()],
        dim=0,
    )


def flatten_grads(model: torch.nn.Module) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if p.grad is None:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append(p.grad.reshape(-1))
    return torch.cat(chunks, dim=0)


def assign_flat_params(model: torch.nn.Module, flat: torch.Tensor):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset+n].reshape_as(p))
            offset += n

    if offset != flat.numel():
        raise RuntimeError("Flat-parameter size mismatch.")


# =============================================================================
# Provenance
# =============================================================================

def preflight(
    stage3_script: Path,
    stage29_script: Path,
    stage31_script: Path,
    stage33_script: Path,
    stage33_dir: Path,
):
    manifest_path = stage33_dir / "manifest.json"
    decision_path = stage33_dir / "decision.json"
    tracking_path = stage33_dir / "matched_branch_tracking.csv"

    for p in (manifest_path, decision_path, tracking_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    m = read_json(manifest_path)
    d = read_json(decision_path)

    shas = {
        "s3": sha256_file(stage3_script),
        "s29": sha256_file(stage29_script),
        "s31": sha256_file(stage31_script),
        "s33": sha256_file(stage33_script),
    }

    checks = (
        ("stage3_solver_sha256", "s3"),
        ("stage29_script_sha256", "s29"),
        ("stage31_script_sha256", "s31"),
        ("stage33r_script_sha256", "s33"),
    )

    for key, skey in checks:
        if m.get(key) != shas[skey]:
            raise RuntimeError(f"Stage-33 provenance mismatch: {key}")

    expected = {
        "U1_exact_stage31_refined_endpoint_replay": True,
        "U2_common_quadrature_control_equivalence": True,
        "U3_persistent_union_visibility_rescue": True,
        "U4_union_certified_escape": True,
        "U5_matched_trajectory_benefit": False,
        "persistent_union_causal_rescue": False,
        "route_class":
            "union_restores_visibility_but_does_not_cleanly_rescue_trajectory",
        "next_route":
            "stage34R_union_visible_residual_optimizer_response_audit",
    }

    for key, value in expected.items():
        if d.get(key) != value:
            raise RuntimeError(
                f"Unexpected Stage-33 decision {key}: "
                f"{d.get(key)!r} != {value!r}"
            )

    rows = read_csv(tracking_path)
    union_rows = [
        r for r in rows
        if r["branch"] == "UNION26_25"
    ]

    union_rows.sort(key=lambda r: int(r["epoch"]))

    if len(union_rows) != 21:
        raise RuntimeError(
            f"Expected 21 Stage-33 union tracking rows, got {len(union_rows)}."
        )

    return {
        **shas,
        "union_rows": union_rows,
        "decision": d,
    }


# =============================================================================
# Exact Adam decomposition
# =============================================================================

def predict_adam_components(exp, gflat: torch.Tensor):
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

    curr = []
    hist = []
    cand = []

    offset = 0
    steps = []

    for p in exp.model.parameters():
        n = p.numel()
        gp = gflat[offset:offset+n].reshape_as(p)
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
        steps.append(step_new)

        v_new = (
            beta2 * v_old
            + (1.0 - beta2) * gp.square()
        )

        bc1 = 1.0 - beta1 ** step_new
        bc2 = 1.0 - beta2 ** step_new

        D = 1.0 / (
            v_new.sqrt() / math.sqrt(bc2)
            + eps
        )

        d_curr = (
            -lr
            * D
            * ((1.0 - beta1) / bc1)
            * gp
        )

        d_hist = (
            -lr
            * D
            * (beta1 / bc1)
            * m_old
        )

        d = d_curr + d_hist

        curr.append(d_curr.reshape(-1))
        hist.append(d_hist.reshape(-1))
        cand.append(d.reshape(-1))

    if len(set(steps)) != 1:
        raise RuntimeError("Inconsistent Adam step counters.")

    return (
        torch.cat(curr),
        torch.cat(hist),
        torch.cat(cand),
    )


# =============================================================================
# Loss / metrics
# =============================================================================

def exact_metrics(exp):
    rm = exp.residual_metrics()

    return {
        "relative_l2_error":
            exp.relative_l2_error(),

        "scaled_vpinn_loss":
            float(rm["scaled_vpinn_loss"]),

        "residual_l2_norm":
            float(rm["residual_l2_norm"]),

        "target_mode_residual_energy_share":
            float(
                rm[
                    "target_mode_residual_energy_share"
                ]
            ),

        "target_template_abs_residual":
            float(
                rm["target_template_abs_residual"]
            ),
    }


def loss_and_gradient(exp, create_graph: bool = False):
    exp.optimizer.zero_grad(set_to_none=True)

    loss = exp.loss_tensor()

    loss.backward(
        create_graph=create_graph
    )

    g = flatten_grads(exp.model)

    return loss, g


# =============================================================================
# Curvature / line scan
# =============================================================================

def event_curvature(exp, direction: torch.Tensor):
    params = list(exp.model.parameters())

    exp.optimizer.zero_grad(set_to_none=True)

    r = exp.weak_residuals()
    loss = torch.sum(r.square()) / LOSS_DENOMINATOR

    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
    )

    gflat = torch.cat(
        [g.reshape(-1) for g in grads]
    )

    first = torch.dot(
        gflat,
        direction.detach()
    )

    hv = torch.autograd.grad(
        first,
        params,
        retain_graph=True,
        allow_unused=False,
    )

    hflat = torch.cat(
        [h.reshape(-1) for h in hv]
    )

    kappa = float(
        torch.dot(
            hflat,
            direction.detach()
        ).item()
    )

    # Gauss-Newton curvature.
    jv = []

    for i in range(r.numel()):
        gri = torch.autograd.grad(
            r[i],
            params,
            retain_graph=True,
            create_graph=False,
        )

        gri_flat = torch.cat(
            [g.reshape(-1) for g in gri]
        )

        jv.append(
            torch.dot(
                gri_flat,
                direction.detach()
            )
        )

    jv = torch.stack(jv)

    kappa_gn = float(
        (
            (2.0 / LOSS_DENOMINATOR)
            * torch.dot(jv, jv)
        ).item()
    )

    return {
        "kappa_total":
            kappa,

        "kappa_GN":
            kappa_gn,

        "kappa_NL":
            kappa - kappa_gn,

        "first_order":
            float(first.item()),

        "second_order_predicted_delta_loss":
            float(
                first.item()
                + 0.5 * kappa
            ),
    }


def line_scan(exp, base_flat: torch.Tensor, direction: torch.Tensor, base_loss: float):
    rows = []

    for alpha in ALPHAS:
        assign_flat_params(
            exp.model,
            base_flat + alpha * direction
        )

        m = exact_metrics(exp)

        rows.append(
            {
                "alpha":
                    alpha,

                "loss":
                    m["scaled_vpinn_loss"],

                "delta_loss":
                    m["scaled_vpinn_loss"]
                    - base_loss,

                "relative_l2_error":
                    m["relative_l2_error"],
            }
        )

    assign_flat_params(
        exp.model,
        base_flat
    )

    return rows


# =============================================================================
# Reconstruction
# =============================================================================

def build_union_at_4000(stage3, stage29, stage31, stage33, device, out_dir):
    cfg, replay25, refined31 = (
        stage33.reconstruct_stage31_refined_endpoint(
            stage3=stage3,
            stage29=stage29,
            stage31=stage31,
            device=device,
            out_dir=out_dir / "reconstruction",
        )
    )

    endpoint_state = stage33.capture_state(
        refined31
    )

    union = stage33.CommonSpanExperiment(
        stage3=stage3,
        cfg=cfg,
        device=device,
        meshes=(26,25),
        base_mode=2,
        target_mode=9,
        base_amplitude=float(
            replay25.base_amplitude
        ),
        target_amplitude=float(
            replay25.amplitude
        ),
        sigma=float(replay25.sigma),
        out_dir=out_dir / "union_replay",
        loss_denominator=LOSS_DENOMINATOR,
    )

    stage33.restore_state(
        union,
        endpoint_state,
    )

    return union


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    def resolve(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else root / p

    stage3_script = resolve(args.stage3_script)
    stage29_script = resolve(args.stage29_script)
    stage31_script = resolve(args.stage31_script)
    stage33_script = resolve(args.stage33_script)
    stage33_dir = resolve(args.stage33_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage29_script=stage29_script,
        stage31_script=stage31_script,
        stage33_script=stage33_script,
        stage33_dir=stage33_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage34R",
    )
    stage29 = load_module(
        stage29_script,
        "vpinn_stage29_stage34R",
    )
    stage31 = load_module(
        stage31_script,
        "vpinn_stage31_stage34R",
    )
    stage33 = load_module(
        stage33_script,
        "vpinn_stage33_stage34R",
    )

    union = build_union_at_4000(
        stage3=stage3,
        stage29=stage29,
        stage31=stage31,
        stage33=stage33,
        device=device,
        out_dir=out_dir,
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
            pf["s3"],

        "stage29_script_sha256":
            pf["s29"],

        "stage31_script_sha256":
            pf["s31"],

        "stage33_script_sha256":
            pf["s33"],

        "stage34r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "readonly_union_visible_residual_adam_response_audit",

            "seed":
                SEED,

            "trajectory":
                "Stage33 UNION26_25 epochs 4000..4500",

            "windows":
                [
                    {
                        "name": name,
                        "start": start,
                        "end": end,
                    }
                    for name, start, end in WINDOWS
                ],

            "event_rule":
                "largest positive one-step exact weak-loss change in each fixed window",

            "A1":
                "all 21 Stage33 union rows reproduce <=1e-10 and Adam delta prediction <=1e-10",

            "A2":
                "both windows contain positive one-step weak-loss spike",

            "A3":
                "same non-mixed mechanism class in both selected events",

            "no_new_training_condition":
                True,

            "no_rescue_sweep":
                True,
        },
    }

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    print("=" * 184)
    print(
        "VPINN — STAGE 34R UNION-VISIBLE RESIDUAL / ADAM RESPONSE AUDIT"
    )
    print("=" * 184)
    print(f"device                    : {device}")
    print(f"seed                      : {SEED}")
    print(f"trajectory                : UNION26_25")
    print(f"epoch range               : {START_EPOCH}..{FINAL_EPOCH}")
    print("new training condition    : NONE")
    print("=" * 184)

    # ---------------------------------------------------------------------
    # Replay with per-step instrumentation.
    # ---------------------------------------------------------------------
    stage33_expected = {
        int(r["epoch"]): r
        for r in pf["union_rows"]
    }

    track_repro_rows = []
    step_rows = []

    max_adam_pred_rel_gap = 0.0

    def compare_if_tracking_epoch(epoch: int):
        if epoch not in stage33_expected:
            return

        actual = exact_metrics(union)
        expected = stage33_expected[epoch]

        diffs = {
            "relative_l2_error":
                abs(
                    actual["relative_l2_error"]
                    - float(
                        expected[
                            "relative_l2_error"
                        ]
                    )
                ),

            "scaled_vpinn_loss":
                abs(
                    actual["scaled_vpinn_loss"]
                    - float(
                        expected[
                            "scaled_vpinn_loss"
                        ]
                    )
                ),

            "residual_l2_norm":
                abs(
                    actual["residual_l2_norm"]
                    - float(
                        expected[
                            "residual_l2_norm"
                        ]
                    )
                ),

            "target_share":
                abs(
                    actual[
                        "target_mode_residual_energy_share"
                    ]
                    - float(
                        expected[
                            "target_mode_residual_energy_share"
                        ]
                    )
                ),

            "target_abs_residual":
                abs(
                    actual[
                        "target_template_abs_residual"
                    ]
                    - float(
                        expected[
                            "target_template_abs_residual"
                        ]
                    )
                ),
        }

        gap = max(diffs.values())

        track_repro_rows.append(
            {
                "epoch":
                    epoch,

                "max_abs_difference":
                    gap,

                "pass":
                    bool(gap <= REPLAY_TOL),

                **{
                    f"gap_{k}": v
                    for k, v in diffs.items()
                },
            }
        )

        if gap > REPLAY_TOL:
            raise RuntimeError(
                f"Stage33 union replay failed at epoch {epoch}: "
                f"gap={gap:.3e}, diffs={diffs}"
            )

    compare_if_tracking_epoch(
        START_EPOCH
    )

    # Keep a small bank of exact pre-step snapshots for the largest-loss step
    # in each fixed window.
    best_event = {
        name: None
        for name, _, _ in WINDOWS
    }

    for epoch in range(
        START_EPOCH,
        FINAL_EPOCH,
    ):

        pre = exact_metrics(union)

        flat_before = flatten_params(
            union.model
        ).clone()

        union.optimizer.zero_grad(
            set_to_none=True
        )

        loss = union.loss_tensor()
        loss.backward()

        g = flatten_grads(
            union.model
        ).detach().clone()

        d_curr, d_hist, d = (
            predict_adam_components(
                union,
                g,
            )
        )

        slope_curr = float(
            torch.dot(g, d_curr).item()
        )
        slope_hist = float(
            torch.dot(g, d_hist).item()
        )
        slope_total = float(
            torch.dot(g, d).item()
        )

        state_before = {
            "model":
                copy.deepcopy(
                    union.model.state_dict()
                ),

            "optimizer":
                copy.deepcopy(
                    union.optimizer.state_dict()
                ),
        }

        union.optimizer.step()

        flat_after = flatten_params(
            union.model
        ).clone()

        actual_delta = (
            flat_after - flat_before
        )

        rel_pred_gap = float(
            torch.linalg.vector_norm(
                actual_delta - d
            ).item()
            /
            max(
                float(
                    torch.linalg.vector_norm(
                        actual_delta
                    ).item()
                ),
                1.0e-300,
            )
        )

        max_adam_pred_rel_gap = max(
            max_adam_pred_rel_gap,
            rel_pred_gap,
        )

        if rel_pred_gap > ADAM_PRED_REL_TOL:
            raise RuntimeError(
                f"Adam prediction mismatch epoch={epoch}: "
                f"{rel_pred_gap:.3e}"
            )

        post = exact_metrics(union)

        delta_loss = (
            post["scaled_vpinn_loss"]
            - pre["scaled_vpinn_loss"]
        )

        delta_rel = (
            post["relative_l2_error"]
            - pre["relative_l2_error"]
        )

        row = {
            "step_from_epoch":
                epoch,

            "step_to_epoch":
                epoch + 1,

            "loss_before":
                pre["scaled_vpinn_loss"],

            "loss_after":
                post["scaled_vpinn_loss"],

            "delta_loss":
                delta_loss,

            "relL2_before":
                pre["relative_l2_error"],

            "relL2_after":
                post["relative_l2_error"],

            "delta_relL2":
                delta_rel,

            "grad_norm":
                float(
                    torch.linalg.vector_norm(g).item()
                ),

            "delta_candidate_norm":
                float(
                    torch.linalg.vector_norm(d).item()
                ),

            "g_dot_delta_current":
                slope_curr,

            "g_dot_delta_history":
                slope_hist,

            "g_dot_delta_candidate":
                slope_total,

            "adam_prediction_relative_gap":
                rel_pred_gap,
        }

        step_rows.append(row)

        for name, start, end in WINDOWS:
            if start <= epoch < end:
                if delta_loss > 0.0:
                    current_best = best_event[name]

                    if (
                        current_best is None
                        or
                        delta_loss
                        >
                        current_best["row"]["delta_loss"]
                    ):
                        best_event[name] = {
                            "row":
                                copy.deepcopy(row),

                            "state":
                                state_before,

                            "direction":
                                d.detach().clone(),

                            "g":
                                g.detach().clone(),
                        }

        if (
            (epoch + 1)
            % TRACK_INTERVAL
            == 0
        ):
            compare_if_tracking_epoch(
                epoch + 1
            )

    # ---------------------------------------------------------------------
    # A1/A2.
    # ---------------------------------------------------------------------
    A1 = bool(
        len(track_repro_rows) == 21
        and
        all(
            bool(r["pass"])
            for r in track_repro_rows
        )
        and
        max_adam_pred_rel_gap
        <= ADAM_PRED_REL_TOL
    )

    A2 = bool(
        all(
            best_event[name] is not None
            for name, _, _ in WINDOWS
        )
    )

    if not A2:
        raise RuntimeError(
            "At least one precommitted spike window contained no positive "
            "one-step weak-loss increase."
        )

    # ---------------------------------------------------------------------
    # Detailed event audits.
    # ---------------------------------------------------------------------
    event_rows = []
    line_rows = []

    for name, _, _ in WINDOWS:
        event = best_event[name]
        row = event["row"]
        state = event["state"]
        direction = event["direction"]

        stage33.restore_state(
            union,
            state,
        )

        base_flat = flatten_params(
            union.model
        ).clone()

        base_metrics = exact_metrics(
            union
        )

        scan = line_scan(
            union,
            base_flat=base_flat,
            direction=direction,
            base_loss=base_metrics[
                "scaled_vpinn_loss"
            ],
        )

        for sr in scan:
            line_rows.append(
                {
                    "window":
                        name,

                    "step_from_epoch":
                        row["step_from_epoch"],

                    **sr,
                }
            )

        smaller_safe = any(
            (
                float(sr["alpha"]) < 1.0
                and
                float(sr["delta_loss"]) < 0.0
            )
            for sr in scan
        )

        curvature = event_curvature(
            union,
            direction,
        )

        history_flip = bool(
            row[
                "g_dot_delta_candidate"
            ]
            >= 0.0
            and
            row[
                "g_dot_delta_current"
            ]
            < 0.0
            and
            row[
                "g_dot_delta_history"
            ]
            >
            abs(
                row[
                    "g_dot_delta_current"
                ]
            )
        )

        curvature_overshoot = bool(
            row[
                "g_dot_delta_candidate"
            ]
            < 0.0
            and
            row["delta_loss"] > 0.0
            and
            smaller_safe
        )

        if history_flip:
            mechanism = "HISTORY_FLIP"
        elif curvature_overshoot:
            mechanism = "CURVATURE_OVERSHOOT"
        else:
            mechanism = "MIXED_OR_OTHER"

        event_rows.append(
            {
                "window":
                    name,

                **row,

                "smaller_alpha_safe":
                    smaller_safe,

                **curvature,

                "history_flip":
                    history_flip,

                "curvature_overshoot":
                    curvature_overshoot,

                "mechanism_class":
                    mechanism,
            }
        )

    classes = [
        r["mechanism_class"]
        for r in event_rows
    ]

    consistent_class = (
        classes[0]
        if len(set(classes)) == 1
        else "MIXED_OR_OTHER"
    )

    A3 = bool(
        len(set(classes)) == 1
        and
        consistent_class
        in (
            "HISTORY_FLIP",
            "CURVATURE_OVERSHOOT",
        )
    )

    if A1 and A2 and A3:
        if consistent_class == "HISTORY_FLIP":
            route_class = (
                "late_union_instability_consistently_driven_by_Adam_history_first_order_direction_reversal"
            )
        else:
            route_class = (
                "late_union_instability_consistently_driven_by_finite_step_curvature_overshoot"
            )
    else:
        route_class = (
            "late_union_instability_is_mixed_or_not_cleanly_localized"
        )

    next_route = (
        "stage35R_current_literature_novelty_claim_audit"
    )

    decision = {
        "A1_exact_stage33_union_replay":
            A1,

        "max_adam_prediction_relative_gap":
            max_adam_pred_rel_gap,

        "A2_both_spike_windows_have_one_step_loss_spike":
            A2,

        "selected_event_mechanisms":
            {
                r["window"]:
                    r["mechanism_class"]
                for r in event_rows
            },

        "A3_consistent_instability_mechanism":
            A3,

        "consistent_mechanism_class":
            consistent_class,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "This is a one-trajectory diagnostic of why Stage-33 U5 failed. "
            "It does not authorize another optimizer sweep and does not "
            "generalize the identified instability class beyond this union "
            "trajectory."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    write_csv(
        out_dir / "stage33_union_replay_checks.csv",
        track_repro_rows,
    )

    write_csv(
        out_dir / "per_step_adam_geometry.csv",
        step_rows,
    )

    write_csv(
        out_dir / "selected_spike_event_audits.csv",
        event_rows,
    )

    write_csv(
        out_dir / "selected_spike_line_scans.csv",
        line_rows,
    )

    # ---------------------------------------------------------------------
    # Console.
    # ---------------------------------------------------------------------
    lines = []

    lines.append("=" * 184)
    lines.append(
        "VPINN — STAGE 34R UNION / ADAM RESPONSE SUMMARY"
    )
    lines.append("=" * 184)

    lines.append(
        f"A1 exact Stage33 union replay         : "
        f"{len(track_repro_rows)}/21 -> {A1}"
    )

    lines.append(
        f"max Adam prediction relative gap      : "
        f"{max_adam_pred_rel_gap:.3e}"
    )

    lines.append("-" * 184)

    for r in event_rows:
        lines.append(
            f"{r['window']} step "
            f"{int(r['step_from_epoch'])}->{int(r['step_to_epoch'])}: "
            f"dL={float(r['delta_loss']):+.6e}, "
            f"dRel={float(r['delta_relL2']):+.6e}, "
            f"g.d_cur={float(r['g_dot_delta_current']):+.6e}, "
            f"g.d_hist={float(r['g_dot_delta_history']):+.6e}, "
            f"g.d={float(r['g_dot_delta_candidate']):+.6e}, "
            f"kappa={float(r['kappa_total']):+.6e}, "
            f"GN={float(r['kappa_GN']):+.6e}, "
            f"NL={float(r['kappa_NL']):+.6e}, "
            f"class={r['mechanism_class']}"
        )

    lines.append("-" * 184)

    lines.append(
        f"A2 both windows contain true one-step spike: {A2}"
    )

    lines.append(
        f"A3 consistent mechanism               : "
        f"{consistent_class} -> {A3}"
    )

    lines.append(
        f"route class                            : "
        f"{route_class}"
    )

    lines.append(
        f"next route                             : "
        f"{next_route}"
    )

    lines.append("=" * 184)

    lines.append(
        "Guardrail: Stage33 U5 remains FAIL; Stage34 is diagnostic only."
    )

    lines.append("=" * 184)

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
