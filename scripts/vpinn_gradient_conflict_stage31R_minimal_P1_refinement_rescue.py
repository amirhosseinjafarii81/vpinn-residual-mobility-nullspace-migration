#!/usr/bin/env python3
"""
VPINN Gradient Geometry — Stage 31R
Matched-State Minimal P1 Mesh-Refinement Rescue
================================================

Status entering Stage 31R
-------------------------
Stage 30R established for the two Stage-29 b=2 nonescaping controls
(seeds 25 and 27):

    N1 exact endpoint replay                  PASS 2/2
    N2 structural blind direction             PASS
    N3 weakly solved but L2 unresolved        PASS 2/2
    N4 endpoint error weak-test invisible     PASS 2/2
    N5 exact blind-mode L2 dominance          FAIL 1/2

So the failure is a DISTRIBUTED weak-test nullspace / near-nullspace floor,
not a single-mode phenomenon.

Stage 29 S4 remains FAIL. Stage 31 does not rewrite that history.

Why minimal mesh refinement?
----------------------------
The original test space is continuous P1 H0^1 on 25 uniform elements
(24 interior tests). Its exact sine-blind family includes k=25,50,75,...

A one-direction enrichment is not justified because seed 27 has distributed
error around k=23,25,27,...

Before any new continuation, Stage 31 performs a READ-ONLY candidate screen
over uniform P1 meshes with

    n_elements in {26,27,28,29,30}.

For each candidate and each Stage-30 endpoint sine spectrum, compute the
energy-weighted visibility proxy

    V = sum_k E_k^a * capture_k^2 / sum_k E_k^a,
    E_k^a = 0.5 * lambda_k * c_k^2,

for k=1..80.

A candidate is admissible only if:

    * none of the union of the top-3 L2 error modes of seeds 25 and 27
      is structurally blind (capture<=1e-10);
    * proxy visibility >=0.30 for BOTH endpoints.

Choose the SMALLEST admissible mesh.

The expected precommitted result from the Stage-30 read-only evidence is
n_elements=26. If the screen does not select 26, abort before training.

Matched-state causal branch
---------------------------
Each seed is deterministically replayed under the ORIGINAL 25-element P1
test space to epoch 3000.

From that exact same model AND Adam state create two branches:

    CONTROL-25:
        continue the original Stage-29 test space.

    REFINE-26:
        replace only the weak-test space by continuous P1 H0^1 on
        26 uniform elements (25 interior tests).

The PDE, exact solution, forcing, network, parameters, Adam moments and Adam
step counter are identical at branching.

Important:
The manufactured amplitudes are NOT recalibrated after refinement.
The physical PDE problem is fixed. Only the test space changes.

Loss-scale control
------------------
The original loss is

    L25 = (1/24) sum_{j=1}^{24} R_j^2.

The refined branch uses 25 test functions but preserves the old denominator:

    L26 = (1/24) sum_{j=1}^{25} R_j^2.

Thus the intervention does not silently reduce the old gradient scale by
switching from mean-24 to mean-25.

Endpoint visibility rescue
--------------------------
At the exact epoch-3000 branch state compute, with the SAME physical error:

    seen25 = ||r25||^2 / ||e||_a^2
    seen26 = ||r26||^2 / ||e||_a^2

using a common high-order independent energy quadrature for ||e||_a^2.

Precommitted endpoint rescue:

    seen26 / seen25 >= 100
    AND seen26 >= 0.05

for BOTH seeds.

Continuation
------------
Continue BOTH branches unconditionally from epoch 3000 through epoch 4000.

No early stop.
No backtracking.
No adaptive learning rate.
No optimizer reset.

Track every 25 epochs.

Branch-specific certified escape keeps the inherited physical criterion:

    relL2 <= 1e-2
    AND target-template residual share <= 0.20

for THREE consecutive 25-grid observations.

Because the target template is defined from the weak response of sin(9*pi*x)
inside each test space, it remains basis invariant within each branch.

Trajectory burden
-----------------
For each branch compute trapezoidal AUC of relL2 over epochs 3000..4000.

The causal rescue is considered trajectory-level supported only if, for BOTH
seeds:

    AUC_refined < AUC_control
    AND relL2_refined(4000) < relL2_control(4000).

Primary gates
-------------

M1 — EXACT STAGE-29 ENDPOINT REPLAY
    2/2 epoch-3000 replays <=1e-10.

M2 — READ-ONLY / EXACT VISIBILITY RESCUE
    candidate screen selects 26 elements;
    exact endpoint seen26/seen25 >=100 and seen26>=0.05 in 2/2.

M3 — REFINED BRANCH ESCAPE
    REFINE-26 certifies escape by epoch 4000 in 2/2.

M4 — MATCHED TRAJECTORY IMPROVEMENT
    AUC_refined < AUC_control
    AND final relL2_refined < final relL2_control
    in 2/2.

MINIMAL P1 REFINEMENT CAUSAL RESCUE:
    M1 & M2 & M3 & M4.

Interpretation if PASS
----------------------
The Stage-29 b=2 nonescape was not a deep mobility lock. A one-element
uniform P1 refinement, applied from the exact same network/Adam state,
increases visibility of the existing error and causally removes the
finite-test-space error floor.

This still does NOT retroactively turn Stage-29 S4 into PASS.

Next if PASS
------------
Stop benchmark expansion.

Stage 32R =
    current-literature / novelty / claim audit,
then choose at most ONE architecture robustness control only if the
literature audit shows it is needed for a defensible paper claim.

If visibility rescue passes but M3/M4 fail:
    do not refine further blindly;
    inspect optimizer-state response to the newly visible residual.

Guardrail
---------
This is a matched-state test-space intervention on two previously censored
controls. It identifies the cause of that control failure; it is not a new
general theorem about P1 spaces.
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
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss


SEEDS = (25, 27)
BASE_MODE = 2
TARGET_MODE = 9

BRANCH_EPOCH = 3000
FINAL_EPOCH = 4000
TRACK_INTERVAL = 25
CERTIFY_POINTS = 3

OLD_ELEMENTS = 25
REFINED_ELEMENTS = 26
GL_PER_ELEMENT = 10
OLD_LOSS_DENOMINATOR = 24.0

CANDIDATE_ELEMENTS = (26, 27, 28, 29, 30)
SCREEN_K_MAX = 80
STRUCTURAL_BLIND_TOL = 1.0e-10
SCREEN_VISIBILITY_MIN = 0.30

REPLAY_TOL = 1.0e-10
VISIBILITY_GAIN_MIN = 100.0
REFINED_SEEN_FRACTION_MIN = 0.05

CONVERGENCE_REL_L2 = 1.0e-2
CONVERGENCE_TARGET_SHARE = 0.20

COMMON_ENERGY_ELEMENTS = 50
COMMON_ENERGY_GL = 12
FINAL_SPECTRUM_K_MAX = 80
FINAL_DENSE_N = 16001


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-31R matched-state minimal P1 mesh-refinement rescue."
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
        "--stage30-script",
        default="vpinn_gradient_conflict_stage30R_p1_nullspace_localization.py",
    )

    p.add_argument(
        "--stage30-dir",
        default="vpinn_gradient_conflict_stage30R_p1_nullspace_localization",
    )

    p.add_argument(
        "--output-dir",
        default="vpinn_gradient_conflict_stage31R_minimal_P1_refinement_rescue",
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
# Provenance
# =============================================================================

def preflight(
    stage3_script: Path,
    stage29_script: Path,
    stage29_dir: Path,
    stage30_script: Path,
    stage30_dir: Path,
) -> dict:

    s29_manifest_path = stage29_dir / "manifest.json"
    s29_tracking_path = stage29_dir / "tracking_metrics.csv"

    s30_manifest_path = stage30_dir / "manifest.json"
    s30_decision_path = stage30_dir / "decision.json"
    s30_spec_path = stage30_dir / "endpoint_error_sine_spectrum.csv"
    s30_endpoint_path = stage30_dir / "endpoint_nullspace_summary.csv"

    for path in (
        s29_manifest_path,
        s29_tracking_path,
        s30_manifest_path,
        s30_decision_path,
        s30_spec_path,
        s30_endpoint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    s29m = read_json(s29_manifest_path)
    s30m = read_json(s30_manifest_path)
    s30d = read_json(s30_decision_path)

    s3 = sha256_file(stage3_script)
    s29 = sha256_file(stage29_script)
    s30 = sha256_file(stage30_script)

    if s29m.get("stage3_solver_sha256") != s3:
        raise RuntimeError("Stage-3 SHA mismatch against Stage 29.")

    if s29m.get("stage29r_script_sha256") != s29:
        raise RuntimeError("Stage-29 SHA mismatch.")

    if s30m.get("stage29_script_sha256") != s29:
        raise RuntimeError("Stage-29 SHA mismatch against Stage 30.")

    if s30m.get("stage30r_script_sha256") != s30:
        raise RuntimeError("Stage-30 SHA mismatch.")

    if not bool(
        s30d.get("weak_test_nullspace_localization_supported", False)
    ):
        raise RuntimeError(
            "Stage 30 did not establish weak-test nullspace localization."
        )

    if bool(
        s30d.get(
            "sharp_structural_blind_mode_localization_supported",
            True,
        )
    ):
        raise RuntimeError(
            "Stage 30 unexpectedly supported a one-mode sharp localization."
        )

    if s30d.get("next_route") != (
        "stage31R_minimal_P1_mesh_refinement_control"
    ):
        raise RuntimeError("Unexpected Stage-30 next route.")

    tracking = read_csv(s29_tracking_path)
    expected_endpoint = {}

    for row in tracking:
        seed = int(row["seed"])
        b = int(row["base_mode"])
        epoch = int(row["epoch"])

        if (
            seed in SEEDS
            and b == BASE_MODE
            and epoch == BRANCH_EPOCH
        ):
            expected_endpoint[seed] = row

    if set(expected_endpoint) != set(SEEDS):
        raise RuntimeError("Missing Stage-29 branch endpoint rows.")

    spectrum = read_csv(s30_spec_path)
    endpoint = read_csv(s30_endpoint_path)

    spectrum_by_seed = {
        seed: [
            row for row in spectrum
            if int(row["seed"]) == seed
        ]
        for seed in SEEDS
    }

    endpoint_by_seed = {
        int(row["seed"]): row
        for row in endpoint
    }

    if set(endpoint_by_seed) != set(SEEDS):
        raise RuntimeError("Incomplete Stage-30 endpoint summary.")

    return {
        "stage3_sha256": s3,
        "stage29_sha256": s29,
        "stage30_sha256": s30,
        "expected_endpoint": expected_endpoint,
        "spectrum_by_seed": spectrum_by_seed,
        "endpoint_by_seed": endpoint_by_seed,
    }


# =============================================================================
# General P1 test-space experiment
# =============================================================================

class GeneralP1Experiment:
    """
    Same physical reaction-diffusion problem and network, arbitrary uniform
    P1 test mesh.

    The exact-solution amplitudes are supplied explicitly so a test-space
    change does NOT change the PDE/forcing.
    """

    def __init__(
        self,
        stage3,
        cfg,
        device: torch.device,
        n_elements: int,
        base_mode: int,
        target_mode: int,
        base_amplitude: float,
        target_amplitude: float,
        out_dir: Path,
        loss_denominator: float,
    ):
        self.stage3 = stage3
        self.cfg = cfg
        self.device = device
        self.dtype = torch.float64

        self.n_elements = int(n_elements)
        self.n_test = self.n_elements - 1
        self.gl_per_element = GL_PER_ELEMENT

        self.base_mode = int(base_mode)
        self.mode = int(target_mode)

        self.sigma = float(stage3.math.pi if False else (5.0 * math.pi) ** 2)

        self.lambda_base = (
            (self.base_mode * math.pi) ** 2 + self.sigma
        )
        self.lambda_target = (
            (self.mode * math.pi) ** 2 + self.sigma
        )

        self.base_amplitude = float(base_amplitude)
        self.amplitude = float(target_amplitude)
        self.loss_denominator = float(loss_denominator)

        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        stage3.seed_everything(cfg.seed)

        self.model = stage3.VPINNNetwork(
            cfg.width,
            cfg.depth,
        ).to(
            device=device,
            dtype=self.dtype,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
        )

        self._build_test_space()
        self._build_templates_and_forcing()

    def _build_test_space(self):
        xi, wi = leggauss(self.gl_per_element)

        h = 1.0 / self.n_elements

        x_all = []
        w_all = []
        raw_v = []
        raw_d = []

        for e in range(self.n_elements):
            a = e * h
            b = (e + 1) * h

            x_e = (xi + 1.0) * (b - a) / 2.0 + a
            w_e = wi * (b - a) / 2.0

            for q in range(self.gl_per_element):
                xq = float(x_e[q])

                vals = np.zeros(self.n_test, dtype=np.float64)
                ders = np.zeros(self.n_test, dtype=np.float64)

                N_left = (b - xq) / h
                N_right = (xq - a) / h

                left_node = e
                right_node = e + 1

                if 1 <= left_node <= self.n_test:
                    vals[left_node - 1] = N_left
                    ders[left_node - 1] = -1.0 / h

                if 1 <= right_node <= self.n_test:
                    vals[right_node - 1] = N_right
                    ders[right_node - 1] = 1.0 / h

                x_all.append(xq)
                w_all.append(float(w_e[q]))
                raw_v.append(vals)
                raw_d.append(ders)

        x = torch.as_tensor(
            np.asarray(x_all, dtype=np.float64).reshape(-1, 1),
            dtype=self.dtype,
            device=self.device,
        )

        w = torch.as_tensor(
            np.asarray(w_all, dtype=np.float64).reshape(-1, 1),
            dtype=self.dtype,
            device=self.device,
        )

        V = torch.as_tensor(
            np.asarray(raw_v, dtype=np.float64),
            dtype=self.dtype,
            device=self.device,
        )

        D = torch.as_tensor(
            np.asarray(raw_d, dtype=np.float64),
            dtype=self.dtype,
            device=self.device,
        )

        G = (
            D.T @ (w * D)
            + self.sigma * V.T @ (w * V)
        )

        vals, vecs = torch.linalg.eigh(G)

        if float(torch.min(vals).item()) <= 0.0:
            raise RuntimeError("P1 energy Gram is not SPD.")

        inv_sqrt = (
            vecs
            @ torch.diag(torch.rsqrt(vals))
            @ vecs.T
        )

        self.test_values = V @ inv_sqrt
        self.test_derivatives = D @ inv_sqrt

        gram = (
            self.test_derivatives.T
            @ (w * self.test_derivatives)
            + self.sigma
            * self.test_values.T
            @ (w * self.test_values)
        )

        eye = torch.eye(
            self.n_test,
            dtype=self.dtype,
            device=self.device,
        )

        self.gram_error = float(
            torch.max(torch.abs(gram - eye)).item()
        )

        if self.gram_error > 1.0e-10:
            raise RuntimeError(
                f"Refined P1 Gram failed: {self.gram_error:.3e}"
            )

        self.x_quad = x.detach().clone().requires_grad_(True)
        self.w_quad = w

    def unit_mode_response(self, k: int) -> torch.Tensor:
        x = self.x_quad.detach()

        s = torch.sin(k * math.pi * x)
        ds = k * math.pi * torch.cos(k * math.pi * x)

        return torch.sum(
            self.w_quad
            * (
                ds * self.test_derivatives
                + self.sigma * s * self.test_values
            ),
            dim=0,
        )

    def _build_templates_and_forcing(self):
        q_target = self.unit_mode_response(self.mode)
        qtn = torch.linalg.vector_norm(q_target)

        if float(qtn.item()) <= 0.0:
            raise RuntimeError("Degenerate refined target template.")

        self.target_template = (q_target / qtn).detach()

        with torch.no_grad():
            self.forcing_values = self.forcing(
                self.x_quad.detach()
            )

            uex = self.exact_solution(self.x_quad.detach())
            duex = self.exact_derivative(self.x_quad.detach())

            exact_r = torch.sum(
                self.w_quad
                * (
                    duex * self.test_derivatives
                    + self.sigma * uex * self.test_values
                    - self.forcing_values * self.test_values
                ),
                dim=0,
            )

            self.exact_weak_residual_error = float(
                torch.max(torch.abs(exact_r)).item()
            )

        if self.exact_weak_residual_error > 1.0e-10:
            raise RuntimeError(
                f"Refined exact weak residual failed: "
                f"{self.exact_weak_residual_error:.3e}"
            )

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * torch.sin(self.base_mode * math.pi * x)
            + self.amplitude
            * torch.sin(self.mode * math.pi * x)
        )

    def exact_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * self.base_mode
            * math.pi
            * torch.cos(self.base_mode * math.pi * x)
            + self.amplitude
            * self.mode
            * math.pi
            * torch.cos(self.mode * math.pi * x)
        )

    def forcing(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base_amplitude
            * self.lambda_base
            * torch.sin(self.base_mode * math.pi * x)
            + self.amplitude
            * self.lambda_target
            * torch.sin(self.mode * math.pi * x)
        )

    def weak_residuals(self) -> torch.Tensor:
        u = self.model(self.x_quad)

        du = torch.autograd.grad(
            u,
            self.x_quad,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        return torch.sum(
            self.w_quad
            * (
                du * self.test_derivatives
                + self.sigma * u * self.test_values
                - self.forcing_values * self.test_values
            ),
            dim=0,
        )

    @torch.no_grad()
    def relative_l2_error(self) -> float:
        x = torch.linspace(
            0.0,
            1.0,
            self.cfg.n_eval,
            dtype=self.dtype,
            device=self.device,
        ).reshape(-1, 1)

        pred = self.model(x)
        truth = self.exact_solution(x)

        return float(
            (
                torch.linalg.vector_norm(pred - truth)
                /
                torch.linalg.vector_norm(truth)
            ).item()
        )

    def residual_metrics(self) -> Dict[str, float]:
        r = self.weak_residuals().detach()
        e = r.square()
        total = e.sum().clamp_min(1.0e-300)

        target_proj = torch.dot(r, self.target_template)

        return {
            "vpinn_loss_scaled":
                float(
                    (torch.sum(e) / self.loss_denominator).item()
                ),

            "residual_l2_norm":
                float(torch.linalg.vector_norm(r).item()),

            "target_template_abs_residual":
                float(torch.abs(target_proj).item()),

            "target_mode_residual_energy_share":
                float((target_proj.square() / total).item()),
        }

    def train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        r = self.weak_residuals()

        loss = (
            torch.sum(r.square())
            / self.loss_denominator
        )

        loss.backward()
        self.optimizer.step()

        return float(loss.detach().item())


# =============================================================================
# Candidate mesh screen
# =============================================================================

def analytic_capture_for_mesh(
    n_elements: int,
    sigma: float,
    kmax: int,
):
    ntest = n_elements - 1
    xi, wi = leggauss(GL_PER_ELEMENT)
    h = 1.0 / n_elements

    x_all = []
    w_all = []
    V_all = []
    D_all = []

    for e in range(n_elements):
        a = e * h
        b = (e + 1) * h
        xe = (xi + 1.0) * (b - a) / 2.0 + a
        we = wi * (b - a) / 2.0

        for xq, wq in zip(xe, we):
            vals = np.zeros(ntest)
            ders = np.zeros(ntest)

            N_left = (b - xq) / h
            N_right = (xq - a) / h

            if 1 <= e <= ntest:
                vals[e - 1] = N_left
                ders[e - 1] = -1.0 / h

            if 1 <= e + 1 <= ntest:
                vals[e] = N_right
                ders[e] = 1.0 / h

            x_all.append(float(xq))
            w_all.append(float(wq))
            V_all.append(vals)
            D_all.append(ders)

    x = np.asarray(x_all)
    w = np.asarray(w_all)[:, None]
    V = np.asarray(V_all)
    D = np.asarray(D_all)

    G = D.T @ (w * D) + sigma * V.T @ (w * V)

    vals, vecs = np.linalg.eigh(G)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    Vo = V @ inv_sqrt
    Do = D @ inv_sqrt

    capture = {}

    for k in range(1, kmax + 1):
        s = np.sin(k * math.pi * x)[:, None]
        ds = (k * math.pi * np.cos(k * math.pi * x))[:, None]

        q = np.sum(
            w * (ds * Do + sigma * s * Vo),
            axis=0,
        )

        lam = (k * math.pi) ** 2 + sigma

        capture[k] = float(
            np.linalg.norm(q)
            / math.sqrt(lam / 2.0)
        )

    return capture


def select_minimal_refinement(
    stage30_spectrum_by_seed: dict,
    sigma: float,
):
    top3_union = set()

    for seed in SEEDS:
        rows = sorted(
            stage30_spectrum_by_seed[seed],
            key=lambda r: float(r["mode_l2_energy"]),
            reverse=True,
        )[:3]

        top3_union.update(
            int(r["mode"]) for r in rows
        )

    screen_rows = []
    selected = None

    for n_elements in CANDIDATE_ELEMENTS:
        capture = analytic_capture_for_mesh(
            n_elements=n_elements,
            sigma=sigma,
            kmax=SCREEN_K_MAX,
        )

        top3_blind = any(
            capture[k] <= STRUCTURAL_BLIND_TOL
            for k in top3_union
        )

        seed_proxy = {}

        for seed in SEEDS:
            numer = 0.0
            denom = 0.0

            for row in stage30_spectrum_by_seed[seed]:
                k = int(row["mode"])
                c = float(row["sine_coefficient"])
                lam = (k * math.pi) ** 2 + sigma

                energy = 0.5 * lam * c * c

                denom += energy
                numer += energy * capture[k] ** 2

            seed_proxy[seed] = (
                numer / denom if denom > 0.0 else float("nan")
            )

        admissible = bool(
            not top3_blind
            and
            all(
                seed_proxy[seed] >= SCREEN_VISIBILITY_MIN
                for seed in SEEDS
            )
        )

        screen_rows.append(
            {
                "n_elements":
                    n_elements,

                "n_test":
                    n_elements - 1,

                "top3_union_modes":
                    ",".join(
                        str(k)
                        for k in sorted(top3_union)
                    ),

                "any_top3_structurally_blind":
                    top3_blind,

                "seed25_visibility_proxy":
                    seed_proxy[25],

                "seed27_visibility_proxy":
                    seed_proxy[27],

                "min_visibility_proxy":
                    min(seed_proxy.values()),

                "admissible":
                    admissible,
            }
        )

        if admissible and selected is None:
            selected = n_elements

    return selected, screen_rows


# =============================================================================
# Common physical error energy
# =============================================================================

def common_error_energy(exp) -> float:
    xi, wi = leggauss(COMMON_ENERGY_GL)

    x_all = []
    w_all = []

    h = 1.0 / COMMON_ENERGY_ELEMENTS

    for e in range(COMMON_ENERGY_ELEMENTS):
        a = e * h
        b = (e + 1) * h

        xe = (xi + 1.0) * (b - a) / 2.0 + a
        we = wi * (b - a) / 2.0

        x_all.extend(float(v) for v in xe)
        w_all.extend(float(v) for v in we)

    x = torch.as_tensor(
        np.asarray(x_all).reshape(-1, 1),
        dtype=exp.dtype,
        device=exp.device,
    ).requires_grad_(True)

    w = torch.as_tensor(
        np.asarray(w_all).reshape(-1, 1),
        dtype=exp.dtype,
        device=exp.device,
    )

    u = exp.model(x)

    du = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=False,
        retain_graph=False,
    )[0]

    with torch.no_grad():
        e = u.detach() - exp.exact_solution(x.detach())
        de = du.detach() - exp.exact_derivative(x.detach())

        return float(
            torch.sum(
                w * (de.square() + exp.sigma * e.square())
            ).item()
        )


# =============================================================================
# Tracking helpers
# =============================================================================

def capture_training_state(exp):
    return {
        "model": copy.deepcopy(exp.model.state_dict()),
        "optimizer": copy.deepcopy(exp.optimizer.state_dict()),
    }


def restore_training_state(exp, state):
    exp.model.load_state_dict(copy.deepcopy(state["model"]))
    exp.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))


def trapezoid_auc(rows: List[dict], key: str) -> float:
    rows = sorted(rows, key=lambda r: int(r["epoch"]))

    x = np.asarray(
        [float(r["epoch"]) for r in rows],
        dtype=np.float64,
    )

    y = np.asarray(
        [float(r[key]) for r in rows],
        dtype=np.float64,
    )

    return float(np.trapezoid(y, x))


@torch.no_grad()
def sine_error_summary(exp, kmax: int = FINAL_SPECTRUM_K_MAX):
    x = torch.linspace(
        0.0,
        1.0,
        FINAL_DENSE_N,
        dtype=exp.dtype,
        device=exp.device,
    ).reshape(-1, 1)

    e = (
        exp.model(x)
        - exp.exact_solution(x)
    ).reshape(-1).cpu().numpy()

    x_np = x.reshape(-1).cpu().numpy()

    total = float(np.trapezoid(e * e, x_np))

    rows = []

    for k in range(1, kmax + 1):
        s = np.sin(k * math.pi * x_np)

        c = 2.0 * float(
            np.trapezoid(e * s, x_np)
        )

        E = 0.5 * c * c

        rows.append(
            {
                "mode": k,
                "coefficient": c,
                "l2_energy": E,
                "fraction_total_l2_error":
                    E / total if total > 0.0 else float("nan"),
            }
        )

    top = sorted(
        rows,
        key=lambda r: float(r["l2_energy"]),
        reverse=True,
    )[:10]

    return {
        "dense_l2_error_sq": total,
        "top10_modes":
            ",".join(
                f"{int(r['mode'])}:{float(r['fraction_total_l2_error']):.6g}"
                for r in top
            ),
        "mode25_fraction":
            next(
                float(r["fraction_total_l2_error"])
                for r in rows
                if int(r["mode"]) == 25
            ),
    }


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
    stage30_script = resolve(args.stage30_script)
    stage30_dir = resolve(args.stage30_dir)
    out_dir = resolve(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    pf = preflight(
        stage3_script=stage3_script,
        stage29_script=stage29_script,
        stage29_dir=stage29_dir,
        stage30_script=stage30_script,
        stage30_dir=stage30_dir,
    )

    stage3 = load_module(
        stage3_script,
        "vpinn_stage3_stage31R",
    )

    stage29 = load_module(
        stage29_script,
        "vpinn_stage29_stage31R",
    )

    sigma = float(stage29.SIGMA)

    selected_elements, screen_rows = select_minimal_refinement(
        stage30_spectrum_by_seed=pf["spectrum_by_seed"],
        sigma=sigma,
    )

    if selected_elements != REFINED_ELEMENTS:
        raise RuntimeError(
            f"Read-only candidate screen selected {selected_elements}; "
            f"precommitted Stage-31 design expected {REFINED_ELEMENTS}."
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

        "stage30_script_sha256":
            pf["stage30_sha256"],

        "stage31r_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "precommitment": {
            "stage":
                "matched_state_minimal_P1_mesh_refinement_rescue",

            "seeds":
                list(SEEDS),

            "branch_epoch":
                BRANCH_EPOCH,

            "final_epoch":
                FINAL_EPOCH,

            "candidate_elements":
                list(CANDIDATE_ELEMENTS),

            "screen_visibility_min":
                SCREEN_VISIBILITY_MIN,

            "selected_elements":
                selected_elements,

            "old_elements":
                OLD_ELEMENTS,

            "loss_denominator_both_branches":
                OLD_LOSS_DENOMINATOR,

            "visibility_gain_min":
                VISIBILITY_GAIN_MIN,

            "refined_seen_fraction_min":
                REFINED_SEEN_FRACTION_MIN,

            "M1":
                "2/2 exact epoch-3000 Stage-29 replay",

            "M2":
                "screen selects 26; exact seen gain>=100 and refined seen>=0.05 in 2/2",

            "M3":
                "refined escape by 4000 in 2/2",

            "M4":
                "refined AUC and final relL2 both lower than control in 2/2",

            "optimizer_reset":
                False,

            "adaptive_steps":
                False,
        },
    }

    write_json(out_dir / "manifest.json", manifest)

    print("=" * 184)
    print(
        "VPINN — STAGE 31R MATCHED-STATE MINIMAL P1 MESH-REFINEMENT RESCUE"
    )
    print("=" * 184)
    print(f"device                    : {device}")
    print(f"seeds                     : {list(SEEDS)}")
    print(f"branch epoch              : {BRANCH_EPOCH}")
    print(f"final epoch               : {FINAL_EPOCH}")
    print(f"old mesh                  : {OLD_ELEMENTS} elements / 24 tests")
    print(f"selected refined mesh     : {selected_elements} elements / {selected_elements-1} tests")
    print("optimizer reset           : NONE")
    print("adaptive intervention     : NONE")
    print("=" * 184)

    write_csv(
        out_dir / "read_only_mesh_candidate_screen.csv",
        screen_rows,
    )

    replay_rows = []
    visibility_rows = []
    tracking_rows = []
    branch_rows = []
    final_spectrum_rows = []

    for seed in SEEDS:

        seed_dir = out_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # Exact reconstruction under original Stage-29 test space.
        # ---------------------------------------------------------------------
        cfg25 = stage29.make_config(
            stage3=stage3,
            seed=seed,
            device=device,
            out_dir=seed_dir / "replay25",
        )

        exp25 = stage29.P1ReactionDiffusionExperiment(
            stage3=stage3,
            cfg=cfg25,
            device=device,
            base_mode=BASE_MODE,
            target_mode=TARGET_MODE,
            out_dir=seed_dir / "replay25",
        )

        for _ in range(BRANCH_EPOCH):
            exp25.train_step()

        rm25 = exp25.residual_metrics()
        rel25 = exp25.relative_l2_error()

        expected = pf["expected_endpoint"][seed]

        diffs = {
            "relative_l2_error":
                abs(
                    rel25
                    - float(expected["relative_l2_error"])
                ),

            "vpinn_loss":
                abs(
                    float(rm25["vpinn_loss"])
                    - float(expected["vpinn_loss"])
                ),

            "residual_l2_norm":
                abs(
                    float(rm25["residual_l2_norm"])
                    - float(expected["residual_l2_norm"])
                ),

            "target_share":
                abs(
                    float(
                        rm25["target_mode_residual_energy_share"]
                    )
                    - float(
                        expected["target_mode_residual_energy_share"]
                    )
                ),

            "target_abs_residual":
                abs(
                    float(
                        rm25["target_template_abs_residual"]
                    )
                    - float(
                        expected["target_template_abs_residual"]
                    )
                ),
        }

        replay_gap = max(diffs.values())

        if replay_gap > REPLAY_TOL:
            raise RuntimeError(
                f"Stage-29 replay failed seed={seed}: "
                f"gap={replay_gap:.3e}, diffs={diffs}"
            )

        replay_rows.append(
            {
                "seed": seed,
                "max_abs_difference": replay_gap,
                "pass": True,
                **{f"gap_{k}": v for k, v in diffs.items()},
            }
        )

        branch_state = capture_training_state(exp25)

        # ---------------------------------------------------------------------
        # Build matched control and refined branches.
        # ---------------------------------------------------------------------
        control = stage29.P1ReactionDiffusionExperiment(
            stage3=stage3,
            cfg=cfg25,
            device=device,
            base_mode=BASE_MODE,
            target_mode=TARGET_MODE,
            out_dir=seed_dir / "control25",
        )

        restore_training_state(control, branch_state)

        refined = GeneralP1Experiment(
            stage3=stage3,
            cfg=cfg25,
            device=device,
            n_elements=selected_elements,
            base_mode=BASE_MODE,
            target_mode=TARGET_MODE,
            base_amplitude=float(exp25.base_amplitude),
            target_amplitude=float(exp25.amplitude),
            out_dir=seed_dir / "refined26",
            loss_denominator=OLD_LOSS_DENOMINATOR,
        )

        restore_training_state(refined, branch_state)

        # ---------------------------------------------------------------------
        # Exact endpoint visibility gain before any continuation.
        # ---------------------------------------------------------------------
        common_energy = common_error_energy(control)

        r25 = control.weak_residuals().detach()
        r26 = refined.weak_residuals().detach()

        seen25 = float(torch.sum(r25.square()).item()) / common_energy
        seen26 = float(torch.sum(r26.square()).item()) / common_energy

        gain = seen26 / max(seen25, 1.0e-300)

        visibility_rows.append(
            {
                "seed":
                    seed,

                "common_error_energy_sq":
                    common_energy,

                "seen25_fraction":
                    seen25,

                "seen26_fraction":
                    seen26,

                "visibility_gain_26_over_25":
                    gain,

                "gain_ge_100":
                    bool(gain >= VISIBILITY_GAIN_MIN),

                "seen26_ge_005":
                    bool(seen26 >= REFINED_SEEN_FRACTION_MIN),

                "pass":
                    bool(
                        gain >= VISIBILITY_GAIN_MIN
                        and
                        seen26 >= REFINED_SEEN_FRACTION_MIN
                    ),
            }
        )

        print()
        print(
            f"seed={seed}: replay={replay_gap:.3e}, "
            f"seen25={seen25:.6e}, "
            f"seen26={seen26:.6e}, "
            f"gain={gain:.3e}"
        )

        # ---------------------------------------------------------------------
        # Matched unconditional continuation.
        # ---------------------------------------------------------------------
        branch_exps = {
            "CONTROL25": control,
            "REFINED26": refined,
        }

        branch_tracking = {
            "CONTROL25": [],
            "REFINED26": [],
        }

        escape_streak = {
            "CONTROL25": 0,
            "REFINED26": 0,
        }

        escape_candidate = {
            "CONTROL25": None,
            "REFINED26": None,
        }

        escape_onset = {
            "CONTROL25": -1,
            "REFINED26": -1,
        }

        escape_confirm = {
            "CONTROL25": -1,
            "REFINED26": -1,
        }

        for epoch in range(BRANCH_EPOCH, FINAL_EPOCH + 1):

            if epoch % TRACK_INTERVAL == 0:

                for branch_name, exp in branch_exps.items():
                    rm = exp.residual_metrics()
                    rel = exp.relative_l2_error()

                    # Stage-29 control names differ only in the loss key.
                    loss_value = (
                        float(rm["vpinn_loss"])
                        if "vpinn_loss" in rm
                        else float(rm["vpinn_loss_scaled"])
                    )

                    target_share = float(
                        rm["target_mode_residual_energy_share"]
                    )

                    qualifies = bool(
                        rel <= CONVERGENCE_REL_L2
                        and
                        target_share <= CONVERGENCE_TARGET_SHARE
                    )

                    if escape_onset[branch_name] < 0:
                        if qualifies:
                            if escape_streak[branch_name] == 0:
                                escape_candidate[branch_name] = epoch

                            escape_streak[branch_name] += 1
                        else:
                            escape_streak[branch_name] = 0
                            escape_candidate[branch_name] = None

                        if escape_streak[branch_name] >= CERTIFY_POINTS:
                            escape_onset[branch_name] = int(
                                escape_candidate[branch_name]
                            )
                            escape_confirm[branch_name] = epoch

                    row = {
                        "seed":
                            seed,

                        "branch":
                            branch_name,

                        "epoch":
                            epoch,

                        "relative_l2_error":
                            rel,

                        "scaled_vpinn_loss":
                            loss_value,

                        "residual_l2_norm":
                            float(rm["residual_l2_norm"]),

                        "target_mode_residual_energy_share":
                            target_share,

                        "target_template_abs_residual":
                            float(
                                rm["target_template_abs_residual"]
                            ),

                        "escape_streak":
                            escape_streak[branch_name],

                        "certified_escape_onset_so_far":
                            escape_onset[branch_name],
                    }

                    tracking_rows.append(row)
                    branch_tracking[branch_name].append(row)

            if epoch < FINAL_EPOCH:
                control.train_step()
                refined.train_step()

        # ---------------------------------------------------------------------
        # Branch summaries.
        # ---------------------------------------------------------------------
        per_branch = {}

        for branch_name, exp in branch_exps.items():
            rows = branch_tracking[branch_name]

            auc = trapezoid_auc(
                rows,
                "relative_l2_error",
            )

            final_row = rows[-1]
            spectrum = sine_error_summary(exp)

            per_branch[branch_name] = {
                "auc_relL2":
                    auc,

                "final_relL2":
                    float(final_row["relative_l2_error"]),

                "escape_onset":
                    escape_onset[branch_name],

                "escape_confirm":
                    escape_confirm[branch_name],

                **spectrum,
            }

            final_spectrum_rows.append(
                {
                    "seed":
                        seed,

                    "branch":
                        branch_name,

                    **spectrum,
                }
            )

        auc_improved = bool(
            per_branch["REFINED26"]["auc_relL2"]
            <
            per_branch["CONTROL25"]["auc_relL2"]
        )

        final_improved = bool(
            per_branch["REFINED26"]["final_relL2"]
            <
            per_branch["CONTROL25"]["final_relL2"]
        )

        refined_escape = bool(
            per_branch["REFINED26"]["escape_onset"] >= 0
        )

        branch_rows.append(
            {
                "seed":
                    seed,

                "control_escape_onset":
                    per_branch["CONTROL25"]["escape_onset"],

                "refined_escape_onset":
                    per_branch["REFINED26"]["escape_onset"],

                "control_auc_relL2":
                    per_branch["CONTROL25"]["auc_relL2"],

                "refined_auc_relL2":
                    per_branch["REFINED26"]["auc_relL2"],

                "refined_auc_lower":
                    auc_improved,

                "control_final_relL2":
                    per_branch["CONTROL25"]["final_relL2"],

                "refined_final_relL2":
                    per_branch["REFINED26"]["final_relL2"],

                "refined_final_relL2_lower":
                    final_improved,

                "refined_certified_escape":
                    refined_escape,

                "control_final_mode25_fraction":
                    per_branch["CONTROL25"]["mode25_fraction"],

                "refined_final_mode25_fraction":
                    per_branch["REFINED26"]["mode25_fraction"],

                "trajectory_improvement_pass":
                    bool(
                        auc_improved
                        and final_improved
                    ),
            }
        )

        print(
            f"  CONTROL25: escape={per_branch['CONTROL25']['escape_onset']}, "
            f"AUC={per_branch['CONTROL25']['auc_relL2']:.6e}, "
            f"final={per_branch['CONTROL25']['final_relL2']:.6e}"
        )

        print(
            f"  REFINED26: escape={per_branch['REFINED26']['escape_onset']}, "
            f"AUC={per_branch['REFINED26']['auc_relL2']:.6e}, "
            f"final={per_branch['REFINED26']['final_relL2']:.6e}"
        )

    # =========================================================================
    # Persist
    # =========================================================================
    write_csv(
        out_dir / "endpoint_replay_checks.csv",
        replay_rows,
    )

    write_csv(
        out_dir / "endpoint_visibility_rescue.csv",
        visibility_rows,
    )

    write_csv(
        out_dir / "matched_branch_tracking.csv",
        tracking_rows,
    )

    write_csv(
        out_dir / "matched_branch_summary.csv",
        branch_rows,
    )

    write_csv(
        out_dir / "final_error_spectrum_summary.csv",
        final_spectrum_rows,
    )

    # =========================================================================
    # Gates
    # =========================================================================
    M1 = bool(
        len(replay_rows) == 2
        and all(bool(r["pass"]) for r in replay_rows)
    )

    M2 = bool(
        selected_elements == REFINED_ELEMENTS
        and
        len(visibility_rows) == 2
        and
        all(bool(r["pass"]) for r in visibility_rows)
    )

    refined_escape_count = sum(
        int(bool(r["refined_certified_escape"]))
        for r in branch_rows
    )

    M3 = bool(
        refined_escape_count == 2
    )

    trajectory_improvement_count = sum(
        int(bool(r["trajectory_improvement_pass"]))
        for r in branch_rows
    )

    M4 = bool(
        trajectory_improvement_count == 2
    )

    rescue = bool(
        M1 and M2 and M3 and M4
    )

    if rescue:
        route_class = (
            "minimal_uniform_P1_refinement_causally_rescues_stage29_control_error_floor"
        )

        next_route = (
            "stage32R_current_literature_novelty_claim_audit_then_at_most_one_architecture_control"
        )

    elif M1 and M2 and not (M3 and M4):
        route_class = (
            "refinement_restores_visibility_but_does_not_cleanly_rescue_training"
        )

        next_route = (
            "stage32R_refined_visibility_optimizer_state_response_audit"
        )

    else:
        route_class = (
            "minimal_refinement_nullspace_rescue_hypothesis_not_supported"
        )

        next_route = (
            "stage32R_stop_refinement_and_reassess_control_failure"
        )

    decision = {
        "candidate_elements":
            list(CANDIDATE_ELEMENTS),

        "selected_elements":
            selected_elements,

        "M1_exact_stage29_endpoint_replay":
            M1,

        "M2_exact_visibility_rescue":
            M2,

        "refined_escape_count":
            refined_escape_count,

        "M3_refined_branch_escape":
            M3,

        "trajectory_improvement_count":
            trajectory_improvement_count,

        "M4_matched_trajectory_improvement":
            M4,

        "minimal_P1_refinement_causal_rescue":
            rescue,

        "route_class":
            route_class,

        "next_route":
            next_route,

        "interpretation_guardrail": (
            "A PASS causally explains the two Stage-29 b=2 nonescapes as a "
            "finite-test-space visibility floor. Stage-29 S4 remains "
            "historically FAIL; the matched refinement is a separate rescue "
            "experiment."
        ),
    }

    write_json(
        out_dir / "decision.json",
        decision,
    )

    # =========================================================================
    # Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10.0, 5.6))

    for seed in SEEDS:
        for branch in ("CONTROL25", "REFINED26"):
            rows = [
                r for r in tracking_rows
                if int(r["seed"]) == seed
                and r["branch"] == branch
            ]

            rows.sort(key=lambda r: int(r["epoch"]))

            ax.plot(
                [int(r["epoch"]) for r in rows],
                [float(r["relative_l2_error"]) for r in rows],
                linewidth=1.1,
                label=f"seed {seed} {branch}",
            )

    ax.axhline(
        CONVERGENCE_REL_L2,
        linestyle="--",
        linewidth=1.0,
        label="relL2 escape threshold",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Matched-state control vs one-element P1 refinement")
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(
        out_dir / "matched_refinement_relL2.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    # =========================================================================
    # Console
    # =========================================================================
    lines = []

    lines.append("=" * 186)
    lines.append(
        "VPINN — STAGE 31R MATCHED-STATE MINIMAL P1 REFINEMENT RESCUE SUMMARY"
    )
    lines.append("=" * 186)

    lines.append(
        f"read-only selected mesh: {selected_elements} elements"
    )

    lines.append(
        "seed | seen25 | seen26 | gain | control escape | refined escape | "
        "control AUC | refined AUC | control final | refined final"
    )

    lines.append("-" * 186)

    vis_map = {
        int(r["seed"]): r
        for r in visibility_rows
    }

    for r in branch_rows:
        seed = int(r["seed"])
        v = vis_map[seed]

        lines.append(
            f"{seed:4d} | "
            f"{float(v['seen25_fraction']):.6e} | "
            f"{float(v['seen26_fraction']):.6e} | "
            f"{float(v['visibility_gain_26_over_25']):.3e} | "
            f"{int(r['control_escape_onset']):14d} | "
            f"{int(r['refined_escape_onset']):14d} | "
            f"{float(r['control_auc_relL2']):.6e} | "
            f"{float(r['refined_auc_relL2']):.6e} | "
            f"{float(r['control_final_relL2']):.6e} | "
            f"{float(r['refined_final_relL2']):.6e}"
        )

    lines.append("-" * 186)

    lines.append(
        f"M1 exact endpoint replay             : "
        f"{sum(int(r['pass']) for r in replay_rows)}/2 -> {M1}"
    )

    lines.append(
        f"M2 exact visibility rescue           : "
        f"{sum(int(r['pass']) for r in visibility_rows)}/2 -> {M2}"
    )

    lines.append(
        f"M3 refined certified escape          : "
        f"{refined_escape_count}/2 -> {M3}"
    )

    lines.append(
        f"M4 matched trajectory improvement    : "
        f"{trajectory_improvement_count}/2 -> {M4}"
    )

    lines.append(
        f"MINIMAL P1 REFINEMENT CAUSAL RESCUE  : {rescue}"
    )

    lines.append(
        f"route class                           : {route_class}"
    )

    lines.append(
        f"next route                            : {next_route}"
    )

    lines.append("=" * 186)

    lines.append(
        "Guardrail: Stage-29 S4 remains FAIL; Stage 31 is a separate matched-state causal rescue."
    )

    lines.append("=" * 186)

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
