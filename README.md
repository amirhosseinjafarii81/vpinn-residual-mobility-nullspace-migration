# Residual Mobility Collapse and Test-Space Nullspace Migration in VPINNs

A reproducible experimental repository for a controlled study of optimization geometry,
finite weak-test visibility, and hidden-error migration in Variational Physics-Informed
Neural Networks (VPINNs).

> **Status:** research artifact / controlled numerical study.  
> **Scope:** 1D manufactured problems, one finite-width MLP architecture, multiple operators,
> Fourier and non-Fourier test spaces, matched-state interventions.

## Strongest supported observations

1. **Residual mobility collapse appears before certified Adam conflict.**
   In Stage 27R it occurred earlier in 5/5 seeds, with a median lead of
   **1475 epochs**.

2. **A finite VPINN test space can miss physically meaningful trial error.**
   In Stage 30R, one run reached weak loss **4.159e-13**
   while retaining **1.67% relative L2 error**.

3. **Matched-state test-space refinement can expose previously invisible error.**
   For seed 27, the visible energy fraction changed from
   **2.323e-10** to
   **0.659**,
   a **2.837e+09x** increase,
   with the same network parameters and Adam state.

4. **The remaining error can migrate into the complement of the replacement test space.**
   For seed 27, V26 visibility fell from **0.659**
   at the branch state to **5.685e-11** after training,
   while the discarded V25 space could still see **0.446**
   of the endpoint error energy.

5. **Persistent union enrichment restores visibility, but late optimizer stability is a separate issue.**
   V26+V25 produced certified escape, while Stage 34R traced the late spikes to
   **Gauss-Newton-dominated finite-step curvature overshoot**, not renewed test-space blindness.

## Core diagnostics

Weak residuals:
```text
R_k(theta) = a(u_theta, v_k) - l(v_k)
```

Variational loss:
```text
L(theta) = (1/M) sum_k R_k(theta)^2
```

Residual Jacobian:
```text
J = dR/dtheta
```

Basis-invariant normalized residual mobility:
```text
mu(theta) = r^T (J J^T) r / ( ||r||^2 tr(J J^T) )
```

Basis-invariant visibility of physical error `e` in a finite test space `V`:
```text
chi_V = ||P_V e||_a^2 / ||e||_a^2
      = b^T G^-1 b / ||e||_a^2
```
where `b_i = a(e, phi_i)` and `G_ij = a(phi_i, phi_j)`.

## Repository layout

```text
scripts/                    Experiment + runner scripts
results_archives/           Complete stage result ZIP archives (Stages 1-34)
results_highlights/
  original_figures/         Selected original experiment plots
  tables/                   Selected exact CSV evidence
  key_findings.csv          Compact evidence table
linkedin_assets/            Final carousel PNGs + LinkedIn post
docs/
  STAGE_INDEX.md
  CLAIMS_AND_LIMITATIONS.md
  RELATED_WORK.md
  REPRODUCIBILITY.md
repo_manifest.csv           SHA256 manifest of public package files
```

## Reproduction

The later mechanism stages used Python 3.12, PyTorch 2.7.1+cu118, NumPy 2.5.1
and float64. Individual scripts contain exact precommitments, seeds, thresholds
and provenance checks.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then inspect `docs/STAGE_INDEX.md`. Many later stages intentionally refuse to
run unless expected predecessor artifacts and source SHA256 values match.

## Scientific claim discipline

This repository does **not** claim:

- that gradient conflict is new in PINNs,
- that small VPINN loss with spurious modes is new,
- that adaptive test-function enrichment is new,
- that `J J^T` / NTK-style residual dynamics are new,
- that a new optimizer has been demonstrated,
- or that the mechanisms are universal across VPINNs.

The strongest candidate contribution is the **controlled temporal evidence for
residual-mobility collapse and test-space nullspace migration under matched-state
test-space interventions**, together with separation of hidden-error failure from
later curvature-driven optimizer instability.

## License

Code is released under the MIT License. Experimental outputs are included for
research reproducibility.
