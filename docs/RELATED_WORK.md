# Related Work and Novelty Guardrails

This is a short, non-exhaustive reading list used to avoid overclaiming.

1. Berrone, Canuto & Pintore (2022), *Variational Physics Informed Neural Networks:
   the Role of Quadratures and Test Functions*, Journal of Scientific Computing.  
   https://doi.org/10.1007/s10915-022-01950-4  
   Establishes the importance of test spaces, inf-sup stability, quadrature, and spurious modes.

2. Rojas et al. (2024), *Robust Variational Physics-Informed Neural Networks*,
   Computer Methods in Applied Mechanics and Engineering.  
   https://doi.org/10.1016/j.cma.2024.116904

3. Berrone & Pintore (2024), *Meshfree Variational-Physics-Informed Neural Networks:
   An Adaptive Training Strategy*.  
   https://doi.org/10.3390/a17090415  
   Adaptive test-function addition already exists in VPINN literature.

4. Radin, Klinkel & Altay (2026), *A new set of test functions for variational
   physics-informed neural networks in solid mechanics*, Journal of Computational Physics.  
   https://doi.org/10.1016/j.jcp.2026.115174

5. Adaptive VPINN error estimator for stationary Navier-Stokes (2026).  
   https://doi.org/10.1016/j.cma.2026.118876

The present repository therefore focuses its candidate novelty on the **temporal,
matched-state mechanism**: residual-mobility collapse, hidden-error visibility,
and migration of residual error between complements when finite test spaces are
replaced.
