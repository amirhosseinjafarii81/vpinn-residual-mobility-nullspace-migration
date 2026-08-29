# Claims and Limitations

## Supported by the current evidence

- A basis-invariant residual-mobility collapse can precede certified Adam conflict by a large margin in the tested family.
- Deep lock is not explained by residual direction or tangent-kernel shape alone; paired residual/kernel geometry matters.
- In a finite P1 weak-test space, trial error can be almost completely invisible even when weak loss is tiny.
- A matched-state test-space replacement can make the previously hidden error visible and substantially improve the solution.
- After training on the replacement space, the remaining error can become nearly invisible to that new space while becoming visible again to the discarded space.
- Retaining previous test information through a union restores visibility and produced a strong transient rescue.
- The late instability of the tested union trajectory was classified as finite-step curvature overshoot dominated by the Gauss-Newton term.

## Not supported / not claimed

- "First-ever" discovery.
- Universal behavior across VPINNs, dimensions, PDE classes, architectures or optimizers.
- A superior new optimizer.
- Gradient conflict as the root cause of deep VPINN lock.
- Persistent union enrichment as a universally stable cure.
- Small VPINN loss with spurious modes as a new observation by itself.

## Main remaining limitation

The evidence spans two operators and two test-space families, but still uses one
finite-width MLP architecture and one-dimensional manufactured problems.
