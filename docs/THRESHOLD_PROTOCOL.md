# CPA threshold sensitivity — 2026-09-16

Post-primary analysis of already stored predictions; the original results and reviewer comments are known.

- Fixed thresholds: 0.5, 1.0, 1.5, 2.0. They rescale the 50 m / 15 m ellipsoid together and do not test an independently changed aspect ratio or a TTC surrogate.
- Reuse all five previously exported low-altitude methods, all eight seeds, all five historical tasks and every test window. No retraining, revised memory selection or replacement test examples.
- Compute immediately-after-learning to final FDE changes. Match simulation run and the original initial-error deciles, then recompute the exact conditional control expectation for each CPA mask.
- Report ordinary forgetting, CPA forgetting, CPA-minus-overall and CPA-minus-matched effects. Average tasks equally; 20,000 paired bootstrap draws use training seeds as the unit.
- Keep all methods and thresholds, regardless of sign or interval. These are descriptive post-primary sensitivity intervals without multiplicity-adjusted discovery claims.
- Require the original threshold 1.0 to reproduce all corresponding published means and intervals to 1e-12; verify nested masks and an exactly zero no-change control.
