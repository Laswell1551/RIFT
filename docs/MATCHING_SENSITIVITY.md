# Matching-resolution sensitivity

Defined after the primary ten-decile analysis and before calculating these sensitivity outcomes. The balance audit found absolute standardized initial-error residuals below 0.037, but mean initial-error differences around -0.14 m. Because this is not negligible relative to the forgetting contrast, test finer matching rather than treating decile matching as exact continuous balance.

Retain the primary result and add both of the following for every method and seed:

1. Repeat the exact expected matched-control calculation with 50 within-run initial-error quantile bins. Ties stay in one bin. Group-size and run-composition constraints remain unchanged; the control population still includes all held-out examples.
2. For every CPA example, find its nearest initial-FDE non-CPA example within the same simulation run. Allow replacement, resolve ties by row index, and retain donor IDs. Report the mean paired forgetting contrast, mean absolute initial-FDE distance, maximum distance, and donor reuse. This compares the CPA stratum with non-CPA neighbors in initial difficulty and has a different estimand from the full-population random controls.

Use the same equal historical-task weights and 20,000 paired bootstrap draws over eight training seeds. No subgroup, method, task, seed or threshold is removed based on its effect. These are explicitly post-primary sensitivity analyses, not a retrospectively claimed original preregistration.
