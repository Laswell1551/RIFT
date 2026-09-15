# RIFT evidence extension, 2026-09-15

This dated extension is specified before the new training runs. Existing main-study results and the earlier training-free road diagnostic are already known. This is not a retrospective claim of independent preregistration.

## 1. Stratification validity

- Reuse the six fixed synthetic tasks, identity partitions, 4,000 training windows per task, all 2,000 held-out windows per task, original 15-epoch optimizer, and 500-window replay budget.
- Run fine-tuning, reservoir, GSS-adapted, two-support, and Dual-LS-adapted with seeds 7, 11, 23, 47, 59, 71, 89, and 107.
- Save prediction snapshots immediately after each task is learned and after the final task. The original triangular task-evaluation matrix is also retained.
- Primary sample response: final FDE minus FDE immediately after learning its task. Average over the first five historical tasks with equal task weights.
- CPA group: unchanged observed d_CPA <= 1. No future coordinates enter group construction or replay selection.
- Random control: equal group size and simulation-run composition; sampling without replacement from the complete held-out task population.
- Difficulty-matched control: preserve CPA counts within each simulation run and deciles of the immediately-after-learning FDE; sample without replacement inside these strata. This matches initial prediction difficulty rather than conditioning on final errors.
- Report exact expected means of both sampling controls and their conditional permutation distributions (1,000 deterministic draws); controls may overlap the CPA group under the null. Matching cells, counts, and indices are auditable.
- Primary contrast: CPA forgetting minus difficulty-matched forgetting. Also report CPA-minus-overall and CPA-minus-random contrasts, all method results, and per-task effects.
- Uncertainty: 20,000 paired bootstrap resamples of the eight training seeds for mean contrasts. Permutation draws do not count as additional experimental replicates. Show intervals and signs without requiring a favorable outcome.

## 2. Independent continual-learning study

Use a public road-trajectory source with actual training updates and evaluation of all historical domains after each update. Establish the acquisition manifest, splits, task order, backbone, prediction horizon, and replay settings before its training. Prefer a documented interaction-aware model. Distinguish an official implementation from a local adaptation. The prior fixed-predictor diagnostic does not count as this experiment.

## 3. Application connection

Evaluate detection of future close approaches using actual future trajectories as labels, if those trajectories are available. Keep a fixed geometric event threshold and input-only neighbor eligibility. Compare missed-event and false-alarm rates immediately after task learning and after the final update. Report event support, eligibility and exclusions. Predicted-neighbor or constant-velocity future paths must not be presented as actual future labels. This experiment is offline event detection, not a closed-loop collision-avoidance evaluation.

## 4. Manuscript integration

Center RIFT on evaluation contribution. Use CPA-stratified/observed-encounter terminology, preserve supported statistical distinctions, connect theoretical quantities to measured quantities, add concise memory pseudocode and main-text hardware details. Keep full numerical tables and reproducibility details in the supplement. Target a 10-page main PDF without changing the IEEE font or margins.
