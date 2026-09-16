# TTC-style time-to-entry control, 2026-09-16

## Timing and question

This post-primary protocol is fixed before extracting entry times or calculating new retention effects. The CPA-threshold results and reviewer comments are already known. The question is whether time-to-entry adds useful resolution to the CPA retention profile, including the distinction between ongoing and future proximity.

## Definition and fixed parameters

- Use only the final observation's relative positions and measured velocities. Retain the original ellipsoid axes (50,50,15) meters and constant-velocity assumption.
- For scaled relative position a and velocity b, define entry time as inf{t >= 0: ||a+t b|| <= 1}. It is zero for an already-inside pair, infinity for a pair that never enters, and the first nonnegative root of ||a+t b||^2=1 otherwise.
- The window score is the earliest entry across the original neighborhood query: k=min(64,snapshot size) followed by focal-aircraft exclusion. This reproduces the actual original CPA candidate set (at most 63 other aircraft), rather than silently expanding it to 64 other aircraft. Preserve snapshot order and the original KD-tree query.
- Call this time-to-ellipsoid-entry (TTE), a TTC-style proximity surrogate. It is neither literal aircraft-body collision time nor time to the closest point of approach.
- Time cutoffs are 3, 5 and 10 seconds, fixed as fractions of the original ten-second forecast horizon. They are evaluation parameters, not regulatory thresholds.

## Declared groups and estimands

Report all eight groups, irrespective of result sign:

1. TTE <= 3 s (includes ongoing proximity).
2. TTE <= 5 s (includes ongoing proximity).
3. TTE <= 10 s (includes ongoing proximity).
4. TTE = 0 (ongoing proximity).
5. 0 < TTE <= 3 s (new early entries).
6. 3 < TTE <= 5 s (new middle entries).
7. 5 < TTE <= 10 s (new later entries).
8. TTE > 10 s, including infinity (no entry in the forecast horizon).

The last five groups form a disjoint partition of all frozen windows. Groups 4-7 partition the original CPA stratum. Group 3 is mathematically identical to CPA <= 1 for the same candidate set, motion and horizon; it is an implementation/equivalence check, not independent corroborating evidence. New early entries (group 5) are the primary temporal contrast because they separate upcoming proximity from ongoing proximity. The full disjoint profile and cumulative windows remain visible even if this contrast is inconclusive.

Reuse five methods (fine-tuning, reservoir, GSS, two-support and Dual-LS), all eight existing seeds and the first five historical tasks. Keep every frozen test window and both saved checkpoints per historical task. Do not retrain or modify memory selection.

For each group, report ordinary forgetting, group forgetting, the exact run/initial-FDE-decile-matched expectation, group minus ordinary and group minus matched forgetting. Initial-error cells remain the original cells across groups; only their group counts change. Also report CPA-conditional matching for the four inside/entry bands: restrict the matching pool to original CPA examples while preserving simulation run and initial-error decile. This asks whether a time band differs within the original CPA stratum. Label its different reference population explicitly.

Average historical tasks equally within seed, then summarize eight seeds with 20,000 seed-bootstrap draws (seed 613 + fixed method index). Intervals condition on fixed windows and task order, are descriptive, and have no multiplicity adjustment. Empty task/group cells, if any, are explicitly recorded and preclude a complete five-task mean for that group; do not silently drop them.

## Audits and release

- Verify core geometry on analytic examples (approach, stationary pair, receding pair, tangent and already inside), and against closest-approach membership on random kinematics.
- Reconstruct original CPA/current distances and require the original CPA mask to match every frozen test window.
- Require TTE <= 10 to match CPA <= 1 exactly and reproduce the published mean/interval results to 1e-12.
- Check nested cumulative masks, disjoint/exhaustive bands and exact count-weighted reconstruction of ordinary and CPA forgetting from bands.
- Retain source hashes, per-task/per-seed aggregates, all effects and support/overlap counts. Raw states and per-window observations stay local; publish analysis code and aggregate experimental materials under the existing release authorization.

## Terminology sources

- FHWA SSAM User Manual, FHWA-HRT-08-050, May 2008, appendix: TTC is a current-state-based surrogate; its definition must be distinguished from post-encroachment time. https://www.fhwa.dot.gov/publications/research/safety/08050/
- NASA/TM-2010-216857, *Time of Closest Approach in Three-Dimensional Airspace*: constant-relative-motion conflict is entry into a declared protected region within a lookahead interval, and closest-approach time is a different quantity. https://shemesh.larc.nasa.gov/fm/papers/NASA-TM-2010-216857.pdf

These references inform terminology. The ellipsoid and the eight groups above are the present evaluation design, not a reproduced FHWA/NASA operational standard.
