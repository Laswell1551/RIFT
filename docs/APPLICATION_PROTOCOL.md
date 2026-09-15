# Offline future-approach detection (fixed before label construction)

For each frozen synthetic test window, select up to eight other aircraft with the smallest current ellipsoidal distance at the observation endpoint. This selection uses observed positions only and is independent of CPA labels and all future paths. Resolve distance ties by aircraft ID. Require the selected aircraft and target to have records at all ten subsequent integer seconds; otherwise exclude the window and report it. No neighbor is replaced because its future is unavailable.

The binary reference event is any selected pair entering the same 50 m horizontal / 15 m vertical ellipsoid (normalized distance <= 1) at any of the next ten integer seconds. Use both aircraft's recorded future trajectories to construct this label. This is a sampled close-approach event, not a collision label. Events already ongoing at the observation endpoint remain eligible, and their number is reported.

The detector forecasts the target with the learned predictor and each selected neighbor with its observed constant velocity. An alarm uses the identical ellipsoid and integer times. Keep neighbor forecasts fixed when comparing immediately-after-learning and final target predictions; thus only the target predictor's continual update changes. Report target-window FNR and FPR, event/non-event counts, and paired changes. Average historical-task rates with equal weights; bootstrap the eight training seeds 20,000 times. All five methods and task effects are retained. No threshold selection on outcomes is allowed.

This assay connects retention to offline encounter alerting under a fixed surrounding-traffic forecast. It does not measure closed-loop control or collision reduction. A zero-positive or zero-negative task is reported explicitly and omitted only from the corresponding undefined macro rate.

## Incident-only sensitivity addendum
After the primary full-window analysis, the label-support audit identified many windows already inside the ellipsoid at observation time. As an explicitly secondary sensitivity, repeat the same fixed detector and threshold among eligible windows with no selected neighbor initially inside the ellipsoid. Preserve and report the original primary result. This subgroup is defined by observed positions, and its analysis is not retrospectively presented as part of the initial freeze. Also verify the declared aircraft-ID tie-break at the nearest-neighbor cutoff.

