# Data contracts

## Low-altitude cache

Files: `data/lowaltitude/cache/{task}__{train,val,test}.npz` for baseline,
wind5, speed15, alt180, speed10, alt200. The locked paper protocol uses five
observed seconds, ten predicted seconds and an identity-grouped 70/10/20 split.
`obs`: float32 [N,5,3]; `future`: float32 [N,10,3]; `context`: float32 [N,6].
`risk_score` and `risk_distance`: [N], calculated from observed relative motion.
CPA membership is `risk_distance <= 1` under the 50 m horizontal / 15 m vertical
separation ellipsoid. `run_index`: [N]. `meta` records target ID, observation
time, world x/y, altitude and (for the metadata cache) heading. Optional
`interaction` and `interaction_set` supply observed neighbor features.

Initial-difficulty matching uses `run_index`, initial prediction errors, and
the fixed CPA mask. It never reads the final predictor to construct cells.

## Alerting inputs

`data/lowaltitude/states/{task}__{scenario}.parquet` contains the five original
scenarios per task, with columns `sim_time_s`, `drone_id`, `lat_deg`, `lon_deg`,
`alt_m`, `gs_ms`, `vs_ms`, `trk_deg`. These state files and the matching
`data/lowaltitude/metadata_cache/{task}__test.npz` are needed to recover actual
future neighbor positions. See `APPLICATION_PROTOCOL.md` for eligibility,
censoring, input-selected neighbors and the separate incident-only assay.

## Road inputs

The downloader writes sixteen trusted upstream files below
`data/interaction/interaction_merge/{train,val}/`. SHA-256 checks precede
`prepare_interaction.py`, which reads the source object arrays and emits only
numeric cached arrays. No original dataset is uploaded by this release.

Regenerating low-altitude caches from state files requires the original
development identity exclusions to reproduce the manuscript sample exactly.
The code can operate on new datasets; new samples should be reported as new
experiments, not as reproductions of the published numerical tables.
