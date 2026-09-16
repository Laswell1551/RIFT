# RIFT: CPA-stratified continual trajectory evaluation

Code and numerical evidence for **Beyond Average Forgetting: CPA-Stratified
Continual 3-D Trajectory Prediction for Low-Altitude Traffic**.

Jiaqi Lin, Shi Yan and Mugen Peng, Beijing University of Posts and Telecommunications.

RIFT evaluates where forecasting skills change during sequential updates.
It fixes an observed closest-point-of-approach (CPA) stratum and reports
ordinary, CPA-stratified and worst-regime error and forgetting. The two-support
memory is a replaceable replay probe. CPA indicates observed interaction
severity; the alerting experiment separately uses recorded future approaches.

## Install and check

Python 3.12 was used. Training ran on an AMD Ryzen 7 7700 CPU with 32 GB RAM
and two PyTorch threads per process. `requirements-tested.txt` records the
measured environment; `requirements.txt` provides compatible dependency ranges.

```bash
git clone https://github.com/Laswell1551/RIFT.git
cd RIFT
python -m pip install -r requirements.txt
python code/reproduce_metrics.py
python code/smoke_test.py
python code/make_validation_figure.py
python code/make_dense_figures.py
python code/make_ttc_figure.py --export
```

The first check reconstructs ordinary/CPA historical forgetting from all 576
road checkpoint metrics and checks published summaries. It also reconstructs
all 100 threshold-sensitivity summaries/intervals from 800 task records and
verifies that threshold 1 reproduces the primary matched-control analysis.
It also reconstructs every time-to-entry summary from 1,600 task records,
checks the disjoint-band accounting, and verifies the ten-second CPA equivalence.
The smoke test trains
small artificial inputs through actual sequential updates and historical
back-testing; its numbers are software checks, never paper evidence.
These commands need no dataset download. Figures regenerate from the released
source tables; PDF and SVG are vector masters, PNG is a 300-dpi preview.

## Independent INTERACTION continual training

The processed dataset is linked from the
[official H2C repository](https://github.com/BIT-Jack/H2C-lifelong).
Read `THIRD_PARTY.md` and the original data/source terms before acquisition.

```bash
python code/fetch_uqnet.py --accept-upstream-terms
python code/acquire_interaction.py --accept-data-terms
python code/prepare_interaction.py
python code/run_interaction.py --seed 7 --method reservoir
# Full four-method x four-seed sequence:
python code/run_interaction_queue.py
python code/aggregate_interaction.py
python code/road_pairwise_contrasts.py
python code/make_dense_figures.py
```

The encoder is fetched at a fixed commit and verified before import edits.
Its source is not bundled. The road runner updates all model parameters,
saves eight checkpoints per run, and evaluates every domain seen so far.
Order: MA, FT, LN, ZS2, OF, EP0, GL, ZS0. Seeds: 7, 11, 23, 47. Each domain
uses a fixed 2,000-example training subset (or all examples if fewer) and all
released validation examples. The official processed split lacks original
recording/vehicle IDs; `docs/INTERACTION_PROTOCOL.md` describes its split audit.
CPA counts vary by domain, including six in LN and two in the final ZS0 domain.

## Synthetic low-altitude continual learning

Place the frozen train/val/test NPZ files in `data/lowaltitude/cache/`.
Their required schema is documented in `docs/DATA.md`. The original synthetic
corpus and caches are not included in this code release. Existing state files
can be converted with `code/prepare_windows.py --help` and augmented with
`code/prepare_multi_neighbor.py --help`. Exact manuscript reproduction needs
the frozen windows, identity exclusions and sampling choices; regenerating a
different sample is a new experiment.

```bash
python code/run_continual.py --config configs/main_13_methods.json --cache data/lowaltitude/cache --output results/main-seed7 --seed 7
# Eight-seed, five-method prediction-export extension used in the new controls:
python code/run_synthetic_seeds.py
python code/analyze_stratification.py
python code/matching_sensitivity.py
python code/analyze_cpa_thresholds.py
```

Run `configs/main_13_methods.json` with seeds 7, 11, 23, 47, 59, 71, 89, 107
for the main comparison. The original locked configurations for backbone,
input, budget and noise studies are under `configs/historical/`.

The post-primary threshold sweep reuses frozen historical predictions at
CPA thresholds 0.5, 1, 1.5 and 2; it changes only evaluation membership.
For reservoir, the initial-difficulty-matched excess decreases from 0.422 m
to 0.060 m over this range, with positive seed-bootstrap intervals throughout.
Smaller selector-specific effects vary with the threshold. All five methods,
thresholds and both excess contrasts are retained; see
`docs/THRESHOLD_PROTOCOL.md`. The sweep changes the ellipsoid's uniform scale,
keeping its horizontal-to-vertical ratio and CPA horizon fixed.

### TTC-style temporal decomposition

With the original observation-time state files available, run:

```bash
python code/analyze_ttc.py check
python code/analyze_ttc.py prepare
python code/analyze_ttc.py analyze
python code/check_ttc_metrics.py
python code/make_ttc_figure.py --export
```

The last two commands need only the released aggregates. Time-to-ellipsoid-entry
(TTE) is a TTC-style proximity diagnostic, using the fixed 50/50/15 m ellipsoid
and observation-time relative velocities. The 3, 5 and 10 s cumulative groups
include ongoing proximity. Five disjoint bands separate ongoing proximity,
new entries in (0,3], (3,5] and (5,10] s, and no entry by 10 s.
The original 64-entry neighbor query is retained, with the focal aircraft
removed. On the frozen windows, current/CPA distances reproduce exactly.

Entry within 10 s is mathematically equivalent to the original CPA stratum;
it is an equivalence check. The temporal bands expose an additional difference:
reservoir's matched excess is +0.339 m for ongoing proximity and -0.156 m
for new entries within 3 s. Two-support's corresponding contrasts are
-0.081 m and +0.324 m. Its absolute forgetting in the new-entry band is
+0.148 m [0.058,0.245], despite negative whole-CPA forgetting.
The six-panel figure shows all five methods and all disjoint bands with eight
seeds. Full-population and CPA-conditional matching have distinct reference
pools; all estimates, intervals and support counts are retained. TTE is not
literal aircraft-body collision time. See `docs/TTC_PROTOCOL.md` for the
pre-computation design and terminology sources.

For the future-approach assay, also place original states and the heading-aware
metadata caches as described in `docs/DATA.md`, then run:

```bash
python code/application_detection.py prepare
python code/application_detection.py analyze
python code/application_detection.py incident
```

## Evidence and interpretation

| Artifact | Role |
|---|---|
| `results/stratification/` | Run-matched and initial-difficulty-matched controls, all seeds and tasks |
| `results/cpa_thresholds_20260916/` | Four-threshold evaluation sweep, all five methods/eight seeds, support counts and source hashes |
| `results/ttc_control_20260916/` | Three time cutoffs, five disjoint bands, two matching pools, all seed/task aggregates and geometry audits |
| `results/interaction_summary/` | All 576 checkpoint/domain metrics, per-domain results and paired contrasts |
| `results/application*/` | Before/after detection metrics and incident-only sensitivity |
| `source_data/` | Mechanism diagnostics and derived road curves/cells |
| `docs/` | Frozen protocols, data schema and plotting rationale |
| `provenance/` | Pinned external-source hashes and release inventory |

Forgetting is **final minus immediately-after-learning error**, retaining
negative values. At intermediate steps, only earlier domains are averaged;
the historical set grows with the sequence. Final road errors average all
eight domains, final forgetting only the first seven. Equal domain weights
are used. Seed bootstrap intervals are conditional on the fixed data and
domain order; matching draws and individual windows are not extra training
replicates. The four road seeds are shown individually in the dynamic panels.
Intervals crossing zero do not establish superiority. The offline alerting
test does not measure collision probability or closed-loop avoidance.

## License and citation

Original code: MIT. External code and datasets: see `THIRD_PARTY.md`.
Please cite the manuscript title and this repository. `CITATION.cff` provides
software metadata; no publication DOI is asserted before publication.
