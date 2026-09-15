# Figure rationale and statistical units

## Figure 2: validation signal (2 x 2)

Paired forests answer whether the CPA stratum adds information after matching,
whether it changes retention profiles on independent recorded traffic, and
whether updates change missed-event rates under the fixed offline detector.
Open points show individual seeds where drawn. Intervals are paired 95% seed
bootstrap intervals: eight low-altitude seeds, four road seeds. The incident
analysis excludes already-close observations and remains a secondary assay.
Its results are retained even when a predictor with better CPA-FDE exhibits
an increase in incident missed events. Height: 97 mm, width: 182 mm.

## Figure 3: road continual dynamics (2 x 3)

All 576 checkpoint evaluations are used: 128 diagonal baselines and 448
historical-domain evaluations. Four panels compare ordinary and CPA forgetting
through updates, with every training seed shown as a thin curve and the
four-seed mean as a thick curve. The historical task set grows at each step.
Three replay panels share vertical limits; fine-tuning uses a larger range.
Two numeric heatmaps show final per-domain ordinary and CPA forgetting with
one symmetric zero-centered scale. Domain categories are not connected by
lines. The heatmaps show descriptive means, not significance decisions.
The LN CPA stratum contains six windows; the final ZS0 task has no historical
forgetting entry. Height: 98 mm, width: 182 mm.

## Figure 4: mechanism diagnostics (1 x 4)

Sample-level rank concordance (4,000 windows), aggregate coverage association,
within-budget-and-seed association (48 runs), and held-out selector effects
(eight paired seeds) have separate panel labels. All data are retained.
The opposite aggregate/matched associations motivate distinguishing capacity
effects from selector effects. Correlations and fitted lines are descriptive.
Selector intervals crossing zero do not establish superiority over GSS.
Height: 61 mm, width: 182 mm.

## Export and checks

Figures use native double-column dimensions, 6 pt minimum explicit font size,
embedded TrueType fonts, vector PDF/SVG and 300-dpi previews. Color is paired
with marker/line styles; heatmaps carry numerical annotations. No new
significance tests, downsampling or omitted method outcomes are introduced.
Source CSVs, all-seed curves and the plotting scripts are included. The plotting
rationale follows the scientific-figure-making and scipilot-figure-skill
workflows used during manuscript preparation.
