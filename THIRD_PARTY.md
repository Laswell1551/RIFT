# Sources and scope of this release

The MIT license applies to the original RIFT implementation and documentation
in this repository. It does not relicense datasets or externally obtained code.

## Independent road predictor

The experiments adapt the public UQnet VectorEncoder in
[BIT-Jack/H2C-lifelong](https://github.com/BIT-Jack/H2C-lifelong), commit
`ecc9cddb7e4ab478768e01ae1ff8f50b69f0d368`. At this revision no repository license
file was found. Its source is not redistributed in this repository.
`code/fetch_uqnet.py` obtains the two source files directly from the pinned
upstream commit after `--accept-upstream-terms`, checks their SHA-256 hashes,
and applies only the import edits documented in `provenance/`.
Users must have the rights required by the upstream authors for their use.
The original base-layer attribution is retained in the downloaded file.

The RIFT endpoint head, training procedure, replay integration and evaluation
are provided in `code/run_interaction.py`. This is a 64-channel encoder
adaptation with 165,570 trainable parameters, not a reproduction of the
original complete UQnet heatmap predictor or the original H2C system.

## Data

[INTERACTION](https://interaction-dataset.com/) and the processed release
linked by H2C retain their original data terms. Acquisition instructions and
the sixteen source-file hashes are provided; data are not bundled. Only
download and load the trusted upstream release: its NPZ files contain Python
objects. The synthetic low-altitude trajectories and frozen caches are also
not redistributed. Public numerical evidence consists of aggregate metrics
and figure-source tables, without individual recorded road trajectories.

## Baselines

The continual runner implements shared-backbone adaptations of EWC, A-GEM,
DER++, GSS, H2C, SYREM and Dual-LS. Internal `cgsm_*` identifiers name the
two-support probe and its controls. These are matched-budget implementations
for this evaluation, not claims of upstream implementation equivalence.
