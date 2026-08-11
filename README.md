# HTA.lung.NE_plasticity

Analysis and figure-generation code for the manuscript on neuroendocrine (NE) lineage
plasticity and histologic transformation in lung cancer.

This repository contains **code only** — no data, intermediate results, or generated
figures/outputs are tracked here. Scripts assume access to the underlying
single-cell, spatial, genomic, and clinical datasets described in the manuscript.
Only code that produced panels in the final figure set is included; exploratory,
superseded, or standalone/one-off analyses (alternative pseudotime methods, abundance
regression tests that came back non-significant, per-gene ad hoc investigations,
diagnostic-only QC scripts, and the RU325 standalone patient pipeline) have been left out.

## Structure

Directories are organized by manuscript figure, with subfolders named for what the
code does. Numeric filename prefixes (e.g. `1_`, `2.1_`) indicate the intended run
order within a subdirectory.

- `figures/figure3/` — clinical/genomic landscape and tumor subtype diversity
  - `snv_shared_mutations/` — shared SNV analysis across tumor subtypes
  - `drug_target_dotplot/` — drug target expression dotplot data prep and plotting
  - `numbat_cnv_phylogeny/` — allele-specific CNV inference and phylogenetic reconstruction
    (**note**: this pipeline's outputs are also consumed by `figure5/` and `figure7/`
    analyses, not just figure3 panels)
- `figures/figure4/` — basal-like intermediate state analysis
  - `basal_module_scoring/` — basal module scoring and characterization
- `figures/figure5/` — non-NE reversion following NE transformation (GSEA of NonNE/EMT
  vs LUAD/SCLC); depends on CNV/phylogeny output from `figure3/numbat_cnv_phylogeny/`
- `figures/figure6/` — treatment timeline visualization (`Fig6a_...`)
- `figures/figure7/` — PHOX2B regulatory analysis (GSEA, ChIP-seq, multiome, chromatin
  accessibility); also depends on `figure3/numbat_cnv_phylogeny/` for CNV-based panels
- `figures/figureS6/` — CellRank/Palantir trajectory robustness analysis

## Related repository

This code was extracted from the working repository
[`HTA.lung.NE_plasticity_scDeBussy`](https://github.com/joechanlab/HTA.lung.NE_plasticity_scDeBussy),
which also contains the `scDeBussy` pseudotime alignment package and other in-progress
analysis code.
