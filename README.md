# HTA.lung.NE_plasticity

Analysis and figure-generation code for the manuscript on neuroendocrine (NE) lineage
plasticity and histologic transformation in lung cancer.

## Structure

Directories are organized by manuscript figure, with subfolders named for what the
code does. Numeric filename prefixes (e.g. `1_`, `2.1_`) indicate the intended run
order within a subdirectory.

- `figures/figure3/` — clinical/genomic landscape and tumor subtype diversity
  - `snv_shared_mutations/` — shared SNV analysis across tumor subtypes
  - `drug_target_dotplot/` — drug target expression dotplot data prep and plotting
  - `numbat_cnv_phylogeny/` — allele-specific CNV inference and phylogenetic reconstruction
    (**note**: this pipeline's outputs are also consumed by `figure5/` and `figure7/`
    analyses)
- `figures/figure4/` — basal-like intermediate state analysis
  - `basal_module_scoring/` — basal module scoring and characterization
- `figures/figure5/` — non-NE reversion following NE transformation (GSEA of NonNE/EMT
  vs LUAD/SCLC); depends on CNV/phylogeny output from `figure3/numbat_cnv_phylogeny/`
- `figures/figure6/` — treatment alluvial plot visualization
- `figures/figure7/` — PHOX2B regulatory analysis (GSEA, ChIP-seq, PHOX2B expression/
  correlation panels); also depends on `figure3/numbat_cnv_phylogeny/` for CNV-based panels
- `figures/figureS6/` — CellRank/Palantir trajectory robustness analysis
- `figures/figureS10/` — PHOX2B/FOXA1 single-cell chromatin accessibility (snATAC
  multiome, ChromVAR motif accessibility)
