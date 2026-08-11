"""
Table S17 - PHOX2B-associated differential expression analysis in engineered
cell lines.

Accompanies the PHOX2B-overexpression GSEA analysis (Fig 7, Table S18): the
wilcoxon rank_genes_groups (PHOX2B-overexpressing vs. LV105-control) results
for H1975 and PC9, on the same mito/ribo-filtered gene set used to rank genes
for the GSEA in Table S18/Fig 7 - so Table S17 is the exact DEG table
underlying that pathway analysis and figure.

An alternative MAST-based DEG run also exists
(mast/PHOX2B_vs_LV105_MAST_*_results.csv in
PHOX2B.scRNAseq.2024/processed/nf-scrnaseq/report) but was computed on the
unfiltered gene set (dominated at the top by ribosomal/mitochondrial/
housekeeping genes excluded here) and does not feed the GSEA/figure, so it is
not used for this table.
"""
import os

import pandas as pd

from phox2b_gsea_helper import CELL_LINES, get_wilcoxon_de

out_dir = "../../results/tables/figure7"
os.makedirs(out_dir, exist_ok=True)

frames = []
for cell_line in CELL_LINES:
    de = get_wilcoxon_de(cell_line)
    de = de.copy()
    de["Cell_line"] = cell_line
    frames.append(de)

table = pd.concat(frames, ignore_index=True)

table = table.rename(columns={
    "names": "Gene",
    "scores": "Wilcoxon_score",
    "logfoldchanges": "Log2FC",
    "pvals": "P_value",
    "pvals_adj": "FDR",
})

col_order = ["Gene", "Cell_line", "Wilcoxon_score", "Log2FC", "P_value", "FDR"]
table = table[col_order]
table = table.sort_values(["Cell_line", "FDR", "P_value"]).reset_index(drop=True)

csv_path = os.path.join(out_dir, "Table_S17_PHOX2B_differential_expression.csv")
table.to_csv(csv_path, index=False)

print(f"Wrote {len(table)} gene-level DE rows across {table['Cell_line'].nunique()} cell lines")
print(table.groupby("Cell_line").size())
print(f"CSV -> {csv_path}")
