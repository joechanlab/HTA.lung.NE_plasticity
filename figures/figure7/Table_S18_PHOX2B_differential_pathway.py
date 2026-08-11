"""
Table S18 - PHOX2B-associated differential pathway analysis in engineered
cell lines.

Full GSEA prerank results (curated_small gene sets, 500 permutations, ranked
by wilcoxon PHOX2B vs. LV105 scores) for H1975 and PC9, the same analysis
rendered in Fig 7's dotplot (Fig7_PHOX2B_GSEA_dotplot_main.png, reproducing
main_plot_cleaned_narrow.png from PHOX2B_DEG_MAST_GSEA.ipynb). See Table S17
for the underlying per-gene differential expression results.
"""
import os

import pandas as pd

from phox2b_gsea_helper import run_gsea_pipeline

out_dir = "../../results/tables/figure7"
os.makedirs(out_dir, exist_ok=True)

results = run_gsea_pipeline()

col_order = ["Term", "Cell_line", "ES", "NES", "NOM p-val", "FDR q-val",
             "FWER p-val", "Tag %", "Gene %", "Lead_genes"]

frames = []
for cell_line, res2d in results.items():
    df = res2d.copy()
    df["Cell_line"] = cell_line
    frames.append(df[col_order])

table = pd.concat(frames, ignore_index=True)
table = table.sort_values(["Cell_line", "FDR q-val", "NOM p-val"]).reset_index(drop=True)

csv_path = os.path.join(out_dir, "Table_S18_PHOX2B_differential_pathway.csv")
table.to_csv(csv_path, index=False)

print(f"Wrote {len(table)} gene-set rows across {table['Cell_line'].nunique()} cell lines")
print(table.groupby("Cell_line").size())
print(f"CSV -> {csv_path}")
