"""
Table S19 - PHOX2B ChIP-seq peaks and candidate direct target genes.

Accompanies the text: "ChIP-seq confirmed direct PHOX2B binding at FOXA1 and
EGFR in both cell lines (Fig S10a - combined ChIP-seq, Table S19)".

Restricts the DESeq2 differential-binding results (PHOX2B ChIP vs. input) to
significant peaks (fold >= 1.5 / padj <= 0.05, the threshold established in
find_chip-seq_threhsold.py as the cutoff that recovers known PHOX2B targets
under both normalizations) annotated to FOXA1 and EGFR, across both cell
lines (PC9, H1975) and both normalization strategies (DepthNorm, PeakNorm).
"""
import os
import re
import pandas as pd

base_dir = "/data1/chanj3/PHOX2B.scRNAseq.2024/processed/ChIPseq/"
norm_methods = ["DepthNorm", "PeakNorm"]
cell_lines = ["PC9", "H1975"]
file_title = "DESeq2_Named_AllResults_foldpv_Sorted_Protein.txt"

target_genes = ["FOXA1", "EGFR"]

FOLD_THRESH = 1.5
PADJ_THRESH = 0.05

out_dir = "../../results/tables/figure7"
os.makedirs(out_dir, exist_ok=True)

region_re = re.compile(r"^(chr[\w]+)_(\d+)_(\d+)$")

frames = []
for nm in norm_methods:
    for cl in cell_lines:
        path = os.path.join(base_dir, nm, cl, file_title)
        df = pd.read_csv(path, sep="\t")
        df = df[df["gene"].isin(target_genes)].copy()
        parsed = df["region"].str.extract(region_re)
        df["Chromosome"] = parsed[0]
        df["Peak_start"] = parsed[1].astype(int)
        df["Peak_end"] = parsed[2].astype(int)
        df["Cell_line"] = cl
        df["Normalization"] = nm
        frames.append(df)

table = pd.concat(frames, ignore_index=True)

table = table[
    (table["log2FoldChange"] >= FOLD_THRESH) & (table["padj"] <= PADJ_THRESH)
].copy()

table = table.rename(columns={
    "gene": "Gene",
    "distance": "Distance_to_TSS_bp",
})

col_order = [
    "Gene", "Cell_line", "Normalization", "Chromosome", "Peak_start", "Peak_end",
    "Distance_to_TSS_bp", "baseMean", "log2FoldChange", "lfcSE", "stat",
    "pvalue", "padj",
]
table = table[col_order]

gene_order = {g: i for i, g in enumerate(target_genes)}
table["_gene_order"] = table["Gene"].map(gene_order)
table = table.sort_values(
    ["_gene_order", "Cell_line", "Normalization", "padj"]
).drop(columns="_gene_order").reset_index(drop=True)

csv_path = os.path.join(out_dir, "Table_S19_PHOX2B_ChIPseq_peaks_FOXA1_EGFR.csv")
table.to_csv(csv_path, index=False)

print(f"Wrote {len(table)} peak rows across {table['Gene'].nunique()} genes")
print(table.groupby(["Gene", "Cell_line", "Normalization"]).size())
print(f"CSV -> {csv_path}")
