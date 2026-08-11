import os
import pandas as pd
base_dir = "/data1/chanj3/PHOX2B.scRNAseq.2024/processed/ChIPseq/"
norm_methods = ["DepthNorm", "PeakNorm"]
cell_lines = ["H1975", "PC9"]
file_title = "DESeq2_Named_AllResults_foldpv_Sorted_Protein.txt"
paths = [os.path.join(base_dir, nm, ft, file_title) for nm in norm_methods for ft in cell_lines]

target_genes = {"FOXA2", "SOX2", "PHOX2B"}
thresholds_fold = [1.5, 2, 3, 4, 5]
thresholds_p = [0.05, 0.01, 0.001, 0.0001]

results = {}  # (fold, p) -> set of norms where all target_genes are present

for path in paths:
    df = pd.read_csv(path, sep="\t")
    norm = os.path.basename(os.path.dirname(os.path.dirname(path)))
    # get gene identifiers (prefer a 'gene' column if present, else index)
    def genes_from_df(d):
        if "gene" in d.columns:
            return d["gene"].astype(str).str.upper()
        for col in ("Gene", "gene_name", "symbol"):
            if col in d.columns:
                return d[col].astype(str).str.upper()
        return d.index.astype(str).str.upper()

    for fold in thresholds_fold:
        for pval in thresholds_p:
            filt = df[(df["log2FoldChange"].abs() >= fold) & (df["padj"] <= pval)]
            found = set(genes_from_df(filt))
            if target_genes.issubset(found):
                results.setdefault((fold, pval), set()).add(norm)

# print combinations where at least one norm contains all three genes
if results:
    for (fold, pval), norms in sorted(results.items()):
        print(f"Fold >= {fold}, padj <= {pval} => norms with all targets: {', '.join(sorted(norms))}")
else:
    print("No threshold combination found that selects all target genes in any norm.")
# {(1.5, 0.05): {'DepthNorm', 'PeakNorm'},
#  (1.5, 0.01): {'DepthNorm'},
#  (2, 0.05): {'DepthNorm', 'PeakNorm'},
#  (2, 0.01): {'DepthNorm'}}
# DepthNorm appears more sensitive overall