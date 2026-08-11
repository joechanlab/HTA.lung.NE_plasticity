from enum import unique
import os
from functools import reduce
import re
import scanpy as sc
import scanpy.external as sce
import pandas as pd
import numpy as np
import gseapy as gp
from gseapy.plot import gseaplot2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
from scipy.spatial import ConvexHull
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.stats import pearsonr
import matplotlib.patheffects as pe

sc.settings.figdir = "../../results/figures/figure7/"
sc.set_figure_params(
    dpi=120,
    dpi_save=600,
    figsize=(2, 2),
    fontsize=9,             # base font size for labels and ticks
    color_map='Spectral_r',
    frameon=False,
    vector_friendly=True,
)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.grid": False,
    "axes.titlesize": 10,       # titles like “PHOX2B expression”
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,         # embeds text as editable in Illustrator
    "ps.fonttype": 42,
})

# ------------ cell cycle gene sets (Tirosh et al., human) ------------
S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
    "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP",
    "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2",
    "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2",
    "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"
]

G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2",
    "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2",
    "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1",
    "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA",
    "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3",
    "CBX5", "CENPA"
]
# ------------ helpers ------------
def get_gene_vec(adata, gene, mask=None, log1p=False):
    X = adata[:, gene].X
    v = X.toarray().flatten() if hasattr(X, "toarray") else X.flatten()
    if mask is not None: v = v[mask]
    return np.log1p(v) if log1p else v

def pearson_obs(x, y):
    # guard against 0 variance
    if np.std(x)==0 or np.std(y)==0:
        return np.nan, np.nan
    return pearsonr(x, y)

def permutation_null(x, y, n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        r, _ = pearson_obs(x, y_shuf)
        null[i] = r
    return null

def empirical_p(null, r_obs, tail="two-sided"):
    null = null[~np.isnan(null)]
    if np.isnan(r_obs): return np.nan
    if tail == "greater":
        return (np.sum(null >= r_obs) + 1) / (len(null) + 1)
    if tail == "less":
        return (np.sum(null <= r_obs) + 1) / (len(null) + 1)
    # two-sided
    return (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (len(null) + 1)

def random_gene_null(adata, y, mask, n_bg=1000, log1p=True, seed=42):
    rng = np.random.default_rng(seed)
    genes = np.array(adata.var_names)
    # remove EGFR/PHOX2B to avoid trivialities
    genes = genes[(genes!="EGFR") & (genes!="PHOX2B")]
    # pre-sample candidates
    picks = rng.choice(genes, size=min(n_bg, len(genes)), replace=False)
    null = []
    for g in picks:
        x = get_gene_vec(adata, g, mask=mask, log1p=log1p)
        r, _ = pearson_obs(x, y)
        if not np.isnan(r):
            null.append(r)
    return np.array(null, dtype=float)

def hist_with_observed(ax, null, r_obs, title, bins=40):
    ax.hist(null, bins=bins, alpha=0.75, edgecolor="none")
    ax.axvline(r_obs, color="red", lw=2, label=f"Observed r = {r_obs:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)

cell_lines = ["H1975", "PC9"]
# adata_dict = {}
# for cell_line in cell_lines:
#     adata = sc.read_h5ad(f'../../external_data/phox2b/{cell_line}_outlier_filtered_scran_scvi_postprocessing_scvi.h5ad')
#     adata.obs['transduction'] = adata.obs['transduction'].replace('PHOX2B', 'PHOX2B OE')
#     adata.obs['transduction'] = adata.obs['transduction'].replace('LV105', 'Control')
#     #genes_of_interest = ["PHOX2B", "FOXA1", "SOX2", "REST", "EGFR"]
#     #genes_of_interest = ["PHOX2B", "EGFR", "ITGA2"]
#     genes_of_interest = ["transduction", "PHOX2B", "EGFR", "FOXA1"] #, "SOX2"]
#     #genes_of_interest = ["PHOX2B", "LINC00682"]
#     s_genes = [g for g in S_GENES if g in adata.var_names]
#     g2m_genes = [g for g in G2M_GENES if g in adata.var_names]
#     sc.tl.score_genes_cell_cycle(
#         adata,
#         s_genes=s_genes,
#         g2m_genes=g2m_genes
#     )

#     # 8A
#     adata_magic = sce.pp.magic(
#         adata,
#         knn=5,
#         t=1,
#         name_list=genes_of_interest,
#         copy=True
#     )
#     adata_magic.obsm['X_umap'] = adata.obsm["X_umap"].copy()
#     fig = sc.pl.umap(
#         adata_magic,
#         color=genes_of_interest,
#         frameon=False,
#         legend_fontoutline=2,
#         wspace=0,
#         ncols=6,
#         use_raw=False,
#         return_fig=True,
#         title = ['Condition', 'PHOX2B', 'EGFR', 'FOXA1'], #, 'SOX2'], #["PHOX2B", "LINC00682"], 
#         legend_loc='on data',
#         legend_fontweight='bold',
#         add_outline=True,
#     )
#     fig.figure.text(
#         0.02, 0.5,  
#         f"{cell_line}", 
#         va='center',
#         ha='center',
#         rotation='vertical',
#         fontsize=18,
#         fontweight='bold'
#         )
#     for ax in fig.axes:
#         ax.title.set_fontsize(18)
#     fig.figure.savefig(os.path.join(sc.settings.figdir, 
#                                     f"umap_{cell_line}_fig7_genes.pdf"), 
#                                     #f"umap_{cell_line}_fig7_LINC00682.pdf"),
#                                     dpi=300, bbox_inches="tight")
#     adata_dict[cell_line] = adata

# 8B Pearson correlation analysis
# def phox2b_cor_all_genes(adata, log1p=False):
#     n_cells = adata.n_obs
#     X = adata.X
#     if sparse.isspmatrix(X):
#         X = X.toarray()
#     if log1p:
#         X = np.log1p(X)

#     phox_idx = adata.var_names.get_loc("PHOX2B")
#     x = X[:, phox_idx].astype(float)
#     x_mean = x.mean()
#     x_std = x.std(ddof=1)
#     if x_std == 0:
#         r = np.full(adata.n_vars, np.nan)
#         p = np.full(adata.n_vars, np.nan)
#     else:
#         x_z = (x - x_mean) / x_std
#         Y = X.astype(float)
#         Y_mean = Y.mean(axis=0)
#         Y_std = Y.std(axis=0, ddof=1)
#         nonzero = Y_std > 0
#         Y_z = np.zeros_like(Y, dtype=float)
#         Y_z[:, nonzero] = (Y[:, nonzero] - Y_mean[nonzero]) / Y_std[nonzero]
#         r = np.full(adata.n_vars, np.nan)
#         r[nonzero] = x_z @ Y_z[:, nonzero] / (n_cells - 1)
#         df = n_cells - 2
#         denom = np.clip(1 - r**2, 1e-15, None)
#         t_stat = r * np.sqrt(df / denom)
#         p = 2 * stats.t.sf(np.abs(t_stat), df)
#         p[~np.isfinite(r)] = np.nan

#     df = pd.DataFrame({
#         "gene": adata.var_names.to_list(),
#         "r_obs": r,
#         "pearson_p (scipy)": p,
#         "n_cells": n_cells
#     })
#     return df

# summary_dict = {}
# for cell_line in cell_lines:
#     adata = adata_dict[cell_line]
#     summary = phox2b_cor_all_genes(adata)
#     summary["cell_line"] = cell_line
#     summary.to_csv(os.path.join(sc.settings.figdir, f"PHOX2B_correlation_{cell_line}.csv"), index=False)
#     summary_dict[cell_line] = summary


# def run_chea_from_phox2b_corr(
#     df,
#     N=200,
#     gene_sets=None,
#     outdir="../../data/processed/",
#     organism="Human",
#     r_min=0.0,  # minimum correlation to keep (e.g. > 0 for "upregulated")
# ):
#     if gene_sets is None:
#         gene_sets = ["ChEA_2022"]

#     res_dict = {}

#     # Optionally drop PHOX2B itself from the gene list
#     df = df[df["gene"] != "PHOX2B"]

#     # Keep only "upregulated" (positively correlated) genes if desired
#     df = df[df["r_obs"] >= r_min]

#     # Rank by correlation with PHOX2B (analogous to your 'score')
#     rnk = df[["gene", "r_obs"]].sort_values("r_obs", ascending=False)

#     # Top N genes
#     df_up = rnk.head(N)
#     gene_list = (
#         df_up.iloc[:, 0]
#         .dropna()
#         .drop_duplicates()
#         .tolist()
#     )

#     print(f"ORA input genes (n={len(gene_list)}): first 10 →", gene_list[:10])
#     enr = gp.enrichr(
#         gene_list=gene_list,
#         gene_sets=gene_sets,
#         organism=organism,   # "Human" or "Mouse"
#         outdir=outdir,
#         cutoff=0.05,         # adjusted p-value cutoff
#         no_plot=True,
#         verbose=True,
#     )
#     res = enr.results.sort_values(["Gene_set", "Adjusted P-value"])
#     return res

# res_dict = {}
# for cell_line in cell_lines:
#     summary = summary_dict[cell_line]
#     res = run_chea_from_phox2b_corr(
#         summary,
#         N=200,
#         gene_sets=["ChEA_2022"],
#         outdir="../../data/processed/",
#         organism="Human",
#         r_min=0.0,   # or e.g. 0.1 if you want a stricter correlation cutoff
#     )
#     res_dict[cell_line] = res

# # Plot the TFs
# res_dict_list = []
# for cell_line in cell_lines:
#     res_dict_list.append(res_dict[cell_line][['Term', 'Combined Score', 'Adjusted P-value']].rename(columns={"Combined Score": f"Combined Score ({cell_line})",
#                                                                                                            "Adjusted P-value": f"Adjusted P-value ({cell_line})"}))
# res_merged = reduce(lambda left, right: pd.merge(left, right, on="Term", how="outer"), res_dict_list)
# res_merged.to_csv('../../data/figures/figure7/ORA_results.csv', index=False)

# ---------- Build tidy DF from your dict ----------
res_merged = pd.read_csv('../../data/figures/figure7/ORA_results.csv')
def tf_from_term(term: str) -> str:
    s = re.sub(r"\b\d{6,}\b", "", term or "")  # remove PMID-like numbers
    s = re.sub(r"\s+", " ", s).strip()
    return s.split(" ", 1)[0] if s else ""

def parse_species(term: str) -> str:
    s = (term or "").upper()
    if re.search(r'\b(HUMAN|HOMO SAPIENS)\b', s): return "Human"
    if re.search(r'\b(MOUSE|MUS MUSCULUS)\b', s): return "Mouse"
    return "Unknown"

df = res_merged.copy()
df["TF"] = df["Term"].apply(tf_from_term)
df["Species"] = df["Term"].apply(parse_species)
df = df.copy()
df["sum_score"] = df["Combined Score (H1975)"] + df["Combined Score (PC9)"]
df.dropna(inplace=True)
df = df.loc[df.groupby("TF")["sum_score"].idxmax()].reset_index(drop=True)
df = df.sort_values("sum_score", ascending=False).reset_index(drop=True)

# Use PeakNorm files for ChIP-seq hits
# Peak-level normalization controls for library size differences at the level of peak signal strength
chipseq_dir = "/data1/chanj3/PHOX2B.scRNAseq.2024/processed/ChIPseq/"
chip_dict = {}
for cell_line in cell_lines:
    if cell_line == "PC9":
        chip_data = pd.read_csv(os.path.join(chipseq_dir, "DepthNorm", cell_line, "DESeq2_Named_AllResults_foldpv_Sorted_Protein.txt"),
                            sep="\t")
    else:
        chip_data = pd.read_csv(os.path.join(chipseq_dir, "PeakNorm", cell_line, "DESeq2_Named_AllResults_foldpv_Sorted_Protein.txt"),
                            sep="\t")
    chip_data = chip_data[(chip_data["log2FoldChange"].abs() >= 1) & (chip_data["padj"] <= 1e-3)]
    chip_dict[cell_line] = set(chip_data['gene'].dropna().tolist())

df["PC9_ChIP"] = df["TF"].isin(chip_dict["PC9"])
df["H1975_ChIP"] = df["TF"].isin(chip_dict["H1975"])
df["any_ChIP"] = df["PC9_ChIP"] | df["H1975_ChIP"]

def plot_scatter_with_annotations(df, xcol, ycol, xlab, ylab, save_path):
    # threshold for “high in both”
    q = 0.95
    x_thr = df[xcol].quantile(q)
    y_thr = df[ycol].quantile(q)
    high_mask = (df[xcol] >= x_thr) & (df[ycol] >= y_thr)

    fig, ax = plt.subplots(figsize=(3.8, 3))
    #markers = {"Human": "o", "Mouse": "s"}      # species → shape
    color_map = {False: "gray", True: "tab:red"}            # ChIP status → color

    texts = []

    for species in ["Human"]:#, "Mouse"]:
        sub = df[df["Species"] == species]
        if sub.empty:
            continue

        # background points for this species
        sub_bg = sub[~high_mask]
        if not sub_bg.empty:
            ax.scatter(
                sub_bg[xcol], sub_bg[ycol],
                s=18,
                alpha=0.2,
                #marker=markers[species],
                c=sub_bg["any_ChIP"].map(color_map),
                edgecolor="none",
            )

        # highlighted “high in both” points
        sub_hi = sub[high_mask]
        if not sub_hi.empty:
            ax.scatter(
                sub_hi[xcol], sub_hi[ycol],
                s=40,
                alpha=0.9,
                #marker=markers[species],
                c=sub_hi["any_ChIP"].map(color_map),
                edgecolor="black",
                linewidth=0.6,
            )

            # text annotations: same color mapping (ChIP status), any species
            for _, row in sub_hi.iterrows():
                col = color_map[bool(row["any_ChIP"])]
                t = ax.text(
                    row[xcol], row[ycol],
                    row["TF"],
                    color=col,
                    fontweight="bold",
                    ha="center", va="center",
                    zorder=5,
                )
                # optional subtle white stroke for readability
                t.set_path_effects([
                    pe.Stroke(linewidth=1.0, foreground="white"),
                    pe.Normal()
                ])
                texts.append(t)

    # adjust label overlaps
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5)
    )
    ax.axvline(x_thr, linestyle="--", linewidth=1, color="gray", alpha=0.6)
    ax.axhline(y_thr, linestyle="--", linewidth=1, color="gray", alpha=0.6)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(False)

    # species_handles = [
    #     Line2D([], [], marker=markers[s], linestyle="None",
    #            markerfacecolor="none", markeredgecolor="black",
    #            label=s)
    #     for s in ["Human", "Mouse"]
    #     if (df["Species"] == s).any()
    # ]
    chip_handles = [
        Line2D([], [], marker="o", linestyle="None",
               color=color_map[False], label="No PHOX2B peak (PC9/H1975)"),
        Line2D([], [], marker="o", linestyle="None",
               color=color_map[True], label="PHOX2B peak (PC9/H1975)"),
    ]
    # leg1 = ax.legend(handles=species_handles, title="Species",
    #                  loc="upper left", frameon=False)
    # ax.add_artist(leg1)
    ax.legend(handles=chip_handles, title="PHOX2B ChIP-seq",
              loc="upper left", frameon=False) # upper center
    plt.tight_layout()
    plt.savefig(os.path.join(sc.settings.figdir, save_path), dpi=300)
    plt.show()

plot_scatter_with_annotations(df, 
                              "Combined Score (H1975)",
                              "Combined Score (PC9)",
                              "Enrichment of ChEA TF targets\n in PHOX2B OE (H1975)",
                              "Enrichment of ChEA TF targets\n in PHOX2B OE (PC9)",
                              "ORA_scatter_PHOX2B.pdf")

# # ------------ Pearson Correlation analysis ------------
# rng = np.random.default_rng(42)
# cell_lines = ["PC9", "H1975"]
# fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)

# summary = []
# for i, line in enumerate(cell_lines):
#     mask = (adata.obs["cell_line"].astype(str) == line).values
#     x = get_gene_vec(adata, "PHOX2B", mask=mask, log1p=True)
#     y = get_gene_vec(adata, "EGFR",   mask=mask, log1p=True)

#     # observed Pearson
#     r_obs, p_obs = pearson_obs(x, y)

#     # 2) random-gene null (PHOX2B vs random genes)
#     null_rand = random_gene_null(adata, y=x, mask=mask, n_bg=1000, log1p=True, seed=999 + i)
#     p_emp_rand = empirical_p(null_rand, r_obs, tail="two-sided")

#     # plots
#     hist_with_observed(axes[i], null_rand, r_obs,
#                        f"{line}: Random-gene null\n(empirical p={p_emp_rand:.3g})")

#     summary.append({
#         "cell_line": line,
#         "r_obs": r_obs,
#         "pearson_p (scipy)": p_obs,
#         "empirical_p (random_gene)": p_emp_rand,
#         "n_cells": mask.sum()
#     })

# plt.tight_layout()
# plt.savefig(os.path.join(sc.settings.figdir, "Fig8_PHOX2B_EGFR_correlation.pdf"), dpi=300)
# plt.show()

# pd.DataFrame(summary)


# Examine ChIP-data
# import matplotlib.pyplot as plt
# import seaborn as sns

# H1975_data = pd.read_csv(os.path.join(chipseq_dir, "PeakNorm", "H1975", "DESeq2_Named_AllResults_foldpv_Sorted_Protein.txt"),
#                         sep="\t")
# PC9_data = pd.read_csv(os.path.join(chipseq_dir, "DepthNorm", "PC9", "DESeq2_Named_AllResults_foldpv_Sorted_Protein.txt"),
#                         sep="\t")
# merged_chip = pd.merge(
#     H1975_data[['gene', 'log2FoldChange']], 
#     PC9_data[['gene', 'log2FoldChange']], 
#     on='gene', 
#     suffixes=('_H1975', '_PC9'))

# plt.figure(figsize=(6, 6))
# sns.scatterplot(
#     data=merged_chip, 
#     x='log2FoldChange_H1975', 
#     y='log2FoldChange_PC9',
#     alpha=0.6
# )

# # Add an identity line (y=x) for comparison
# plt.plot([0, 4], [0, 4], color='red', linestyle='--')

# plt.title('PHOX2B ChIP-seq Correlation')
# plt.xlabel('H1975 log2FoldChange')
# plt.ylabel('PC9 log2FoldChange')
# plt.grid(True, linestyle=':', alpha=0.6)
# # List of genes to highlight
# highlight_genes = ['PHOX2B', 'FOXA1', 'SOX2']

# # Loop through the genes and add labels
# for gene in highlight_genes:
#     # Find the data point for the specific gene
#     target = merged_chip[merged_chip['gene'] == gene]
    
#     if not target.empty:
#         x_val = target['log2FoldChange_H1975'].values[0]
#         y_val = target['log2FoldChange_PC9'].values[0]
        
#         # Add the text label with an arrow
#         plt.annotate(
#             gene,
#             xy=(x_val, y_val), 
#             xytext=(x_val + 0.2, y_val + 0.2),  # Adjust offset as needed
#             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
#             fontsize=10,
#             fontweight='bold'
#         )
        
#         # Optionally make the specific point stand out
#         plt.scatter(x_val, y_val, color='red', s=50, edgecolor='black', zorder=5)
# plt.show()