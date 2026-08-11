import os
from functools import reduce
import re
import scanpy as sc
import pandas as pd
import numpy as np
import gseapy as gp
from gseapy.plot import gseaplot2
import matplotlib as mpl
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.stats import pearsonr
sc.settings.figdir = "../../results/figures/figure8/"
sc.set_figure_params(
    dpi=120,
    dpi_save=600,
    figsize=(2, 2.5),
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

# ------------ main analysis ------------

# adata = sc.read_h5ad('../../external_data/phox2b/PHOX2B_postprocessing_scvi.h5ad')
# adata.obs.cell_line = adata.obs.cell_line.str.split('_').str[0]

# # 8A
# sc.pl.umap(adata, color=["cell_line"], frameon=False, 
#            legend_loc="on data", 
#            legend_fontoutline=2,
#            title=["Cell line"], 
#            save="_8A.pdf", wspace=0)

# # 8B Subclustering
# k_list   = [10, 15, 20, 30, 40]
# res_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
# random_state = 0

# for kk in k_list:
#     sc.pp.neighbors(adata, n_neighbors=kk, n_pcs=None, use_rep="X_scVI", key_added=f'neighbors_k{kk}')
#     for rr in res_list:
#         key = f"leiden_k{kk}_r{rr}"
#         sc.tl.leiden(
#             adata,
#             resolution=rr,
#             random_state=random_state,
#             key_added=key,
#             neighbors_key=f'neighbors_k{kk}',
#         )
#         # make categorical
#         adata.obs[key] = adata.obs[key].astype('category')

# leiden_df = adata.obs[[f"leiden_k{kk}_r{rr}" for kk in k_list for rr in res_list]].copy()
# leiden_df.insert(0, "cell_id", adata.obs_names)
# leiden_df.to_csv('../../data/figures/figure8/8B_subclustering.csv', index=False)

# # S? ARI plot
# leiden_df = pd.read_csv('../../data/figures/figure8/8B_subclustering.csv')
# k0  = 30
# r0  = 1.4

# def pairwise_ari_matrix(items, getter):
#     n = len(items)
#     mat = np.ones((n, n), dtype=float)
#     labels = [getter(v) for v in items]
#     for i in range(n):
#         for j in range(i+1, n):
#             mat[i, j] = mat[j, i] = ARI(labels[i], labels[j])
#     return mat

# ari_res_mat = pairwise_ari_matrix(
#     res_list,
#     getter=lambda r: leiden_df[f"leiden_k{k0}_r{r}"],
# )
# ari_k_mat = pairwise_ari_matrix(
#     k_list,
#     getter=lambda k: leiden_df[f"leiden_k{k}_r{r0}"],
# )

# fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), constrained_layout=True)

# im0 = axes[0].imshow(ari_res_mat, vmin=0, vmax=1, aspect='equal', cmap='Spectral_r')
# axes[0].set_title(f"ARI across resolutions")
# axes[0].set_xticks(range(len(res_list))); axes[0].set_xticklabels(res_list, rotation=45, ha='right')
# axes[0].set_yticks(range(len(res_list))); axes[0].set_yticklabels(res_list)
# axes[0].set_xlabel("Resolution"); axes[0].set_ylabel("Resolution")
# if r0 in res_list:
#     idx = res_list.index(r0)
#     axes[0].text(
#         idx, idx, "*", fontsize=12,
#         color="black", ha="center", va="center", weight="bold"
#     )

# im1 = axes[1].imshow(ari_k_mat, vmin=0, vmax=1, aspect='equal', cmap='Spectral_r')
# axes[1].set_title(f"ARI across k")
# axes[1].set_xticks(range(len(k_list))); axes[1].set_xticklabels(k_list, rotation=45, ha='right')
# axes[1].set_yticks(range(len(k_list))); axes[1].set_yticklabels(k_list)
# axes[1].set_xlabel("k neighbors"); axes[1].set_ylabel("k neighbors")
# if k0 in k_list:
#     idx = k_list.index(k0)
#     axes[1].text(
#         idx, idx, "*", fontsize=12,
#         color="black", ha="center", va="center", weight="bold"
#     )

# cbar = fig.colorbar(im1, ax=axes, shrink=0.9)
# cbar.set_label("Adjusted Rand Index (ARI)")
# plt.show()
# fig.savefig(os.path.join(sc.settings.figdir, "8B_subclustering_ARI_heatmaps.pdf"))

# # 8B UMAP
# plt.rcParams.update({"figure.figsize": (2, 2.5)})
# adata.obs["leiden"] = leiden_df["leiden_k30_r1.4"].astype('category').values
# adata.obs["leiden"] = adata.obs["leiden"].replace({21: 0})
# adata.obs["leiden"] = adata.obs["leiden"].astype("category")
# sc.pl.umap(adata, color=["leiden"], frameon=False, legend_loc="on data", 
#            legend_fontoutline=2, 
#            title="Leiden (k = 30, res = 1.4)",
#            save="_8B.pdf")

# # 8B PHOX2B
# cluster_means = adata.to_df()[["PHOX2B"]].groupby(adata.obs["leiden"]).mean()
# cluster_order = (
#     cluster_means.sort_values(by=["PHOX2B"], ascending=[False])
#     .index
#     .tolist()
# )
# cluster_order = [str(i) for i in cluster_order]
# plt.rcParams.update({"figure.figsize": (5, 2.5)})
# fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), constrained_layout=True)
# ax = sc.pl.violin(
#     adata,
#     keys=["PHOX2B"],
#     groupby="leiden",
#     order=cluster_order,
#     stripplot=False,
#     legend=None,
#     ax=ax,
#     show=False
# )
# ax.set_title("PHOX2B")
# for pos in [cluster_order.index("7"), cluster_order.index("0"), cluster_order.index("17")]:
#     ax.axvspan(pos-0.5, pos+0.5, color='gold', alpha=0.5, zorder=0)
# fig.savefig(os.path.join(sc.settings.figdir, "8B_PHOX2B_violin.pdf"))

# # 8B UMAP highlight 0 & 7
# target_pairs = [("PC9", "0"), ("H1975", "7"), ("H1975", "17")]  # (cell_line, leiden cluster)
# edge_colors  = {("PC9","0"): "#FF4D00", ("H1975","7"): "steelblue", ("H1975", "17"): "steelblue"}  # orange / blue
# dense_quantile = 0.80   # keep top 80% densest points for hull
# k_knn         = 20      # neighbors for density score
# fill_alpha    = 0.15

# # fonts export-friendly
# mpl.rcParams.update({
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
#     "pdf.fonttype": 42, "ps.fonttype": 42
# })

# # ---- data ----
# X        = adata.obsm["X_umap"]
# expr     = adata[:,"PHOX2B"].X.toarray()
# leiden   = adata.obs["leiden"].astype(str).to_numpy()
# cellline = adata.obs["cell_line"].astype(str).to_numpy()

# vmin, vmax = np.percentile(expr, [1, 99])

# def dense_core(pts, keep_q=0.90, k=20):
#     if len(pts) < 3:
#         return np.arange(len(pts))
#     k = min(k, len(pts)-1)
#     nn = NearestNeighbors(n_neighbors=k).fit(pts)
#     d, _ = nn.kneighbors(pts)
#     score = d[:, 1:].mean(axis=1)                 # smaller = denser
#     thr = np.quantile(score, keep_q)
#     keep = np.where(score <= thr)[0]
#     return keep if len(keep) >= 3 else np.arange(len(pts))

# def draw_hull(ax, pts, edge, alpha=0.15, lw=2):
#     if pts.shape[0] < 3: 
#         return
#     hull = ConvexHull(pts)
#     poly = pts[hull.vertices]
#     ax.plot(poly[:,0], poly[:,1], color=edge, lw=lw)
#     ax.add_patch(plt.Polygon(poly, closed=True, facecolor=edge, alpha=alpha, edgecolor="none"))

# # ---- plot ----
# fig, ax = plt.subplots(figsize=(2.3, 2.5))
# scatter = ax.scatter(X[:,0], X[:,1], c=expr, s=3, linewidths=0, cmap="Spectral_r", vmin=vmin, vmax=vmax)
# ax.axis("off")
# ax.set_title("PHOX2B-high subclusters", weight="bold", fontsize=10)

# for cl, c in target_pairs:
#     mask = (cellline == cl) & (leiden == c)
#     pts = X[mask]
#     if pts.shape[0] == 0:
#         continue
#     pts_core = pts[dense_core(pts, keep_q=dense_quantile, k=k_knn)]
#     edge = edge_colors[(cl, c)]
#     draw_hull(ax, pts_core, edge=edge, alpha=fill_alpha, lw=1.5)
#     cx, cy = pts_core.mean(axis=0) if len(pts_core) else pts.mean(axis=0)
#     ax.text(cx, cy + 1.5, f"{cl}(c{c})",
#             ha="center", va="bottom", fontsize=8, weight="bold", color="black",
#             bbox=dict(facecolor="white", alpha=0.2, lw=0, pad=1.5))
# cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
# #cbar.set_label("PHOX2B expression", fontsize=8, weight="bold")
# cbar.ax.tick_params(labelsize=7)
# plt.savefig(os.path.join(sc.settings.figdir, "8B_PHOX2B_circled.pdf"), dpi=600, bbox_inches="tight", transparent=True)
# plt.show()

# #----------------------------------------------------------------------------
# # Fig8
# sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
# # ORA (upregulated genes)
# def remove_pmid(s):
#     s = re.sub(r"\b\d{6,}\b", "", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def bold_tf(s):
#     # remove PMID
#     s = re.sub(r"\b\d{6,}\b", "", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     # wrap first token in mathtext bold
#     return re.sub(r"^(\S+)", r"$\\mathbf{\1}$", s)

# cluster_of_interest = ['7', '0', '17']
# res_dict = {}
# for cluster in cluster_of_interest:
#     rg = adata.uns['rank_genes_groups']
#     df = pd.DataFrame({
#         'gene': rg['names'][cluster],
#         'pval_adj': rg['pvals_adj'][cluster],
#         'logFC': rg['logfoldchanges'][cluster],
#         'score': rg['scores'][cluster]
#     })
#     rnk = df[['gene', 'score']].sort_values('score', ascending=False)
#     N = 200
#     df_up = (
#         rnk
#         .head(N)
#     )
#     gene_list = df_up.iloc[:,0].dropna().drop_duplicates().tolist()
#     print(f"ORA input genes (n={len(gene_list)}): first 10 →", gene_list[:10])
#     gene_sets = ['ChEA_2022']
#     outdir = f"../../data/processed/"
#     enr = gp.enrichr(
#         gene_list=gene_list,
#         gene_sets=gene_sets,
#         organism="Human",      # or "Mouse"
#         outdir=outdir,
#         cutoff=0.05,           # adjusted p-value cutoff
#         no_plot=True,          # we’ll make custom plots below
#         verbose=True,
#     )
    
#     res = enr.results.sort_values(["Gene_set", "Adjusted P-value"])
#     res_dict[cluster] = res.copy()
#     res["Term"] = res["Term"].apply(remove_pmid).apply(bold_tf)


# res_dict_list = []
# for cluster in cluster_of_interest:
#     res_dict_list.append(res_dict[cluster][['Term', 'Combined Score', 'Adjusted P-value']].rename(columns={"Combined Score": f"Combined Score (cluster {cluster})",
#                                                                                                            "Adjusted P-value": f"Adjusted P-value (cluster {cluster})"}))
# res_merged = reduce(lambda left, right: pd.merge(left, right, on="Term", how="outer"), res_dict_list)
# res_merged.to_csv('../../data/figures/figure8/8B_ORA_results.csv', index=False)

# # ---------- Build tidy DF from your dict ----------
# res_merged = pd.read_csv('../../data/figures/figure8/8B_ORA_results.csv')
# def tf_from_term(term: str) -> str:
#     s = re.sub(r"\b\d{6,}\b", "", term or "")  # remove PMID-like numbers
#     s = re.sub(r"\s+", " ", s).strip()
#     return s.split(" ", 1)[0] if s else ""

# def parse_species(term: str) -> str:
#     s = (term or "").upper()
#     if re.search(r'\b(HUMAN|HOMO SAPIENS)\b', s): return "Human"
#     if re.search(r'\b(MOUSE|MUS MUSCULUS)\b', s): return "Mouse"
#     return "Unknown"

# df = res_merged.copy()
# df["TF"] = df["Term"].apply(tf_from_term)
# df["Species"] = df["Term"].apply(parse_species)
# df = df.copy()
# df["sum_score"] = df["Combined Score (cluster 7)"] + df["Combined Score (cluster 0)"] + df["Combined Score (cluster 7)"] + df["Combined Score (cluster 17)"]
# df.dropna(inplace=True)
# df = df.loc[df.groupby("TF")["sum_score"].idxmax()].reset_index(drop=True)
# df = df.sort_values("sum_score", ascending=False).reset_index(drop=True)

# def plot_scatter_with_annotations(df, xcol, ycol, xlab, ylab, save_path):
#     # 2) Choose how to define "high in both"
#     q = 0.9
#     x_thr = df[xcol].quantile(q)
#     y_thr = df[ycol].quantile(q)
#     mask = (df[xcol] >= x_thr) & (df[ycol] >= y_thr)

#     # --- plotting ---
#     fig, ax = plt.subplots(figsize=(3, 3))

#     # 1) background points (unannotated) in neutral color
#     ax.scatter(df.loc[~mask, xcol], df.loc[~mask, ycol], s=18, alpha=0.25, color="lightgray", edgecolor="none")

#     # 2) highlighted points, colored by species
#     palette = {"Human": "tab:blue", "Mouse": "tab:orange", "Unknown": "tab:green"}
#     markers = {"Human": "o", "Mouse": "s", "Unknown": "D"}  # optional: different markers

#     texts = []  # collect annotations

#     for species in ["Human", "Mouse", "Unknown"]:
#         sub = df[mask & (df["Species"] == species)]
#         if not sub.empty:
#             ax.scatter(
#                 sub[xcol], sub[ycol],
#                 s=10, alpha=0.9,
#                 color=palette[species],
#                 marker=markers[species],
#                 edgecolor="black",
#                 linewidth=0.6,
#                 label=species
#             )
#             # store text objects instead of drawing immediately
#             for _, row in sub.iterrows():
#                 t = ax.text(
#                     row[xcol], row[ycol],
#                     row["TF"],
#                     fontweight="bold",
#                     color=palette[species],
#                     ha="center", va="center"
#                 )
#                 texts.append(t)

#     # After all labels are added:
#     adjust_text(
#         texts,
#         ax=ax,
#         arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5)
#     )

#     # thresholds to show annotated quadrant (optional)
#     ax.axvline(x_thr, linestyle="--", linewidth=1, color="gray", alpha=0.6)
#     ax.axhline(y_thr, linestyle="--", linewidth=1, color="gray", alpha=0.6)

#     ax.set_xlabel(xlab)
#     ax.set_ylabel(ylab)
#     ax.grid(False)

#     # legend only for species that appear
#     handles, labels = ax.get_legend_handles_labels()
#     if handles:
#         ax.legend(title="Species", loc="upper right", 
#                 handletextpad=0.1,
#                 labelspacing=0.1,
#                 bbox_to_anchor=(1.0, 1.0), frameon=False)
#     plt.tight_layout()
#     plt.savefig(os.path.join(sc.settings.figdir, save_path), dpi=300)
#     plt.show()

# plot_scatter_with_annotations(df, "Combined Score (cluster 7)", 
#                               "Combined Score (cluster 0)", 
#                               "Enriched TF in PHOX2B-high H1975 (c7)",
#                               "Enriched TF in PHOX2B-high PC9 (c0)",
#                               "8B_ORA_scatter_PHOX2B_clusters_7_vs_0.pdf")
# plot_scatter_with_annotations(df, 
#                               "Combined Score (cluster 17)", 
#                               "Combined Score (cluster 7)", 
#                               "Enriched TF in PHOX2B-high H1975 (c17)",
#                               "Enriched TF in PHOX2B-high H1975 (c7)",
#                               "8B_ORA_scatter_PHOX2B_clusters_17_vs_7.pdf")
# adata.write("../../data/figures/figure8/8B_ORA_scatter_PHOX2B_clusters.h5ad")
# sc.pl.rank_genes_groups(adata, groups = ['0', '7', '17'], 
#                         save="_8B_ranked_genes.pdf")
phox2b_clusters = ['0', '7', '17']
adata = sc.read_h5ad("../../data/figures/figure8/8B_ORA_scatter_PHOX2B_clusters.h5ad")
sc.pl.umap(adata, color =['EGFR', 'PHOX2B'], save="_Fig8_PHOX2B_EGFR.pdf")

# GSEA (MAPK pathway)
rank = sc.get.rank_genes_groups_df(adata, group=None)
kegg = gp.get_library(name="KEGG_2021_Human")
mapk_term = "MAPK signaling pathway"
pat = re.compile(mapk_term, re.I)
gene_sets = {k:v for k,v in {**kegg}.items() if pat.search(k)}
plots = {}
results = {}
for cl in phox2b_clusters:
    df = rank[rank["group"] == cl]
    prerank = df[["names", "logfoldchanges"]].dropna()
    prerank = prerank.sort_values("logfoldchanges", ascending=False)

    gsea_res = gp.prerank(
        rnk=prerank,
        gene_sets=gene_sets,
        processes=4,
        min_size=10,
        max_size=500,
        permutation_num=10000,
        outdir=None,
        seed=42
    )
    fig = gseaplot2(
        rank_metric=gsea_res.ranking,
        terms=[mapk_term],
        hits=[gsea_res.results[mapk_term]['hits']],
        RESs=[gsea_res.results[mapk_term]['RES']],     # provides es/nes/fdr/hits/RES/leading-edge
        figsize=(4,3),
        legend_kws={"loc": None},
        ofname=os.path.join(sc.settings.figdir, f"Fig8_PHOX2B_GSEA_cluster{cl}.pdf")
    )
    #fig[1].set_title(f"Cluster {cl} — {mapk_term}")
    plots[cl] = fig
    results[cl] = gsea_res.res2d

summary = pd.concat([
    res.assign(cluster=cl)
    for cl, res in results.items()
])
summary = summary[["cluster", "Term", "NES", "FDR q-val"]]
summary = summary.sort_values(["cluster", "NES"])


# ------------ Pearson Correlation analysis ------------
rng = np.random.default_rng(42)
cell_lines = ["PC9", "H1975"]
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)

summary = []
for i, line in enumerate(cell_lines):
    mask = (adata.obs["cell_line"].astype(str) == line).values
    x = get_gene_vec(adata, "PHOX2B", mask=mask, log1p=True)
    y = get_gene_vec(adata, "EGFR",   mask=mask, log1p=True)

    # observed Pearson
    r_obs, p_obs = pearson_obs(x, y)

    # 2) random-gene null (PHOX2B vs random genes)
    null_rand = random_gene_null(adata, y=x, mask=mask, n_bg=1000, log1p=True, seed=999 + i)
    p_emp_rand = empirical_p(null_rand, r_obs, tail="two-sided")

    # plots
    hist_with_observed(axes[i], null_rand, r_obs,
                       f"{line}: Random-gene null\n(empirical p={p_emp_rand:.3g})")

    summary.append({
        "cell_line": line,
        "r_obs": r_obs,
        "pearson_p (scipy)": p_obs,
        "empirical_p (random_gene)": p_emp_rand,
        "n_cells": mask.sum()
    })

plt.tight_layout()
plt.savefig(os.path.join(sc.settings.figdir, "Fig8_PHOX2B_EGFR_correlation.pdf"), dpi=300)
plt.show()

pd.DataFrame(summary)