"""
SEACells quality assessment — compactness and separation in diffusion space.

Metrics follow Persad et al. 2023 (Nature Biotechnology, Fig. 2):

  Compactness: for each metacell, compute the std of each diffusion component
    across its member cells, then average across components.
    Lower = cells within a metacell are more transcriptionally homogeneous.
    (The trivial zeroth diffusion component is excluded.)

  Separation: for each metacell, the Euclidean distance to its nearest
    metacell-centroid neighbour in diffusion space (computed within sample).
    Higher = metacells are more distinct from one another.

Outputs (figures/figure3/Numbat/):
  seacells_quality_compactness.pdf  — violin plots (compactness + separation)  [panel B]
  seacells_quality_umap.pdf         — UMAP with all metacell centroids (all samples)  [panel A]
  seacells_quality_summary.csv      — per-cell-type numeric table

Usage:
  /usersoftware/chanj3/SEACells/bin/python 1.5b_seacells_quality.py

-------------------------------------------------------------------------------
Figure caption
-------------------------------------------------------------------------------
Supplementary Figure 5. Analysis of shared single nucleotide and copy number
alterations patient samples with combined histology.

a. UMAP of the full single-cell atlas (gray background, 271k cells) with cells
   from the 36 analyzed samples colored by cell type. Filled circles with black
   outlines denote SEACells metacell centroids computed per sample; centroids
   are positioned at the mean UMAP coordinate of their member cells.

b. SEACells metacell quality metrics (Persad et al. 2023, Nat. Biotechnol.)
   across cell types. Left: compactness, defined as the mean standard deviation
   of diffusion-map coordinates across member cells within each metacell (lower
   values indicate greater transcriptional homogeneity). Right: separation,
   defined as the Euclidean distance in diffusion space to the nearest
   neighboring metacell centroid within the same sample (higher values indicate
   greater distinctness between metacells). Violins are colored by broad cell
   category (Tumor, Immune, Stromal/Epithelial); dots show individual metacell
   values; black horizontal lines indicate medians. Lower strips show the number
   of patients and total cells contributing to each cell type.

c. Per-patient SNV sharing and clonal phylogenies in tumors with admixed
   histology. Each panel shows one patient with ≥2 tumor cell types detected in
   scRNA-seq. Left: UpSet plots of somatic SNVs detected in scRNA-seq reads from
   MSK-IMPACT–genotyped variants. Bars indicate the number of mutations present
   in each intersection of tumor cell types (dot matrix below); gold bars mark
   intersections containing driver gene mutations. Right: Numbat single-cell
   phylogenetic trees inferred from allele-specific copy number profiles with
   MSK-IMPACT CNAs as prior information; nodes are colored by majority-vote tumor
   cell type assignment.

d. Box plot of median CNV burden (fraction of genes with copy-number
   alterations) per patient in LUAD (gold, n = 11) and NE (red, n = 16)
   populations. Each dot represents the median CNV burden of one patient's
   metacells for the indicated cell type. Gray lines connect paired patients with
   both LUAD and NE populations (n = 9). Box shows p-value from Mann–Whitney U
   test, two-sided.
"""

import argparse
import colorsys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import euclidean_distances

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
args = parser.parse_args()

IN_DIR      = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/"
H5AD_ALL    = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/"
               "scvi_latent150/adata.scvi.all_genes.final_subset.051826.h5ad")
SAMPLE_LIST = os.path.join(os.path.dirname(__file__), "sample_list.txt")
OUT_DIR     = os.path.dirname(__file__)

cell_type_col = "cell_type_final2"

# Diffusion components to use: skip index 0 (trivial constant eigenvector)
DIFF_COMPS = slice(1, None)

CT_PALETTE = {
    "SCLC-A":               "#E64B35",
    "SCLC-N":               "#4DBBD5",
    "SCLC-P":               "#9A3CC7",
    "NonNE_SCLC":           "#F39B7F",
    "FOXI1+ NonNE SCLC-N":  "#2ca02c",
    "SCLC-N (FOXI1+)":      "#2ca02c",
    "EMT":                  "#8491B4",
    "LUAD":                 "#00A087",
    "LUSC":                 "#3C5488",
    "CD4 TRM":         "#91D1C2",
    "CD4_EM/Effector":  "#B09C85",
    "CD4_naive/CM":     "#FCCDE5",
    "CD8 TRM/EM":      "#7570B3",
    "T_reg":            "#D95F02",
    "NK":               "#1B9E77",
    "ILC":              "#E7298A",
    "B_memory":         "#66A61E",
    "Mφ/Mono":        "#A6761D",
    "Mφ/Mono CCL":     "#666666",
    "Mφ/Mono CD11c":   "#6BAED6",
    "Mφ/Mono CD14":    "#2171B5",
    "cDC":              "#9ECAE1",
    "pDC":              "#AAAAAA",
    "Mast_cell ":       "#FD8D3C",
    "PMN":              "#FDAE6B",
    "Fibroblast":       "#74C476",
    "Endothelial":      "#A1D99B",
    "Club":             "#31A354",
    "Goblet":           "#006D2C",
    "Ciliated":         "#C7E9C0",
    "Basal":            "#FC9272",
}
DEFAULT_COLOR = "#BBBBBB"

CT_RENAME    = {"FOXI1+ NonNE SCLC-N": "SCLC-N (FOXI1+)"}

TUMOR_TYPES  = {"SCLC-A","SCLC-N","SCLC-P","NonNE_SCLC","SCLC-N (FOXI1+)","EMT","LUAD","LUSC"}

# Legend order for tumor cell types (non-tumor types follow, sorted by colour hue)
TUMOR_LEGEND_ORDER = [
    "LUAD", "LUSC", "SCLC-A", "SCLC-N", "SCLC-P",
   "SCLC-N (FOXI1+)", "EMT",  "NonNE SCLC", 
]
IMMUNE_TYPES = {"CD4 TRM","CD4_EM/Effector","CD4_naive/CM","CD8 TRM/EM",
                "T_reg","NK","ILC","B_memory",
                "Mφ/Mono","Mφ/Mono CCL","Mφ/Mono CD11c","Mφ/Mono CD14",
                "cDC","pDC","Mast_cell ","PMN"}
CAT_COLORS   = {"Tumor":"#E64B35","Immune":"#4DBBD5","Stromal/Epithelial":"#00A087"}

def cell_category(ct):
    if ct in TUMOR_TYPES:  return "Tumor"
    if ct in IMMUNE_TYPES: return "Immune"
    return "Stromal/Epithelial"

# ---------------------------------------------------------------------------
# 1. Load embeddings from h5ad (backed mode — skip loading X)
# ---------------------------------------------------------------------------
print("Loading h5ad embeddings ...")
adata = sc.read_h5ad(H5AD_ALL, backed="r")
X_diffmap = pd.DataFrame(
    adata.obsm["X_diffmap"],
    index=adata.obs_names,
)
X_umap = pd.DataFrame(
    adata.obsm["X_umap"],
    index=adata.obs_names,
    columns=["UMAP1","UMAP2"],
)
obs_meta = adata.obs[[cell_type_col, "sample"]].copy()

# Pull colour palette from adata.uns (scanpy convention: "{col}_colors" paired
# with the categorical category order).  Normalise \n → space to match labels.
uns_col_key = f"{cell_type_col}_colors"
if uns_col_key in adata.uns and hasattr(adata.obs[cell_type_col], "cat"):
    raw_cats   = adata.obs[cell_type_col].cat.categories.tolist()
    raw_colors = list(adata.uns[uns_col_key])
    CT_PALETTE = {c.replace("\n", " "): col
                  for c, col in zip(raw_cats, raw_colors)}
    print(f"  Using adata.uns palette ({len(CT_PALETTE)} cell types)")
else:
    print("  adata.uns palette not found; using hard-coded palette")

# Rename FOXI1+ label in palette (pre-rename key may appear if loaded from adata.uns)
for _old, _new in CT_RENAME.items():
    if _old in CT_PALETTE:
        CT_PALETTE[_new] = CT_PALETTE.pop(_old)

adata.file.close()
del adata
obs_meta[cell_type_col] = obs_meta[cell_type_col].str.replace("\n", " ", regex=False)
obs_meta[cell_type_col] = obs_meta[cell_type_col].replace(CT_RENAME)
print(f"  {len(X_diffmap)} cells, {X_diffmap.shape[1]} diffusion components "
      f"(using {X_diffmap.iloc[:, DIFF_COMPS].shape[1]} non-trivial)")

# ---------------------------------------------------------------------------
# 2. Per-sample: load metacell assignments, compute compactness + separation
# ---------------------------------------------------------------------------
samples     = open(SAMPLE_LIST).read().splitlines()
all_metrics = []
asgn_store  = {}   # for UMAP panel

print("\nComputing per-metacell quality metrics ...")
for sample_id in samples:
    patient_id = sample_id.split("_")[0]
    asgn_path  = os.path.join(IN_DIR, patient_id, "GEX", sample_id,
                               "metacell", f"{sample_id}.metacell_assignment.tsv")
    bc_path    = os.path.join(IN_DIR, patient_id, "GEX", sample_id, "barcodes.tsv")

    if not os.path.exists(asgn_path) or not os.path.exists(bc_path):
        print(f"  [{sample_id}] missing files; skipping.")
        continue

    decoded_barcodes = pd.read_csv(bc_path, header=None)[0].tolist()
    h5ad_barcodes    = obs_meta.index[obs_meta["sample"] == sample_id].tolist()

    if len(h5ad_barcodes) != len(decoded_barcodes):
        print(f"  [{sample_id}] barcode count mismatch; skipping.")
        continue

    decoded_to_h5ad = dict(zip(decoded_barcodes, h5ad_barcodes))

    # Load cell→metacell assignments
    asgn = pd.read_csv(asgn_path, sep="\t", index_col=0)
    asgn.index.name = "cell"

    # Map decoded → h5ad barcodes for embedding lookup
    h5ad_bcs = pd.Series(asgn.index.map(decoded_to_h5ad), index=asgn.index)
    asgn["h5ad_bc"]     = h5ad_bcs.values
    asgn[cell_type_col] = obs_meta.loc[h5ad_bcs.values, cell_type_col].values

    # Diffusion coords for this sample's cells
    diff_sample = X_diffmap.loc[asgn["h5ad_bc"].values, :].iloc[:, DIFF_COMPS]
    diff_sample.index = asgn.index   # re-index by decoded barcode

    # ---- Per-metacell compactness -------------------------------------------
    # Compactness = mean std of diffusion components within metacell (Persad 2023)
    records   = []
    centroids = {}

    for mc, group in asgn.groupby("SEACell"):
        cells = group.index.tolist()
        emb   = diff_sample.loc[cells].values   # n_cells × n_diff_comps
        n     = len(cells)

        compactness = emb.std(axis=0).mean() if n > 1 else 0.0
        centroid    = emb.mean(axis=0)
        centroids[mc] = centroid

        records.append({
            "sample":      sample_id,
            "metacell":    mc,
            "cell_type":   group[cell_type_col].iloc[0],
            "n_cells":     n,
            "compactness": compactness,
            "passthrough": n == 1,
        })

    df_s = pd.DataFrame(records)

    # ---- Per-metacell separation --------------------------------------------
    # Separation = distance to nearest other metacell centroid (Persad 2023)
    mc_ids    = list(centroids.keys())
    cent_mat  = np.array([centroids[m] for m in mc_ids])

    if len(mc_ids) > 1:
        dist_mat = euclidean_distances(cent_mat)
        np.fill_diagonal(dist_mat, np.inf)
        nn_dist  = dist_mat.min(axis=1)
    else:
        nn_dist = np.zeros(len(mc_ids))

    sep_map = dict(zip(mc_ids, nn_dist))
    df_s["separation"] = df_s["metacell"].map(sep_map)

    all_metrics.append(df_s)
    asgn_store[sample_id] = (asgn, diff_sample)
    print(f"  [{sample_id}] {len(df_s)} metacells | "
          f"compactness={df_s['compactness'].mean():.4f} | "
          f"separation={df_s['separation'].mean():.4f}")

if not all_metrics:
    print("No completed samples found. Run after 1.5_seacells_aggregation.py finishes.")
    raise SystemExit(0)

metrics = pd.concat(all_metrics, ignore_index=True)
metrics["category"] = metrics["cell_type"].map(cell_category).fillna("Stromal/Epithelial")

n_samples   = metrics["sample"].nunique()
n_mc        = len(metrics)
n_pass      = metrics["passthrough"].sum()
print(f"\nTotal: {n_samples} samples, {n_mc} metacells ({n_pass} passthrough)")

# ---------------------------------------------------------------------------
# 3. Collect UMAP coords for ALL sample cells and ALL metacell centroids
# ---------------------------------------------------------------------------
print("\nBuilding UMAP coordinates for all samples ...")
cell_umap_records     = []
centroid_umap_records = []

for sample_id, (asgn_s, _) in asgn_store.items():
    umap_s = X_umap.loc[asgn_s["h5ad_bc"].values].copy()
    umap_s.index = asgn_s.index

    # Individual cells
    cell_umap_records.append(pd.DataFrame({
        "UMAP1":       umap_s["UMAP1"].values,
        "UMAP2":       umap_s["UMAP2"].values,
        cell_type_col: asgn_s[cell_type_col].values,
    }))

    # Metacell centroids = mean UMAP of member cells
    tmp = asgn_s[[cell_type_col, "SEACell"]].copy()
    tmp["UMAP1"] = umap_s["UMAP1"].values
    tmp["UMAP2"] = umap_s["UMAP2"].values
    centroids_s = (
        tmp.groupby("SEACell")[["UMAP1", "UMAP2", cell_type_col]]
        .agg({"UMAP1": "mean", "UMAP2": "mean", cell_type_col: "first"})
        .reset_index()
    )
    centroid_umap_records.append(centroids_s)

all_cells_umap    = pd.concat(cell_umap_records,    ignore_index=True)
all_centroids_umap = pd.concat(centroid_umap_records, ignore_index=True)

# Keep only aggregated metacells (n_cells > 1) for centroid overlay
passthrough_ids = set(metrics.loc[metrics["passthrough"], "metacell"])
all_centroids_umap = all_centroids_umap[
    ~all_centroids_umap["SEACell"].isin(passthrough_ids)
].reset_index(drop=True)

# Sort: tumour types first (in defined order), then non-tumour by colour hue
def _ct_legend_key(ct):
    if ct in TUMOR_LEGEND_ORDER:
        return (0, TUMOR_LEGEND_ORDER.index(ct), 0, 0)
    hex_col = CT_PALETTE.get(ct, DEFAULT_COLOR).lstrip("#")
    r, g, b = (int(hex_col[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (1, 0, h, 1 - v)

ct_all = sorted(all_cells_umap[cell_type_col].unique(), key=_ct_legend_key)

# ---------------------------------------------------------------------------
# 4. Figure 1 — UMAP with all metacell centroids (all samples)
# ---------------------------------------------------------------------------
print("Plotting UMAP ...")
fig_u, ax_u = plt.subplots(figsize=(6, 5))
fig_u.subplots_adjust(right=0.70)   # reserve space for legend on the right

# Layer 1: full atlas background (all 271k cells, gray, rasterized)
ax_u.scatter(X_umap["UMAP1"], X_umap["UMAP2"],
             s=0.3, c="#DDDDDD", alpha=0.15, rasterized=True, zorder=1)

# Layer 2: cells from analyzed samples, coloured by cell type
for ct in ct_all:
    mask  = all_cells_umap[cell_type_col] == ct
    color = CT_PALETTE.get(ct, DEFAULT_COLOR)
    ax_u.scatter(all_cells_umap.loc[mask, "UMAP1"],
                 all_cells_umap.loc[mask, "UMAP2"],
                 s=2, c=color, alpha=0.5, rasterized=True, zorder=2)

# Layer 3: metacell centroids — same colour, larger, black border
for ct in ct_all:
    mask  = all_centroids_umap[cell_type_col] == ct
    color = CT_PALETTE.get(ct, DEFAULT_COLOR)
    ax_u.scatter(all_centroids_umap.loc[mask, "UMAP1"],
                 all_centroids_umap.loc[mask, "UMAP2"],
                 s=18, c=color, edgecolors="black", linewidths=0.2,
                 zorder=3)

# Legend — outside the axes to the right
legend_els = [
    Line2D([0],[0], marker="o", color="w",
           markerfacecolor=CT_PALETTE.get(ct, DEFAULT_COLOR),
           markeredgecolor="none", markersize=6, label=ct)
    for ct in ct_all
]
# Glyph explanation at the bottom of the legend
legend_els += [
    Line2D([0],[0], linestyle="none"),   # blank spacer
    Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="black", markeredgewidth=0.8,
           markersize=6,  label="Metacell centroid"),
]
ax_u.legend(
    handles=legend_els,
    fontsize=7,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    framealpha=0.9,
    handlelength=1,
    ncol=2,
    title="Cell type",
    title_fontsize=8,
)

n_centroids = len(all_centroids_umap)
ax_u.set_title("SEACells metacell centroids", fontsize=10)
ax_u.set_xlabel("UMAP1", fontsize=10)
ax_u.set_ylabel("UMAP2", fontsize=10)
ax_u.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for sp in ax_u.spines.values():
    sp.set_visible(False)

out_u = os.path.join(OUT_DIR, "seacells_quality_umap.pdf")
fig_u.savefig(out_u, dpi=200, bbox_inches="tight")
plt.close(fig_u)
print(f"Saved UMAP → {out_u}")

# ---------------------------------------------------------------------------
# 5. Figure 2 — Compactness + Separation violins with n-patients and n-cells strips
# ---------------------------------------------------------------------------
# Aggregated metacells only (exclude passthrough n=1)
metrics_agg = metrics[~metrics["passthrough"]].copy()
metrics_agg["patient"] = metrics_agg["sample"].str.split("_").str[0]

ct_npatients = metrics_agg.groupby("cell_type")["patient"].nunique()
ct_ncells    = metrics_agg.groupby("cell_type")["n_cells"].sum()

# Order cell types by median compactness (ascending — tighter first)
ct_order = (
    metrics_agg.groupby("cell_type")["compactness"]
    .median().sort_values().index.tolist()
)
ct_present = [ct for ct in ct_order if ct in metrics_agg["cell_type"].values]

# Layout: violin | n_patients strip | n_cells strip  (3 rows × 2 cols)
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(
    3, 2,
    height_ratios=[5, 1, 1],
    hspace=0.05,
    wspace=0.25,
    top=0.93, bottom=0.22, left=0.07, right=0.98,
)
ax_compact = fig.add_subplot(gs[0, 0])
ax_sep     = fig.add_subplot(gs[0, 1])
ax_np_l    = fig.add_subplot(gs[1, 0], sharex=ax_compact)
ax_np_r    = fig.add_subplot(gs[1, 1], sharex=ax_sep)
ax_nc_l    = fig.add_subplot(gs[2, 0], sharex=ax_compact)
ax_nc_r    = fig.add_subplot(gs[2, 1], sharex=ax_sep)

plot_specs = [
    ("compactness", "Compactness", ax_compact, ax_np_l, ax_nc_l),
    ("separation",  "Separation",  ax_sep,     ax_np_r, ax_nc_r),
]

rng = np.random.default_rng(42)
for metric, ylabel, ax_v, ax_np, ax_nc in plot_specs:
    ct_in     = [ct for ct in ct_present if ct in metrics_agg["cell_type"].values]
    positions = list(range(len(ct_in)))
    bp_data   = [metrics_agg.loc[metrics_agg["cell_type"] == ct, metric].values
                 for ct in ct_in]
    cat_cols  = [CAT_COLORS[cell_category(ct)] for ct in ct_in]

    # Violin
    parts = ax_v.violinplot(bp_data, positions=positions, widths=0.7,
                            showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], cat_cols):
        pc.set_facecolor(col); pc.set_alpha(0.55)
        pc.set_edgecolor("#333333"); pc.set_linewidth(0.5)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.8)
    for i, (d, col) in enumerate(zip(bp_data, cat_cols)):
        jitter = rng.uniform(-0.15, 0.15, size=len(d))
        ax_v.scatter(i + jitter, d, s=4, c=col, alpha=0.35, zorder=3, linewidths=0)
    ax_v.set_ylabel(ylabel, fontsize=14)
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)
    plt.setp(ax_v.get_xticklabels(), visible=False)

    # n-patients lollipop strip
    npts = [ct_npatients.get(ct, 0) for ct in ct_in]
    for i, (n, col) in enumerate(zip(npts, cat_cols)):
        ax_np.plot([i, i], [0, n], color=col, linewidth=1.5, alpha=0.7, zorder=2)
    ax_np.scatter(positions, npts, c=cat_cols, s=45, zorder=3, edgecolors="none")
    ax_np.set_ylim(0, max(npts) * 1.35)
    ax_np.set_ylabel("# patients", fontsize=9)
    ax_np.spines["top"].set_visible(False)
    ax_np.spines["right"].set_visible(False)
    ax_np.tick_params(axis="y", labelsize=8)
    plt.setp(ax_np.get_xticklabels(), visible=False)

    # n-cells lollipop strip (total cells aggregated per cell type)
    ncells = [ct_ncells.get(ct, 0) for ct in ct_in]
    for i, (n, col) in enumerate(zip(ncells, cat_cols)):
        ax_nc.plot([i, i], [0, n], color=col, linewidth=1.5, alpha=0.7, zorder=2)
    ax_nc.scatter(positions, ncells, c=cat_cols, s=45, zorder=3, edgecolors="none")
    ax_nc.set_xticks(positions)
    ax_nc.set_xticklabels(ct_in, rotation=90, ha="center", fontsize=9)
    ax_nc.set_ylim(0, max(ncells) * 1.15)
    ax_nc.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax_nc.set_ylabel("# cells", fontsize=9)
    ax_nc.spines["top"].set_visible(False)
    ax_nc.spines["right"].set_visible(False)
    ax_nc.tick_params(axis="y", labelsize=8)

legend_patches = [
    mpatches.Patch(color=CAT_COLORS[cat], alpha=0.7, label=cat)
    for cat in ["Tumor", "Immune", "Stromal/Epithelial"]
]
ax_compact.legend(handles=legend_patches, fontsize=10,
                  title="Cell category", title_fontsize=11,
                  loc="upper left", framealpha=0.9)

out_q = os.path.join(OUT_DIR, "seacells_quality_compactness.pdf")
fig.savefig(out_q, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved quality violins → {out_q}")

# ---------------------------------------------------------------------------
# 6. Summary CSV
# ---------------------------------------------------------------------------
summary = (
    metrics.groupby("cell_type")
    .agg(
        n_metacells      =("metacell",     "count"),
        n_cells_median   =("n_cells",      "median"),
        compactness_median=("compactness", "median"),
        separation_median =("separation",  "median"),
        passthrough_frac  =("passthrough", "mean"),
    )
    .round(4)
    .sort_values("compactness_median")
)
print("\nPer-cell-type summary:")
print(summary.to_string())

out_csv = os.path.join(OUT_DIR, "seacells_quality_summary.csv")
summary.to_csv(out_csv)
print(f"\nSummary table → {out_csv}")
print("Done.")
