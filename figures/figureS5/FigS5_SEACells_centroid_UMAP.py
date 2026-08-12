"""
SEACells metacell centroid UMAP — Supplementary Figure 5, panel a.

Outputs (figures/figureS5/):
  seacells_quality_umap.pdf

Usage:
  /usersoftware/chanj3/SEACells/bin/python seacells_centroid_umap.py

-------------------------------------------------------------------------------
Figure caption
-------------------------------------------------------------------------------
a. UMAP of the full single-cell atlas (gray background, 271k cells) with cells
   from the 36 analyzed samples colored by cell type. Filled circles with black
   outlines denote SEACells metacell centroids computed per sample; centroids
   are positioned at the mean UMAP coordinate of their member cells.
"""

import colorsys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUMBAT_DIR  = "/data1/chanj3/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/figure3/Numbat"
IN_DIR      = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/"
H5AD_ALL    = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/"
               "scvi_latent150/adata.scvi.all_genes.final_subset.081126.h5ad")
SAMPLE_LIST = os.path.join(NUMBAT_DIR, "sample_list.txt")
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))

cell_type_col = "cell_type_final2"

# Colors extracted (visual estimate) from the reference atlas UMAP figure.
CT_PALETTE = {
    'AT1': '#f3f3f3ff',
    'AT2': '#e2e2e2ff',
    'B_memory': '#fdd9b4ff',
    'B_naive': '#fda762ff',
    'B_plasma': '#f3701bff',
    'Basal': '#cececeff',
    'CD4\nTRM': '#deebf7ff',
    'CD4_EM/Effector': '#c6dbefff',
    'CD4_naive/CM': '#9dcae1ff',
    'CD8\nTRM/EM': '#6aaed6ff',
    'Cholangiocytes': '#404040ff',
    'Ciliated': '#b5b5b5ff',
    'Club': '#959595ff',
    'EMT': '#8C564B',
    'Endothelial': '#e32f27ff',
    'SCLC-N (FOXI1+)': '#2CA02C',
    'Fibroblast': '#fca082ff',
    'Goblet': '#5f5f5fff',
    'Hepatocytes': '#1d1d1dff',
    'ILC': '#08509bff',
    'LUAD': '#ffd700ff',
    'LUSC': '#FF7F0E',
    'Mast_cell ': '#2f974eff',
    'Mesothelia': '#7a7a7aff',
    'Mφ/Mono': '#e9f7e5ff',
    'Mφ/Mono CCL': '#d3eecdff',
    'Mφ/Mono CD11c': '#b8e3b2ff',
    'Mφ/Mono CD14': '#98d594ff',
    'Mφ_alveolar': '#73c476ff',
    'NK': '#2070b4ff',
    'NonNE SCLC': '#9467bdff',
    'PMN': '#4bb062ff',
    'RBC': '#c54102ff',
    'SCLC-A': '#d62728ff',
    'SCLC-N': '#17becfff',
    'SCLC-P': '#1f77b4ff',
    'T_reg': '#4191c6ff',
    'cDC': '#157f3bff',
    'pDC': '#006428ff',
}
DEFAULT_COLOR = "#BBBBBB"

CT_RENAME = {"FOXI1+ NonNE SCLC-N": "SCLC-N (FOXI1+)"}

TUMOR_LEGEND_ORDER = [
    "LUAD", "LUSC", "SCLC-A", "SCLC-N", "SCLC-P",
    "SCLC-N (FOXI1+)", "EMT", "NonNE SCLC",
]

# ---------------------------------------------------------------------------
# 1. Load UMAP embedding + cell-type labels from h5ad (backed mode)
# ---------------------------------------------------------------------------
print("Loading h5ad embeddings ...")
adata = sc.read_h5ad(H5AD_ALL, backed="r")
X_umap = pd.DataFrame(
    adata.obsm["X_umap"],
    index=adata.obs_names,
    columns=["UMAP1", "UMAP2"],
)
obs_meta = adata.obs[[cell_type_col, "sample"]].copy()

# All cell types (tumor and normal) are drawn from the hard-coded, curated
# CT_PALETTE above — adata.uns colors are scanpy-assigned defaults and are
# never used, so labels must be normalised (multi-word labels use embedded
# "\n" for compact legends elsewhere; collapse to a single space here) to
# match the curated keys exactly.
adata.file.close()
del adata
obs_meta[cell_type_col] = (
    obs_meta[cell_type_col].astype(str)
    .str.replace("/\n", "/", regex=False)   # "Mφ/\nMono" -> "Mφ/Mono" (no space)
    .str.replace("\n", " ", regex=False)    # "CD4\nTRM"  -> "CD4 TRM"
)
obs_meta[cell_type_col] = obs_meta[cell_type_col].replace(CT_RENAME)

missing = sorted(set(obs_meta[cell_type_col].unique()) - set(CT_PALETTE))
if missing:
    print(f"  WARNING: no curated color for {missing}; falling back to {DEFAULT_COLOR}")
print(f"  {len(X_umap)} cells")

# ---------------------------------------------------------------------------
# 2. Per-sample: load metacell assignments, collect UMAP coords + centroids
# ---------------------------------------------------------------------------
samples = open(SAMPLE_LIST).read().splitlines()

cell_umap_records     = []
centroid_umap_records = []

print("\nBuilding UMAP coordinates for all samples ...")
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

    umap_s = X_umap.loc[asgn["h5ad_bc"].values].copy()
    umap_s.index = asgn.index

    # Individual cells
    cell_umap_records.append(pd.DataFrame({
        "UMAP1":       umap_s["UMAP1"].values,
        "UMAP2":       umap_s["UMAP2"].values,
        cell_type_col: asgn[cell_type_col].values,
    }))

    # Metacell centroids = mean UMAP of member cells (exclude n=1 passthrough)
    tmp = asgn[[cell_type_col, "SEACell"]].copy()
    tmp["UMAP1"] = umap_s["UMAP1"].values
    tmp["UMAP2"] = umap_s["UMAP2"].values
    sizes = tmp.groupby("SEACell").size()
    centroids_s = (
        tmp.groupby("SEACell")[["UMAP1", "UMAP2", cell_type_col]]
        .agg({"UMAP1": "mean", "UMAP2": "mean", cell_type_col: "first"})
        .reset_index()
    )
    centroids_s = centroids_s[centroids_s["SEACell"].map(sizes) > 1].reset_index(drop=True)
    centroid_umap_records.append(centroids_s)

    print(f"  [{sample_id}] {len(cell_umap_records[-1])} cells, "
          f"{len(centroids_s)} metacell centroids")

all_cells_umap     = pd.concat(cell_umap_records,     ignore_index=True)
all_centroids_umap = pd.concat(centroid_umap_records, ignore_index=True)

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
# 3. Plot — UMAP with all metacell centroids (all samples)
# ---------------------------------------------------------------------------
print("Plotting UMAP ...")
fig_u, ax_u = plt.subplots(figsize=(6, 5))
fig_u.subplots_adjust(right=0.70)   # reserve space for legend on the right

# Layer 1: full atlas background (all cells, gray, rasterized)
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
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=CT_PALETTE.get(ct, DEFAULT_COLOR),
           markeredgecolor="none", markersize=6, label=ct)
    for ct in ct_all
]
legend_els += [
    Line2D([0], [0], linestyle="none"),   # blank spacer
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="black", markeredgewidth=0.8,
           markersize=6, label="Metacell centroid"),
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
