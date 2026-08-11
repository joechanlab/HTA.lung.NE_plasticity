#!/usr/bin/env python3
"""
5b_ru1518_umap_distance_to_root.py

UMAP for RU1518 colored by distance_to_root (metacell Numbat phylogeny,
propagated to constituent single cells).

Propagation chain:
  adata obs_name (DNA3Bit integer) → decoded barcode
  → metacell assignment TSV (RU1518_T cells only) → SEACell ID
  → metacell leaf_distances.csv → distance_to_root

Embedding: X_draw_graph_slalom_knn_15  (from figure_4_adata_path.xlsx)
Colormap: Spectral_r
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scanpy as sc

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ADATA_PATH   = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/"
                "fresh.SCLC_NSCLC.062124/individual.021626/"
                "RU1518.no_cc.hvg_2000.021626.h5ad")
COMBINED_H5AD = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/"
                 "fresh.SCLC_NSCLC.062124/combined/adata.SCLC_NSCLC.051226.h5ad")
ASSIGN_PATH  = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/"
                "Numbat_All/RU1518/GEX/RU1518_T/metacell/"
                "RU1518_T.metacell_assignment.tsv")
LEAF_PATH    = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/"
                "Numbat_All/RU1518/numbat_output_FACETS_HISENS_metacell/"
                "RU1518_leaf_distances.csv")
FDL_KEY      = "X_draw_graph_slalom_knn_15"
OUT_PNG      = os.path.join(SCRIPT_DIR, "figures",
                             "RU1518_umap_distance_to_root.png")

CELL_TYPE_COLORS = {
    "SCLC-A":              "#d62728",
    "SCLC-N":              "#17becf",
    "SCLC-P":              "#1f77b4",
    "FOXI1+ SCLC-N":       "#2ca02c",
    "FOXI1+ NonNE SCLC-N": "#2ca02c",
    "NonNE SCLC":          "#4B0082",
    "EMT":                 "#673147",
    "LUAD":                "#ffd700",
    "LUSC":                "#FFA500",
    "NSCLC":               "#ffd700",
}

PATIENT_CELL_TYPE_REMAP = {
    "FOXI1+ NonNE SCLC-N": "FOXI1+ SCLC-N",
}

# ── DNA3Bit decode ─────────────────────────────────────────────────────────────
_BIN2STR = {0b100: "A", 0b110: "C", 0b101: "G", 0b011: "T", 0b111: "N"}

def decode_barcode(i: int) -> str:
    r = ""
    while i > 0:
        r = _BIN2STR[i & 0b111] + r
        i >>= 3
    return r

def decode_obs_name(name: str):
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        return f"{parts[0]}_{decode_barcode(int(parts[1]))}-1"
    except (ValueError, KeyError):
        return None


# ── Load adata ─────────────────────────────────────────────────────────────────
print("Loading adata …")
adata = sc.read_h5ad(ADATA_PATH)
print(f"  {adata.n_obs} cells total")

# Filter to RU1518_T cells only
t_mask = adata.obs_names.str.startswith("RU1518_T_")
adata  = adata[t_mask].copy()
print(f"  {adata.n_obs} RU1518_T cells retained")

if FDL_KEY not in adata.obsm:
    raise SystemExit(f"Embedding '{FDL_KEY}' not found in adata.obsm")
print(f"  embedding: {FDL_KEY}")

# ── Cell type labels from combined adata ───────────────────────────────────────
print("Loading cell_type_final2 from combined adata …")
combined = sc.read_h5ad(COMBINED_H5AD, backed="r")
# Build decoded_cell → cell_type_final2 map
ct_map = {}
for obs_name, ct in combined.obs["cell_type_final2"].items():
    parts = obs_name.rsplit("_", 1)
    if len(parts) != 2:
        continue
    try:
        ct_map[f"{parts[0]}_{decode_barcode(int(parts[1]))}-1"] = str(ct)
    except (ValueError, KeyError):
        pass
del combined
print(f"  {len(ct_map):,} cells mapped")

# Match to adata: decode obs_names then look up combined adata cell type
decoded_names = [decode_obs_name(n) for n in adata.obs_names]
adata.obs["decoded_cell"] = decoded_names
adata.obs["cell_type_combined"] = (
    pd.Series(decoded_names, index=adata.obs_names)
    .map(ct_map)
    .fillna("Unknown")
    .replace(PATIENT_CELL_TYPE_REMAP)
    .values
)
n_typed = (adata.obs["cell_type_combined"] != "Unknown").sum()
print(f"  {n_typed}/{adata.n_obs} cells with combined cell type")
print(adata.obs["cell_type_combined"].value_counts().to_string())

# ── Build distance_to_root per single cell ─────────────────────────────────────
print("Building distance_to_root map …")

# 1. Load metacell leaf distances: SEACell full name → distance
leaf_df = pd.read_csv(LEAF_PATH)
leaf_df.columns = [c.strip('"') for c in leaf_df.columns]
# full SEACell name: "RU1518_T_SEACell-0"
leaf_df["seacell_full"] = leaf_df["cell"].str.strip('"')
dist_map = leaf_df.set_index("seacell_full")["distance_to_root"].to_dict()

# 2. Load assignment: decoded_cell (RU1518_T_BARCODE-1) → SEACell-X
assign = pd.read_csv(ASSIGN_PATH, sep="\t",
                     names=["decoded_cell", "seacell"], header=0)
assign["seacell_full"] = "RU1518_T_" + assign["seacell"]
assign["distance_to_root"] = assign["seacell_full"].map(dist_map)
decoded_to_dist = assign.set_index("decoded_cell")["distance_to_root"].to_dict()

# 3. Decode adata obs_names and look up distance
decoded_names = [decode_obs_name(n) for n in adata.obs_names]
distances     = [decoded_to_dist.get(d, np.nan) for d in decoded_names]
adata.obs["distance_to_root"] = distances

n_matched = adata.obs["distance_to_root"].notna().sum()
print(f"  {n_matched}/{adata.n_obs} cells with distance_to_root")

# cell_type_plot = combined adata labels (already remapped above)
adata.obs["cell_type_plot"] = adata.obs["cell_type_combined"]

# ── Plot ───────────────────────────────────────────────────────────────────────
print("Plotting …")
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

coords = adata.obsm[FDL_KEY]
x, y   = coords[:, 0], coords[:, 1]

# Sort: NaN first (background), then by distance ascending (low = grey)
order = np.argsort(
    np.where(np.isnan(distances), -np.inf, distances),
    kind="stable"
)

fig, axes = plt.subplots(1, 2, figsize=(5, 4))

# ── Panel A: colored by distance_to_root ──────────────────────────────────────
ax = axes[0]
dist_arr = np.array(distances)

# Color by distance_to_root (Spectral_r); NaN cells drawn grey
nan_mask = np.isnan(dist_arr)
if nan_mask.any():
    ax.scatter(x[nan_mask], y[nan_mask], s=1.5, c="#dddddd", alpha=0.4,
               linewidths=0, rasterized=True)

val_mask = ~nan_mask
sc_plot = ax.scatter(
    x[val_mask], y[val_mask],
    c=dist_arr[val_mask],
    s=2.5, alpha=0.7, linewidths=0,
    cmap="Spectral_r", rasterized=True,
)
cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.04, pad=0.02, shrink=0.8)
cbar.set_label("Distance to root\n(metacell phylogeny)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

ax.set_title("RU1518 — Distance to root\n(propagated from metacell tree)",
             fontsize=9, fontweight="bold")
ax.set_axis_off()

# ── Panel B: colored by cell type ─────────────────────────────────────────────
ax2 = axes[1]
cell_types = adata.obs["cell_type_plot"].values
present_cts = [ct for ct in CELL_TYPE_COLORS if ct in cell_types]
other_cts   = [ct for ct in set(cell_types) if ct not in CELL_TYPE_COLORS]

# Background (unlabeled types)
for ct in other_cts:
    m = cell_types == ct
    ax2.scatter(x[m], y[m], s=1.5, c="#cccccc", alpha=0.4,
                linewidths=0, rasterized=True)

for ct in present_cts:
    m = cell_types == ct
    ax2.scatter(x[m], y[m], s=2.5, c=CELL_TYPE_COLORS[ct], alpha=0.75,
                linewidths=0, label=ct, rasterized=True)

ax2.set_title("RU1518 — Cell type", fontsize=9, fontweight="bold")
ax2.set_axis_off()
ax2.legend(markerscale=4, fontsize=7.5, loc="lower left",
           framealpha=0.85, handlelength=1)

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"→ {OUT_PNG}")
