#!/usr/bin/env python3
"""
7d_build_metacell_violin_data.py

Build cnv_burden_metacell_joined.tsv — metacell-level equivalent of
cnv_burden_singlecell_sc.tsv, used to run 5_cnv_burden_violin.py at
metacell resolution.

For each SEACell (metacell):
  cnv_burden_gene_bin  — from cnv_burden_per_cell.tsv (metacell Numbat run)
  cell_type            — plurality cell_type_final2 from constituent single cells
                         (combined adata, authoritative)
  infercnv_score       — mean of constituent single cells (combined adata)
  CNV_clock            — mean of constituent single cells (combined adata)
  decoded_cell         — SEACell identifier (for downstream join)

Distance-to-root is added separately by 7d_compute_metacell_leaf_distances.R.
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BURDEN_TSV    = os.path.join(SCRIPT_DIR, "cnv_burden_per_cell.tsv")
COMBINED_H5AD = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/"
                 "fresh.SCLC_NSCLC.062124/combined/adata.SCLC_NSCLC.051226.h5ad")
ASSIGN_ROOT   = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
OUT_TSV       = os.path.join(SCRIPT_DIR, "cnv_burden_metacell_joined.tsv")

# ── DNA3Bit decode ─────────────────────────────────────────────────────────────
_BIN2STR = {0b100: "A", 0b110: "C", 0b101: "G", 0b011: "T", 0b111: "N"}

def decode_barcode(i: int) -> str:
    r = ""
    while i > 0:
        r = _BIN2STR[i & 0b111] + r
        i >>= 3
    return r


def decode_obs_name(obs_name: str) -> str | None:
    """RU151_PDX_120786804718942  →  RU151_PDX_TTTGG...-1"""
    parts = obs_name.rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        return f"{parts[0]}_{decode_barcode(int(parts[1]))}-1"
    except (ValueError, KeyError):
        return None


# ── Load combined adata ────────────────────────────────────────────────────────
print("Loading combined adata …")
adata = ad.read_h5ad(COMBINED_H5AD, backed="r")
avail = [c for c in ["patient", "sample", "cell_type_final2",
                      "infercnv_score", "CNV_clock"] if c in adata.obs.columns]
obs   = adata.obs[avail].copy()
# Convert Categorical to str so value_counts() only returns present values,
# not zeros for all unused categories (which causes idxmax() to return a
# spurious first-alphabetically label when no cells actually matched).
if "cell_type_final2" in obs.columns:
    obs["cell_type_final2"] = obs["cell_type_final2"].astype(str)

# Build decoded_cell → (cell_type_final2, infercnv_score, CNV_clock)
print("  Decoding obs_names …")
obs["decoded_cell"] = [decode_obs_name(n) for n in obs.index]
obs = obs.dropna(subset=["decoded_cell"])
# Index by decoded_cell for fast lookup
obs_indexed = obs.set_index("decoded_cell")
print(f"  {len(obs_indexed):,} cells available")


# ── Load metacell burden ───────────────────────────────────────────────────────
print("Loading metacell burden …")
burden = pd.read_csv(BURDEN_TSV, sep="\t")
print(f"  {len(burden)} metacells, {burden['patient_id'].nunique()} patients")


# ── For each patient/sample, load metacell assignments and aggregate ──────────
print("Aggregating single-cell metrics to metacell level …")

all_rows = []
for patient, grp in burden.groupby("patient_id"):
    # Discover sample IDs from burden data
    for sample_id, sgrp in grp.groupby("sample_id"):
        assign_path = os.path.join(
            ASSIGN_ROOT, patient, "GEX", sample_id, "metacell",
            f"{sample_id}.metacell_assignment.tsv"
        )
        if not os.path.exists(assign_path):
            # Fall back: use cell_type from Numbat directly, no aggregation
            for _, row in sgrp.iterrows():
                all_rows.append({
                    "patient_id":         patient,
                    "sample_id":          sample_id,
                    "decoded_cell":       row["cell"],
                    "cell_type":          row["cell_type"],
                    "cnv_burden_bp":      row.get("cnv_burden_bp"),
                    "cnv_burden_fga":     row.get("cnv_burden_fga"),
                    "cnv_burden_segs":    row.get("cnv_burden_segs"),
                    "cnv_burden_gene":    row.get("cnv_burden_gene"),
                    "cnv_burden_gene_bin": row.get("cnv_burden_gene_bin"),
                    "infercnv_score":     np.nan,
                    "CNV_clock":          np.nan,
                })
            continue

        assign = pd.read_csv(assign_path, sep="\t",
                             names=["decoded_cell", "seacell"], header=0)
        # Qualify SEACell name to match burden table: "{sample_id}_SEACell-X"
        assign["seacell_full"] = sample_id + "_" + assign["seacell"]

        # Join assign → adata metrics
        agg_cols = [c for c in ["cell_type_final2", "infercnv_score", "CNV_clock"]
                    if c in obs_indexed.columns]
        assign = assign.join(obs_indexed[agg_cols], on="decoded_cell", how="left")

        # Aggregate per SEACell
        for seacell_full, sc_grp in assign.groupby("seacell_full"):
            # Majority cell_type from combined adata (authoritative, updated labels)
            ct_counts = sc_grp["cell_type_final2"].dropna().value_counts() \
                if "cell_type_final2" in sc_grp.columns else pd.Series(dtype=str)
            # Filter out "nan" strings that can appear after astype(str)
            ct_counts = ct_counts[ct_counts.index != "nan"]
            cell_type_adata = ct_counts.idxmax() if len(ct_counts) else np.nan

            infercnv  = sc_grp["infercnv_score"].mean() \
                if "infercnv_score" in sc_grp.columns else np.nan
            cnv_clock = sc_grp["CNV_clock"].mean() \
                if "CNV_clock" in sc_grp.columns else np.nan

            # Match to burden row
            burden_row = sgrp[sgrp["cell"] == seacell_full]
            if burden_row.empty:
                continue
            r = burden_row.iloc[0]

            # Cell type priority: combined adata (updated) > Numbat burden table
            cell_type = cell_type_adata if pd.notna(cell_type_adata) \
                        else r.get("cell_type")

            all_rows.append({
                "patient_id":          patient,
                "sample_id":           sample_id,
                "decoded_cell":        seacell_full,
                "cell_type":           cell_type,
                "cnv_burden_bp":       r.get("cnv_burden_bp"),
                "cnv_burden_fga":      r.get("cnv_burden_fga"),
                "cnv_burden_segs":     r.get("cnv_burden_segs"),
                "cnv_burden_gene":     r.get("cnv_burden_gene"),
                "cnv_burden_gene_bin": r.get("cnv_burden_gene_bin"),
                "infercnv_score":      infercnv,
                "CNV_clock":           cnv_clock,
            })

out = pd.DataFrame(all_rows)
out.to_csv(OUT_TSV, sep="\t", index=False)
matched_infercnv = out["infercnv_score"].notna().sum()
print(f"\nOutput: {OUT_TSV}")
print(f"  {len(out)} metacells, {out['patient_id'].nunique()} patients")
print(f"  infercnv_score matched: {matched_infercnv}")
print(f"  cnv_burden_gene_bin not-null: {out['cnv_burden_gene_bin'].notna().sum()}")
print("\nCell type distribution (tumor types):")
tumor_types = {"SCLC-A","SCLC-N","SCLC-P","NonNE SCLC","EMT",
               "FOXI1+ NonNE SCLC-N","LUAD","LUSC"}
print(out[out["cell_type"].isin(tumor_types)]["cell_type"].value_counts().to_string())
