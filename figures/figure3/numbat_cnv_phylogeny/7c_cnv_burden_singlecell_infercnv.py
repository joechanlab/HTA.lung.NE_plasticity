#!/usr/bin/env python3
"""
7c_cnv_burden_singlecell_infercnv.py

Compare single-cell Numbat CNV burden against inferCNV score, CNV_clock,
and pseudotime (palantir) from per-patient h5ad files listed in
figure_4_adata_path.xlsx.

Two modes:
  default (metacell propagation):
    Loads combined adata.SCLC_NSCLC.051226.h5ad, propagates metacell burden
    to single cells via SEACell assignment TSVs.

  --sc-numbat:
    Loads cnv_burden_per_cell_sc.tsv (computed by 7b --single-cell), joins
    to per-patient h5ad files from figure_4_adata_path.xlsx via DNA3Bit
    barcode decode.  Produces three comparisons per patient:
      1. cnv_burden_bp / fga / segs  vs  infercnv_score
      2. cnv_burden_bp / fga / segs  vs  CNV_clock
      3. cnv_burden_bp               vs  palantir_pseudotime_slalom
         + start-cell test: are pseudotime-low cells also CNV-burden-low?

Outputs (--sc-numbat):
  cnv_burden_singlecell_sc.tsv
  figures/cnv_burden_infercnv_sc/{patient}_burden_infercnv.png
  figures/cnv_burden_infercnv_sc/{patient}_burden_pseudotime.png
  figures/cnv_burden_infercnv_sc/all_patients_correlation.png
  cnv_burden_infercnv_correlation_sc.tsv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
import scanpy as sc

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH     = os.path.join(os.path.dirname(SCRIPT_DIR), "Basal",
                              "figure_4_adata_path.xlsx")
BASE          = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
ADATA_PATH    = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/"
                 "fresh.SCLC_NSCLC.062124/combined/"
                 "adata.SCLC_NSCLC.051226.h5ad")
BURDEN_TSV_MC = os.path.join(SCRIPT_DIR, "cnv_burden_per_cell.tsv")
BURDEN_TSV_SC = os.path.join(SCRIPT_DIR, "cnv_burden_per_cell_sc.tsv")
OUT_TSV_MC    = os.path.join(SCRIPT_DIR, "cnv_burden_singlecell.tsv")
OUT_TSV_SC    = os.path.join(SCRIPT_DIR, "cnv_burden_singlecell_sc.tsv")
OUT_FIG_DIR   = os.path.join(SCRIPT_DIR, "figures", "cnv_burden_infercnv")

CELL_TYPE_COLORS = {
    "SCLC-A":              "#d62728",
    "SCLC-N":              "#17becf",
    "SCLC-P":              "#1f77b4",
    "NonNE SCLC":          "#4B0082",
    "EMT":                 "#673147",
    "FOXI1+ NonNE SCLC-N": "#2ca02c",
    "LUAD":                "#ffd700",
    "LUSC":                "#FFA500",
}
TUMOR_TYPES = set(CELL_TYPE_COLORS)

BURDEN_COLS = {
    "cnv_burden_bp":   "Probabilistic FGA\n(p_cnv-weighted bp)",
    "cnv_burden_fga":  "Binary FGA\n(p_cnv ≥ 0.5)",
    "cnv_burden_segs": "Mean segment p(CNV)",
    "cnv_burden_gene": "Gene-weighted FGA\n(fraction of genes altered)",
}

# Fraction of lowest-pseudotime cells considered "start cells"
START_QUANTILE = 0.10


# ── DNA3Bit decode ─────────────────────────────────────────────────────────────

_BIN2STR = {0b100: "A", 0b110: "C", 0b101: "G", 0b011: "T", 0b111: "N"}

def decode_barcode(i: int) -> str:
    r = ""
    while i > 0:
        r = _BIN2STR[i & 0b111] + r
        i >>= 3
    return r


def decode_obs_names(obs_names: pd.Index) -> pd.Series:
    """obs_name (sample_integer) → decoded_cell (sample_barcode-1)."""
    result = {}
    for name in obs_names:
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            try:
                bc = decode_barcode(int(parts[1]))
                result[name] = f"{parts[0]}_{bc}-1"
            except (ValueError, KeyError):
                result[name] = None
        else:
            result[name] = None
    return pd.Series(result, name="decoded_cell")


# ── Per-patient adata loading (SC mode) ───────────────────────────────────────

OBS_COLS_WANTED = ["cell_type_final2", "infercnv_score", "CNV_clock"]


def load_patient_obs(path: str, pseudotime_col: str) -> pd.DataFrame | None:
    """
    Load obs from a per-patient h5ad (backed), decode obs_names, and return
    a DataFrame with decoded_cell, pseudotime, infercnv_score, CNV_clock,
    cell_type_final2.
    """
    if not os.path.exists(path):
        return None
    try:
        adata = sc.read_h5ad(path, backed="r")
    except Exception as e:
        print(f"    Could not load {path}: {e}")
        return None

    obs = adata.obs.copy()
    cols = [c for c in OBS_COLS_WANTED + [pseudotime_col] if c in obs.columns]
    obs = obs[cols].copy()

    decoded = decode_obs_names(obs.index)
    obs["decoded_cell"] = decoded.values
    obs.index.name = "obs_name"
    obs = obs.reset_index()

    if pseudotime_col != "pseudotime":
        obs = obs.rename(columns={pseudotime_col: "pseudotime"})
    return obs


def build_singlecell_burden_sc(burden_sc: pd.DataFrame,
                                xlsx_df: pd.DataFrame,
                                subset_patients: set | None) -> pd.DataFrame:
    """
    For each patient in xlsx_df, load their per-patient adata obs, decode
    barcodes, join with SC Numbat burden, and assemble a combined DataFrame.
    """
    all_parts = []

    for _, row in xlsx_df.iterrows():
        patient       = row["patient"]
        adata_path    = row["path"]
        pseudotime_col = row["pseudotime_col"]

        if subset_patients and patient not in subset_patients:
            continue

        pat_burden = burden_sc[burden_sc["patient_id"] == patient]
        if pat_burden.empty:
            continue   # no Numbat SC output for this patient

        print(f"  [{patient}]  loading {os.path.basename(adata_path)} …")
        obs = load_patient_obs(adata_path, pseudotime_col)
        if obs is None or obs.empty:
            print(f"    adata not found or empty — skipping")
            continue

        # Build lookup: decoded_cell → burden
        burden_cols = [c for c in ["cnv_burden_bp", "cnv_burden_fga",
                                    "cnv_burden_segs", "cnv_burden_gene",
                                    "cnv_burden_gene_bin", "cell_type"]
                       if c in pat_burden.columns]
        burden_lookup = pat_burden.set_index("cell")[burden_cols]

        merged = obs.merge(
            burden_lookup.reset_index().rename(columns={"cell": "decoded_cell"}),
            on="decoded_cell", how="left"
        )
        merged["patient_id"] = patient
        n_matched = merged["cnv_burden_bp"].notna().sum()
        print(f"    {len(merged)} cells, {n_matched} matched to Numbat SC burden")

        all_parts.append(merged)

    if not all_parts:
        return pd.DataFrame()

    out = pd.concat(all_parts, ignore_index=True)
    col_order = (["patient_id", "obs_name", "decoded_cell",
                  "cell_type", "cell_type_final2",
                  "pseudotime", "infercnv_score", "CNV_clock",
                  "cnv_burden_bp", "cnv_burden_fga",
                  "cnv_burden_segs", "cnv_burden_gene", "cnv_burden_gene_bin"])
    return out[[c for c in col_order if c in out.columns]]


# ── Metacell propagation (original mode) ──────────────────────────────────────

def build_barcode_map(patient: str, sample_ids: list) -> pd.DataFrame:
    rows = []
    for sid in sample_ids:
        assign_path = os.path.join(
            BASE, patient, "GEX", sid, "metacell",
            f"{sid}.metacell_assignment.tsv"
        )
        if not os.path.exists(assign_path):
            continue
        df = pd.read_csv(assign_path, sep="\t")
        df.columns = ["decoded_cell", "seacell"]
        df["seacell"] = f"{sid}_" + df["seacell"]
        df["sample_id"] = sid
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["decoded_cell", "seacell", "sample_id"])
    return pd.concat(rows, ignore_index=True)


def build_singlecell_burden_mc(patients: list, burden_df: pd.DataFrame,
                                adata_obs: pd.DataFrame) -> pd.DataFrame:
    all_records = []
    for patient in patients:
        pat_obs = adata_obs[adata_obs["patient"] == patient].copy()
        if pat_obs.empty:
            continue
        sample_ids = pat_obs["sample"].unique().tolist()
        decoded = decode_obs_names(pat_obs.index)
        pat_obs["decoded_cell"] = decoded.values

        barcode_map = build_barcode_map(patient, sample_ids)
        if barcode_map.empty:
            continue

        burden_pat = burden_df[burden_df["patient_id"] == patient].copy()
        if burden_pat.empty:
            continue

        burden_cols_mc = [c for c in ["cnv_burden_bp", "cnv_burden_fga",
                                       "cnv_burden_segs", "cnv_burden_gene",
                                       "cnv_burden_gene_bin"]
                          if c in burden_pat.columns]
        burden_lookup = burden_pat.set_index("cell")[burden_cols_mc]
        merged = pat_obs.merge(
            barcode_map[["decoded_cell", "seacell"]], on="decoded_cell", how="left"
        )
        merged = merged.merge(
            burden_lookup.reset_index().rename(columns={"cell": "seacell"}),
            on="seacell", how="left"
        )
        n_matched = merged["cnv_burden_bp"].notna().sum()
        print(f"  [{patient}]  {len(merged)} cells, {n_matched} matched")
        all_records.append(merged)

    if not all_records:
        return pd.DataFrame()
    out = pd.concat(all_records, ignore_index=True)
    out = out.rename(columns={"patient": "patient_id", "sample": "sample_id",
                               "cell_type_final2": "cell_type"})
    return out


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _scatter_metric(ax, df, x_col, y_col, xlabel, ylabel, tumor_only=True):
    """Scatter x vs y, color by cell_type; show Spearman r."""
    ct_col = "cell_type" if "cell_type" in df.columns else "cell_type_final2"
    sub = df.dropna(subset=[x_col, y_col])
    if sub.empty:
        ax.set_axis_off(); return float("nan"), float("nan")

    non_tumor = sub[~sub[ct_col].isin(TUMOR_TYPES)]
    ax.scatter(non_tumor[x_col], non_tumor[y_col],
               color="#CCCCCC", s=3, alpha=0.25, linewidths=0)

    for ct in [c for c in CELL_TYPE_COLORS if c in sub[ct_col].values]:
        g = sub[sub[ct_col] == ct]
        ax.scatter(g[x_col], g[y_col], color=CELL_TYPE_COLORS[ct],
                   s=6, alpha=0.6, linewidths=0, label=ct)

    sub_t = sub[sub[ct_col].isin(TUMOR_TYPES)] if tumor_only else sub
    r, p = (float("nan"), float("nan"))
    if len(sub_t) >= 10:
        r, p = stats.spearmanr(sub_t[x_col], sub_t[y_col])
        m, b, *_ = stats.linregress(sub_t[x_col], sub_t[y_col])
        xs = np.linspace(sub_t[x_col].min(), sub_t[x_col].max(), 100)
        ax.plot(xs, m * xs + b, color="#333333", lw=1.1, ls="--", zorder=5)
        pstr = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
        ax.text(0.03, 0.97, f"r={r:.2f}\n{pstr}", transform=ax.transAxes,
                va="top", fontsize=5.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))

    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=5.5)
    return r, p


def plot_infercnv_scclock(patient: str, df: pd.DataFrame, out_path: str):
    """3×3 grid: rows = cnv_burden_bp/fga/segs, cols = infercnv/CNV_clock + legend."""
    has_infercnv = "infercnv_score" in df.columns and df["infercnv_score"].notna().any()
    has_cnvclock  = "CNV_clock"         in df.columns and df["CNV_clock"].notna().any()
    if not has_infercnv and not has_cnvclock:
        return {}

    n_x   = sum([has_infercnv, has_cnvclock])
    n_met = len(BURDEN_COLS)
    fig, axes = plt.subplots(n_met, n_x,
                             figsize=(4.5 * n_x, 3 * n_met), squeeze=False)
    fig.suptitle(f"{patient} — CNV burden vs inferCNV / CNV_clock (tumor cells)",
                 fontsize=8, fontweight="bold")

    corr = {}
    for ri, (metric, m_label) in enumerate(BURDEN_COLS.items()):
        xi = 0
        for x_col, x_label in [("infercnv_score", "inferCNV score"),
                                ("CNV_clock", "CNV_clock")]:
            if (x_col == "infercnv_score" and not has_infercnv) or \
               (x_col == "CNV_clock"        and not has_cnvclock):
                continue
            ax = axes[ri][xi]
            r, p = _scatter_metric(ax, df, x_col, metric,
                                   x_label, m_label.replace("\n", " "))
            corr[f"{metric}_vs_{x_col}_r"] = round(r, 3) if not np.isnan(r) else np.nan
            corr[f"{metric}_vs_{x_col}_p"] = round(p, 4) if not np.isnan(p) else np.nan
            xi += 1

    # Legend on last panel
    ct_col = "cell_type" if "cell_type" in df.columns else "cell_type_final2"
    cts = [c for c in CELL_TYPE_COLORS if c in df[ct_col].values]
    handles = [Patch(color=CELL_TYPE_COLORS[ct], label=ct) for ct in cts]
    if handles:
        fig.legend(handles=handles, fontsize=5.5, loc="lower center",
                   ncol=len(cts), frameon=False,
                   title="Cell type", title_fontsize=6)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return corr


def plot_pseudotime_burden(patient: str, df: pd.DataFrame, out_path: str):
    """
    2-row figure per patient:
      Row 1: scatter pseudotime vs cnv_burden_bp | infercnv_score | CNV_clock
      Row 2: boxplot start (bottom 10% PT) vs rest for each metric
    Also returns dict of Spearman correlations and Mann-Whitney p-values.
    """
    if "pseudotime" not in df.columns or df["pseudotime"].isna().all():
        return {}

    metrics_to_plot = {
        "cnv_burden_bp":   "Probabilistic FGA",
        "cnv_burden_gene": "Gene-weighted FGA",
        "infercnv_score":  "inferCNV score",
        "CNV_clock":       "CNV clock",
    }
    metrics_to_plot = {k: v for k, v in metrics_to_plot.items()
                       if k in df.columns and df[k].notna().any()}
    if not metrics_to_plot:
        return {}

    n = len(metrics_to_plot)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 6), squeeze=False)
    fig.suptitle(f"{patient} — Pseudotime vs CNV metrics (tumor cells)",
                 fontsize=8, fontweight="bold")

    ct_col = "cell_type" if "cell_type" in df.columns else "cell_type_final2"
    tumor  = df[df[ct_col].isin(TUMOR_TYPES)].copy()

    # Define start cells
    if "pseudotime" in tumor.columns and tumor["pseudotime"].notna().any():
        pt_thresh = tumor["pseudotime"].quantile(START_QUANTILE)
        tumor["is_start"] = tumor["pseudotime"] <= pt_thresh
    else:
        tumor["is_start"] = False

    start_ct = tumor["cell_type"].value_counts().idxmax() \
        if not tumor[tumor["is_start"]].empty else "N/A"

    corr = {"patient_id": patient,
            "n_tumor": len(tumor),
            "n_start": int(tumor["is_start"].sum()),
            "dominant_start_ct": start_ct}

    for xi, (metric, label) in enumerate(metrics_to_plot.items()):
        ax_top = axes[0][xi]
        ax_bot = axes[1][xi]

        # ── Row 1: scatter pseudotime vs metric ───────────────────────────────
        sub = tumor.dropna(subset=["pseudotime", metric])
        for ct in [c for c in CELL_TYPE_COLORS if c in sub[ct_col].values]:
            g = sub[sub[ct_col] == ct]
            ax_top.scatter(g["pseudotime"], g[metric],
                           color=CELL_TYPE_COLORS[ct], s=6, alpha=0.6,
                           linewidths=0, label=ct)

        r, p = (float("nan"), float("nan"))
        if len(sub) >= 10:
            r, p = stats.spearmanr(sub["pseudotime"], sub[metric])
            m_s, b_s, *_ = stats.linregress(sub["pseudotime"], sub[metric])
            xs = np.linspace(sub["pseudotime"].min(), sub["pseudotime"].max(), 100)
            ax_top.plot(xs, m_s * xs + b_s, color="#333333", lw=1.1,
                        ls="--", zorder=5)
            pstr = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
            ax_top.text(0.03, 0.97, f"r={r:.2f}\n{pstr}",
                        transform=ax_top.transAxes, va="top", fontsize=5.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax_top.set_xlabel("Pseudotime", fontsize=6)
        ax_top.set_ylabel(label, fontsize=6)
        ax_top.spines[["top", "right"]].set_visible(False)
        ax_top.tick_params(labelsize=5.5)

        corr[f"{metric}_spearman_r"] = round(r, 3) if not np.isnan(r) else np.nan
        corr[f"{metric}_spearman_p"] = round(p, 4) if not np.isnan(p) else np.nan

        # ── Row 2: boxplot start vs non-start ─────────────────────────────────
        start_vals    = tumor.loc[tumor["is_start"],     metric].dropna()
        nonstart_vals = tumor.loc[~tumor["is_start"],    metric].dropna()

        parts = ax_bot.boxplot(
            [start_vals, nonstart_vals],
            labels=[f"Start\n(PT≤{pt_thresh:.2f})", "Non-start"],
            patch_artist=True, widths=0.5,
            medianprops=dict(color="black", lw=1.5),
            whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
            flierprops=dict(marker=".", markersize=2, alpha=0.4),
        )
        parts["boxes"][0].set_facecolor("#FFF3CD")
        parts["boxes"][1].set_facecolor("#DDEEFF")

        mwu_u, mwu_p = (float("nan"), float("nan"))
        if len(start_vals) >= 3 and len(nonstart_vals) >= 3:
            mwu_u, mwu_p = stats.mannwhitneyu(
                start_vals, nonstart_vals, alternative="two-sided")
            pstr = f"MWU p={mwu_p:.2e}" if mwu_p < 0.001 else f"MWU p={mwu_p:.3f}"
            ax_bot.set_title(pstr, fontsize=6)
        ax_bot.set_ylabel(label, fontsize=6)
        ax_bot.spines[["top", "right"]].set_visible(False)
        ax_bot.tick_params(labelsize=5.5)

        corr[f"{metric}_mwu_p"]      = round(mwu_p, 4) if not np.isnan(mwu_p) else np.nan
        corr[f"{metric}_start_med"]  = round(start_vals.median(), 4) if len(start_vals) else np.nan
        corr[f"{metric}_rest_med"]   = round(nonstart_vals.median(), 4) if len(nonstart_vals) else np.nan

    # Cell-type legend
    cts = [c for c in CELL_TYPE_COLORS if c in tumor[ct_col].values]
    handles = [Patch(color=CELL_TYPE_COLORS[ct], label=ct) for ct in cts]
    if handles:
        fig.legend(handles=handles, fontsize=5.5, loc="lower center",
                   ncol=len(cts), frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")
    return corr


def plot_correlation_summary(corr_rows: list, out_path: str, mode: str = "infercnv"):
    """Dot-matrix: patients × metrics, dot size ∝ |r|, color = direction."""
    if not corr_rows:
        return
    df = pd.DataFrame(corr_rows).set_index("patient_id")
    if mode == "pseudotime":
        r_cols = [c for c in df.columns if c.endswith("_spearman_r")]
        labels = [c.replace("_spearman_r", "").replace("_", " ") for c in r_cols]
    else:
        r_cols = [c for c in df.columns if c.endswith("_r") and "_vs_" in c]
        labels = [c.replace("_r", "").replace("_vs_", "\nvs\n").replace("_", " ")
                  for c in r_cols]

    if not r_cols:
        return

    n_pat = len(df)
    fig, ax = plt.subplots(figsize=(max(5, len(r_cols) * 1.8),
                                    max(3, n_pat * 0.42 + 1.5)))
    title = ("Spearman r: Numbat CNV burden vs pseudotime"
             if mode == "pseudotime"
             else "Spearman r: Numbat CNV burden vs inferCNV / CNV_clock")
    fig.suptitle(title + "\n(tumor cells per patient)",
                 fontsize=9, fontweight="bold")

    for xi, r_col in enumerate(r_cols):
        for yi, pat in enumerate(df.index):
            r_val = df.at[pat, r_col]
            if pd.isna(r_val):
                continue
            color = "#d62728" if r_val > 0 else "#1f77b4"
            ax.scatter(xi, yi, s=abs(r_val) * 800, color=color,
                       alpha=0.85, linewidths=0.5, edgecolors="white", zorder=3)
            ax.text(xi, yi, f"{r_val:.2f}", ha="center", va="center",
                    fontsize=5.5,
                    color="white" if abs(r_val) > 0.3 else "#333333")

    ax.set_xticks(range(len(r_cols)))
    ax.set_xticklabels(labels, fontsize=6, rotation=20, ha="right")
    ax.set_yticks(range(n_pat))
    ax.set_yticklabels(df.index, fontsize=6)
    ax.set_xlim(-0.5, len(r_cols) - 0.5)
    ax.set_ylim(-0.5, n_pat - 0.5)
    ax.grid(True, lw=0.4, color="#EEEEEE", zorder=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=8, label="r > 0 (higher burden → higher score)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=8, label="r < 0 (higher burden → lower score)"),
    ], fontsize=7, loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Correlation summary → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare Numbat CNV burden with inferCNV, CNV_clock, and pseudotime")
    parser.add_argument("--patients", nargs="*",
                        help="Restrict to these patient IDs")
    parser.add_argument("--sc-numbat", action="store_true",
                        help="Use true single-cell Numbat output "
                             "(cnv_burden_per_cell_sc.tsv) and per-patient "
                             "adatas from figure_4_adata_path.xlsx. "
                             "Run '7b --single-cell' first.")
    args = parser.parse_args()

    sc_mode = args.sc_numbat
    fig_dir = OUT_FIG_DIR + ("_sc" if sc_mode else "")
    os.makedirs(fig_dir, exist_ok=True)
    subset  = set(args.patients) if args.patients else None

    if sc_mode:
        # ── SC Numbat + per-patient adatas ────────────────────────────────────
        if not os.path.exists(BURDEN_TSV_SC):
            print(f"ERROR: {BURDEN_TSV_SC} not found.")
            print("  Run: python 7b_cnv_burden_per_cell.py --single-cell")
            return
        if not os.path.exists(XLSX_PATH):
            print(f"ERROR: {XLSX_PATH} not found.")
            return

        print(f"Loading SC burden …")
        burden_sc = pd.read_csv(BURDEN_TSV_SC, sep="\t")
        print(f"  {len(burden_sc)} cells, {burden_sc['patient_id'].nunique()} patients")

        xlsx_df = pd.read_excel(XLSX_PATH)
        if subset:
            xlsx_df = xlsx_df[xlsx_df["patient"].isin(subset)]
        print(f"  {len(xlsx_df)} patient entries from xlsx")

        print("\nBuilding single-cell dataset …")
        sc_df = build_singlecell_burden_sc(burden_sc, xlsx_df, subset)

        if sc_df.empty:
            print("No data produced.")
            return

        sc_df.to_csv(OUT_TSV_SC, sep="\t", index=False)
        n_matched = sc_df["cnv_burden_bp"].notna().sum()
        print(f"\nOutput: {OUT_TSV_SC}")
        print(f"  {len(sc_df)} cells, {n_matched} with Numbat burden")

        patients = sc_df["patient_id"].dropna().unique().tolist()
        corr_infercnv = []
        corr_pt       = []

        for patient in patients:
            df_pat = sc_df[sc_df["patient_id"] == patient].copy()
            if df_pat.empty:
                continue

            # Plot 1: burden vs infercnv + CNV_clock
            c1 = plot_infercnv_scclock(
                patient, df_pat,
                os.path.join(fig_dir, f"{patient}_burden_infercnv.png")
            )
            if c1:
                c1["patient_id"] = patient
                corr_infercnv.append(c1)

            # Plot 2: burden vs pseudotime + start-cell test
            c2 = plot_pseudotime_burden(
                patient, df_pat,
                os.path.join(fig_dir, f"{patient}_burden_pseudotime.png")
            )
            if c2:
                corr_pt.append(c2)

        # Summary figures and TSVs
        if corr_infercnv:
            df_ci = pd.DataFrame(corr_infercnv)
            df_ci.to_csv(
                os.path.join(SCRIPT_DIR, "cnv_burden_infercnv_correlation_sc.tsv"),
                sep="\t", index=False)
            plot_correlation_summary(
                corr_infercnv,
                os.path.join(fig_dir, "all_patients_infercnv_correlation.png"),
                mode="infercnv")

        if corr_pt:
            df_pt = pd.DataFrame(corr_pt)
            df_pt.to_csv(
                os.path.join(SCRIPT_DIR, "cnv_burden_pseudotime_correlation_sc.tsv"),
                sep="\t", index=False)
            plot_correlation_summary(
                corr_pt,
                os.path.join(fig_dir, "all_patients_pseudotime_correlation.png"),
                mode="pseudotime")
            # Print start-cell summary
            print("\n── Start-cell summary (bottom 10% pseudotime) ──────────────")
            print(df_pt[["patient_id", "n_start", "dominant_start_ct",
                         "cnv_burden_bp_start_med", "cnv_burden_bp_rest_med",
                         "cnv_burden_bp_mwu_p"]].to_string(index=False))

    else:
        # ── Metacell propagation (original mode) ──────────────────────────────
        with open(os.path.join(SCRIPT_DIR, "patient_list.txt")) as f:
            pid_entries = [l.strip() for l in f if l.strip()]
        all_patients = list(dict.fromkeys(p.split("_")[0] for p in pid_entries))
        patients = [p for p in all_patients if not subset or p in subset]

        print(f"Loading metacell burden …")
        burden_df = pd.read_csv(BURDEN_TSV_MC, sep="\t")

        print("Loading combined adata obs …")
        adata = sc.read_h5ad(ADATA_PATH, backed="r")
        adata_obs = adata.obs.copy()

        sc_df = build_singlecell_burden_mc(patients, burden_df, adata_obs)
        if sc_df.empty:
            print("No data produced.")
            return

        sc_df.to_csv(OUT_TSV_MC, sep="\t", index=False)
        print(f"\nOutput: {OUT_TSV_MC}  ({len(sc_df)} cells)")

        corr_rows = []
        for patient in patients:
            df_pat = sc_df[sc_df["patient_id"] == patient]
            if df_pat.empty or df_pat["cnv_burden_bp"].isna().all():
                continue
            out_png = os.path.join(fig_dir, f"{patient}_burden_infercnv.png")
            c1 = plot_infercnv_scclock(patient, df_pat, out_png)
            if c1:
                c1["patient_id"] = patient
                corr_rows.append(c1)

        if corr_rows:
            pd.DataFrame(corr_rows).to_csv(
                os.path.join(SCRIPT_DIR, "cnv_burden_infercnv_correlation.tsv"),
                sep="\t", index=False)
            plot_correlation_summary(
                corr_rows,
                os.path.join(fig_dir, "all_patients_infercnv_correlation.png"))

    print(f"\nAll figures → {fig_dir}/")


if __name__ == "__main__":
    main()
