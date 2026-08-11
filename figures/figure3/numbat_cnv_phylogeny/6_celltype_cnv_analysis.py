#!/usr/bin/env python3
"""
6_celltype_cnv_analysis.py

Cell-type-specific CNV analysis for each patient.

For each patient:
  1. Loads patient-level joint_post_filtered.csv
     (columns: cell, seg, CHROM, seg_start, seg_end, cnv_state, p_cnv, cell_type, ...)
  2. Filters to tumor cell types
  3. Computes mean p_cnv per (segment × cell_type)
  4. Identifies cell-type-specific CNV events via pairwise Mann-Whitney U tests
     + Benjamini-Hochberg FDR correction
  5. Saves:
       - cnv_celltype_enrichment.tsv   (per patient × seg × cell_type statistics)
       - figures/cnv_celltype/{patient}_cnv_celltype.png  (heatmap per patient)

Usage:
    /path/to/python 6_celltype_cnv_analysis.py [--patients RU1083 ...]
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE        = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
OUTDIR_NAME = "numbat_output_FACETS_HISENS_metacell"
OUT_TSV     = os.path.join(SCRIPT_DIR, "cnv_celltype_enrichment.tsv")
OUT_PNG_DIR = os.path.join(SCRIPT_DIR, "figures", "cnv_celltype")

TUMOR_TYPES = ["SCLC-A", "SCLC-N", "SCLC-P", "NonNE SCLC",
               "EMT", "FOXI1+ NonNE SCLC-N", "LUAD", "LUSC"]

# GRCh38 centromere midpoints (Mb)
CENTROMERE = {
    "1":123.4,"2":93.9,"3":90.9,"4":50.4,"5":48.8,
    "6":59.8,"7":60.6,"8":45.2,"9":43.0,"10":39.8,
    "11":53.4,"12":35.5,"13":17.7,"14":17.2,"15":19.0,
    "16":36.8,"17":25.1,"18":18.5,"19":26.2,"20":28.1,
    "21":12.0,"22":15.0,"X":60.6,"Y":10.4,
}

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

CNV_STATE_COLORS = {
    "amp": "#d62728", "del": "#1f77b4", "loh": "#9467bd",
    "bamp": "#8B0000", "bdel": "#003f8e",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "freq", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def arm_label(chrom, seg_start, seg_end, cnv_state):
    """Return 'chrXx state' label (X=chr, x=arm, state=amp/del/loh/...)."""
    c = str(chrom)
    cen = CENTROMERE.get(c, None)
    if cen is None:
        arm = ""
    else:
        mid_mb = (float(seg_start) + float(seg_end)) / 2e6
        arm = "p" if mid_mb < cen else "q"
    state_str = {"amp":"gain","del":"loss","loh":"LOH","bamp":"bamp","bdel":"homdel"}.get(
        cnv_state, cnv_state)
    return f"chr{c}{arm} {state_str}"


def bh_correct(pvals):
    """Benjamini-Hochberg FDR correction; returns array of q-values."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    qvals = pvals * n / ranks
    # enforce monotonicity from right
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals = qvals[order]          # re-index to original order
    return np.minimum(qvals, 1.0)


def pairwise_mwu(df_seg, cell_types):
    """
    For a single segment's data (rows = cells with p_cnv and cell_type),
    return dict of {(ct_a, ct_b): (U, p)} for all pairs.
    """
    results = {}
    for i, cta in enumerate(cell_types):
        for ctb in cell_types[i+1:]:
            a = df_seg.loc[df_seg["cell_type"] == cta, "p_cnv"].values
            b = df_seg.loc[df_seg["cell_type"] == ctb, "p_cnv"].values
            if len(a) < 3 or len(b) < 3:
                results[(cta, ctb)] = (np.nan, 1.0)
                continue
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            results[(cta, ctb)] = (U, p)
    return results


# ── Per-patient analysis ───────────────────────────────────────────────────────

def analyze_patient(patient, df_raw):
    """
    Given the patient's joint_post_filtered DataFrame, return:
      - enrich_df : long-format enrichment table
      - heatmap_data : dict with 'mat', 'seg_labels', 'cell_types', 'sig_mask', 'cnv_states'
    """
    df_raw = df_raw.copy()

    # Add arm_label column
    df_raw["arm_label"] = [
        arm_label(row.CHROM, row.seg_start, row.seg_end, row.cnv_state)
        for _, row in df_raw.iterrows()
    ]

    # Filter to tumor cell types present in this patient
    present_ct = [ct for ct in TUMOR_TYPES if ct in df_raw["cell_type"].values]
    if len(present_ct) < 1:
        return None, None

    df_tumor = df_raw[df_raw["cell_type"].isin(present_ct)].copy()
    segs = df_tumor[["seg","CHROM","seg_start","seg_end","cnv_state","arm_label"]]\
               .drop_duplicates("seg")\
               .sort_values(["CHROM","seg_start"])
    seg_order = segs["seg"].tolist()

    # ── Mean p_cnv per (seg × cell_type) ──────────────────────────────────────
    mean_mat = df_tumor.groupby(["seg","cell_type"])["p_cnv"].mean().unstack(
        fill_value=np.nan)
    # Ensure all present_ct columns
    for ct in present_ct:
        if ct not in mean_mat.columns:
            mean_mat[ct] = np.nan
    mean_mat = mean_mat[present_ct]
    mean_mat = mean_mat.reindex(seg_order)

    # ── Statistical tests ─────────────────────────────────────────────────────
    # For each segment, pairwise MWU across all tumor cell-type pairs
    all_pvals_info = []   # (seg, ct_a, ct_b, U, p)
    for seg_id in seg_order:
        seg_data = df_tumor[df_tumor["seg"] == seg_id]
        if len(present_ct) < 2:
            continue
        pairs = pairwise_mwu(seg_data, present_ct)
        for (cta, ctb), (U, p) in pairs.items():
            all_pvals_info.append({"seg": seg_id, "ct_a": cta, "ct_b": ctb, "U": U, "p": p})

    if all_pvals_info:
        pvals_df = pd.DataFrame(all_pvals_info)
        pvals_df["q"] = bh_correct(pvals_df["p"].fillna(1.0).values)
    else:
        pvals_df = pd.DataFrame(columns=["seg","ct_a","ct_b","U","p","q"])

    # ── Significance mask for heatmap ─────────────────────────────────────────
    # sig_mask[i, j] = True if cell_type j is significantly different from
    # at least one other cell_type for segment i (q < 0.05)
    sig_mask = np.zeros(mean_mat.shape, dtype=bool)
    for row_q in pvals_df[pvals_df["q"] < 0.05].itertuples():
        if row_q.seg in seg_order:
            ri = seg_order.index(row_q.seg)
            for ct in [row_q.ct_a, row_q.ct_b]:
                if ct in present_ct:
                    ci = present_ct.index(ct)
                    sig_mask[ri, ci] = True

    # ── Compute specificity score (range of mean p_cnv across cell types) ─────
    p_range = mean_mat.max(axis=1) - mean_mat.min(axis=1)

    # ── Build enrichment output DataFrame ─────────────────────────────────────
    enrich_rows = []
    seg_meta = segs.set_index("seg")
    for seg_id in seg_order:
        for ct in present_ct:
            mean_p = mean_mat.at[seg_id, ct] if seg_id in mean_mat.index else np.nan
            # is this CT the max?
            row_vals = mean_mat.loc[seg_id, present_ct]
            max_ct   = row_vals.idxmax() if not row_vals.isna().all() else None
            enrich_rows.append({
                "patient_id":  patient,
                "seg_id":      seg_id,
                "arm_label":   seg_meta.at[seg_id, "arm_label"],
                "cnv_state":   seg_meta.at[seg_id, "cnv_state"],
                "CHROM":       seg_meta.at[seg_id, "CHROM"],
                "seg_start":   seg_meta.at[seg_id, "seg_start"],
                "cell_type":   ct,
                "mean_p_cnv":  round(mean_p, 4) if not np.isnan(mean_p) else np.nan,
                "p_range":     round(float(p_range.get(seg_id, np.nan)), 4),
                "is_max_ct":   (max_ct == ct),
            })
    enrich_df = pd.DataFrame(enrich_rows)

    # Attach pairwise q-values (best q across all pairs involving each ct)
    if not pvals_df.empty:
        for ct in present_ct:
            q_for_ct = pvals_df[
                (pvals_df["ct_a"] == ct) | (pvals_df["ct_b"] == ct)
            ].groupby("seg")["q"].min()
            enrich_df.loc[enrich_df["cell_type"] == ct, "min_q"] = \
                enrich_df.loc[enrich_df["cell_type"] == ct, "seg_id"].map(
                    q_for_ct).values
    else:
        enrich_df["min_q"] = np.nan

    # ── Heatmap data dict ──────────────────────────────────────────────────────
    arm_labels  = [seg_meta.at[s, "arm_label"] for s in seg_order]
    cnv_states  = [seg_meta.at[s, "cnv_state"] for s in seg_order]

    heatmap_data = {
        "mat":        mean_mat.values,         # (n_segs, n_ct)
        "seg_ids":    seg_order,
        "arm_labels": arm_labels,
        "cnv_states": cnv_states,
        "cell_types": present_ct,
        "sig_mask":   sig_mask,
        "p_range":    p_range.values,
    }

    return enrich_df, heatmap_data


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_patient(patient, hm, out_path):
    """
    Two-panel figure:
      Left  : heatmap of mean p_cnv per (seg × cell_type), sorted by variability
      Right : per-cell-type specificity — dot-plot of top differentially-carried CNVs
    """
    mat        = hm["mat"]            # (n_segs, n_ct)
    arm_labels = hm["arm_labels"]
    cnv_states = hm["cnv_states"]
    cell_types = hm["cell_types"]
    sig_mask   = hm["sig_mask"]
    p_range    = hm["p_range"]
    n_segs, n_ct = mat.shape

    if n_segs == 0 or n_ct == 0:
        return

    # Sort rows by p_range descending (most variable first), then by chromosome
    sort_idx = np.argsort(-p_range)
    mat        = mat[sort_idx]
    arm_labels = [arm_labels[i] for i in sort_idx]
    cnv_states = [cnv_states[i] for i in sort_idx]
    sig_mask   = sig_mask[sort_idx]

    # Column colours
    ct_colors = [CELL_TYPE_COLORS.get(ct, "#888888") for ct in cell_types]

    # ── Figure layout ──────────────────────────────────────────────────────────
    row_h   = max(0.28, min(0.45, 9.0 / max(n_segs, 1)))
    fig_h   = max(5, n_segs * row_h + 2.5)
    col_w   = max(0.7, min(1.2, 8.0 / max(n_ct, 1)))
    fig_w   = max(8, n_ct * col_w + 6)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(f"Patient {patient} — Cell-type-specific CNV events\n"
                 f"(mean posterior CNV probability per cell type)",
                 fontsize=11, fontweight="bold", y=0.995)

    # Left panel: heatmap
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[n_ct, max(n_ct, 2)],
                           wspace=0.50, left=0.20, right=0.97,
                           top=0.93, bottom=0.08)
    ax_hm   = fig.add_subplot(gs[0, 0])
    ax_bar  = fig.add_subplot(gs[0, 1])

    # ── Heatmap ────────────────────────────────────────────────────────────────
    im = ax_hm.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP,
                      vmin=0, vmax=1, interpolation="nearest")

    # X-axis: cell type labels coloured
    ax_hm.set_xticks(range(n_ct))
    ax_hm.set_xticklabels(cell_types, rotation=40, ha="right", fontsize=8)
    for tick, color in zip(ax_hm.get_xticklabels(), ct_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")

    # Y-axis: arm labels coloured by CNV state
    ax_hm.set_yticks(range(n_segs))
    ax_hm.set_yticklabels([])
    for ri, (lbl, state) in enumerate(zip(arm_labels, cnv_states)):
        col = CNV_STATE_COLORS.get(state, "#333333")
        ax_hm.text(-0.12, ri, lbl,
                   transform=ax_hm.get_yaxis_transform(),
                   ha="right", va="center", fontsize=7, color=col)

    # Significance stars
    for ri in range(n_segs):
        for ci in range(n_ct):
            if sig_mask[ri, ci]:
                ax_hm.text(ci, ri, "★", ha="center", va="center",
                           fontsize=5.5, color="white", alpha=0.85)

    # Colourbar
    cbar = plt.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.02, shrink=0.6)
    cbar.set_label("Mean P(CNV)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax_hm.set_title("Mean P(CNV) per cell type", fontsize=8.5, pad=4)

    # Top coloured header bar for cell types
    for ci, (ct, col) in enumerate(zip(cell_types, ct_colors)):
        ax_hm.add_patch(plt.Rectangle(
            (ci - 0.5, -0.8), 1, 0.4,
            color=col, transform=ax_hm.transData, clip_on=False))

    # ── Right panel: variability bar chart ──────────────────────────────────
    # Show top 20 most variable segments as horizontal bars coloured by max CT
    top_n  = min(20, n_segs)
    top_ri = np.arange(top_n)  # already sorted
    colors_bar = []
    for ri in top_ri:
        max_ci = np.nanargmax(mat[ri])
        colors_bar.append(ct_colors[max_ci])

    ax_bar.barh(top_ri, p_range[sort_idx[:top_n]],
                color=colors_bar, edgecolor="none", height=0.65)
    ax_bar.set_yticks(top_ri)
    ax_bar.set_yticklabels([arm_labels[i] for i in top_ri], fontsize=7)
    ax_bar.set_xlabel("P(CNV) range\n(max − min across cell types)", fontsize=7.5)
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_ylim(top_n - 0.5, -0.5)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(labelsize=7)
    ax_bar.set_title("Top variable events\n(bar colour = highest CT)", fontsize=8.5, pad=4)
    ax_bar.axvline(0.3, color="#aaaaaa", lw=0.7, ls="--")

    # Legend for cell type colours
    from matplotlib.patches import Patch
    handles = [Patch(color=CELL_TYPE_COLORS.get(ct, "#888888"), label=ct)
               for ct in cell_types]
    fig.legend(handles=handles, loc="lower center", ncol=len(cell_types),
               fontsize=7, frameon=False, title="Cell type", title_fontsize=7.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="*")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "patient_list.txt")) as f:
        pid_entries = [l.strip() for l in f if l.strip()]

    subset = set(args.patients) if args.patients else None

    all_enrich = []

    for pid_entry in pid_entries:
        patient = pid_entry.split("_")[0]
        if subset and patient not in subset and pid_entry not in subset:
            continue

        jp_path = os.path.join(BASE, patient, OUTDIR_NAME,
                               f"{patient}_joint_post_filtered.csv")
        if not os.path.exists(jp_path):
            print(f"[{patient}] joint_post_filtered not found — skipping")
            continue

        print(f"[{patient}] Loading {jp_path} …")
        usecols = ["cell","seg","CHROM","seg_start","seg_end",
                   "cnv_state","p_cnv","p_amp","p_del","p_loh","cell_type"]
        # add seg_label if present
        try:
            df_raw = pd.read_csv(jp_path, usecols=usecols)
        except ValueError:
            usecols_try = [c for c in usecols if c != "seg_label"]
            df_raw = pd.read_csv(jp_path, usecols=usecols_try)

        present_ct = [ct for ct in TUMOR_TYPES if ct in df_raw["cell_type"].values]
        print(f"  Tumor types: {present_ct}")

        if len(present_ct) < 2:
            print(f"  < 2 tumor types — still computing (single-type statistics)")

        enrich_df, hm = analyze_patient(patient, df_raw)
        if enrich_df is None:
            print(f"  No tumor cells found — skipping")
            continue

        all_enrich.append(enrich_df)

        # Plot
        if hm and len(hm["cell_types"]) >= 1:
            out_png = os.path.join(OUT_PNG_DIR, f"{patient}_cnv_celltype.png")
            plot_patient(patient, hm, out_png)
        else:
            print(f"  Skipping plot (< 2 cell types or no data)")

    # Save enrichment TSV
    if all_enrich:
        out_df = pd.concat(all_enrich, ignore_index=True)
        out_df.to_csv(OUT_TSV, sep="\t", index=False)
        print(f"\nEnrichment TSV: {OUT_TSV}  ({len(out_df)} rows, "
              f"{out_df.patient_id.nunique()} patients)")
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()
