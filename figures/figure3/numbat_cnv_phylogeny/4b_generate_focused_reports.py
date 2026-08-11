#!/usr/bin/env python3
"""
4b_generate_focused_reports.py

Focused per-patient PDF reports containing:
  Page 1 — Numbat phylogenies (majority-vote coloring)
              • Patient-level tree (majority)
              • Timepoint-colored tree (if ≥2 timepoints)
              • Sample-level tree thumbnails (majority)
  Page 2 — SNV overlap + CNV burden
              • SNV UpSet plot  ({patient}_SNV_venn.png)
              • Boxplot of gene-weighted CNV burden by tumor cell type

Usage:
    python 4b_generate_focused_reports.py [--patients RU1083 RU581 ...]
"""

import argparse, math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE        = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
OUTDIR_NAME = "numbat_output_FACETS_HISENS_metacell"
SNV_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "SNV"))
BURDEN_TSV  = os.path.join(SCRIPT_DIR, "cnv_burden_metacell_joined.tsv")
TP_FILE     = os.path.join(SCRIPT_DIR, "sample_timepoints.tsv")
SAMPLE_LIST = os.path.join(SCRIPT_DIR, "sample_list.txt")
PATIENT_LIST = os.path.join(SCRIPT_DIR, "patient_list.txt")
REPORT_DIR  = os.path.join(SCRIPT_DIR, "figures", "patient_reports_focused")

CELL_TYPE_COLORS = {
    "SCLC-A":              "#d62728",
    "SCLC-N":              "#17becf",
    "SCLC-P":              "#1f77b4",
    "NonNE SCLC":          "#4B0082",
    "EMT":                 "#673147",
    "FOXI1+ NonNE SCLC-N": "#2ca02c",
    "FOXI1+ SCLC-N":       "#1f77b4",  # NE subtype (RU1518)
    "LUAD":                "#ffd700",
    "LUSC":                "#FFA500",
}
TUMOR_TYPES = set(CELL_TYPE_COLORS)

PATIENT_CELL_TYPE_REMAP = {
    "RU1518": {"FOXI1+ NonNE SCLC-N": "FOXI1+ SCLC-N"},
}


# ── PNG helpers ────────────────────────────────────────────────────────────────

def embed_png(ax, path, title=None):
    if path and os.path.exists(path):
        ax.imshow(np.asarray(Image.open(path)))
    else:
        ax.text(0.5, 0.5, "Not\navailable", ha="center", va="center",
                fontsize=7, color="#AAAAAA", transform=ax.transAxes)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def phylo_path(patient, sample_id=None, kind="majority"):
    """Return path to a majority-vote (or timepoint) phylogeny PNG."""
    d = os.path.join(BASE, patient, OUTDIR_NAME)
    if kind == "timepoint":
        p = os.path.join(d, f"{patient}_phylogeny_timepoint.png")
    elif sample_id:
        p = os.path.join(d, sample_id, f"{sample_id}_phylogeny_{kind}.png")
    else:
        p = os.path.join(d, f"{patient}_phylogeny_{kind}.png")
    return p if os.path.exists(p) else None


def snv_path(patient):
    p = os.path.join(SNV_DIR, f"{patient}_SNV_venn.png")
    return p if os.path.exists(p) else None


# ── CNV burden boxplot ─────────────────────────────────────────────────────────

def draw_cnv_burden(ax, burden_pat):
    """
    Horizontal strip + IQR box of cnv_burden_gene per tumor cell type,
    ordered ascending by median (lowest = putative ancestral state).
    """
    metric = "cnv_burden_gene_bin" if "cnv_burden_gene_bin" in burden_pat.columns \
             else "cnv_burden_gene"
    sub = burden_pat[burden_pat["cell_type"].isin(TUMOR_TYPES)].dropna(subset=[metric])
    if sub.empty:
        ax.text(0.5, 0.5, "No burden data", ha="center", va="center",
                fontsize=8, color="#AAAAAA", transform=ax.transAxes)
        ax.set_axis_off()
        return

    order = (sub.groupby("cell_type")[metric]
               .median().sort_values().index.tolist())
    rng = np.random.default_rng(42)

    for yi, ct in enumerate(order):
        vals = sub[sub["cell_type"] == ct][metric].values
        color = CELL_TYPE_COLORS.get(ct, "#888888")
        q1, med, q3 = np.percentile(vals, [25, 50, 75])

        ax.barh(yi, q3 - q1, left=q1, height=0.45,
                color=color, alpha=0.25, zorder=2)
        ax.plot([med, med], [yi - 0.22, yi + 0.22],
                color=color, lw=2.0, zorder=3)
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(vals, yi + jitter, color=color, s=10,
                   alpha=0.6, linewidths=0, zorder=4)
        ax.text(1.02, (yi + 0.5) / len(order), f"n={len(vals)}",
                va="center", fontsize=8, color="#555555",
                transform=ax.transAxes)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10)
    for tick, ct in zip(ax.get_yticklabels(), order):
        tick.set_color(CELL_TYPE_COLORS.get(ct, "#333333"))
        tick.set_fontweight("bold")

    if order:
        med0 = sub[sub["cell_type"] == order[0]][metric].median()
        ax.annotate("★ lowest", xy=(med0, 0),
                    xytext=(med0 + 0.003, 0.4),
                    fontsize=8, color="#8B0000",
                    arrowprops=dict(arrowstyle="-", color="#8B0000", lw=0.8))

    ax.set_xlabel("Fraction of genes altered\n(binarized p_cnv ≥ 0.5)", fontsize=10)
    ax.set_title("CNV burden by tumor cell type\n"
                 "(ascending → putative ancestral at bottom)", fontsize=10, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


# ── Per-patient report ─────────────────────────────────────────────────────────

def make_report(patient, samples, tp_df):
    """
    Single-page PDF: two panels side-by-side, no suptitle header.

    Col 1 — SNV UpSet plot (embedded PNG).
    Col 2 — Numbat phylogenies (majority vote):
               Patient tree at top; timepoint tree below it (if ≥2 TPs);
               sample thumbnails at bottom (if multi-sample).
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_pdf = os.path.join(REPORT_DIR, f"{patient}_focused_report.pdf")

    n_tp         = int(tp_df["date_rank"].max()) if len(tp_df) else 1
    multi_tp     = n_tp >= 2
    multi_sample = len(samples) > 1
    tp_rank      = tp_df.set_index("sample_id")["date_rank"].to_dict() if len(tp_df) else {}

    sorted_sids = tp_df.sort_values("date_rank")["sample_id"].tolist()
    sorted_sids += [s for s in samples if s not in sorted_sids]

    # ── Determine row structure inside the phylogeny column ───────────────────
    n_samp_cols = min(len(sorted_sids), 4) if multi_sample else 0
    n_samp_rows = math.ceil(len(sorted_sids) / n_samp_cols) if n_samp_cols else 0

    phy_row_h   = [3.5]
    if multi_tp:
        phy_row_h.append(3.5)
    if multi_sample:
        phy_row_h.append(n_samp_rows * 2.0)

    fig_h = max(6.0, sum(phy_row_h) + 0.3)
    fig_w = 12.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Outer 2-column gridspec: SNV | phylogeny (equal width)
    outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1, 1],
        wspace=0.08,
        left=0.02, right=0.97, top=0.98, bottom=0.04,
    )

    # ── Column 1: SNV UpSet ───────────────────────────────────────────────────
    ax_snv = fig.add_subplot(outer[0, 0])
    embed_png(ax_snv, snv_path(patient))

    # ── Column 2: phylogenies ─────────────────────────────────────────────────
    phy_gs = gridspec.GridSpecFromSubplotSpec(
        len(phy_row_h), 1,
        subplot_spec=outer[0, 1],
        hspace=0.06,
        height_ratios=phy_row_h,
    )

    phy_row = 0
    ax_pat = fig.add_subplot(phy_gs[phy_row, 0])
    embed_png(ax_pat, phylo_path(patient, kind="majority"))
    phy_row += 1

    if multi_tp:
        ax_tp = fig.add_subplot(phy_gs[phy_row, 0])
        embed_png(ax_tp, phylo_path(patient, kind="timepoint"))
        phy_row += 1

    if multi_sample:
        samp_inner = gridspec.GridSpecFromSubplotSpec(
            n_samp_rows, n_samp_cols,
            subplot_spec=phy_gs[phy_row, 0],
            hspace=0.05, wspace=0.03,
        )
        for i, sid in enumerate(sorted_sids):
            r, c = divmod(i, n_samp_cols)
            ax_s = fig.add_subplot(samp_inner[r, c])
            embed_png(ax_s,
                      phylo_path(patient, sample_id=sid, kind="majority"),
                      title=f"{sid} (TP{tp_rank.get(sid,'?')})")

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved → {out_pdf}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="*")
    args = parser.parse_args()

    with open(PATIENT_LIST) as f:
        pid_entries = [l.strip() for l in f if l.strip()]
    with open(SAMPLE_LIST) as f:
        all_samples = [l.strip() for l in f if l.strip()]

    tp_all = pd.read_csv(TP_FILE, sep="\t")


    subset = set(args.patients) if args.patients else None

    for pid_entry in pid_entries:
        patient = pid_entry.split("_")[0]
        if subset and patient not in subset and pid_entry not in subset:
            continue

        samples = [s for s in all_samples if s.split("_")[0] == patient]
        tp_df   = tp_all[tp_all["patient_id"] == patient].copy()
        if len(tp_df) == 0 and pid_entry != patient:
            tp_df = tp_all[tp_all["sample_id"] == pid_entry].copy()

        n_tp = int(tp_df["date_rank"].max()) if len(tp_df) else 1
        print(f"[{patient}]  {len(samples)} sample(s), {n_tp} timepoint(s)")
        make_report(patient, samples, tp_df)

    print(f"\nAll reports → {REPORT_DIR}/")


if __name__ == "__main__":
    main()
