#!/usr/bin/env python3
"""
Generate per-patient CNV phylogeny interpretation reports (PDF).

Usage:
    python 4_generate_patient_reports.py [--patients RU1083 RU222 ...]

For each patient in patient_list.txt, produces a PDF with:
  Page 1 : Clinical overview | patient-level tree (cell type) |
           cell type composition bar chart | interpretation
  Page 2 : CNV analysis — clone-fraction heatmap + trunk/subclonal listing
  Page 3 : Cell-type-specific CNV analysis
           (mean P(CNV) per seg × cell type; pre-computed by 6_celltype_cnv_analysis.py)
  Page 4 : Sample-level trees colored by cell type (if > 1 sample)
  Page 5 : Timepoint-colored patient tree + temporal summary (if ≥ 2 timepoints)
"""

import argparse, csv, math, os, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH   = os.path.join(SCRIPT_DIR,
    "Rudin Lab NE transformation 040723_AG_CF.JC.deidentified.xlsx")
BASE         = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
OUTDIR_NAME  = "numbat_output_FACETS_HISENS_metacell"
TP_FILE      = os.path.join(SCRIPT_DIR, "sample_timepoints.tsv")
PATIENT_LIST = os.path.join(SCRIPT_DIR, "patient_list.txt")
SAMPLE_LIST  = os.path.join(SCRIPT_DIR, "sample_list.txt")
REPORT_DIR   = os.path.join(SCRIPT_DIR, "figures", "patient_reports")
CNV_SUMMARY    = os.path.join(SCRIPT_DIR, "cnv_summary.tsv")
CNV_DETAIL     = os.path.join(SCRIPT_DIR, "cnv_mutations.tsv")
CT_ENRICH_TSV  = os.path.join(SCRIPT_DIR, "cnv_celltype_enrichment.tsv")
CNV_BURDEN_TSV = os.path.join(SCRIPT_DIR, "cnv_burden_per_cell.tsv")
CT_PNG_DIR     = os.path.join(SCRIPT_DIR, "figures", "cnv_celltype")
PHYLO_TSV      = os.path.join(SCRIPT_DIR, "phylo_celltype.tsv")
PHYLO_PNG_DIR  = os.path.join(SCRIPT_DIR, "figures", "cnv_phylo_celltype")
SNV_PNG_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "SNV"))

# ── Colour palettes ────────────────────────────────────────────────────────────
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
TUMOR_TYPES = set(CELL_TYPE_COLORS.keys())
# Fixed 4-step timepoint palette
TP_COLORS = ["#4575B4", "#91BFDB", "#FC8D59", "#D73027"]

# CNV state colors
CNV_STATE_COLORS = {
    "amp":  "#d62728",  "gain":   "#d62728",
    "del":  "#1f77b4",  "loss":   "#1f77b4",
    "loh":  "#9467bd",  "LOH":    "#9467bd",
    "bamp": "#8B0000",
    "bdel": "#003f8e",  "homdel": "#003f8e",
}
CNV_CMAP = LinearSegmentedColormap.from_list(
    "cnv_freq", ["#ffffff", "#fee0d2", "#fc9272", "#de2d26", "#8B0000"])

# Known clinically relevant CNV regions for annotation
ONCOGENES  = {"chr3q", "chr5p", "chr8q", "chr17q", "chr19q", "chr20q"}
SUPPRESSORS = {"chr3p", "chr13q", "chr17p", "chr9p", "chr10q", "chr4q",
               "chr16q", "chr18q", "chrXp", "chrXq"}
# RB1 on 13q, TP53 on 17p, CDKN2A on 9p, PTEN on 10q

def arm_key(label):
    """Extract 'chrNX' prefix from 'chrNX gain' labels for clinical relevance."""
    return label.split()[0] if label else ""


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_clinical():
    df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")
    rename = {
        "Rudin Sample ID Standardized":  "sample_id",
        "Rudin Patient ID Standardized": "patient_id",
        "Collection date":               "collection_date",
        "Joe Abbreviation":              "joe_abbrev",
        "Organ site abbreviation":       "organ_abbrev",
        "Pathology Read":                "pathology",
        "Prior Treatment ":              "treatment",
        "Age at Collection":             "age",
        "Sex":                           "sex",
        "Dx Date":                       "dx_date",
        "Dx Stage":                      "dx_stage",
        "Smoking":                       "smoking",
        "TP53":                          "tp53",
        "RB1":                           "rb1",
        "EGFR":                          "egfr",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["sample_id"]  = df["sample_id"].astype(str).str.strip()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df


def load_timepoints():
    return pd.read_csv(TP_FILE, sep="\t")


def load_cnv_data():
    """Load CNV summary and detail, return (summary_df, detail_df)."""
    summary = pd.read_csv(CNV_SUMMARY, sep="\t") if os.path.exists(CNV_SUMMARY) else pd.DataFrame()
    detail  = pd.read_csv(CNV_DETAIL,  sep="\t") if os.path.exists(CNV_DETAIL)  else pd.DataFrame()
    return summary, detail


def load_ct_enrichment():
    """Load cell-type CNV enrichment table from 6_celltype_cnv_analysis.py output."""
    if os.path.exists(CT_ENRICH_TSV):
        return pd.read_csv(CT_ENRICH_TSV, sep="\t")
    return pd.DataFrame()


def load_phylo_celltype():
    """Load clone–depth–cell-type table from 7_extract_phylo_celltype.R output."""
    if os.path.exists(PHYLO_TSV):
        return pd.read_csv(PHYLO_TSV, sep="\t")
    return pd.DataFrame()


def load_cnv_burden():
    """Load per-cell CNV burden from 7b_cnv_burden_per_cell.py output."""
    if os.path.exists(CNV_BURDEN_TSV):
        return pd.read_csv(CNV_BURDEN_TSV, sep="\t")
    return pd.DataFrame()


def load_composition(patient, sample_id):
    csv_path = os.path.join(BASE, patient, OUTDIR_NAME,
                            sample_id, f"{sample_id}_joint_post_filtered.csv")
    if not os.path.exists(csv_path):
        return None
    counts = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            ct = row.get("cell_type", "Unknown").strip()
            counts[ct] = counts.get(ct, 0) + 1
    return counts


def get_snv_png(patient):
    p = os.path.join(SNV_PNG_DIR, f"{patient}_SNV_venn.png")
    return p if os.path.exists(p) else None


def get_png(patient, sample_id=None, kind="celltype"):
    if kind == "timepoint":
        p = os.path.join(BASE, patient, OUTDIR_NAME,
                         f"{patient}_phylogeny_timepoint.png")
        return p if os.path.exists(p) else None
    if sample_id:
        p = os.path.join(BASE, patient, OUTDIR_NAME,
                         sample_id, f"{sample_id}_phylogeny.png")
        return p if os.path.exists(p) else None
    p = os.path.join(BASE, patient, OUTDIR_NAME, f"{patient}_phylogeny.png")
    return p if os.path.exists(p) else None


# ── Drawing helpers ────────────────────────────────────────────────────────────

def embed_png(ax, path, title=None):
    if not path or not os.path.exists(path):
        ax.text(0.5, 0.5, "Plot not\navailable",
                ha="center", va="center", fontsize=8, color="#AAAAAA",
                transform=ax.transAxes)
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=7.5, pad=2)
        return
    ax.imshow(np.asarray(Image.open(path)))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7.5, pad=2)


def draw_clinical(ax, patient, pid_entry, clinical_df, tp_df):
    rows = clinical_df[clinical_df["patient_id"].astype(str).str.startswith(patient)]

    def first(col, maxlen=50):
        s = rows[col].dropna() if col in rows.columns else pd.Series(dtype=str)
        return textwrap.shorten(str(s.iloc[0]), maxlen) if len(s) else "—"

    dx = first("dx_date")[:10]
    fields = [
        ("Sex",                   first("sex")),
        ("Dx date",               dx),
        ("Stage",                 first("dx_stage")),
        ("Smoking",               first("smoking")),
        ("TP53 / RB1 / EGFR",    f"{first('tp53')} / {first('rb1')} / {first('egfr')}"),
        ("Treatment",             first("treatment", 70)),
    ]
    ax.set_axis_off()
    y, dy = 0.97, 0.095

    ax.text(0, y, "Clinical overview", fontsize=8.5, fontweight="bold",
            transform=ax.transAxes, va="top")
    y -= 0.04
    for lbl, val in fields:
        ax.text(0.0,  y, f"{lbl}:", fontsize=7, fontweight="bold",
                transform=ax.transAxes, va="top", color="#222222")
        ax.text(0.38, y, val,       fontsize=7,
                transform=ax.transAxes, va="top", color="#444444")
        y -= dy

    y -= 0.02
    ax.text(0, y, "Samples", fontsize=8.5, fontweight="bold",
            transform=ax.transAxes, va="top")
    y -= 0.04
    hdrs = ["Sample", "Date", "TP", "Pathology"]
    xs   = [0.0, 0.30, 0.52, 0.60]
    for x_, h in zip(xs, hdrs):
        ax.text(x_, y, h, fontsize=6.5, fontweight="bold",
                transform=ax.transAxes, va="top")
    y -= dy * 0.80

    # Adaptive row height: divide remaining space equally among all sample rows
    n_samp = len(tp_df)
    row_dy = min(dy * 0.80, max(y - 0.01, 0) / max(n_samp, 1))
    row_dy = max(row_dy, 0.030)          # hard minimum to avoid text overlap
    row_fs = 6 if row_dy >= 0.055 else 5  # shrink font when rows are tight

    for _, r in tp_df.sort_values("date_rank").iterrows():
        sid  = r["sample_id"]
        date = str(r.get("collection_date", ""))[:10]
        tp   = f"TP{int(r['date_rank'])}"
        cr   = clinical_df[clinical_df["sample_id"] == sid]
        path = textwrap.shorten(str(cr["pathology"].iloc[0]), 28) \
               if len(cr) and pd.notna(cr["pathology"].iloc[0]) else "—"
        for x_, val in zip(xs, [sid, date, tp, path]):
            ax.text(x_, y, val, fontsize=row_fs, transform=ax.transAxes, va="top")
        y -= row_dy
        if y < 0.01:
            break


def draw_composition(ax, comp_dict, tp_df):
    valid = {s: c for s, c in comp_dict.items() if c}
    if not valid:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=8, color="#AAAAAA", transform=ax.transAxes)
        ax.set_axis_off()
        return

    all_cts   = set(ct for c in valid.values() for ct in c)
    tumor_cts = [ct for ct in CELL_TYPE_COLORS if ct in all_cts]
    other_cts = sorted(all_cts - set(tumor_cts))
    ordered   = tumor_cts + other_cts

    tp_order = tp_df.set_index("sample_id")["date_rank"].to_dict() if len(tp_df) else {}
    sids     = sorted(valid.keys(), key=lambda s: (tp_order.get(s, 999), s))
    prefix   = os.path.commonprefix(sids)
    labels   = [s[len(prefix):] or s for s in sids]

    bottom = np.zeros(len(sids))
    for ct in ordered:
        vals = np.array([valid[s].get(ct, 0) / max(sum(valid[s].values()), 1)
                         for s in sids])
        ax.bar(range(len(sids)), vals, bottom=bottom,
               color=CELL_TYPE_COLORS.get(ct, "#CCCCCC"),
               label=ct, width=0.72, edgecolor="none")
        bottom += vals

    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction", fontsize=7)
    ax.set_title("Cell type composition", fontsize=8, pad=3)
    ax.tick_params(labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)


def draw_interpretation(ax, sections):
    ax.set_axis_off()
    y = 0.99
    ax.text(0, y, "Interpretation", fontsize=9, fontweight="bold",
            transform=ax.transAxes, va="top", color="#111111")
    y -= 0.06

    for heading, body in sections:
        if y < 0.02:
            break
        ax.text(0, y, f"▸  {heading}", fontsize=8, fontweight="bold",
                transform=ax.transAxes, va="top", color="#223366")
        y -= 0.045
        for line in textwrap.wrap(body, width=130):
            if y < 0.02:
                break
            ax.text(0.015, y, line, fontsize=7, transform=ax.transAxes,
                    va="top", color="#333333")
            y -= 0.038
        y -= 0.02


# ── CNV page helpers ───────────────────────────────────────────────────────────

def _summarise_cnv_list(rows, max_items=None):
    """Compact comma-separated labels, deduplicating identical arm+state combos."""
    labels = list(dict.fromkeys(rows["seg_label"].tolist()))  # ordered unique
    if max_items:
        labels = labels[:max_items]
    return labels


def draw_cnv_heatmap(ax, detail_pat, tp_df, n_top=25):
    """
    Heatmap: rows = top-N CNV segments (by mean clone_frac),
             cols = samples (ordered by timepoint),
             colour = clone_frac value.
    """
    if detail_pat.empty:
        ax.text(0.5, 0.5, "No CNV data", ha="center", va="center",
                fontsize=8, color="#AAAAAA", transform=ax.transAxes)
        ax.set_axis_off()
        return

    # Order samples by timepoint
    tp_order = tp_df.set_index("sample_id")["date_rank"].to_dict() if len(tp_df) else {}
    samples  = sorted(detail_pat["sample_id"].unique(),
                      key=lambda s: (tp_order.get(s, 999), s))
    prefix   = os.path.commonprefix(samples)
    xlabels  = [s[len(prefix):] or s for s in samples]

    # Top segments by mean clone_frac
    seg_mean = detail_pat.groupby("seg_id")["clone_frac"].mean().sort_values(ascending=False)
    top_segs = seg_mean.index[:n_top].tolist()

    # Get seg_label for each seg_id
    id2label = detail_pat.drop_duplicates("seg_id").set_index("seg_id")["seg_label"].to_dict()
    ylabels  = [id2label.get(s, s) for s in top_segs]

    # Build matrix
    mat = np.zeros((len(top_segs), len(samples)))
    pivot = detail_pat[detail_pat["seg_id"].isin(top_segs)].pivot_table(
        index="seg_id", columns="sample_id", values="clone_frac", aggfunc="first")
    for ri, seg in enumerate(top_segs):
        for ci, sid in enumerate(samples):
            if seg in pivot.index and sid in pivot.columns:
                v = pivot.at[seg, sid]
                mat[ri, ci] = v if not np.isnan(v) else 0.0

    im = ax.imshow(mat, aspect="auto", cmap=CNV_CMAP, vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels(xlabels, fontsize=6.5, rotation=35, ha="right")
    ax.set_yticks(range(len(top_segs)))

    # Colour y-tick labels by CNV state
    ax.set_yticklabels([])
    for ri, (seg_id, label) in enumerate(zip(top_segs, ylabels)):
        state = label.split()[-1] if label else ""
        col   = CNV_STATE_COLORS.get(state, "#333333")
        ax.text(-0.02, ri, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.0, color=col)

    ax.set_title(f"Clone fraction per CNV event (top {len(top_segs)})", fontsize=8, pad=4)
    ax.tick_params(left=False)

    # Colourbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Clone fraction", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)


def draw_cnv_table(ax, summary_pat, classification_order=None):
    """
    Text-based table listing CNV events by classification tier.
    classification_order defaults to trunk → major_subclonal → minor_subclonal → sample_specific.
    """
    if classification_order is None:
        classification_order = ["trunk", "major_subclonal",
                                "minor_subclonal", "sample_specific"]

    tier_labels = {
        "trunk":           "Trunk (≥90% in all samples) — likely founding events",
        "major_subclonal": "Major subclonal (≥50% in all samples)",
        "minor_subclonal": "Minor subclonal (<50% in some samples)",
        "sample_specific": "Sample-specific / heterogeneous",
    }

    ax.set_axis_off()
    y = 0.99
    ax.text(0, y, "CNV events by clonal tier", fontsize=8.5, fontweight="bold",
            transform=ax.transAxes, va="top", color="#111111")
    y -= 0.07

    if summary_pat.empty:
        ax.text(0.02, y, "No CNV data available.", fontsize=7,
                transform=ax.transAxes, va="top", color="#888888")
        return

    for cls in classification_order:
        sub = summary_pat[summary_pat["classification"] == cls].copy()
        if sub.empty:
            continue
        if y < 0.02:
            break

        # Section header
        ax.text(0, y, f"▸  {tier_labels.get(cls, cls)}  [{len(sub)} events]",
                fontsize=7.5, fontweight="bold",
                transform=ax.transAxes, va="top", color="#223366")
        y -= 0.042

        # Group by state and list
        sub_sorted = sub.sort_values(["cnv_state","CHROM","seg_start"])
        for state, grp in sub_sorted.groupby("cnv_state", sort=False):
            labels = list(dict.fromkeys(grp["seg_label"].tolist()))
            state_col = CNV_STATE_COLORS.get(state, "#333333")
            # Print in wrapped lines of ~6 events each
            chunk = 6
            for i in range(0, len(labels), chunk):
                line = "  " + ",  ".join(labels[i:i+chunk])
                ax.text(0.01, y, line, fontsize=6.5,
                        transform=ax.transAxes, va="top", color=state_col)
                y -= 0.035
                if y < 0.02:
                    break
        y -= 0.015
        if y < 0.02:
            break


def draw_cnv_stats(ax, summary_pat, detail_pat, patient):
    """
    Summary statistics panel: polyclonality note, trunk count, etc.
    """
    ax.set_axis_off()
    if summary_pat.empty:
        return

    n_total  = len(summary_pat)
    n_trunk  = (summary_pat["classification"] == "trunk").sum()
    n_major  = (summary_pat["classification"] == "major_subclonal").sum()
    n_minor  = (summary_pat["classification"] == "minor_subclonal").sum()
    n_spec   = (summary_pat["classification"] == "sample_specific").sum()
    n_absent = (summary_pat["classification"] == "absent").sum()
    n_nonneu = n_total - n_absent

    # Polyclonality: average clone_frac across all events and samples
    if not detail_pat.empty:
        mean_frac = detail_pat["clone_frac"].mean()
        # Clonality score: fraction of events with clone_frac > 0.5
        pct_clonal = (detail_pat["clone_frac"] > 0.5).mean() * 100
    else:
        mean_frac = np.nan
        pct_clonal = np.nan

    y = 0.98
    ax.text(0, y, "CNV summary statistics", fontsize=8.5, fontweight="bold",
            transform=ax.transAxes, va="top")
    y -= 0.08

    stats = [
        ("Non-neutral segments",     f"{n_nonneu}"),
        ("Trunk (clonal)",           f"{n_trunk}   ({100*n_trunk/max(n_nonneu,1):.0f}%)"),
        ("Major subclonal",          f"{n_major}"),
        ("Minor subclonal",          f"{n_minor}"),
        ("Sample-specific",          f"{n_spec}"),
        ("Mean clone fraction",      f"{mean_frac:.2f}" if not np.isnan(mean_frac) else "—"),
        ("Events >50% in any clone", f"{pct_clonal:.0f}%" if not np.isnan(pct_clonal) else "—"),
    ]
    for lbl, val in stats:
        ax.text(0.0, y, f"{lbl}:", fontsize=7, fontweight="bold",
                transform=ax.transAxes, va="top", color="#222222")
        ax.text(0.60, y, val, fontsize=7,
                transform=ax.transAxes, va="top", color="#444444")
        y -= 0.085

    # Highlight clinically relevant trunk events
    if n_trunk > 0:
        trunk_segs = summary_pat[summary_pat["classification"] == "trunk"]["seg_label"].tolist()
        relevant = [s for s in trunk_segs
                    if arm_key(s) in SUPPRESSORS or arm_key(s) in ONCOGENES]
        if relevant:
            y -= 0.04
            ax.text(0, y, "Clinically relevant trunk events:", fontsize=7,
                    fontweight="bold", transform=ax.transAxes, va="top", color="#8B0000")
            y -= 0.045
            for line in textwrap.wrap(", ".join(relevant), width=42):
                ax.text(0.02, y, line, fontsize=6.5,
                        transform=ax.transAxes, va="top", color="#8B0000")
                y -= 0.04


# ── CNV burden summary ────────────────────────────────────────────────────────

BURDEN_METRICS = {
    "cnv_burden_bp":   "Probabilistic FGA\n(p_cnv-weighted bp / genome)",
    "cnv_burden_fga":  "Binary FGA\n(p_cnv ≥ 0.5 threshold)",
    "cnv_burden_segs": "Mean segment p(CNV)\n(unweighted average across segments)",
}


def _burden_tumor_stats(burden_pat, metric="cnv_burden_bp"):
    """
    Return per-tumor-cell-type stats (median, Q1, Q3, n) for the given metric,
    sorted ascending by median of cnv_burden_bp (consistent ordering across plots).
    """
    sub = burden_pat[burden_pat["cell_type"].isin(TUMOR_TYPES)].copy()
    if sub.empty or metric not in sub.columns:
        return pd.DataFrame()
    # Sort order always fixed by cnv_burden_bp median
    order = (
        sub.groupby("cell_type")["cnv_burden_bp"]
        .median()
        .sort_values()
        .index.tolist()
    )
    stats = (
        sub.groupby("cell_type")[metric]
        .agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            n="count",
        )
        .reindex(order)
        .reset_index()
    )
    return stats


def draw_cnv_burden_plot(ax, burden_pat, metric="cnv_burden_bp", show_yticks=True):
    """
    Horizontal strip + IQR box per tumor cell type for the given metric.
    Cell types ordered ascending by cnv_burden_bp median (consistent across panels).
    """
    stats = _burden_tumor_stats(burden_pat, metric)
    if stats.empty:
        ax.text(0.5, 0.5, "CNV burden data\nnot available",
                ha="center", va="center", fontsize=9, color="#AAAAAA",
                transform=ax.transAxes)
        ax.set_axis_off()
        return

    sub = burden_pat[burden_pat["cell_type"].isin(stats["cell_type"].tolist())].copy()
    ct_order = stats["cell_type"].tolist()
    rng = np.random.default_rng(42)

    for yi, ct in enumerate(ct_order):
        cells = sub[sub["cell_type"] == ct][metric].values
        color = CELL_TYPE_COLORS.get(ct, "#888888")

        row = stats[stats["cell_type"] == ct].iloc[0]
        # IQR box
        ax.barh(yi, row["q3"] - row["q1"], left=row["q1"],
                height=0.45, color=color, alpha=0.25, zorder=2)
        # Median line
        ax.plot([row["median"], row["median"]], [yi - 0.22, yi + 0.22],
                color=color, lw=2.0, zorder=3)
        # Jittered dots
        jitter = rng.uniform(-0.18, 0.18, size=len(cells))
        ax.scatter(cells, yi + jitter, color=color, s=10, alpha=0.7,
                   zorder=4, linewidths=0)
        # n= label on rightmost panel only (avoids clutter)
        if not show_yticks:
            ax.text(1.01, (yi + 0.5) / len(ct_order), f"n={len(cells)}",
                    va="center", fontsize=6, color="#555555",
                    transform=ax.transAxes)

    ax.set_yticks(range(len(ct_order)))
    if show_yticks:
        ax.set_yticklabels(ct_order, fontsize=7.5)
        for tick, ct in zip(ax.get_yticklabels(), ct_order):
            tick.set_color(CELL_TYPE_COLORS.get(ct, "#333333"))
            tick.set_fontweight("bold")
    else:
        ax.set_yticklabels([])

    ax.set_xlabel(BURDEN_METRICS[metric], fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)

    # Star the lowest-burden cell type on the primary metric panel
    if metric == "cnv_burden_bp":
        ax.annotate("★ lowest", xy=(stats.iloc[0]["median"], 0),
                    xytext=(stats.iloc[0]["median"] + 0.005, 0.42),
                    fontsize=6.5, color="#8B0000",
                    arrowprops=dict(arrowstyle="-", color="#8B0000", lw=0.8))


def draw_cnv_burden_text(ax, burden_pat, phylo_pat=None):
    """
    Text panel: 3-metric median table per tumor cell type + starting-cell-type inference.
    """
    ax.set_axis_off()
    stats = _burden_tumor_stats(burden_pat, metric="cnv_burden_bp")

    y = 0.99
    ax.text(0, y, "CNV burden summary (median per cell type)",
            fontsize=8.5, fontweight="bold", transform=ax.transAxes, va="top")
    y -= 0.07

    if stats.empty:
        ax.text(0, y, "No tumor cell burden data.", fontsize=7,
                transform=ax.transAxes, va="top", color="#888888")
        return

    # Column positions: cell type | FGA_bp | FGA_bin | segs | n
    xs = [0.0, 0.36, 0.54, 0.72, 0.88]
    hdrs = ["Cell type", "FGA_bp", "FGA_bin", "Segs", "n"]
    for x_, h in zip(xs, hdrs):
        ax.text(x_, y, h, fontsize=6.5, fontweight="bold",
                transform=ax.transAxes, va="top", color="#222222")
    y -= 0.05

    # Pre-compute medians for the other two metrics in same ct order
    stats_fga  = _burden_tumor_stats(burden_pat, "cnv_burden_fga").set_index("cell_type")
    stats_segs = _burden_tumor_stats(burden_pat, "cnv_burden_segs").set_index("cell_type")

    for _, row in stats.iterrows():
        ct    = row["cell_type"]
        color = CELL_TYPE_COLORS.get(ct, "#333333")
        m_fga  = stats_fga.at[ct,  "median"] if ct in stats_fga.index  else float("nan")
        m_segs = stats_segs.at[ct, "median"] if ct in stats_segs.index else float("nan")
        ax.text(xs[0], y, ct,                    fontsize=6.5, transform=ax.transAxes,
                va="top", color=color, fontweight="bold")
        ax.text(xs[1], y, f"{row['median']:.3f}", fontsize=6.5, transform=ax.transAxes,
                va="top", color="#333333")
        ax.text(xs[2], y, f"{m_fga:.3f}",         fontsize=6.5, transform=ax.transAxes,
                va="top", color="#333333")
        ax.text(xs[3], y, f"{m_segs:.3f}",         fontsize=6.5, transform=ax.transAxes,
                va="top", color="#333333")
        ax.text(xs[4], y, str(int(row["n"])),      fontsize=6.5, transform=ax.transAxes,
                va="top", color="#555555")
        y -= 0.052

    y -= 0.04
    ax.plot([0, 1], [y + 0.02, y + 0.02], color="#CCCCCC", lw=0.5,
            transform=ax.transAxes, clip_on=False)
    y -= 0.02

    # Starting cell type inference
    ax.text(0, y, "Interpretation", fontsize=8.5, fontweight="bold",
            transform=ax.transAxes, va="top")
    y -= 0.065

    lowest_ct  = stats.iloc[0]["cell_type"]
    highest_ct = stats.iloc[-1]["cell_type"]
    lowest_med = stats.iloc[0]["median"]
    highest_med = stats.iloc[-1]["median"]
    fold = highest_med / lowest_med if lowest_med > 0 else float("inf")

    body_lines = [
        f"Lowest burden: {lowest_ct} (median FGA {lowest_med:.3f})",
        f"Highest burden: {highest_ct} (median FGA {highest_med:.3f})",
    ]
    if fold > 1.5:
        body_lines.append(
            f"{highest_ct} carries {fold:.1f}× more altered genome than "
            f"{lowest_ct}, suggesting {highest_ct} accumulated more CNVs "
            f"and may represent a later or more genomically unstable state."
        )
        body_lines.append(
            f"Putative starting (ancestral) cell type: {lowest_ct}."
        )
    else:
        body_lines.append(
            "Burden is similar across cell types — CNV profiles do not "
            "strongly distinguish an ancestral cell type by this metric alone."
        )

    # Cross-reference phylogenetic ordering if available
    if phylo_pat is not None and not phylo_pat.empty:
        from_interpret = _get_celltype_first_depths(phylo_pat)
        if from_interpret:
            phylo_first = min(from_interpret, key=from_interpret.get)
            if phylo_first == lowest_ct:
                body_lines.append(
                    f"Phylogenetic ordering independently places {phylo_first} "
                    f"shallowest in the clone tree — consistent with CNV burden ranking."
                )
            else:
                body_lines.append(
                    f"Note: phylogenetic ordering places {phylo_first} shallowest "
                    f"in the clone tree, while CNV burden ranks {lowest_ct} lowest — "
                    f"consider both lines of evidence."
                )

    body_lines.append(
        "Caveat: CNV burden is computed at metacell resolution "
        "(~25 cells/metacell); values reflect the aggregated clone state."
    )

    for line in body_lines:
        for wrapped in textwrap.wrap(line, width=48):
            if y < 0.02:
                break
            ax.text(0, y, wrapped, fontsize=6.5, transform=ax.transAxes,
                    va="top", color="#333333")
            y -= 0.046
        y -= 0.01


# ── Auto-interpretation ────────────────────────────────────────────────────────

def _get_celltype_first_depths(phylo_pat):
    """
    Abundance-weighted mean clone depth for each tumor cell type.

    Each clone×sample observation is weighted by (n_cells × ct_frac), i.e. the
    approximate number of cells of this type in that clone.  The weighted mean
    depth is far more stable than the minimum depth, which is sensitive to a
    single clone with just a handful of cells.
    """
    if phylo_pat.empty:
        return {}
    sub = phylo_pat[phylo_pat["cell_type"].isin(TUMOR_TYPES)].copy()
    if sub.empty:
        return {}
    sub["weight"] = sub["n_cells"] * sub["ct_frac"]
    ct_depths = {}
    for ct, grp in sub.groupby("cell_type"):
        w_total = grp["weight"].sum()
        if w_total > 0:
            ct_depths[ct] = float((grp["depth"] * grp["weight"]).sum() / w_total)
    return ct_depths


def _get_divergence_depth(phylo_pat):
    """
    Lineage divergence depth, estimated as the 2nd-smallest abundance-weighted
    mean depth across tumor cell types.  The first (shallowest) cell type is
    treated as ancestral; the second defines when a distinct lineage branches off.
    """
    ct_depths = _get_celltype_first_depths(phylo_pat)
    if len(ct_depths) < 2:
        return None
    return sorted(ct_depths.values())[1]


def _get_branch_cnv_depths(phylo_pat):
    """Return dict arm_label → (min_depth, dominant_ct) across all samples."""
    result = {}
    for _, row in phylo_pat.iterrows():
        lbls = row.get("branch_labels", "")
        if not isinstance(lbls, str) or not lbls:
            continue
        for lbl in lbls.split("; "):
            lbl = lbl.strip()
            if not lbl:
                continue
            d  = int(row["depth"])
            ct = row["cell_type"]
            if lbl not in result or d < result[lbl][0]:
                result[lbl] = (d, ct)
    return result


def interpret(patient, pid_entry, samples, tp_df, comp_dict,
              cnv_summary_pat, cnv_detail_pat,
              ct_enrich_pat=None, phylo_pat=None, burden_pat=None):
    """Return list of (heading, body) interpretation sections."""
    sections = []

    # ── 1. Cell type composition ───────────────────────────────────────────────
    all_tumor  = set()
    has_normal = False
    for comp in comp_dict.values():
        if not comp:
            continue
        for ct, n in comp.items():
            if ct in TUMOR_TYPES:
                all_tumor.add(ct)
            else:
                has_normal = True

    normal_note = " Admixed normal/stromal cells are present." if has_normal else ""
    if not all_tumor:
        body = "No tumor cells detected in the available data."
    elif len(all_tumor) == 1:
        body = (f"The tumor consists of a single cell type "
                f"({next(iter(all_tumor))}), suggesting a relatively "
                f"homogeneous neoplastic population.{normal_note}")
    else:
        body = (f"The tumor harbours {len(all_tumor)} distinct cell types: "
                f"{', '.join(sorted(all_tumor))}. Co-existence of multiple "
                f"cell states is consistent with active neuroendocrine lineage "
                f"plasticity or ongoing histological transformation.{normal_note}")
    sections.append(("Cell type composition", body))

    # ── 2. Clonality & ancestral mutations (actual CNV data) ──────────────────
    luad     = "LUAD"   in all_tumor
    lusc     = "LUSC"   in all_tumor
    any_sclc = any(t.startswith("SCLC") for t in all_tumor)

    if not all_tumor:
        body = "Clonality cannot be assessed without confirmed tumor cells."
    else:
        cnv_parts = []

        if not cnv_summary_pat.empty:
            n_nonneu = (cnv_summary_pat["classification"] != "absent").sum()
            n_trunk  = (cnv_summary_pat["classification"] == "trunk").sum()
            n_major  = (cnv_summary_pat["classification"] == "major_subclonal").sum()
            n_samples = int(cnv_summary_pat["n_samples_total"].max())

            if n_trunk > 0:
                trunk_lbls = cnv_summary_pat[
                    cnv_summary_pat["classification"] == "trunk"
                ]["seg_label"].drop_duplicates().tolist()
                trunk_str = ", ".join(trunk_lbls[:10])
                if len(trunk_lbls) > 10:
                    trunk_str += f" … (+{len(trunk_lbls)-10} more)"
                cnv_parts.append(
                    f"Trunk CNVs shared by ≥90% of tumor clones across all "
                    f"{n_samples} sample(s): {trunk_str}."
                )
                # Call out tumour-suppressor losses
                ts_lost = [s for s in trunk_lbls
                           if arm_key(s) in SUPPRESSORS and "loss" in s.lower()]
                ts_loh  = [s for s in trunk_lbls
                           if arm_key(s) in SUPPRESSORS and "loh" in s.lower()]
                if ts_lost or ts_loh:
                    ts_all = list(dict.fromkeys(ts_lost + ts_loh))
                    cnv_parts.append(
                        f"Ancestral tumour-suppressor events: {', '.join(ts_all)}."
                    )
                # Call out oncogene gains
                og_amp = [s for s in trunk_lbls
                          if arm_key(s) in ONCOGENES and ("gain" in s or "amp" in s)]
                if og_amp:
                    cnv_parts.append(
                        f"Ancestral oncogene gains: {', '.join(og_amp)}."
                    )

            elif n_major > 0:
                major_lbls = cnv_summary_pat[
                    cnv_summary_pat["classification"] == "major_subclonal"
                ]["seg_label"].drop_duplicates().tolist()[:8]
                cnv_parts.append(
                    f"No single CNV is shared by >90% of clones across all samples. "
                    f"Most prevalent events (>50% in all samples): "
                    f"{', '.join(major_lbls)}."
                )

            else:
                # Highly polyclonal — report by mean clone_frac
                if not cnv_detail_pat.empty:
                    top_segs = (cnv_detail_pat.groupby("seg_label")["clone_frac"]
                                .mean().sort_values(ascending=False).head(8).index.tolist())
                    cnv_parts.append(
                        f"Highly polyclonal tumor: no CNV event is consistently "
                        f"present in >50% of clones across all {n_samples} sample(s). "
                        f"Most prevalent events (by mean clone fraction): "
                        f"{', '.join(top_segs)}."
                    )
                else:
                    cnv_parts.append(
                        "Highly polyclonal tumor: no consistent trunk CNV identified."
                    )

            cnv_parts.append(
                f"Total non-neutral CNV segments: {n_nonneu} (trunk: {n_trunk}, "
                f"major subclonal: {n_major})."
            )
        else:
            cnv_parts.append("CNV mutation data not available for this patient.")

        # Transformation context
        if (luad or lusc) and any_sclc:
            src = "LUAD" if luad else "LUSC"
            cnv_parts.append(
                f"{src} and SCLC co-exist — consistent with treatment-induced "
                f"lineage transformation. Trunk CNVs (if any) predate the transformation "
                f"event; SCLC-branch CNVs represent post-transformation diversification."
            )

        body = "  ".join(cnv_parts).strip()
    sections.append(("Tumor clonality & ancestral CNV mutations", body))

    # ── 3. Cell type ordering & temporal dynamics ──────────────────────────────
    n_tp = int(tp_df["date_rank"].max()) if len(tp_df) else 1

    if n_tp <= 1:
        body = ("All samples collected at a single timepoint; temporal evolution "
                "cannot be assessed from longitudinal data. The phylogeny instead "
                "reflects spatial heterogeneity across concurrent biopsy sites.")
    else:
        tp_type_map = {}
        for rank in sorted(tp_df["date_rank"].unique()):
            sids_at = tp_df[tp_df["date_rank"] == rank]["sample_id"].tolist()
            date    = tp_df[tp_df["date_rank"] == rank]["collection_date"].iloc[0]
            types   = set()
            for s in sids_at:
                types |= {ct for ct in (comp_dict.get(s) or {}) if ct in TUMOR_TYPES}
            tp_type_map[int(rank)] = {"date": str(date)[:10], "samples": sids_at,
                                      "types": types}

        tp_lines = []
        for rank, info in sorted(tp_type_map.items()):
            tp_lines.append(
                f"TP{rank} ({info['date']}, {'/'.join(info['samples'])}): "
                + (", ".join(sorted(info["types"])) if info["types"] else "none")
            )

        first_types = tp_type_map[1]["types"]
        last_types  = tp_type_map[n_tp]["types"]
        new_types   = last_types - first_types
        lost_types  = first_types - last_types

        summary = "  •  " + "\n  •  ".join(tp_lines) + "\n"
        if new_types:
            summary += (f"New cell type(s) at later timepoints: "
                        f"{', '.join(sorted(new_types))} — suggesting "
                        f"treatment-driven selection or emerging plasticity. ")
        if lost_types:
            summary += (f"Cell type(s) absent at last collection: "
                        f"{', '.join(sorted(lost_types))} — consistent with "
                        f"clonal extinction or sampling underrepresentation. ")
        if not new_types and not lost_types:
            summary += ("Cell type composition stable across timepoints, suggesting "
                        "maintained tumour cell state despite treatment.")

        if (luad or lusc) and any_sclc:
            src = "LUAD" if luad else "LUSC"
            if src in tp_type_map[1]["types"] and any(
                    t.startswith("SCLC") for t in tp_type_map[n_tp]["types"]):
                summary += (f" Temporal ordering confirms transformation "
                            f"directionality: {src} at earliest timepoint with "
                            f"SCLC enrichment at later timepoints.")

        body = summary.strip()
    sections.append(("Cell type ordering & temporal dynamics", body))

    # ── 4. Cell-type-specific CNV events in phylogenetic context ─────────────
    has_enrich  = ct_enrich_pat is not None and not ct_enrich_pat.empty
    has_phylo   = phylo_pat is not None and not phylo_pat.empty
    if (has_enrich or has_phylo) and len(all_tumor) >= 2:
        present_ct = [ct for ct in TUMOR_TYPES
                      if (has_enrich and ct in ct_enrich_pat["cell_type"].values) or
                         (has_phylo  and ct in phylo_pat["cell_type"].values)]
        ct_parts = []

        # ── Phylogenetic ordering of cell types ────────────────────────────────
        ordered = []   # fallback so later loops are safe
        if has_phylo:
            ct_first_d = _get_celltype_first_depths(phylo_pat)
            div_depth  = _get_divergence_depth(phylo_pat)
            branch_cnv = _get_branch_cnv_depths(phylo_pat)

            if ct_first_d:
                ordered = sorted(ct_first_d, key=lambda c: ct_first_d[c])
                order_str = "  →  ".join(
                    f"{ct} (d={ct_first_d[ct]:.1f})" for ct in ordered)
                ct_parts.append(
                    f"Phylogenetic cell type ordering (abundance-weighted mean "
                    f"clone depth): {order_str}."
                )

            if div_depth is not None:
                # Count pre- vs post-divergence branch CNVs
                pre_d  = [lbl for lbl, (d, _) in branch_cnv.items() if d < div_depth]
                post_d = [lbl for lbl, (d, _) in branch_cnv.items() if d >= div_depth]
                ct_parts.append(
                    f"Estimated lineage divergence at mean depth {div_depth:.1f}. "
                    f"{len(pre_d)} CNV event(s) acquired before divergence "
                    f"(ancestral, shared across cell types); "
                    f"{len(post_d)} after divergence (potentially cell-type-specific)."
                )

                # Which CNVs appear specifically in post-divergence SCLC-enriched clones?
                if has_enrich:
                    for ct in [c for c in ordered if c not in ("LUAD","LUSC")
                                and ct_first_d.get(c, 0) >= div_depth]:
                        post_ct_cnvs = [
                            lbl for lbl, (d, dom) in branch_cnv.items()
                            if d >= div_depth and dom == ct
                        ]
                        if post_ct_cnvs:
                            # Verify enrichment from enrich_pat
                            enriched = ct_enrich_pat[
                                (ct_enrich_pat["cell_type"] == ct) &
                                (ct_enrich_pat["arm_label"].isin(post_ct_cnvs)) &
                                (ct_enrich_pat["p_range"] > 0.30) &
                                (ct_enrich_pat["is_max_ct"])
                            ]["arm_label"].drop_duplicates().tolist()[:5]
                            if enriched:
                                ct_parts.append(
                                    f"{ct} acquires post-divergence CNVs: "
                                    f"{', '.join(enriched)}."
                                )
            else:
                ct_parts.append(
                    "No clear divergence point detected — all cell types appear at "
                    "similar phylogenetic depths, consistent with convergent evolution "
                    "or insufficient clone resolution."
                )

        # ── Overall enrichment summary ─────────────────────────────────────────
        if has_enrich:
            specific = ct_enrich_pat[
                (ct_enrich_pat["p_range"] > 0.50) & ct_enrich_pat["is_max_ct"]
            ]
            for ct in present_ct:
                ev = specific[specific["cell_type"] == ct].sort_values(
                    "mean_p_cnv", ascending=False)
                if len(ev) == 0:
                    continue
                lbls = ev["arm_label"].drop_duplicates().tolist()[:5]
                extra = f" (+{len(ev.drop_duplicates('arm_label'))-5} more)" \
                        if len(ev.drop_duplicates("arm_label")) > 5 else ""
                ct_parts.append(
                    f"{ct} enriched events (P(CNV) range >0.5): "
                    f"{', '.join(lbls)}{extra}."
                )

            low_range = ct_enrich_pat[ct_enrich_pat["p_range"] < 0.15].groupby("seg_id").first()
            n_shared  = (low_range["mean_p_cnv"] > 0.7).sum()
            if n_shared > 0:
                ct_parts.append(
                    f"{n_shared} event(s) uniformly carried by all tumor cell types "
                    f"(ancestral, predating lineage divergence)."
                )

        if ct_parts:
            body = "  ".join(ct_parts)
        else:
            body = ("No strongly cell-type-specific CNV events detected "
                    "(all events shared uniformly, or insufficient clone resolution).")
        sections.append(("Cell-type-specific CNV events & phylogenetic ordering", body))

    # ── 5. CNV burden ordering → starting cell type ───────────────────────────
    if burden_pat is not None and not burden_pat.empty:
        stats = _burden_tumor_stats(burden_pat)
        tumor_stats = stats[stats["cell_type"].isin(all_tumor)]
        if not tumor_stats.empty and len(tumor_stats) >= 2:
            lowest_ct   = tumor_stats.iloc[0]["cell_type"]
            highest_ct  = tumor_stats.iloc[-1]["cell_type"]
            lowest_med  = tumor_stats.iloc[0]["median"]
            highest_med = tumor_stats.iloc[-1]["median"]
            fold = highest_med / lowest_med if lowest_med > 0 else float("inf")

            order_str = "  →  ".join(
                f"{r['cell_type']} ({r['median']:.3f})"
                for _, r in tumor_stats.iterrows()
            )
            burden_parts = [f"CNV burden order (ascending): {order_str}."]

            if fold > 1.5:
                burden_parts.append(
                    f"{lowest_ct} has the lowest median genomic burden (FGA "
                    f"{lowest_med:.3f}), consistent with it being the ancestral / "
                    f"starting cell state.  {highest_ct} carries {fold:.1f}× more "
                    f"altered genome, suggesting later or more unstable emergence."
                )
            else:
                burden_parts.append(
                    "CNV burden is similar across tumor cell types; the metric alone "
                    "does not distinguish an ancestral cell state."
                )
            sections.append(("CNV burden & putative starting cell type",
                              "  ".join(burden_parts)))

    return sections


def _draw_tp_legend(ax, tp_df, comp):
    """Draw a compact timepoint legend (color swatch | TP label | date | samples | cell types)."""
    # Mirror R's interpolation: round(seq(1, 4, length.out=n_tp))
    # so colors always anchor blue=earliest, red=latest, matching the tree plot.
    n_tp = int(tp_df["date_rank"].max())
    idxs = [round(x) - 1 for x in np.linspace(1, len(TP_COLORS), n_tp)]
    tp_palette = {r + 1: TP_COLORS[min(i, len(TP_COLORS) - 1)]
                  for r, i in enumerate(idxs)}

    ax.set_axis_off()
    y = 0.98
    ax.text(0, y, "Timepoints", fontsize=7.5, fontweight="bold",
            transform=ax.transAxes, va="top", color="#111111")
    y -= 0.09

    for rank in sorted(tp_df["date_rank"].unique()):
        rank = int(rank)
        sids = tp_df[tp_df["date_rank"] == rank]["sample_id"].tolist()
        date_val = tp_df[tp_df["date_rank"] == rank]["collection_date"].iloc[0]
        dstr = str(date_val)[:10] if pd.notna(date_val) else "?"
        types = set()
        for s in sids:
            types |= {ct for ct in (comp.get(s) or {}) if ct in TUMOR_TYPES}
        color = tp_palette.get(rank, "#AAAAAA")

        # Color swatch
        ax.add_patch(plt.Rectangle(
            (0.0, y - 0.028), 0.07, 0.055,
            color=color, transform=ax.transAxes, clip_on=False))

        # TP label + date
        ax.text(0.12, y, f"TP{rank}  ({dstr})",
                fontsize=6.5, fontweight="bold",
                transform=ax.transAxes, va="top", color="#222222")
        y -= 0.048
        # Sample IDs
        ax.text(0.12, y, ", ".join(sids),
                fontsize=5.8, transform=ax.transAxes, va="top", color="#444444")
        y -= 0.040
        # Cell types
        ct_str = ", ".join(sorted(types)) if types else "—"
        ax.text(0.12, y, ct_str,
                fontsize=5.5, transform=ax.transAxes, va="top",
                color="#666666", style="italic")
        y -= 0.065
        if y < 0.02:
            break


# ── Per-patient report ─────────────────────────────────────────────────────────

def make_report(pid_entry, samples, tp_df, clinical_df,
                cnv_summary_all, cnv_detail_all, ct_enrich_all,
                phylo_all=None, cnv_burden_all=None):
    patient = pid_entry.split("_")[0]
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_pdf = os.path.join(REPORT_DIR, f"{patient}_report.pdf")

    # Subset CNV data
    cnv_summary_pat = cnv_summary_all[cnv_summary_all["patient_id"] == patient].copy() \
        if not cnv_summary_all.empty else pd.DataFrame()
    cnv_detail_pat  = cnv_detail_all[cnv_detail_all["patient_id"] == patient].copy() \
        if not cnv_detail_all.empty else pd.DataFrame()
    ct_enrich_pat   = ct_enrich_all[ct_enrich_all["patient_id"] == patient].copy() \
        if not ct_enrich_all.empty else pd.DataFrame()
    phylo_pat       = (phylo_all[phylo_all["patient_id"] == patient].copy()
                       if phylo_all is not None and not phylo_all.empty
                       else pd.DataFrame())
    burden_pat      = (cnv_burden_all[cnv_burden_all["patient_id"] == patient].copy()
                       if cnv_burden_all is not None and not cnv_burden_all.empty
                       else pd.DataFrame())

    # Load compositions
    comp = {s: load_composition(patient, s) for s in samples}

    n_tp         = int(tp_df["date_rank"].max()) if len(tp_df) else 1
    multi_sample = len(samples) > 1
    multi_tp     = n_tp >= 2

    # Samples in timepoint order (used throughout)
    sorted_sids = tp_df.sort_values("date_rank")["sample_id"].tolist()
    sorted_sids += [s for s in samples if s not in sorted_sids]
    tp_rank = tp_df.set_index("sample_id")["date_rank"].to_dict()

    # ── Figure height: top strip + sample trees + interpretation ─────────────
    n_samp      = len(sorted_sids)
    n_samp_cols = min(n_samp, 5)
    n_samp_rows = math.ceil(n_samp / n_samp_cols) if multi_sample else 0

    h_top  = 4.0                         # clinical + main trees
    h_samp = n_samp_rows * 2.6           # per-sample tree thumbnails
    h_fig  = max(h_top + h_samp + 0.6, 7)

    snv_path = get_snv_png(patient)

    with PdfPages(out_pdf) as pdf:

        # ── Single combined page ───────────────────────────────────────────────
        fig_w = 14 + (2.8 if snv_path else 0)
        fig = plt.figure(figsize=(fig_w, h_fig))
        fig.suptitle(patient, fontsize=13, fontweight="bold", y=0.99)

        # Outer grid rows: top-strip | [sample row]
        n_outer   = 1 + (1 if multi_sample else 0)
        outer_h   = [h_top] + ([h_samp] if multi_sample else [])
        outer_gs  = gridspec.GridSpec(
            n_outer, 1, figure=fig,
            height_ratios=outer_h,
            hspace=0.28,
            left=0.03, right=0.97, top=0.95, bottom=0.03
        )

        # ── Top strip: clinical | comp(small) | patient_ct | [tp tree] |
        #   [tp legend] | [SNV overlap] ────────────────────────────────────────
        top_widths = [1.5, 0.75, 2.2]          # clinical, comp, patient_ct
        if multi_tp:
            top_widths.append(2.0)              # timepoint tree
            top_widths.append(1.0)              # timepoint legend
        if snv_path:
            top_widths.append(2.8)              # SNV UpSet plot

        top_gs = gridspec.GridSpecFromSubplotSpec(
            1, len(top_widths), subplot_spec=outer_gs[0],
            wspace=0.22, width_ratios=top_widths
        )

        ax_clin    = fig.add_subplot(top_gs[0, 0])
        ax_comp    = fig.add_subplot(top_gs[0, 1])
        ax_tree_ct = fig.add_subplot(top_gs[0, 2])

        draw_clinical(ax_clin, patient, pid_entry, clinical_df, tp_df)
        draw_composition(ax_comp, comp, tp_df)
        embed_png(ax_tree_ct, get_png(patient, kind="celltype"),
                  title=f"{patient} — cell type")

        col = 3
        if multi_tp:
            ax_tree_tp = fig.add_subplot(top_gs[0, col])
            embed_png(ax_tree_tp, get_png(patient, kind="timepoint"),
                      title=f"{patient} — timepoint")
            col += 1
            ax_tp_leg = fig.add_subplot(top_gs[0, col])
            _draw_tp_legend(ax_tp_leg, tp_df, comp)
            col += 1
        if snv_path:
            ax_snv = fig.add_subplot(top_gs[0, col])
            embed_png(ax_snv, snv_path, title=f"{patient} — SNV overlap")

        # Cell-type legend anchored to the right of the comp panel
        handles = [plt.Rectangle((0, 0), 1, 1,
                                  color=CELL_TYPE_COLORS.get(ct, "#CCCCCC"), label=ct)
                   for ct in CELL_TYPE_COLORS]
        ax_comp.legend(handles=handles, loc="upper left",
                       bbox_to_anchor=(1.01, 1), fontsize=5.5,
                       frameon=True, facecolor="white", edgecolor="#CCCCCC",
                       title="Cell type", title_fontsize=6.0,
                       handlelength=1.2, handleheight=1.0,
                       handletextpad=0.6, borderpad=0.5,
                       labelspacing=0.4)

        # ── Sample trees row (multi-sample only) ──────────────────────────────
        if multi_sample:
            samp_gs = gridspec.GridSpecFromSubplotSpec(
                n_samp_rows, n_samp_cols,
                subplot_spec=outer_gs[1],
                hspace=0.10, wspace=0.04
            )
            for i, sid in enumerate(sorted_sids):
                r, c = divmod(i, n_samp_cols)
                ax_s  = fig.add_subplot(samp_gs[r, c])
                rank  = tp_rank.get(sid, "?")
                embed_png(ax_s, get_png(patient, sample_id=sid, kind="celltype"),
                          title=f"{sid}  (TP {rank})")

        pdf.savefig(fig, bbox_inches="tight", dpi=200)
        plt.close(fig)

        # ── Page 2: CNV burden per tumor cell type (3 metrics) ────────────────
        fig2 = plt.figure(figsize=(16, 9))
        fig2.suptitle(f"{patient} — CNV burden per tumor cell type",
                      fontsize=11, fontweight="bold", y=0.99)
        gs2 = gridspec.GridSpec(
            3, 2, figure=fig2,
            width_ratios=[2.2, 1],
            hspace=0.55, wspace=0.30,
            left=0.12, right=0.95, top=0.93, bottom=0.06,
        )
        metrics = list(BURDEN_METRICS.keys())
        metric_titles = [
            "Probabilistic FGA  (p_cnv-weighted bp)",
            "Binary FGA  (p_cnv ≥ 0.5)",
            "Mean segment p(CNV)",
        ]
        for mi, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax_m = fig2.add_subplot(gs2[mi, 0])
            ax_m.set_title(title, fontsize=8, pad=3)
            draw_cnv_burden_plot(ax_m, burden_pat, metric=metric,
                                 show_yticks=(mi == 0))
        ax_txt = fig2.add_subplot(gs2[:, 1])
        draw_cnv_burden_text(ax_txt, burden_pat, phylo_pat=phylo_pat)
        pdf.savefig(fig2, bbox_inches="tight", dpi=200)
        plt.close(fig2)

    print(f"  Saved → {out_pdf}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="*",
                        help="Subset of patient IDs to process (default: all)")
    args = parser.parse_args()

    clinical_df          = load_clinical()
    tp_all               = load_timepoints()
    cnv_summary_all, cnv_detail_all = load_cnv_data()
    ct_enrich_all        = load_ct_enrichment()
    phylo_all            = load_phylo_celltype()
    cnv_burden_all       = load_cnv_burden()

    with open(PATIENT_LIST) as f:
        pid_entries = [l.strip() for l in f if l.strip()]
    with open(SAMPLE_LIST) as f:
        all_samples = [l.strip() for l in f if l.strip()]

    subset = set(args.patients) if args.patients else None

    for pid_entry in pid_entries:
        patient = pid_entry.split("_")[0]
        if subset and patient not in subset and pid_entry not in subset:
            continue

        samples = [s for s in all_samples if s.split("_")[0] == patient]
        tp_df   = tp_all[tp_all["patient_id"] == patient].copy()
        if len(tp_df) == 0 and pid_entry != patient:
            tp_df = tp_all[tp_all["sample_id"] == pid_entry].copy()

        print(f"[{patient}]  {len(samples)} sample(s), "
              f"{int(tp_df['date_rank'].max()) if len(tp_df) else 1} timepoint(s)")
        make_report(pid_entry, samples, tp_df, clinical_df,
                    cnv_summary_all, cnv_detail_all, ct_enrich_all,
                    phylo_all=phylo_all, cnv_burden_all=cnv_burden_all)

    print(f"\nAll reports → {REPORT_DIR}/")


if __name__ == "__main__":
    main()
