#!/usr/bin/env python3
"""
5_cnv_burden_violin.py

Split box plots: NE (SCLC-A + SCLC-N) vs NonNE (LUAD and/or NonNE SCLC /
EMT / FOXI1+ NonNE SCLC-N) for four metrics:
  1. cnv_burden_gene_bin  — fraction of genes altered (binarized p_cnv ≥ 0.5)
  2. CNV_clock            — CNV clock score
  3. infercnv_score       — inferCNV score
  4. distance_to_root     — phylogenetic branch length from root (Numbat treeML)

Key design decisions:
  • cell_type_final2 from combined adata is used (authoritative, overrides Numbat labels)
  • EMT and FOXI1+ NonNE SCLC-N are grouped with NonNE SCLC as the purple group
  • When a patient has BOTH LUAD (yellow) and NonNE/EMT (purple):
      – both halves are drawn; TWO separate MWU tests are run and annotated
  • Non-significant boxes (FDR ≥ 0.05) are greyed out with "ns" label
  • Sorting: sig NE>NonNE | ns NE>NonNE | ns NonNE>NE | sig NonNE>NE
"""

import glob, os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mc
import anndata as ad

# ── Paths ──────────────────────────────────────────────────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--data-tsv", default=None)
_parser.add_argument("--out-suffix", default="")
_args, _ = _parser.parse_known_args()

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TSV  = os.path.join(SCRIPT_DIR, "cnv_burden_metacell_joined.tsv")
DATA_TSV      = _args.data_tsv or _DEFAULT_TSV
OUT_SUFFIX    = _args.out_suffix   # e.g. "_metacell"
COMBINED_H5AD = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/"
                 "fresh.SCLC_NSCLC.062124/combined/adata.SCLC_NSCLC.051226.h5ad")
LEAF_DIST_DIR = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/"
                 "Numbat_All/{patient}/numbat_output_FACETS_HISENS/"
                 "{patient}_leaf_distances.csv")
LEAF_DIST_DIR_MC = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/"
                    "Numbat_All/{patient}/numbat_output_FACETS_HISENS_metacell/"
                    "{patient}_leaf_distances.csv")
OUT_DIR       = os.path.join(SCRIPT_DIR, "figures")

# ── Cell type groups ───────────────────────────────────────────────────────────
NE_TYPES    = {"SCLC-A", "SCLC-N", "SCLC-N (FOXI1+)"}
# Individual NonNE subtypes — each drawn separately on the left half
NONNE_SUBTYPES = [
    ("LUAD",       "#ffd700"),   # gold
    ("LUSC",       "#FFA500"),   # orange
    ("NonNE SCLC", "#4B0082"),   # purple (includes EMT)
]
NONNE_TYPES = {ct for ct, _ in NONNE_SUBTYPES}   # for combined tests
COLORS      = {"SCLC": "#d62728", "NSCLC": "#ffd700", "NonNE": "#4B0082"}

# SCLC subtype colors and display order for the subtype plot
NE_SUBTYPES = [
    ("SCLC-A",         "#d62728"),   # red
    ("SCLC-N",         "#17becf"),   # cyan
    ("SCLC-N (FOXI1+)","#2ca02c"),   # tab:green
]
MIN_CELLS   = 5

# Patient-specific cell-type corrections applied after loading
PATIENT_CELL_TYPE_REMAP = {
    "RU1518": {"FOXI1+ NonNE SCLC-N": "SCLC-N (FOXI1+)",
               "FOXI1+ SCLC-N":       "SCLC-N (FOXI1+)"},
}
NS_ALPHA    = 0.20   # alpha for non-significant box fill (unused — kept for reference)


# ── DNA3Bit decode ─────────────────────────────────────────────────────────────
_BIN2STR = {0b100: "A", 0b110: "C", 0b101: "G", 0b011: "T", 0b111: "N"}

def _decode_barcode(i: int) -> str:
    r = ""
    while i > 0:
        r = _BIN2STR[i & 0b111] + r
        i >>= 3
    return r


def build_cell_type_map(h5ad_path: str) -> dict:
    """decoded_cell → cell_type_final2 from authoritative combined adata."""
    print("Loading cell_type_final2 from combined adata …")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    ct_map = {}
    for obs_name, ct in adata.obs["cell_type_final2"].items():
        parts = obs_name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        try:
            bc = _decode_barcode(int(parts[1]))
            ct_map[f"{parts[0]}_{bc}-1"] = ct
        except (ValueError, KeyError):
            continue
    print(f"  {len(ct_map):,} cells mapped")
    return ct_map


# ── Leaf-distance loader ───────────────────────────────────────────────────────
def load_leaf_distances(patients: list) -> pd.DataFrame:
    """
    Load distance_to_root from leaf_distances.csv.
    Tries metacell path first (when DATA_TSV is the metacell file), then SC path.
    Metacell cells use SEACell IDs (no decode needed);
    SC cells use DNA3Bit-encoded integers (decoded to 10x barcode format).
    """
    is_metacell = "metacell" in DATA_TSV
    parts = []
    for pat in patients:
        path = (LEAF_DIST_DIR_MC if is_metacell else LEAF_DIST_DIR).format(patient=pat)
        if not os.path.exists(path):
            # fallback to the other
            alt = (LEAF_DIST_DIR if is_metacell else LEAF_DIST_DIR_MC).format(patient=pat)
            if not os.path.exists(alt):
                continue
            path = alt
        df = pd.read_csv(path)
        df.columns = [c.strip('"') for c in df.columns]

        if is_metacell:
            # metacell leaf names are plain SEACell identifiers — use as-is
            df["decoded_cell"] = df["cell"].str.strip('"')
        else:
            def decode_name(name):
                p = name.strip('"').rsplit("_", 1)
                if len(p) != 2:
                    return None
                try:
                    return f"{p[0]}_{_decode_barcode(int(p[1]))}-1"
                except (ValueError, KeyError):
                    return None
            df["decoded_cell"] = df["cell"].apply(decode_name)

        df = df.dropna(subset=["decoded_cell"])
        parts.append(df[["decoded_cell", "distance_to_root"]])
    if not parts:
        return pd.DataFrame(columns=["decoded_cell", "distance_to_root"])
    return pd.concat(parts, ignore_index=True)


# ── Half-boxplot ───────────────────────────────────────────────────────────────
def half_box(ax, data, x, side, color, width=0.35, alpha=0.75, x_offset=0.0):
    if len(data) < 4:
        return
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo_whisk = max(float(data.min()), q1 - 1.5 * iqr)
    hi_whisk = min(float(data.max()), q3 + 1.5 * iqr)
    sign = -1 if side == "left" else 1

    # x_base: spine origin, shifted away from center by x_offset (for staggering)
    x_base = x + sign * x_offset

    # Filled box: Q1 to Q3, extending sign*width from x_base
    verts = [[x_base, q1], [x_base + sign * width, q1],
             [x_base + sign * width, q3], [x_base, q3]]
    box = plt.Polygon(verts, closed=True,
                      facecolor=color, alpha=alpha,
                      edgecolor=color, lw=0.4, zorder=2)
    ax.add_patch(box)

    # Median line
    r, g, b, _ = mc.to_rgba(color)
    dark = (r * 0.55, g * 0.55, b * 0.55, 1.0)
    ax.plot([x_base, x_base + sign * width], [med, med], color=dark, lw=0.75, zorder=5)

    # Whiskers at x_base, vertical
    ax.plot([x_base, x_base], [lo_whisk, q1], color=color, lw=0.4, zorder=3)
    ax.plot([x_base, x_base], [q3, hi_whisk], color=color, lw=0.4, zorder=3)

    # Whisker caps
    cap_w = width * 0.25
    ax.plot([x_base, x_base + sign * cap_w], [lo_whisk, lo_whisk], color=color, lw=0.4, zorder=3)
    ax.plot([x_base, x_base + sign * cap_w], [hi_whisk, hi_whisk], color=color, lw=0.4, zorder=3)



# ── Per-patient statistics ─────────────────────────────────────────────────────
def compute_records(df, metric):
    """
    For each patient, identify the primary NonNE subtype (most cells) and run
    one MWU test for that subtype vs NE.  BH FDR is applied across one test per
    patient, keeping the multiple-testing burden proportional to patient count.

    Per-subtype p-values for annotation arrows are computed separately (raw,
    unadjusted) so the violin labels still reflect each subtype's direction and
    rough significance, while the sig/ns status that drives sorting and brackets
    uses only the per-patient primary test.
    """
    from collections import defaultdict

    # ── Step 1: collect all subtype arrays per patient (raw p, unadjusted) ──
    pat_data = {}   # patient → {ct: (vals, raw_p, diff)}
    for patient, grp in df.groupby("patient_id"):
        ne = grp[grp["cell_type"].isin(NE_TYPES)][metric].dropna().values
        if len(ne) < MIN_CELLS:
            continue
        subtypes = {}
        for ct, _ in NONNE_SUBTYPES:
            vals = grp[grp["cell_type"] == ct][metric].dropna().values
            if len(vals) >= MIN_CELLS:
                _, p = stats.mannwhitneyu(ne, vals, alternative="two-sided")
                diff = float(np.median(ne) - np.median(vals))
                subtypes[ct] = (vals, p, diff)
        if subtypes:
            pat_data[patient] = (ne, subtypes)

    if not pat_data:
        return []

    # ── Step 2: one primary test per patient → FDR pool ──────────────────────
    primary_rows = []   # (patient, primary_ct, raw_p, diff, ne, subtypes)
    for patient, (ne, subtypes) in pat_data.items():
        primary_ct = max(subtypes, key=lambda c: len(subtypes[c][0]))
        _, raw_p, diff = subtypes[primary_ct]
        primary_rows.append((patient, primary_ct, raw_p, diff, ne, subtypes))

    _, padj, _, _ = multipletests([r[2] for r in primary_rows], method="fdr_bh")

    records = []
    for (patient, primary_ct, raw_p, diff, ne, subtypes), q in zip(primary_rows, padj):
        grp = df[df["patient_id"] == patient]
        nonne_groups = {ct: v for ct, (v, _, _) in subtypes.items()}
        # annotation arrows use raw (unadjusted) per-subtype p-values
        group_tests  = {ct: (p, d) for ct, (_, p, d) in subtypes.items()}

        ne_subtypes = {ct: grp[grp["cell_type"] == ct][metric].dropna().values
                       for ct, _ in NE_SUBTYPES}

        records.append(dict(
            patient      = patient,
            diff         = diff,
            q            = q,
            sig          = q < 0.05,
            ne           = ne,
            ne_subtypes  = ne_subtypes,
            nonne_groups = nonne_groups,
            group_tests  = group_tests,
            luad         = nonne_groups.get("LUAD", np.array([])),
            nonne        = np.concatenate([v for c, v in nonne_groups.items()
                                           if c != "LUAD"]) if any(
                           c != "LUAD" for c in nonne_groups) else np.array([]),
            luad_q       = group_tests.get("LUAD",   (None, None))[0],
            nonne_q      = None,
            luad_diff    = group_tests.get("LUAD",   (None, None))[1],
            nonne_diff   = None,
        ))

    def sort_key(r):
        sig, ne_gt = r["sig"], r["diff"] > 0
        tier = (0 if (sig and ne_gt)
                else 1 if (not sig and ne_gt)
                else 2 if (not sig and not ne_gt)
                else 3)
        return (tier, -r["diff"] if ne_gt else r["diff"])

    records.sort(key=sort_key)
    return records


# ── Inequality annotation helper ──────────────────────────────────────────────
def _annotate_inequality(ax, xi, y, nonne_color, diff, star, fontsize=6):
    """
    Draw:  ■ <* ■   (or  ■ >* ■)  below a split box.
    Left square = nonne_color (NonNE subtype), right square = SCLC red.
    diff > 0 → NE median > NonNE median → NonNE < NE → '<' sign.
    Stars are placed immediately after the sign.
    """
    sign   = "<" if diff > 0 else ">"
    offset = 0.2
    kw = dict(ha="center", va="top", fontsize=fontsize, clip_on=False)
    ax.text(xi - offset, y, "■", color=nonne_color, **kw)
    ax.text(xi,          y, sign, color="#333333", fontweight="bold", **kw)
    ax.text(xi + offset, y, "■", color=COLORS["SCLC"], **kw)
    ax.text(xi + 0.3, y, star, color="#333333", fontweight="bold",
            ha="left", va="top", fontsize=fontsize * 0.85, clip_on=False)


# ── Draw one panel ─────────────────────────────────────────────────────────────
def draw_panel(ax, records, ylabel, show_xticklabels=True):
    n = len(records)
    if n == 0:
        ax.set_axis_off(); return

    for xi, r in enumerate(records):
        # Left: all NonNE subtypes overlaid, largest first (behind)
        nonne_present = [(ct, color, r["nonne_groups"][ct])
                         for ct, color in NONNE_SUBTYPES
                         if ct in r["nonne_groups"] and len(r["nonne_groups"][ct]) >= 4]
        nonne_present.sort(key=lambda x: -len(x[2]))
        for i, (_, color, vals) in enumerate(nonne_present):
            alpha = 0.80 if i == 0 else max(0.40, 0.80 - i * 0.15)
            half_box(ax, vals, xi, "left", color, width=0.35, alpha=alpha,
                     x_offset=i * 0.12)
        # Right: SCLC combined
        half_box(ax, r["ne"], xi, "right", COLORS["SCLC"],
                 width=0.35, alpha=0.80)

    # Two grey vertical lines separating sig / ns / sig groups
    sig_ne_idx    = [i for i, r in enumerate(records) if r["diff"] > 0 and r["sig"]]
    sig_nonne_idx = [i for i, r in enumerate(records) if r["diff"] < 0 and r["sig"]]
    ns_idx        = [i for i, r in enumerate(records) if not r["sig"]]
    # Line between last sig-NE>NonNE and first ns (or sig-NonNE>NE)
    if sig_ne_idx:
        right_of_sig_ne = max(sig_ne_idx)
        next_idx = right_of_sig_ne + 1
        if next_idx < len(records):
            ax.axvline(right_of_sig_ne + 0.5,
                       color="#aaaaaa", lw=0.6, ls="-", zorder=0)
    # Line between last ns (or sig-NE>NonNE) and first sig-NonNE>NE
    if sig_nonne_idx:
        left_of_sig_nonne = min(sig_nonne_idx)
        prev_idx = left_of_sig_nonne - 1
        if prev_idx >= 0:
            ax.axvline(left_of_sig_nonne - 0.5,
                       color="#aaaaaa", lw=0.6, ls="-", zorder=0)

    ax.relim(); ax.autoscale_view()
    ylo, yhi = ax.get_ylim()
    pad  = (yhi - ylo) * 0.08
    y_br = yhi + pad * 0.3
    y_tx = yhi + pad * 1.1

    # Significance brackets
    sig_ne_gt    = [i for i, r in enumerate(records) if r["diff"] > 0 and r["sig"]]
    sig_nonne_gt = [i for i, r in enumerate(records) if r["diff"] < 0 and r["sig"]]
    for sig_idxs, label in [
        (sig_ne_gt,    "NE transformation\n(FDR < 0.05)"),
        (sig_nonne_gt, "NonNE reversion\n(FDR < 0.05)"),
    ]:
        if not sig_idxs:
            continue
        x0, x1 = min(sig_idxs) - 0.45, max(sig_idxs) + 0.45
        ax.annotate("", xy=(x0, y_br), xytext=(x1, y_br),
                    arrowprops=dict(arrowstyle="<->", color="black",
                                    lw=0.8, mutation_scale=12))
        ax.text((x0 + x1) / 2, y_tx, label,
                ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    # Per-box annotations: colored-square inequality (■ <* ■)
    for xi, r in enumerate(records):
        group_tests = r.get("group_tests", {})
        n_groups = len(group_tests)

        if n_groups > 1:
            dy_start, dy_step = 1.2, 2.0
            for gi, (ct, ct_color) in enumerate(NONNE_SUBTYPES):
                if ct not in group_tests:
                    continue
                q, diff = group_tests[ct]
                star = ("***" if q < 0.001 else "**" if q < 0.01
                        else "*" if q < 0.05 else "ns")
                _annotate_inequality(ax, xi, ylo - pad * (dy_start + gi * dy_step),
                                     ct_color, diff, star)
        elif n_groups == 1:
            ct = next(iter(group_tests))
            q, diff = group_tests[ct]
            star = ("***" if q < 0.001 else "**" if q < 0.01
                    else "*" if q < 0.05 else "ns")
            ct_color = dict(NONNE_SUBTYPES).get(ct, "#333333")
            _annotate_inequality(ax, xi, ylo - pad * 1.2, ct_color, diff, star)
        else:
            # No NonNE group present: fall back to star above
            star = ("***" if r["q"] < 0.001 else "**" if r["q"] < 0.01
                    else "*" if r["q"] < 0.05 else "ns")
            col = "#333333" if r["sig"] else "#999999"
            ax.text(xi, yhi + pad * 0.05, star,
                    ha="center", va="bottom", fontsize=8,
                    color=col, fontweight="bold",
                    style="normal" if r["sig"] else "italic")

    ax.set_xticks(range(n))
    if show_xticklabels:
        ax.set_xticklabels([r["patient"] for r in records],
                           rotation=45, ha="right", fontsize=6)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.7, n - 0.3)
    max_groups = max((len(r.get("group_tests", {})) for r in records), default=1)
    bottom_pad = 1.2 + max_groups * 2.0 + 0.4
    ax.set_ylim(ylo - pad * bottom_pad, y_tx + pad * 0.5)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=5.5)


def draw_panel_subtypes(ax, records, ylabel, show_xticklabels=True):
    """
    Same layout as draw_panel but the right side shows individual SCLC
    subtype half-boxes (SCLC-A / SCLC-N / SCLC-N (FOXI1+)) overlaid.
    Left side NonNE subtypes are also individually overlaid.
    """
    n = len(records)
    if n == 0:
        ax.set_axis_off(); return

    for xi, r in enumerate(records):
        # Left: all NonNE subtypes overlaid (same as draw_panel)
        nonne_present = [(ct, color, r["nonne_groups"][ct])
                         for ct, color in NONNE_SUBTYPES
                         if ct in r["nonne_groups"] and len(r["nonne_groups"][ct]) >= 4]
        nonne_present.sort(key=lambda x: -len(x[2]))
        for i, (_, color, vals) in enumerate(nonne_present):
            alpha = 0.80 if i == 0 else max(0.40, 0.80 - i * 0.15)
            half_box(ax, vals, xi, "left", color, width=0.35, alpha=alpha,
                     x_offset=i * 0.24)
        # Right: SCLC subtypes overlaid
        present = [(ct, color, r["ne_subtypes"].get(ct, np.array([])))
                   for ct, color in NE_SUBTYPES
                   if len(r["ne_subtypes"].get(ct, [])) >= 4]
        present.sort(key=lambda x: -len(x[2]))
        for i, (_, color, vals) in enumerate(present):
            half_box(ax, vals, xi, "right", color,
                     width=0.35, alpha=0.80 if i == 0 else 0.55)

    sig_ne_idx    = [i for i, r in enumerate(records) if r["diff"] > 0 and r["sig"]]
    sig_nonne_idx = [i for i, r in enumerate(records) if r["diff"] < 0 and r["sig"]]
    if sig_ne_idx and max(sig_ne_idx) + 1 < n:
        ax.axvline(max(sig_ne_idx) + 0.5, color="#aaaaaa", lw=0.6, ls="-", zorder=0)
    if sig_nonne_idx and min(sig_nonne_idx) > 0:
        ax.axvline(min(sig_nonne_idx) - 0.5, color="#aaaaaa", lw=0.6, ls="-", zorder=0)

    ax.relim(); ax.autoscale_view()
    ylo, yhi = ax.get_ylim()
    pad  = (yhi - ylo) * 0.08
    y_br = yhi + pad * 0.3
    y_tx = yhi + pad * 1.1

    for sig_idxs, label in [
        (sig_ne_idx,    "NonNE < NE\n(FDR < 0.05)"),
        (sig_nonne_idx, "NonNE > NE\n(FDR < 0.05)"),
    ]:
        if not sig_idxs:
            continue
        x0, x1 = min(sig_idxs) - 0.45, max(sig_idxs) + 0.45
        ax.annotate("", xy=(x0, y_br), xytext=(x1, y_br),
                    arrowprops=dict(arrowstyle="<->", color="black",
                                    lw=0.8, mutation_scale=12))
        ax.text((x0 + x1) / 2, y_tx, label,
                ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    # Per-box annotations: colored-square inequality (■ <* ■)
    for xi, r in enumerate(records):
        group_tests = r.get("group_tests", {})
        n_groups = len(group_tests)
        if n_groups > 1:
            dy_start, dy_step = 1.2, 2.0
            for gi, (ct, ct_color) in enumerate(NONNE_SUBTYPES):
                if ct not in group_tests:
                    continue
                q, diff = group_tests[ct]
                star = ("***" if q < 0.001 else "**" if q < 0.01
                        else "*" if q < 0.05 else "ns")
                _annotate_inequality(ax, xi, ylo - pad * (dy_start + gi * dy_step),
                                     ct_color, diff, star)
        elif n_groups == 1:
            ct = next(iter(group_tests))
            q, diff = group_tests[ct]
            star = ("***" if q < 0.001 else "**" if q < 0.01
                    else "*" if q < 0.05 else "ns")
            ct_color = dict(NONNE_SUBTYPES).get(ct, "#333333")
            _annotate_inequality(ax, xi, ylo - pad * 1.2, ct_color, diff, star)
        else:
            star = ("***" if r["q"] < 0.001 else "**" if r["q"] < 0.01
                    else "*" if r["q"] < 0.05 else "ns")
            col = "#333333" if r["sig"] else "#999999"
            ax.text(xi, yhi + pad * 0.05, star,
                    ha="center", va="bottom", fontsize=8,
                    color=col, fontweight="bold",
                    style="normal" if r["sig"] else "italic")

    ax.set_xticks(range(n))
    if show_xticklabels:
        ax.set_xticklabels([r["patient"] for r in records],
                           rotation=45, ha="right", fontsize=6)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.7, n - 0.3)
    max_g = max((len(r.get("group_tests", {})) for r in records), default=1)
    ax.set_ylim(ylo - pad * (1.2 + max_g * 2.0 + 0.4), y_tx + pad * 0.5)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=5.5)


def _legend_handles():
    """Shared legend: all NonNE subtypes + SCLC subtypes."""
    return (
        [mpatches.Patch(color=c, label=ct) for ct, c in NONNE_SUBTYPES]
        + [mpatches.Patch(color=c, label=ct) for ct, c in NE_SUBTYPES]
    )


def _add_split_cartoon(fig, legend):
    """
    Draw a small schematic split half-box directly above `legend`, matching its
    width and left edge exactly.  A light grey box is drawn around the cartoon.
    Must be called after the legend is added so its rendered bbox is available.
    """
    fig.canvas.draw()                           # flush so legend bbox is computed
    renderer = fig.canvas.get_renderer()
    lb = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())

    gap    = 0.006                              # figure-coord gap between legend and cartoon
    cart_h = max(lb.height * 0.65, 0.13)
    cax = fig.add_axes([lb.x0, lb.y1 + gap, lb.width, cart_h])

    nonne_demo = np.array([0.15, 0.18, 0.22, 0.25, 0.20, 0.28, 0.24, 0.17])
    ne_demo    = np.array([0.50, 0.55, 0.60, 0.68, 0.63, 0.58, 0.72, 0.65])
    half_box(cax, nonne_demo, 0, "left",  "#808080",      width=0.35, alpha=0.75)
    half_box(cax, ne_demo,    0, "right", COLORS["SCLC"], width=0.35, alpha=0.80)

    cax.relim(); cax.autoscale_view()
    ylo, yhi = cax.get_ylim()
    pad_c = (yhi - ylo) * 0.12
    cax.set_xlim(-0.55, 0.55)
    cax.set_ylim(ylo - pad_c * 3.2, yhi + pad_c * 0.5)

    # Labels just below the lower whisker
    cax.text(-0.18, ylo - pad_c * 0.2, "NonNE",
             ha="center", va="top", fontsize=4, color="#555555")
    cax.text( 0.18, ylo - pad_c * 0.2, "NE",
             ha="center", va="top", fontsize=4, color=COLORS["SCLC"])

    # Visible border, no ticks
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#aaaaaa")
    cax.set_xticks([])
    cax.set_yticks([])
    cax.patch.set_facecolor("white")


def add_legend_subtypes(fig, bbox=(0.81, 0.5)):
    leg = fig.legend(handles=_legend_handles(), loc="center left",
                     bbox_to_anchor=bbox, fontsize=5.5, framealpha=0.9)
    _add_split_cartoon(fig, leg)


def add_legend(fig, bbox=(0.81, 0.5)):
    handles = (
        [mpatches.Patch(color=c, label=ct) for ct, c in NONNE_SUBTYPES]
        + [mpatches.Patch(color=COLORS["SCLC"], label="SCLC")]
    )
    leg = fig.legend(handles=handles, loc="center left",
                     bbox_to_anchor=bbox, fontsize=5.5, framealpha=0.9)
    _add_split_cartoon(fig, leg)


# ── Load & prepare data ────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_TSV, sep="\t")
print(f"  {len(df):,} cells, {df['patient_id'].nunique()} patients")

# Replace cell_type with authoritative cell_type_final2
ct_map = build_cell_type_map(COMBINED_H5AD)
if "decoded_cell" in df.columns:
    updated = df["decoded_cell"].map(ct_map)
    df["cell_type"] = updated.fillna(df["cell_type"])
    print(f"  Cell types updated for {updated.notna().sum():,} cells")

# Apply patient-specific cell-type corrections
for pat, remap in PATIENT_CELL_TYPE_REMAP.items():
    mask = df["patient_id"] == pat
    for old_lbl, new_lbl in remap.items():
        n = (mask & (df["cell_type"] == old_lbl)).sum()
        if n:
            df.loc[mask & (df["cell_type"] == old_lbl), "cell_type"] = new_lbl
            print(f"  [{pat}] relabeled {n} '{old_lbl}' → '{new_lbl}'")

# Merge EMT into NonNE SCLC
n_emt = (df["cell_type"] == "EMT").sum()
if n_emt:
    df.loc[df["cell_type"] == "EMT", "cell_type"] = "NonNE SCLC"
    print(f"  Relabeled {n_emt} 'EMT' → 'NonNE SCLC'")

# Merge distance_to_root from leaf_distances.csv files
patients = df["patient_id"].unique().tolist()
leaf_df  = load_leaf_distances(patients)
print(f"  Loaded leaf distances for {leaf_df['decoded_cell'].nunique():,} cells")
if "decoded_cell" in df.columns and not leaf_df.empty:
    df = df.merge(leaf_df, on="decoded_cell", how="left")
    print(f"  distance_to_root matched: {df['distance_to_root'].notna().sum():,}")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Metrics ────────────────────────────────────────────────────────────────────
METRICS = [
    ("cnv_burden_gene_bin", "Fraction of genes altered\n(binarized p_cnv ≥ 0.5)"),
    ("infercnv_score",      "inferCNV score"),
    ("distance_to_root",    "Distance from root"),
    ("CytoTRACE2_Score",    "CytoTRACE2 score"),
    ("pseudotime",          "Pseudotime\n(Palantir)"),
]
METRICS = [(m, lbl) for m, lbl in METRICS if m in df.columns]

# ── Individual plots ───────────────────────────────────────────────────────────
all_records = {}
for metric, ylabel in METRICS:
    records = compute_records(df, metric)
    all_records[metric] = records
    n = len(records)
    print(f"\n{metric}: {n} patients")
    for r in records:
        sig_str = "sig" if r["sig"] else "ns "
        print(f"  {r['patient']}  {sig_str}  diff={r['diff']:+.4f}  q={r['q']:.3f}")

    slug  = metric.lower().replace(" ", "_")

    fig, ax = plt.subplots(figsize=(4, 1.85))
    draw_panel(ax, records, ylabel)
    add_legend(fig, bbox=(1.01, 0.5))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"cnv_burden_violin_{slug}{OUT_SUFFIX}.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")