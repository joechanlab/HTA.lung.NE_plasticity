#!/usr/bin/env python3
"""
8_phylo_cnv_celltype_plot.py

Integrates phylogenetic ordering with cell-type-specific CNV events.

For each patient with ≥2 tumor cell types:
  Panel A: Cell type prevalence vs. phylogenetic depth
           (stacked area / bar showing which cell types dominate each depth level)
  Panel B: CNV event timeline — rows are CNV events ordered by phylogenetic depth
           at which they FIRST appear; columns are cell types; colour = mean P(CNV).
           A vertical dashed line marks the "divergence depth" (depth at which ≥2
           cell types first co-exist as distinct clone majorities).
           CNVs above the line = pre-divergence (ancestral); below = post-divergence.

Outputs: figures/cnv_phylo_celltype/{patient}_phylo_cnv.png  (one per patient)
         Also updates reports via 4_generate_patient_reports.py import.

Usage:
    /path/to/python 8_phylo_cnv_celltype_plot.py [--patients RU1083 ...]
"""

import argparse, os, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PHYLO_TSV    = os.path.join(SCRIPT_DIR, "phylo_celltype.tsv")
ENRICH_TSV   = os.path.join(SCRIPT_DIR, "cnv_celltype_enrichment.tsv")
OUT_DIR      = os.path.join(SCRIPT_DIR, "figures", "cnv_phylo_celltype")
PATIENT_LIST = os.path.join(SCRIPT_DIR, "patient_list.txt")

TUMOR_TYPES = ["SCLC-A", "SCLC-N", "SCLC-P", "NonNE SCLC",
               "EMT", "FOXI1+ NonNE SCLC-N", "LUAD", "LUSC"]

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
    "gain": "#d62728", "loss": "#1f77b4", "LOH": "#9467bd",
    "bamp": "#8B0000", "homdel": "#003f8e",
}
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "freq", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _state_color(label):
    for k, c in CNV_STATE_COLORS.items():
        if k in label:
            return c
    return "#555555"


def compute_divergence_depth(phylo_pat):
    """
    Return the depth at which two distinct tumor cell types first co-exist as
    majority in separate clones within the same sample.
    """
    present_ct = phylo_pat["cell_type"].unique()
    if len(present_ct) < 2:
        return None

    # For each sample × depth, get the set of majority cell types among clones
    # A clone's majority CT = the CT with highest ct_frac in that clone
    maj = (phylo_pat.sort_values("ct_frac", ascending=False)
                    .groupby(["sample_id","clone_id","depth"])
                    .first()
                    .reset_index()[["sample_id","clone_id","depth","cell_type"]])

    for depth in sorted(maj["depth"].unique()):
        d_clones = maj[maj["depth"] == depth]
        # Check if ≥2 different tumor cell types appear as majority in clones AT THIS depth
        cts_at_depth = set(d_clones["cell_type"].unique())
        tumor_cts    = cts_at_depth & set(TUMOR_TYPES)
        if len(tumor_cts) >= 2:
            return depth
    return None


def get_branch_cnv_depths(phylo_pat):
    """
    For each CNV event (arm_label), find the shallowest depth at which it
    first appears on a branch in any tumor-dominant clone.
    Returns dict: arm_label → (min_depth, dominant_ct_at_that_depth)
    """
    result = {}
    # Each row has branch_labels (semicolon-separated arm labels for this branch)
    # and depth + cell_type + ct_frac
    for _, row in phylo_pat.iterrows():
        if not isinstance(row["branch_labels"], str) or row["branch_labels"] == "":
            continue
        for lbl in row["branch_labels"].split("; "):
            lbl = lbl.strip()
            if not lbl:
                continue
            depth = int(row["depth"])
            ct    = row["cell_type"]
            frac  = row["ct_frac"]
            if lbl not in result or depth < result[lbl][0]:
                result[lbl] = (depth, ct, frac)
    return result


def get_celltype_first_depths(phylo_pat):
    """
    Abundance-weighted mean clone depth for each tumor cell type.

    Each clone×sample observation is weighted by (n_cells × ct_frac) so
    clones with more cells of a given type contribute proportionally more
    to the depth estimate.  This is far more robust than taking the minimum
    depth, which is sensitive to a single clone with very few cells.
    """
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


# ── Per-patient plot ───────────────────────────────────────────────────────────

def plot_patient(patient, phylo_pat, enrich_pat):
    present_ct = [ct for ct in TUMOR_TYPES if ct in phylo_pat["cell_type"].values]
    if len(present_ct) < 1:
        return

    # ── Divergence depth ──────────────────────────────────────────────────────
    div_depth = compute_divergence_depth(phylo_pat)
    ct_first_depths = get_celltype_first_depths(phylo_pat)

    # Order cell types by first appearance depth
    ordered_ct = sorted(present_ct, key=lambda c: ct_first_depths.get(c, 999))

    # ── Branch CNV depths ─────────────────────────────────────────────────────
    branch_cnv_d = get_branch_cnv_depths(phylo_pat)

    # ── Per-depth cell type composition (aggregated across all samples) ────────
    # For each depth, aggregate cell-type fractions weighted by n_cells
    all_depths  = sorted(phylo_pat["depth"].unique())
    depth_comp  = {}
    for d in all_depths:
        sub = phylo_pat[phylo_pat["depth"] == d]
        ct_totals = sub.groupby("cell_type")["n_cells"].sum()
        # Keep only tumor types; normalise
        ct_tumor  = ct_totals[ct_totals.index.isin(TUMOR_TYPES)]
        if ct_tumor.sum() == 0:
            continue
        depth_comp[d] = (ct_tumor / ct_tumor.sum()).to_dict()

    if not depth_comp:
        return

    # ── Figure setup ──────────────────────────────────────────────────────────
    n_ct = len(ordered_ct)
    n_hm_rows = len(branch_cnv_d)   # CNV events in heatmap

    # Dynamic height
    hm_row_h = max(0.22, min(0.40, 8.0 / max(n_hm_rows, 1)))
    fig_h    = max(7, n_hm_rows * hm_row_h + 4.5)
    fig_w    = 14

    fig = plt.figure(figsize=(fig_w, fig_h))
    title_lines = [
        f"Patient {patient}   |   CNV events in phylogenetic context",
        "Cell type ordering × ancestral vs. cell-type-specific CNV events",
    ]
    if div_depth is not None:
        title_lines.append(
            f"Cell type divergence at depth {div_depth}  "
            f"(cell types ordered by first appearance: "
            + "  →  ".join(f"{ct} [d={ct_first_depths.get(ct,'?')}]"
                           for ct in ordered_ct) + ")"
        )
    fig.suptitle("\n".join(title_lines), fontsize=9.5,
                 fontweight="bold", y=0.995, linespacing=1.5)

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[2.2, max(n_hm_rows * hm_row_h, 3)],
        width_ratios=[1, 2],
        hspace=0.40, wspace=0.38,
        left=0.18, right=0.97, top=0.91, bottom=0.07)

    ax_depth = fig.add_subplot(gs[0, 0])   # Panel A: depth composition
    ax_order = fig.add_subplot(gs[0, 1])   # Panel A': cell-type ordering timeline
    ax_hm    = fig.add_subplot(gs[1, :])   # Panel B: heatmap (full width)

    ct_colors = [CELL_TYPE_COLORS.get(ct, "#888888") for ct in ordered_ct]

    # ── Panel A: stacked bar — cell type fraction per depth ───────────────────
    depth_list  = sorted(depth_comp.keys())
    x_pos       = np.arange(len(depth_list))
    bottoms     = np.zeros(len(depth_list))

    for ct, col in zip(ordered_ct, ct_colors):
        vals = np.array([depth_comp[d].get(ct, 0.0) for d in depth_list])
        ax_depth.bar(x_pos, vals, bottom=bottoms, color=col,
                     label=ct, width=0.72, edgecolor="none")
        bottoms += vals

    ax_depth.set_xticks(x_pos)
    ax_depth.set_xticklabels([f"d={d}" for d in depth_list], fontsize=7)
    ax_depth.set_ylabel("Tumor cell\ntype fraction", fontsize=7.5)
    ax_depth.set_ylim(0, 1.05)
    ax_depth.set_title("Cell type composition\nper clone depth", fontsize=8.5, pad=3)
    ax_depth.spines[["top","right"]].set_visible(False)
    ax_depth.tick_params(labelsize=7)

    if div_depth is not None:
        # Mark divergence depth with arrow
        if div_depth in depth_list:
            xi = depth_list.index(div_depth)
            ax_depth.annotate("", xy=(xi, 1.02), xytext=(xi, 1.08),
                              arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
            ax_depth.text(xi, 1.10, "div.", ha="center", va="bottom",
                          fontsize=6.5, color="#333333")

    # ── Panel A': cell-type first-appearance timeline ─────────────────────────
    max_d = max(ct_first_depths.values()) if ct_first_depths else 1
    ax_order.set_xlim(-0.5, max_d + 1.5)
    ax_order.set_ylim(-0.5, len(ordered_ct) - 0.5)
    ax_order.set_xlabel("Phylogenetic depth\n(# branches from root)", fontsize=7.5)
    ax_order.set_title("Cell type first appearance\nalong phylogeny", fontsize=8.5, pad=3)
    ax_order.spines[["top","right","left"]].set_visible(False)
    ax_order.set_yticks([])

    for yi, (ct, col) in enumerate(zip(ordered_ct, ct_colors)):
        fd = ct_first_depths.get(ct, 0)
        # Horizontal line from 0 to first depth
        ax_order.plot([0, fd], [yi, yi], color="#cccccc", lw=1.5, zorder=1)
        # Diamond marker at first appearance
        ax_order.scatter([fd], [yi], s=120, color=col, zorder=3,
                         marker="D", edgecolors="white", linewidths=0.5)
        ax_order.text(fd + 0.15, yi, ct, va="center", ha="left",
                      fontsize=8, color=col, fontweight="bold")

    if div_depth is not None:
        ax_order.axvline(div_depth, color="#555555", lw=1.2, ls="--", zorder=2)
        ax_order.text(div_depth + 0.05, len(ordered_ct) - 0.2,
                      f"divergence\n(d={div_depth})", fontsize=6.5,
                      color="#555555", va="top")

    # ── Panel B: CNV heatmap ordered by phylogenetic depth ────────────────────
    if not enrich_pat.empty and branch_cnv_d:
        # Build sorted list of (arm_label, first_depth, dominant_ct)
        cnv_items = sorted(branch_cnv_d.items(), key=lambda x: (x[1][0], x[0]))
        hm_labels = [it[0] for it in cnv_items]
        hm_depths = [it[1][0] for it in cnv_items]
        hm_dom_ct = [it[1][1] for it in cnv_items]

        # Mean P(CNV) per (arm_label × cell_type) from enrich_pat
        enrich_pivot = enrich_pat.pivot_table(
            index="arm_label", columns="cell_type",
            values="mean_p_cnv", aggfunc="mean")
        for ct in ordered_ct:
            if ct not in enrich_pivot.columns:
                enrich_pivot[ct] = np.nan

        mat = np.zeros((len(hm_labels), len(ordered_ct)))
        for ri, lbl in enumerate(hm_labels):
            for ci, ct in enumerate(ordered_ct):
                if lbl in enrich_pivot.index and ct in enrich_pivot.columns:
                    v = enrich_pivot.at[lbl, ct]
                    mat[ri, ci] = v if not np.isnan(v) else 0.0

        im = ax_hm.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP,
                          vmin=0, vmax=1, interpolation="nearest")

        # X-axis
        ax_hm.set_xticks(range(len(ordered_ct)))
        ax_hm.set_xticklabels(ordered_ct, rotation=35, ha="right", fontsize=8.5)
        for tick, col in zip(ax_hm.get_xticklabels(), ct_colors):
            tick.set_color(col); tick.set_fontweight("bold")

        # Y-axis — arm labels with depth prefix and state colour
        ax_hm.set_yticks(range(len(hm_labels)))
        ax_hm.set_yticklabels([])
        for ri, (lbl, dep, dom) in enumerate(zip(hm_labels, hm_depths, hm_dom_ct)):
            state_col = _state_color(lbl)
            dom_col   = CELL_TYPE_COLORS.get(dom, "#555555")
            # Depth badge
            ax_hm.text(-0.01, ri, f"d{dep}", transform=ax_hm.get_yaxis_transform(),
                       ha="right", va="center", fontsize=6.0, color="#888888",
                       fontfamily="monospace")
            ax_hm.text(-0.06, ri, lbl, transform=ax_hm.get_yaxis_transform(),
                       ha="right", va="center", fontsize=6.5, color=state_col)
            # Dominant CT colour dot on left
            ax_hm.scatter([-0.5], [ri], s=22, color=dom_col,
                          transform=ax_hm.transData, clip_on=False, zorder=3)

        # Divergence line across heatmap
        if div_depth is not None:
            # Find last row where depth < div_depth
            last_pre = max((i for i, d in enumerate(hm_depths) if d < div_depth),
                           default=None)
            if last_pre is not None:
                ax_hm.axhline(last_pre + 0.5, color="#333333",
                              lw=1.5, ls="--", zorder=4)
                ax_hm.text(len(ordered_ct) - 0.5, last_pre + 0.5,
                           f" ← depth {div_depth}: cell types diverge",
                           va="bottom", ha="right", fontsize=7,
                           color="#333333", style="italic")
                # Region labels
                if last_pre >= 0:
                    ax_hm.text(-0.5, last_pre / 2,
                               "ancestral\n(pre-divergence)",
                               transform=ax_hm.transData,
                               ha="right", va="center", fontsize=6.5,
                               color="#444444", rotation=90)
                post_mid = (last_pre + 1 + len(hm_labels) - 1) / 2
                if post_mid < len(hm_labels):
                    ax_hm.text(-0.5, post_mid,
                               "cell-type-specific\n(post-divergence)",
                               transform=ax_hm.transData,
                               ha="right", va="center", fontsize=6.5,
                               color="#444444", rotation=90)

        ax_hm.set_title(
            "CNV events ordered by phylogenetic depth of first appearance\n"
            "Left dot = dominant cell type when CNV acquired  |  "
            "Dashed line = cell type divergence point",
            fontsize=8, pad=4)
        cbar = plt.colorbar(im, ax=ax_hm, fraction=0.015, pad=0.01, shrink=0.6)
        cbar.set_label("Mean P(CNV)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    else:
        ax_hm.text(0.5, 0.5, "No CNV data available",
                   ha="center", va="center", fontsize=9, color="#aaaaaa",
                   transform=ax_hm.transAxes)
        ax_hm.set_axis_off()

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=CELL_TYPE_COLORS.get(ct, "#888"), label=ct)
               for ct in ordered_ct]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(ordered_ct), 6),
               fontsize=7.5, frameon=False,
               title="Cell type (ordered by first appearance in phylogeny)",
               title_fontsize=8)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{patient}_phylo_cnv.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="*")
    args = parser.parse_args()

    with open(PATIENT_LIST) as f:
        pid_entries = [l.strip() for l in f if l.strip()]
    subset = set(args.patients) if args.patients else None

    phylo_all  = pd.read_csv(PHYLO_TSV,   sep="\t") if os.path.exists(PHYLO_TSV)   else pd.DataFrame()
    enrich_all = pd.read_csv(ENRICH_TSV,  sep="\t") if os.path.exists(ENRICH_TSV)  else pd.DataFrame()

    if phylo_all.empty:
        print("phylo_celltype.tsv not found — run 7_extract_phylo_celltype.R first.")
        return

    for pid_entry in pid_entries:
        patient = pid_entry.split("_")[0]
        if subset and patient not in subset and pid_entry not in subset:
            continue

        phylo_pat  = phylo_all[phylo_all["patient_id"] == patient].copy()
        enrich_pat = enrich_all[enrich_all["patient_id"] == patient].copy() \
                     if not enrich_all.empty else pd.DataFrame()

        present_ct = [ct for ct in TUMOR_TYPES if ct in phylo_pat["cell_type"].values]
        if not present_ct:
            print(f"[{patient}] No tumor cell types — skipping")
            continue

        print(f"[{patient}] types={present_ct}")
        plot_patient(patient, phylo_pat, enrich_pat)


if __name__ == "__main__":
    main()
