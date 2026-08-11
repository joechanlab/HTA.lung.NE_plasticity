#!/usr/bin/env python3
"""
9_ancestral_position_cnv_burden.py

Single publication plot testing:
  "LUAD populations occupy ancestral phylogenetic positions with lower CNV
   burden; NE populations emerge on more evolved branches with more CNAs."

Background: all Numbat clones (colored by dominant cell type).
Foreground: per-tumor abundance-weighted centroids connected by LUAD → NE arrows.
Statistics:  paired Wilcoxon on clone depth and cumulative CNV count.

Data: phylo_celltype.tsv  (clone depth + cumulative CNV segment count)
"""

import os, argparse, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--out-suffix", default="")
_args, _ = _p.parse_known_args()
OUT_SUFFIX = _args.out_suffix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHYLO_TSV  = os.path.join(SCRIPT_DIR, "phylo_celltype.tsv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

NE_TYPES   = ["SCLC-A", "SCLC-N"]
LUAD_TYPES = ["LUAD"]

CT_COLORS = {
    "LUAD":   "#e6a817",
    "SCLC-A": "#d62728",
    "SCLC-N": "#17becf",
}

# ── Load ─────────────────────────────────────────────────────────────────────────
ph = pd.read_csv(PHYLO_TSV, sep="\t")
ph["n_cum_cnv"] = ph["cum_segs"].apply(
    lambda s: len(str(s).split(",")) if pd.notna(s) and str(s).strip() else 0
)

# Dominant cell type per clone
dom = (ph.sort_values("ct_frac", ascending=False)
         .groupby(["patient_id", "sample_id", "clone_id", "depth", "n_cum_cnv"])
         .first().reset_index()
         [["patient_id", "clone_id", "depth", "n_cum_cnv",
           "cell_type", "ct_frac", "n_cells"]])

# Per-patient abundance-weighted centroids
pat_rows = []
for pat, grp in ph.groupby("patient_id"):
    lu = grp[grp["cell_type"].isin(LUAD_TYPES)]
    ne = grp[grp["cell_type"].isin(NE_TYPES)]
    if lu.empty or ne.empty:
        continue
    wl = (lu["n_cells"] * lu["ct_frac"]).sum()
    wn = (ne["n_cells"] * ne["ct_frac"]).sum()
    if wl == 0 or wn == 0:
        continue
    pat_rows.append(dict(
        patient = pat,
        luad_d  = float((lu["depth"]     * lu["n_cells"] * lu["ct_frac"]).sum() / wl),
        luad_c  = float((lu["n_cum_cnv"] * lu["n_cells"] * lu["ct_frac"]).sum() / wl),
        ne_d    = float((ne["depth"]     * ne["n_cells"] * ne["ct_frac"]).sum() / wn),
        ne_c    = float((ne["n_cum_cnv"] * ne["n_cells"] * ne["ct_frac"]).sum() / wn),
    ))
pm = pd.DataFrame(pat_rows)
N  = len(pm)

# Paired Wilcoxon
diff_d = pm["ne_d"] - pm["luad_d"]
diff_c = pm["ne_c"] - pm["luad_c"]
wx_d = wilcoxon(diff_d[diff_d != 0], alternative="greater")
wx_c = wilcoxon(diff_c[diff_c != 0], alternative="greater")

def star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

print(f"Clone depth  NE>LUAD: {(diff_d>0).sum()}/{N}, p={wx_d.pvalue:.4f} {star(wx_d.pvalue)}")
print(f"Cum. CNVs   NE>LUAD: {(diff_c>0).sum()}/{N}, p={wx_c.pvalue:.4f} {star(wx_c.pvalue)}")

# ── Figure ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))

rng = np.random.default_rng(42)

# ── Background: clone scatter, jittered ─────────────────────────────────────────
for ct in ["SCLC-N", "SCLC-A", "LUAD"]:      # draw LUAD last (on top)
    sub = dom[dom["cell_type"] == ct]
    if sub.empty:
        continue
    col = CT_COLORS[ct]
    jit = rng.uniform(-0.22, 0.22, len(sub))
    sz  = np.clip(np.sqrt(sub["n_cells"].values) * 5, 12, 120)
    ax.scatter(sub["depth"].values + jit, sub["n_cum_cnv"].values,
               s=sz, color=col, alpha=0.38, edgecolors="none",
               zorder=2, rasterized=True)

# ── Foreground: per-tumor LUAD → NE arrows ───────────────────────────────────────
for _, row in pm.iterrows():
    ax.annotate(
        "", xy=(row["ne_d"], row["ne_c"]), xytext=(row["luad_d"], row["luad_c"]),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#555555",
            lw=1.2,
            mutation_scale=10,
            shrinkA=5, shrinkB=5,
        ),
        zorder=5,
    )

LUAD_MARKER_KW = dict(s=130, color=CT_COLORS["LUAD"], edgecolors="#555",
                      linewidths=0.8, zorder=6, marker="o")
NE_MARKER_KW   = dict(s=130, color="#8B0000", edgecolors="#555",
                      linewidths=0.8, zorder=6, marker="^")

ax.scatter(pm["luad_d"], pm["luad_c"], **LUAD_MARKER_KW)
ax.scatter(pm["ne_d"],   pm["ne_c"],   **NE_MARKER_KW)

# Patient labels next to LUAD centroids
for _, row in pm.iterrows():
    ax.text(row["luad_d"] - 0.10, row["luad_c"],
            row["patient"], fontsize=7, color="#333",
            va="center", ha="right", zorder=7)

# ── Statistics annotation ────────────────────────────────────────────────────────
stats_lines = (
    f"Paired Wilcoxon (NE > LUAD, n = {N} tumors)\n"
    f"Clone depth:   {(diff_d>0).sum()}/{N}  {star(wx_d.pvalue)}  (p = {wx_d.pvalue:.3f})\n"
    f"Cum. CNAs:     {(diff_c>0).sum()}/{N}  {star(wx_c.pvalue)}  (p = {wx_c.pvalue:.3f})"
)
ax.text(0.98, 0.04, stats_lines,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#222",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.92))

# ── Legend ───────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(color=CT_COLORS["LUAD"],   alpha=0.5, label="LUAD clones"),
    mpatches.Patch(color=CT_COLORS["SCLC-A"], alpha=0.5, label="SCLC-A clones"),
    mpatches.Patch(color=CT_COLORS["SCLC-N"], alpha=0.5, label="SCLC-N clones"),
    Line2D([0], [0], color="none", marker="o", markerfacecolor=CT_COLORS["LUAD"],
           markeredgecolor="#555", ms=9, label="LUAD centroid (per tumor)"),
    Line2D([0], [0], color="none", marker="^", markerfacecolor="#8B0000",
           markeredgecolor="#555", ms=9, label="NE centroid (per tumor)"),
    Line2D([0], [0], color="#555", lw=1.2, marker=">", ms=6,
           label="LUAD → NE (per tumor)"),
]
ax.legend(handles=legend_elements, loc="upper left",
          fontsize=8.5, framealpha=0.88, edgecolor="#cccccc",
          handlelength=1.4, borderpad=0.6)

ax.set_xlabel("Clone depth from root", fontsize=12)
ax.set_ylabel("Cumulative CNV events\n(# segments accumulated)", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=11)


fig.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(OUT_DIR, f"ancestral_position_cnv_burden{OUT_SUFFIX}.{ext}")
    fig.savefig(out, dpi=300 if ext == "pdf" else 200, bbox_inches="tight")
    print(f"Saved → {out}")
plt.close(fig)
