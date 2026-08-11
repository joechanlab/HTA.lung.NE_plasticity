#!/usr/bin/env python3
"""
9b_luad_ne_cnv_violin.py

Split violin plot: LUAD (left, gold) vs NE subtypes (right, red/cyan)
for cnv_burden_gene_bin, one violin per dual-lineage patient.
Patients sorted: sig NE>LUAD → ns NE>LUAD → ns LUAD>NE → sig LUAD>NE.
Per-patient MWU with BH FDR correction.
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_TSV   = os.path.join(SCRIPT_DIR, "cnv_burden_metacell_joined.tsv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

NE_SUBTYPES = [
    ("SCLC-A",         "#d62728"),
    ("SCLC-N",         "#17becf"),
    ("SCLC-N (FOXI1+)","#2ca02c"),
]
NE_TYPES  = [ct for ct, _ in NE_SUBTYPES]
LUAD_COL  = "#e6a817"
MIN_CELLS = 2

# Metacell data uses "FOXI1+ NonNE SCLC-N" across all patients
GLOBAL_CELL_TYPE_REMAP = {
    "FOXI1+ NonNE SCLC-N": "SCLC-N (FOXI1+)",
    "FOXI1+ SCLC-N":        "SCLC-N (FOXI1+)",
}

# ── Half-violin (mirrors 5_cnv_burden_violin.py) ────────────────────────────────
def half_violin(ax, data, x, side, color, width=0.38, alpha=0.78, bw=0.25):
    if len(data) < 4:
        return
    kde = stats.gaussian_kde(data, bw_method=bw)
    lo, hi = data.min(), data.max()
    if lo >= hi:
        return
    ys   = np.linspace(lo, hi, 400)
    dens = kde(ys)
    dens = dens / dens.max() * width
    sign = -1 if side == "left" else 1
    ax.fill_between(ys, x, x + sign * dens, color=color, alpha=alpha, zorder=2)
    lw = max(0.7, min(2.2, 20.0 / max(len(data), 1)))
    ax.plot(ys, x + sign * dens, color=color, lw=lw,
            alpha=min(alpha + 0.1, 1.0), zorder=3)
    med = float(np.median(data))
    med_dens = float(kde([med])[0]) / kde(ys).max() * width
    r, g, b, _ = mc.to_rgba(color)
    dark = (r * 0.55, g * 0.55, b * 0.55, 1.0)
    ax.plot([med, med], [x, x + sign * med_dens],
            color=dark, lw=2.0, zorder=5)

# ── Load data ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_TSV, sep="\t")
for old, new in GLOBAL_CELL_TYPE_REMAP.items():
    df.loc[df["cell_type"] == old, "cell_type"] = new

# ── Dual-lineage patients + per-patient MWU ──────────────────────────────────────
test_rows = []
for pat, grp in df.groupby("patient_id"):
    luad = grp[grp["cell_type"] == "LUAD"]["cnv_burden_gene_bin"].dropna().values
    ne   = grp[grp["cell_type"].isin(NE_TYPES)]["cnv_burden_gene_bin"].dropna().values
    if len(luad) < MIN_CELLS or len(ne) < MIN_CELLS:
        continue
    _, p = stats.mannwhitneyu(ne, luad, alternative="two-sided")
    test_rows.append(dict(patient=pat, p=p,
                          diff=float(np.median(ne) - np.median(luad)),
                          luad=luad, ne=ne))

if not test_rows:
    raise RuntimeError("No dual-lineage patients found.")

_, padj, _, _ = multipletests([r["p"] for r in test_rows], method="fdr_bh")
for r, q in zip(test_rows, padj):
    r["q"] = q
    r["sig"] = q < 0.05

def sort_key(r):
    sig, ne_gt = r["sig"], r["diff"] > 0
    tier = (0 if (sig and ne_gt) else 1 if (not sig and ne_gt)
            else 2 if (not sig and not ne_gt) else 3)
    return (tier, -r["diff"] if ne_gt else r["diff"])

records = sorted(test_rows, key=sort_key)
n = len(records)
print(f"{n} dual-lineage patients")
for r in records:
    print(f"  {r['patient']:8}  diff={r['diff']:+.4f}  q={r['q']:.3f}  {'*' if r['sig'] else 'ns'}")


# ── Figure ───────────────────────────────────────────────────────────────────────
fig_w = 3.5
fig, ax = plt.subplots(figsize=(fig_w, 4.5))

for xi, r in enumerate(records):
    # Left: LUAD
    half_violin(ax, r["luad"], xi, "left",  LUAD_COL, width=0.38, alpha=0.80)

    # Right: all NE subtypes pooled into one violin
    half_violin(ax, r["ne"], xi, "right", "tab:red", width=0.38, alpha=0.82)

# Significance brackets + per-patient star
ax.relim(); ax.autoscale_view()
xlo, xhi = ax.get_xlim()
pad = (xhi - xlo) * 0.08

sig_ne  = [i for i, r in enumerate(records) if r["diff"] > 0 and r["sig"]]
sig_lu  = [i for i, r in enumerate(records) if r["diff"] < 0 and r["sig"]]

# Divider lines between sorted groups
if sig_ne and max(sig_ne) + 1 < n:
    ax.axhline(max(sig_ne) + 0.5, color="#aaaaaa", lw=1.1, ls="-", zorder=0)
if sig_lu and min(sig_lu) > 0:
    ax.axhline(min(sig_lu) - 0.5, color="#aaaaaa", lw=1.1, ls="-", zorder=0)

# Group brackets
for idxs, label in [(sig_ne, "NE > LUAD\n(FDR < 0.05)"),
                     (sig_lu, "LUAD > NE\n(FDR < 0.05)")]:
    if not idxs:
        continue
    y0, y1 = min(idxs) - 0.44, max(idxs) + 0.44
    x_br = xhi + pad * 0.3
    x_tx = xhi + pad * 1.1
    ax.annotate("", xy=(x_br, y0), xytext=(x_br, y1),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.4,
                                mutation_scale=11))
    ax.text(x_tx, (y0 + y1) / 2, label,
            ha="left", va="center", fontsize=10, fontweight="bold")

# Per-violin star annotation left of y-axis
for xi, r in enumerate(records):
    star = ("***" if r["q"] < 0.001 else "**" if r["q"] < 0.01
            else "*" if r["q"] < 0.05 else "ns")
    col  = "#333333" if r["sig"] else "#999999"
    ax.text(xlo - pad * 0.5, xi, star,
            ha="right", va="center", fontsize=11, color=col, fontweight="bold")

ax.set_yticks(range(n))
ax.set_yticklabels([r["patient"] for r in records],
                   rotation=0, ha="right", fontsize=11)
ax.set_ylim(n - 0.35, -0.65)
ax.set_xlim(xlo - pad * 4, xhi + pad * 2.2)
ax.set_xlabel("Fraction of genes with CNV", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=10)

# Legend
handles = [mpatches.Patch(color=LUAD_COL, label="LUAD"),
           mpatches.Patch(color="tab:red", label="NE")]
ax.legend(handles=handles, loc="center right",
          bbox_to_anchor=(1.4, 0.38),
          fontsize=9, framealpha=0.88, edgecolor="#cccccc")

fig.tight_layout()
out = os.path.join(OUT_DIR, "luad_ne_cnv_burden_violin.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved → {out.replace('.pdf','.png')}")
plt.close(fig)
