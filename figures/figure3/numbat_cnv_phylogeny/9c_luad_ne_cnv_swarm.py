#!/usr/bin/env python3
"""
9c_luad_ne_cnv_swarm.py

Box plot comparing LUAD vs NE vs NonNE SCLC CNV burden across all samples.
Each dot = per-patient median CNV burden for that cell-type group.
Lines trace each patient's path across whichever groups they have data in,
in left-to-right order (skipping missing groups) — one edge per pair of
consecutive groups a patient has, not a full pairwise graph.

RU1250's EMT cells get their own "RU1250 EMT" column at the end instead of
being pooled into "NonNE SCLC" — RU1250 has LUAD and EMT but no NE cells at
all, so pooling it in would misleadingly imply an NE intermediate state.

Pairwise significance is computed via a linear mixed model
(cnv_burden_gene_bin ~ group, random intercept per patient) fit on every
individual metacell, and annotated on the plot (BH-corrected across the
3 pairwise comparisons: LUAD/NE/NonNE SCLC only — RU1250 EMT is a single
patient, so it's excluded from the significance brackets). Per-patient-median
MWU p-values are also printed to stdout for reference.
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_TSV    = os.path.join(SCRIPT_DIR, "cnv_burden_metacell_joined.tsv")
TP_TSV      = os.path.join(SCRIPT_DIR, "sample_timepoints.tsv")
OUT_DIR     = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

NE_TYPES     = ["SCLC-A", "SCLC-N", "SCLC-N (FOXI1+)"]
NONNE_TYPES  = ["NonNE SCLC", "EMT"]
LUAD_COL     = "#e6a817"
NONNE_COL    = "#7a5195"
NE_COL       = "#d62728"
MIN_CELLS    = 1

# RU1250 is the only patient whose EMT population lacks any NE cells at all
# (LUAD -> EMT directly, no NE intermediate) — pooling it into "NonNE SCLC"
# would misleadingly suggest it passed through an NE state. Give it its own
# column instead, appended at the end.
RU1250_EMT_LABEL = "EMT (RU1250)"
RU1250_EMT_COL   = "#c299d1"

GROUPS = [
    ("LUAD",        ["LUAD"],              LUAD_COL),
    ("NE",          NE_TYPES,              NE_COL),
    ("NonNE SCLC",  NONNE_TYPES,           NONNE_COL),
    ("RU1250 EMT",  [RU1250_EMT_LABEL],    RU1250_EMT_COL),
]

GLOBAL_CELL_TYPE_REMAP = {
    "FOXI1+ NonNE SCLC-N": "SCLC-N (FOXI1+)",
    "FOXI1+ SCLC-N":        "SCLC-N (FOXI1+)",
}

# ── Load & remap ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_TSV, sep="\t")
for old, new in GLOBAL_CELL_TYPE_REMAP.items():
    df.loc[df["cell_type"] == old, "cell_type"] = new

# Pull RU1250's EMT cells out of the pooled "NonNE SCLC" group so they land
# in their own "RU1250 EMT" column instead (see RU1250_EMT_LABEL above).
df.loc[(df["patient_id"] == "RU1250") & (df["cell_type"] == "EMT"),
       "cell_type"] = RU1250_EMT_LABEL

# ── Sample timepoints: max date_rank per patient per group ────────────────────
tp = pd.read_csv(TP_TSV, sep="\t")[["sample_id", "date_rank"]].dropna()
tp_rank = dict(zip(tp["sample_id"], tp["date_rank"]))

def max_date_rank(patient, group_mask):
    """Return max date_rank for rows matching patient + cell-type mask, or NaN."""
    sids = df.loc[(df["patient_id"] == patient) & group_mask, "sample_id"].unique()
    ranks = [tp_rank[s] for s in sids if s in tp_rank]
    return max(ranks) if ranks else float("nan")

def luad_is_later(patient):
    """True if max LUAD date_rank > max NE date_rank for this patient."""
    r_luad = max_date_rank(patient, df["cell_type"] == "LUAD")
    r_ne   = max_date_rank(patient, df["cell_type"].isin(NE_TYPES))
    if np.isnan(r_luad) or np.isnan(r_ne):
        return False
    return r_luad > r_ne

# ── Per-patient medians (all patients, each group independently) ──────────────
group_meds = {name: {} for name, _, _ in GROUPS}   # group -> patient -> median CNV

for pat, grp in df.groupby("patient_id"):
    for name, cts, _ in GROUPS:
        vals = grp[grp["cell_type"].isin(cts)]["cnv_burden_gene_bin"].dropna().values
        if len(vals) >= MIN_CELLS:
            group_meds[name][pat] = float(np.median(vals))

group_vecs = {name: np.array(list(meds.values())) for name, meds in group_meds.items()}

for name, meds in group_meds.items():
    print(f"{name} patients: {len(meds)}")

pair_list = [("LUAD", "NE"), ("NE", "NonNE SCLC"), ("LUAD", "NonNE SCLC")]

# ── Pairwise MWU (per-patient medians), BH-corrected — printed for reference ───
raw_mwu = []
for g1, g2 in pair_list:
    _, p = stats.mannwhitneyu(group_vecs[g1], group_vecs[g2], alternative="two-sided")
    raw_mwu.append(p)
_, adj_mwu, _, _ = multipletests(raw_mwu, method="fdr_bh")
for (g1, g2), raw, adj in zip(pair_list, raw_mwu, adj_mwu):
    print(f"MWU {g1} vs {g2}: p = {raw:.4g}  (BH-adj p = {adj:.4g})")

# ── Linear mixed model at the metacell level ───────────────────────────────────
# Per-patient medians collapse each patient to one point, throwing away
# within-patient metacell variance/n. Fit cnv_burden_gene_bin ~ group on every
# metacell instead, with a random intercept per patient to still account for
# repeated measurements within a patient. These p-values (BH-corrected) are
# what get annotated on the plot.
print("\n── Linear mixed model (metacell-level, random intercept per patient) ──")
group_of_celltype = {ct: name for name, cts, _ in GROUPS for ct in cts}
lmm_df = df[df["cell_type"].isin(group_of_celltype)].copy()
lmm_df["group"] = lmm_df["cell_type"].map(group_of_celltype)
lmm_df = lmm_df.dropna(subset=["cnv_burden_gene_bin"])

for name, _, _ in GROUPS:
    n_mc  = (lmm_df["group"] == name).sum()
    n_pat = lmm_df.loc[lmm_df["group"] == name, "patient_id"].nunique()
    print(f"{name}: {n_mc} metacells across {n_pat} patients")

raw_lmm = {}
for ref in ["LUAD", "NonNE SCLC"]:
    lmm_df["group_c"] = pd.Categorical(
        lmm_df["group"],
        categories=[ref] + [g for g, _, _ in GROUPS if g != ref])
    model = smf.mixedlm("cnv_burden_gene_bin ~ C(group_c, Treatment(reference='" + ref + "'))",
                         lmm_df, groups=lmm_df["patient_id"])
    result = model.fit(reml=True)
    print(f"\nReference = {ref}:")
    for term, coef, pval in zip(result.params.index, result.params.values, result.pvalues.values):
        if term.startswith("C(group_c"):
            other = term.split("T.")[-1].rstrip("]")
            print(f"  {other} vs {ref}: coef = {coef:+.4f}, p = {pval:.4g}")
            raw_lmm[frozenset((ref, other))] = pval

raw_lmm_ordered = [raw_lmm[frozenset((g1, g2))] for g1, g2 in pair_list]
_, adj_lmm, _, _ = multipletests(raw_lmm_ordered, method="fdr_bh")
pair_pvals = dict(zip(pair_list, zip(raw_lmm_ordered, adj_lmm)))
print()
for (g1, g2), (raw, adj) in pair_pvals.items():
    print(f"LMM {g1} vs {g2}: p = {raw:.4g}  (BH-adj p = {adj:.4g})")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.8))

X_POS = {name: i for i, (name, _, _) in enumerate(GROUPS)}
BOX_W = 0.45

for name, _, color in GROUPS:
    ax.boxplot(group_vecs[name], positions=[X_POS[name]], widths=BOX_W,
               patch_artist=True, showfliers=False,
               medianprops=dict(color="black", lw=2),
               boxprops=dict(facecolor=color, alpha=0.35, linewidth=1.2),
               whiskerprops=dict(linewidth=1.2),
               capprops=dict(linewidth=1.2))

# Lines connecting the same patient across groups: one path per patient
# through whichever groups they have data in, in left-to-right (x) order —
# NOT every pairwise combination. A patient with all 3 groups gets 2 edges
# (LUAD-NE, NE-NonNE), not the full triangle; a patient missing the middle
# group (e.g. LUAD+NonNE, no NE) still gets a direct single edge between the
# two they do have. This keeps each dot's connections one-to-one per neighbor.
np.random.seed(42)
JITTER = 0.08
jitter_by_group = {}
for name, _, _ in GROUPS:
    pats = sorted(group_meds[name])
    jitter_by_group[name] = dict(zip(pats, np.random.uniform(-JITTER, JITTER, len(pats))))

all_patients = sorted(set().union(*group_meds.values()))
for pat in all_patients:
    present = [name for name, _, _ in GROUPS if pat in group_meds[name]]
    for g1, g2 in zip(present[:-1], present[1:]):
        ax.plot([X_POS[g1] + jitter_by_group[g1][pat], X_POS[g2] + jitter_by_group[g2][pat]],
                [group_meds[g1][pat], group_meds[g2][pat]],
                color="#888888", lw=0.9, alpha=0.6, zorder=3)

# Per-patient median dots (on top of lines)
for name, _, color in GROUPS:
    pats = sorted(group_meds[name])
    jx = np.array([jitter_by_group[name][p] for p in pats])
    ax.scatter(X_POS[name] + jx, [group_meds[name][p] for p in pats],
               color=color, s=45, zorder=5, edgecolors="white", linewidths=0.7)

# ── P-value brackets (LMM, BH-adjusted) ────────────────────────────────────────
y_max = max(v.max() for v in group_vecs.values())
step  = y_max * 0.12
for i, (g1, g2) in enumerate(pair_list):
    y_br = y_max * 1.06 + i * step
    y_tx = y_br + y_max * 0.02
    x1, x2 = X_POS[g1], X_POS[g2]
    ax.plot([x1, x1, x2, x2], [y_br * 0.98, y_br, y_br, y_br * 0.98],
            color="black", lw=1.2)
    _, adj = pair_pvals[(g1, g2)]
    p_label = (f"p = {adj:.2e}" if adj < 0.001 else f"p = {adj:.3f}")
    ax.text((x1 + x2) / 2, y_tx, p_label, ha="center", va="bottom", fontsize=9)

ax.set_xticks(list(X_POS.values()))
ax.set_xticklabels(
    [f"{name}\n(n={len(group_meds[name])})" for name, _, _ in GROUPS],
    fontsize=11)
ax.set_xlim(-0.55, len(GROUPS) - 0.45)
ax.set_ylim(bottom=0)
ax.set_ylabel("Median CNV burden per patient", fontsize=11)
ax.tick_params(labelsize=10)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = os.path.join(OUT_DIR, "luad_ne_cnv_burden_boxplot.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved → {out.replace('.pdf', '.png')}")
plt.close(fig)

