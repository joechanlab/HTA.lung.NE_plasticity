"""
Step 3 of 3 — depends on step 1's (1_gsea_cell_type_all_patients.py) per-patient
gsea_hallmarks_<cell_type>.csv outputs, specifically from
`--reference sclc_pool` mode (figures/gsea_all_patients_sclc_pool/). Run step 1
with that flag for the full 12-patient cohort before this script, or it will
read stale/partial/wrong-mode data.

Convergence/divergence of Hallmark pathway enrichment between the NonNE-lineage
(NonNE SCLC + EMT — non-neuroendocrine transdifferentiated states; FOXI1+ NonNE
SCLC-N groups with SCLC here, not NonNE — see step 1's SCLC_REFERENCE_TYPES) and
LUAD cell types, pooling the per-patient GSEA results produced by step 1.

WHY sclc_pool AND NOT neighbor MODE (2026-07-21): step 1 offers two comparator
designs. --reference neighbor (its default) compares each cell type to its own
single most-similar OTHER cell type per patient — useful for marker discovery,
but that neighbor is NOT uniformly an SCLC subtype (e.g. in RU1444, LUAD's
nearest neighbor is SCLC-A, but in other patients LUAD's neighbor is LUSC), so it
cannot support a "co-divergence FROM SCLC" question, and it isn't
cross-patient-comparable either (the reference identity itself varies with each
patient's cell-type composition). --reference sclc_pool fixes this: every
non-SCLC cell type is compared against the SAME kind of reference in every
patient — a pooled SCLC-A + SCLC-N (+ FOXI1+ NonNE SCLC-N where present) group —
so "recurrently enriched" here means "diverges from actual SCLC in the same
direction," consistently, and NES values are comparable across patients and
across cell-type columns because the reference is always the same population
type, not whatever happened to be nearby. Each patient's actual comparator
(mostly identical by construction, but shown for transparency and to reveal any
patient where only one SCLC subtype was available) is loaded from step 1's
comparators.csv and printed at the start of the run.

A term is called "recurrently enriched" for a phenotype (NonNE or LUAD) if it is
positively enriched (NES > 0, FDR q-val < GSEA_FDR_CUTOFF) in at least
NONNE_MIN_FRACTION / LUAD_MIN_FRACTION of that group's patients (rounded up, min
1). The two fractions are DELIBERATELY different here (NonNE: 2/3, i.e. requires
2 of the 3 NonNE-lineage patients to agree; LUAD: 1/3, i.e. requires 4 of 10) —
NonNE was tightened from an earlier 1/3-for-both version specifically to trim its
list down to only the pathways with real cross-patient support, at the cost of no
longer being symmetric with LUAD's threshold. Both thresholds are stated in the
figure title so the asymmetry is never hidden.

For the genuine "do NonNE and LUAD resemble each other directly?" question, see
2_direct_nonne_vs_luad.py (a real one-vs-one head-to-head test in the 2 patients
that have both populations) — its result is NOT folded into this figure's
coloring (an earlier version did that, but once every co-divergent term landed in
the same category the per-term coloring stopped adding information over just
labeling the whole region, so it was removed); check
direct_nonne_vs_luad.csv / nonne_luad_hallmark_summary.csv directly instead.

Output: figures/gsea_all_patients_sclc_pool/_summary/nonne_luad_hallmark_venn.{pdf,png}
        figures/gsea_all_patients_sclc_pool/_summary/nonne_luad_hallmark_summary.csv
"""
import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

ROOT = "figures/gsea_all_patients_sclc_pool"
OUTDIR = f"{ROOT}/_summary"
os.makedirs(OUTDIR, exist_ok=True)

# Cell-type CSV basenames (as written by 1_gsea_cell_type_all_patients.py) that
# belong to each phenotype group. NonNE = NonNE SCLC + EMT (both non-neuroendocrine
# transdifferentiated states). FOXI1+ NonNE SCLC-N groups with SCLC in the
# sclc_pool design (see step 1's SCLC_REFERENCE_TYPES), not with NonNE, and so
# never appears as its own file here (it has no DEG/GSEA output of its own when
# it's the only SCLC-reference subtype present, or is folded into the "vs SCLC"
# comparators for others when it isn't).
NONNE_FILES = ["gsea_hallmarks_NonNE_SCLC.csv", "gsea_hallmarks_EMT.csv"]
NONNE_CELL_TYPES = ["NonNE SCLC", "EMT"]  # cell_type names as they appear in comparators.csv
LUAD_FILES = ["gsea_hallmarks_LUAD.csv"]

GSEA_FDR_CUTOFF = 0.25
NONNE_MIN_FRACTION = 2 / 3
LUAD_MIN_FRACTION = 1 / 3

NONNE_COLOR = "tab:purple"
LUAD_COLOR = "gold"

# Short forms for Hallmark terms, used for in-figure labels (full names still used
# in the CSV/console output). Falls back to a truncated clean name if not listed.
ABBREV = {
    "TNF-alpha Signaling via NF-kB": "TNFα / NF-κB",
    "IL-6/JAK/STAT3 Signaling": "IL-6 / JAK-STAT3",
    "IL-2/STAT5 Signaling": "IL-2 / STAT5",
    "Interferon Gamma Response": "IFN-γ",
    "Interferon Alpha Response": "IFN-α",
    "Epithelial Mesenchymal Transition": "EMT",
    "Reactive Oxygen Species Pathway": "ROS Pathway",
    "Cholesterol Homeostasis": "Cholesterol",
    "Estrogen Response Early": "Estrogen (early)",
    "Estrogen Response Late": "Estrogen (late)",
    "UV Response Dn": "UV Response ↓",
    "UV Response Up": "UV Response ↑",
    "TGF-beta Signaling": "TGF-β",
    "p53 Pathway": "p53",
    "PI3K/AKT/mTOR  Signaling": "PI3K/AKT/mTOR",
    "Xenobiotic Metabolism": "Xenobiotic Metab.",
    "Fatty Acid Metabolism": "Fatty Acid Metab.",
    "Bile Acid Metabolism": "Bile Acid Metab.",
    "heme Metabolism": "Heme Metabolism",
    "KRAS Signaling Up": "KRAS Up",
    "KRAS Signaling Dn": "KRAS Down",
    "Myc Targets V1": "Myc Targets V1",
    "Myc Targets V2": "Myc Targets V2",
    "Inflammatory Response": "Inflammatory",
    "Androgen Response": "Androgen",
    "Notch Signaling": "Notch",
    "Apical Surface": "Apical Surf.",
    "Apical Junction": "Apical Junc.",
    "Allograft Rejection": "Allograft Rej.",
    "Unfolded Protein Response": "Unfolded Prot.",
}


def load_group(patient_dirs, filenames):
    """Return {patient: DataFrame indexed by Term} for patients having one of filenames."""
    per_patient = {}
    for pdir in patient_dirs:
        patient = os.path.basename(pdir.rstrip("/"))
        for fname in filenames:
            fpath = os.path.join(pdir, fname)
            if os.path.exists(fpath):
                per_patient[patient] = pd.read_csv(fpath, index_col=0)
                break  # a patient has at most one NonNE-lineage cell type
    return per_patient


def recurrent_terms(per_patient, fdr_cutoff=GSEA_FDR_CUTOFF, min_fraction=1 / 3):
    """Terms with NES>0 & FDR<cutoff in enough patients.

    Uses the SAME fraction across groups of any size (>= min_fraction of patients,
    rounded up, at least 1) rather than an absolute count — an earlier version used
    "any 1 of 3" for NonNE but "majority (6 of 10)" for LUAD, which are 33% vs 60%
    and NOT comparable: real cross-patient recurrence in LUAD (e.g. a term hit in
    4-5/10 patients) was being excluded from LUAD's "recurrent" set purely because
    it fell short of that inconsistent 60% bar, so it could never show up as
    "shared" even though it visibly recurred across several patients in both groups.

    Also returns, per term, exactly which patients contributed a hit — so every
    count in the summary table is traceable back to named patients rather than
    needing to be re-derived by hand.
    """
    n = len(per_patient)
    threshold = max(1, math.ceil(min_fraction * n))
    hits = pd.Series(0, index=[], dtype=int)
    contributors = {}
    for patient, df in per_patient.items():
        sig = df[(df["NES"] > 0) & (df["FDR q-val"] < fdr_cutoff)].index
        hits = hits.add(pd.Series(1, index=sig), fill_value=0)
        for term in sig:
            contributors.setdefault(term, []).append(patient)
    hits = hits.sort_values(ascending=False)
    recurrent = set(hits[hits >= threshold].index)
    return recurrent, hits, n, threshold, contributors


def print_comparators(patient_dirs, group_per_patient, group_name):
    """Show which comparator each patient's group members were actually tested
    against — should be a pooled SCLC reference in every patient under
    sclc_pool mode, but printed explicitly (not assumed) since a patient with
    only one SCLC-reference subtype still gets a valid but different-looking
    comparator label (e.g. "SCLC pool (SCLC-A)" instead of "SCLC pool
    (SCLC-A+SCLC-N)")."""
    for pdir in patient_dirs:
        patient = os.path.basename(pdir.rstrip("/"))
        if patient not in group_per_patient:
            continue
        comp_path = os.path.join(pdir, "comparators.csv")
        if not os.path.exists(comp_path):
            print(f"  {patient}: comparators.csv not found (stale/wrong-mode step-1 output?)")
            continue
        comp = pd.read_csv(comp_path)
        for ct in NONNE_CELL_TYPES if group_name == "NonNE" else ["LUAD"]:
            row = comp[comp["cell_type"] == ct]
            if not row.empty:
                print(f"  {patient}: {ct} vs {row.iloc[0]['comparator']}")


def main():
    patient_dirs = sorted(
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith("_")
    )
    patient_dirs = [os.path.join(ROOT, d) for d in patient_dirs]

    nonne_per_patient = load_group(patient_dirs, NONNE_FILES)
    luad_per_patient = load_group(patient_dirs, LUAD_FILES)

    print(f"NonNE-lineage patients ({len(nonne_per_patient)}): {sorted(nonne_per_patient)}")
    print(f"LUAD patients ({len(luad_per_patient)}): {sorted(luad_per_patient)}")

    print("\nComparator actually used per patient (should be a pooled SCLC reference throughout):")
    print_comparators(patient_dirs, nonne_per_patient, "NonNE")
    print_comparators(patient_dirs, luad_per_patient, "LUAD")

    nonne_set, nonne_hits, n_nonne, maj_nonne, nonne_contrib = recurrent_terms(
        nonne_per_patient, min_fraction=NONNE_MIN_FRACTION)
    luad_set, luad_hits, n_luad, maj_luad, luad_contrib = recurrent_terms(
        luad_per_patient, min_fraction=LUAD_MIN_FRACTION)

    print(f"\nNonNE: {len(nonne_set)} terms recurrent in >= {maj_nonne}/{n_nonne} patients")
    print(f"LUAD:  {len(luad_set)} terms recurrent in >= {maj_luad}/{n_luad} patients")

    shared = sorted(nonne_set & luad_set)
    nonne_only = sorted(nonne_set - luad_set)
    luad_only = sorted(luad_set - nonne_set)

    def clean(term):
        return term.replace("HALLMARK_", "").replace("_", " ").title()

    def abbrev(term):
        return ABBREV.get(term, clean(term))

    # ── Summary table — every Hallmark term GSEA was run on, not just the
    #    recurrent ones, so any pathway (including ones significant in 0
    #    patients of either group) can be audited, not only ones that cleared
    #    a threshold.
    any_df = next(iter(nonne_per_patient.values()))
    all_terms = sorted(any_df.index)
    rows = []
    for term in all_terms:
        if term in shared:
            category = "Shared"
        elif term in nonne_only:
            category = "NonNE-only"
        elif term in luad_only:
            category = "LUAD-only"
        else:
            category = "Not recurrent in either group"
        rows.append({
            "Term": term,
            "Category": category,
            "NonNE_n_sig": int(nonne_hits.get(term, 0)),
            "NonNE_n_total": n_nonne,
            "NonNE_patients": ";".join(sorted(nonne_contrib.get(term, []))),
            "LUAD_n_sig": int(luad_hits.get(term, 0)),
            "LUAD_n_total": n_luad,
            "LUAD_patients": ";".join(sorted(luad_contrib.get(term, []))),
        })
    summary = pd.DataFrame(rows).sort_values(["Category", "Term"])
    summary.to_csv(f"{OUTDIR}/nonne_luad_hallmark_summary.csv", index=False)
    print(f"Saved: {OUTDIR}/nonne_luad_hallmark_summary.csv")

    fig, ax = plt.subplots(figsize=(5, 5))
    v = venn2(
        [nonne_set, luad_set],
        set_labels=(f"NonNE (n={n_nonne})", f"LUAD (n={n_luad})"),
        set_colors=(NONNE_COLOR, LUAD_COLOR),
        alpha=0.35,
        ax=ax,
    )
    for text in v.set_labels:
        if text is not None:
            text.set_fontsize(19)
            text.set_fontweight("bold")

    # venn2's automatic overlap color is a blend of NONNE_COLOR and LUAD_COLOR
    # (comes out as a washed-out brown at alpha=0.35), which no longer matches
    # either legend color. Override each patch's fill explicitly instead of
    # relying on that blend, so "10"/"11" both read as solid NONNE_COLOR (no
    # NonNE-only set exists here, so the overlap IS the NonNE region) and "01"
    # reads as solid LUAD_COLOR.
    for region_id, color in (("10", NONNE_COLOR), ("11", NONNE_COLOR), ("01", LUAD_COLOR)):
        patch = v.get_patch_by_id(region_id)
        if patch is not None:
            patch.set_color(color)
            patch.set_alpha(0.55)

    # Text colors are chosen for contrast against each region's fill (set above),
    # not to match it — matching text/background was unreadable. White reads
    # clearly on the solid NONNE_COLOR (purple) fill; a dark near-black brown
    # reads clearly on the solid LUAD_COLOR (gold) fill.
    region_count_color = {"10": "#3d2f00", "11": "#3d2f00", "01": "#3d2f00"}

    def place_columns(x, y, header, names, colors, fontsize=7.5, line_h=0.062,
                       col_w=0.62, max_rows=11, header_color="black", ha="center"):
        """Lay a (optionally multi-column) list of short strings around (x, y).

        ha controls both the text alignment and which side of x the block grows
        from — use "right" to anchor a block's right edge at x (so it grows
        leftward, away from a neighboring region) instead of straddling x.
        """
        ncols = max(1, math.ceil(len(names) / max_rows))
        rows = math.ceil(len(names) / ncols)
        if header:
            ax.text(x, y + (rows / 2) * line_h + line_h * 0.9, header, ha=ha,
                    va="center", fontsize=fontsize + 1, fontweight="bold", color=header_color)
        for i, name in enumerate(names):
            col = i // rows
            row = i % rows
            cx = x + (col - (ncols - 1) / 2) * col_w
            cy = y + (rows / 2 - row - 0.5) * line_h
            ax.text(cx, cy, name, ha=ha, va="center",
                    fontsize=fontsize, fontweight="bold", color=colors[i])

    # NonNE-only.
    patch10 = v.get_label_by_id("10")
    if patch10 is not None:
        if not nonne_only:
            patch10.set_text("0")
            patch10.set_fontsize(18)
            patch10.set_fontweight("bold")
        else:
            patch10.set_text("")
            cx, cy = patch10.get_position()
            names = [abbrev(t) for t in nonne_only]
            colors = [region_count_color["10"]] * len(names)
            # Right-anchored so the block grows leftward, away from the overlap
            # region instead of straddling the centroid toward it.
            place_columns(cx + 0.05, cy, f"{len(nonne_only)} NonNE-only", names, colors,
                          fontsize=9.5, line_h=0.078, header_color=region_count_color["10"],
                          ha="right")

    # Overlap: co-divergent terms (both diverge from SCLC the same way — see
    # the module docstring for why this is NOT the same as NonNE resembling
    # LUAD directly; that's 2_direct_nonne_vs_luad.py's job).
    patch11 = v.get_label_by_id("11")
    if patch11 is not None and shared:
        patch11.set_text("")
        cx, cy = patch11.get_position()
        names = [abbrev(t) for t in shared]
        colors = [region_count_color["11"]] * len(names)
        place_columns(cx, cy, f"{len(shared)} co-divergent", names, colors,
                      fontsize=9.5, line_h=0.078, header_color=region_count_color["11"])

    # LUAD-only: placed inside the LUAD ring as a compact multi-column list.
    # Shifted right (away from the overlap region) for extra clearance from the
    # co-divergent block's text, which sits just to its left.
    patch01 = v.get_label_by_id("01")
    if patch01 is not None and luad_only:
        patch01.set_text("")
        cx, cy = patch01.get_position()
        names = [abbrev(t) for t in luad_only]
        colors = [region_count_color["01"]] * len(names)
        place_columns(cx + 0.1, cy, f"{len(luad_only)} LUAD-only", names, colors,
                      fontsize=8, line_h=0.062, col_w=0.44, max_rows=10,
                      header_color=region_count_color["01"])

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/nonne_luad_hallmark_venn.pdf", bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/nonne_luad_hallmark_venn.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR}/nonne_luad_hallmark_venn.pdf")

    print(f"\n=== Co-divergent from SCLC (NOT nec. similar to each other) — {len(shared)} ===")
    for t in shared:
        print(f"  {clean(t)}")
    print(f"\n=== NonNE-only — {len(nonne_only)} ===")
    for t in nonne_only:
        print(f"  {clean(t)}")
    print(f"\n=== LUAD-only — {len(luad_only)} ===")
    for t in luad_only:
        print(f"  {clean(t)}")


if __name__ == "__main__":
    main()
