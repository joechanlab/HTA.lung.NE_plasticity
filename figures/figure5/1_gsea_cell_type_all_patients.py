"""
Per-patient DEG (one-vs-nearest-neighbor Wilcoxon) + GSEA (MSigDB Hallmarks)
analysis across cell types, generalized from gsea_cell_type_RU1444.py to run over
the full patient cohort. Mitochondrial (MT-) and ribosomal protein
(RPS/RPL/MRPS/MRPL) genes are excluded from the expression matrix before DEG/GSEA,
so they can never appear as markers or drive pathway enrichment.

DEG DESIGN (changed 2026-07-21 from one-vs-rest): each cell type is compared only
to its single most transcriptionally similar OTHER cell type present in that
patient — not to the pooled "rest" — because one-vs-rest folds together
populations that may be nothing alike (e.g. LUAD and SCLC-A), which dilutes and
distorts what "distinguishes" a cell type. The nearest neighbor is determined
per-patient (it can differ across patients) via Pearson correlation of pseudobulk
mean expression on that patient's own HVGs (compute_nearest_neighbors()) — a
purely data-driven choice, not a hardcoded biological assumption. One consequence:
the "vs SCLC" framing used in downstream scripts (nonne_luad_hallmark_venn.py,
direct_nonne_vs_luad.py) no longer holds uniformly, since a cell type's nearest
neighbor need not be an SCLC subtype — those scripts will need re-auditing against
this design before being trusted again.

BALANCING: cell counts are NOT downsampled/balanced before the Wilcoxon test.
Wilcoxon rank-sum is valid for unequal n (its variance formula already accounts
for both group sizes) — the real risk in scRNA-seq DE is pseudoreplication (cells
within a patient/cell-type aren't independent draws), which balancing group sizes
does not fix, since both groups remain internally non-independent either way.
Genuine robustness should come from combining p-values ACROSS PATIENTS (true
biological replicates) rather than from within-patient downsampling — see the
sample-size-weighted Stouffer approach already used in direct_nonne_vs_luad.py.

FDR CONVENTIONS (two different thresholds, deliberately):
  - DEG (gene-level, DEG_FDR_CUTOFF): 0.05 — standard single-gene convention.
    Applied ONLY to the reported/plotted marker gene list, never to the ranked
    list handed to GSEA prerank (pre-filtering genes by significance before GSEA
    would bias the enrichment test, which must run on the full ranked list).
    Previously there was NO gene-level FDR filter at all for markers (just top-30
    by rank + logFC>0) — that was too relaxed; fixed here.
  - GSEA (gene-set-level, GSEA_FDR_CUTOFF): 0.25 — this is the field-standard
    threshold recommended by the original GSEA paper (Subramanian et al. 2005),
    used because gene sets are large, correlated/overlapping blocks with
    inherently less testing power than single genes at a given sample size. NOT
    too relaxed for GSEA specifically, despite being 5x more lenient than the DEG
    cutoff — the two are different kinds of hypotheses with different established
    conventions, not the same threshold applied inconsistently.

SECOND COMPARATOR MODE (added 2026-07-21, --reference sclc_pool): the
one-vs-nearest-neighbor design above answers "what most distinguishes this cell
type from whatever is easiest to confuse it with" — useful for marker discovery,
but NOT for questions that need a comparison to stay consistent across patients
and cell types, e.g. "how does each cell type co-diverge from SCLC" or "compare
cell types without the comparison itself being skewed by which other cell types
happen to be present in a given patient." For that, --reference sclc_pool uses a
FIXED, biologically-anchored reference instead of a data-driven one: every
non-SCLC cell type (NonNE SCLC, EMT, LUAD, LUSC, FOXI1+ NonNE SCLC-N) is compared
against a pooled SCLC-A+SCLC-N reference group (whichever of the two subtypes are
present in that patient — every patient in this cohort has at least one). SCLC-A
and SCLC-N, when both present, are compared directly against each other instead
of against the pool (they can't be "vs SCLC" since they ARE the SCLC reference).
A patient with only ONE SCLC subtype present (e.g. RU1303: SCLC-A only, no
SCLC-N) has no partner for that lone subtype, so it's used only as the reference
for others and gets no DEG/GSEA output of its own that run. Output goes to
figures/gsea_all_patients_sclc_pool/<patient>/ instead, so neighbor-mode and
sclc_pool-mode outputs never collide and both remain available for their
respective purposes.

For each patient this produces, under figures/gsea_all_patients[_sclc_pool]/<patient>/:
  - hallmark_NES_heatmap.{pdf,png}
  - hallmark_barplots.{pdf,png}
  - top_markers_dotplot.{pdf,png}
  - gsea_hallmarks_<cell_type>.csv  (one per testable cell type present in that patient)
  - comparators.csv  (which cell type each was compared against, and why)

Designed to be run once per patient (e.g. as a SLURM job array element via
1_submit_gsea_all_patients.sh):
    python 1_gsea_cell_type_all_patients.py --patient RU1444 [--reference neighbor|sclc_pool]
or sequentially over the whole cohort:
    python 1_gsea_cell_type_all_patients.py --patient all --reference sclc_pool
"""
import os
import re
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

COMBINED_H5AD = (
    "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124"
    "/combined/adata.SCLC_NSCLC.051226.h5ad"
)
# RU151's cell typing in the combined object is stale; use its own, newer
# per-patient embedding/cell-type calls instead.
RU151_H5AD = (
    "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124"
    "/individual.021626/RU151.new_embedding.070326.h5ad"
)

PATIENTS = [
    "RU1444", "RU151", "RU942", "RU1083", "RU263", "RU1303",
    "RU1518", "RU581", "RU831", "RU1068", "RU1646", "RU1314",
]

MIN_CELLS = 20  # minimum cells for a cell type to be included in a patient's analysis
DEG_FDR_CUTOFF = 0.05   # gene-level marker-list filter only (never applied before GSEA prerank)
GSEA_FDR_CUTOFF = 0.25  # gene-set-level, per Subramanian et al. 2005 GSEA convention
NN_HVG_COUNT = 2000     # HVGs used to compute per-patient pseudobulk nearest-neighbor similarity

SCLC_REFERENCE_TYPES = ["SCLC-A", "SCLC-N", "FOXI1+ NonNE SCLC-N"]  # pooled as the fixed
# "SCLC" reference in sclc_pool mode. FOXI1+ NonNE SCLC-N groups with SCLC here (despite
# the name) per user clarification — it's a proliferative SCLC-N-adjacent subtype, not
# part of the NonNE phenotype (EMT is the one that groups with NonNE, see
# 3_nonne_luad_hallmark_venn.py's NONNE_CELL_TYPES).
SCLC_POOL_LABEL = "SCLC_POOL"

PALETTE = {
    "SCLC-N":                "#17becf",
    "SCLC-A":                "#d62728",
    "NonNE SCLC":            "tab:purple",
    "FOXI1+ NonNE SCLC-N":   "#2ca02c",
    "EMT":                   "#673147",
    "LUAD":                  "#d4ac0d",
    "LUSC":                  "#8c564b",
}
CELL_TYPE_ORDER = list(PALETTE.keys())

MITO_RIBO_PATTERN = re.compile(r"^(MT-|(MRP|RP)[SL]\d+[A-Z]?$)")
# Ribosomal protein genes whose HGNC names don't follow the RP[SL]<digit> pattern
# (e.g. the acidic phosphoproteins and the RPS30/RACK1 fusion/scaffold genes).
EXTRA_RIBO_GENES = {"RPSA", "RPLP0", "RPLP1", "RPLP2", "FAU", "RACK1", "GNB2L1"}

OUTDIR_ROOT = "figures/gsea_all_patients"


def load_patient(patient):
    """Return an AnnData subset to one patient with 'new_cell_type' set."""
    if patient == "RU151":
        adata = sc.read_h5ad(RU151_H5AD)
        adata.obs["new_cell_type"] = adata.obs["new_cell_type"].astype(str)
        return adata

    adata = sc.read_h5ad(COMBINED_H5AD, backed="r")
    mask = adata.obs["patient"] == patient
    sub = adata[mask].to_memory()
    adata.file.close()
    sub.obs["new_cell_type"] = sub.obs["cell_type_final2"].astype(str)
    return sub


def compute_nearest_neighbors(sub, cell_types, layer, n_top_genes=NN_HVG_COUNT):
    """For each cell type, find the other cell type it is most similar to.

    Similarity = Pearson correlation of pseudobulk (per-cell-type mean) expression
    on HVGs computed within this patient's own cell-type-filtered subset — purely
    data-driven, computed fresh per patient (a cell type's nearest neighbor can
    differ between patients). Returns (neighbor_of: dict[ct -> ct], corr: DataFrame).
    """
    hvg_src = sub.copy()
    sc.pp.highly_variable_genes(
        hvg_src, n_top_genes=min(n_top_genes, hvg_src.n_vars), layer=layer, flavor="seurat"
    )
    hvg_genes = hvg_src.var_names[hvg_src.var["highly_variable"]]

    X = sub[:, hvg_genes].layers[layer] if layer else sub[:, hvg_genes].X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    expr = pd.DataFrame(X, columns=hvg_genes, index=sub.obs_names)
    expr["new_cell_type"] = sub.obs["new_cell_type"].values
    pseudobulk = expr.groupby("new_cell_type", observed=True).mean().loc[cell_types]

    corr = pseudobulk.T.corr(method="pearson")
    neighbor_of = {ct: corr.loc[ct].drop(ct).idxmax() for ct in cell_types}
    return neighbor_of, corr


def compute_sclc_pool_comparators(sub, cell_types):
    """Fixed-reference design: every non-SCLC cell type is compared against a
    pooled SCLC reference (SCLC-A + SCLC-N + FOXI1+ NonNE SCLC-N — whichever of
    those are present in this patient; FOXI1+ groups with SCLC here despite the
    name, it's a proliferative SCLC-N-adjacent subtype, not part of the NonNE
    phenotype). Members of the SCLC reference itself are compared against each
    other instead of against the pool they're part of:
      - exactly 2 present -> direct one-vs-one between them
      - 3+ present         -> each vs the pool of the OTHER SCLC-reference members
                               (its own one-vs-rest within just the SCLC group)
      - only 1 present     -> no partner; used only as the reference for others,
                               gets no DEG/GSEA output of its own this run
    See module docstring for why this (not nearest-neighbor) is the right choice
    for "co-divergence from SCLC" or cross-patient-comparable questions.

    Returns (comparator_of, groupby_col, reference_of, testable_cell_types):
      comparator_of[ct]  -> human-readable label of what ct was compared against
      groupby_col[ct]    -> which sub.obs column to pass as `groupby`
      reference_of[ct]   -> which value in that column to pass as `reference`
      testable_cell_types -> subset of cell_types that actually get DEG/GSEA
                              output this run
    Mutates sub.obs in place to add pooled-reference grouping column(s).
    """
    sclc_present = [ct for ct in SCLC_REFERENCE_TYPES if ct in cell_types]
    non_sclc = [ct for ct in cell_types if ct not in sclc_present]

    comparator_of, groupby_col, reference_of = {}, {}, {}
    testable_cell_types = []

    if sclc_present:
        pool_label_text = "+".join(sclc_present)
        sub.obs["sclc_pool_group"] = sub.obs["new_cell_type"].apply(
            lambda x, present=set(sclc_present): SCLC_POOL_LABEL if x in present else x
        )
        for ct in non_sclc:
            comparator_of[ct] = f"SCLC pool ({pool_label_text})"
            groupby_col[ct] = "sclc_pool_group"
            reference_of[ct] = SCLC_POOL_LABEL
            testable_cell_types.append(ct)

    if len(sclc_present) == 2:
        a, b = sclc_present
        comparator_of[a], reference_of[a], groupby_col[a] = b, b, "new_cell_type"
        comparator_of[b], reference_of[b], groupby_col[b] = a, a, "new_cell_type"
        testable_cell_types += [a, b]
    elif len(sclc_present) >= 3:
        # Each SCLC-reference member vs the pool of the OTHER SCLC-reference
        # members (never occurs in the current cohort, but handled generally).
        for ct in sclc_present:
            others = set(sclc_present) - {ct}
            col = f"sclc_internal_group_{ct}".replace(" ", "_").replace("+", "pos")
            sub.obs[col] = sub.obs["new_cell_type"].apply(
                lambda x, ct=ct, others=others: ct if x == ct else ("__OTHER_SCLC__" if x in others else x)
            )
            comparator_of[ct] = f"other SCLC ({'+'.join(sorted(others))})"
            groupby_col[ct] = col
            reference_of[ct] = "__OTHER_SCLC__"
            testable_cell_types.append(ct)
    elif len(sclc_present) == 1:
        print(f"  Note: only {sclc_present[0]} present (no SCLC-reference partner) — "
              f"used as reference for others, no DEG/GSEA computed for it this run.")

    return comparator_of, groupby_col, reference_of, testable_cell_types


def run_patient(patient, reference_mode="neighbor", outdir_root=OUTDIR_ROOT):
    print(f"\n{'=' * 70}\nPatient: {patient}  (reference={reference_mode})\n{'=' * 70}")
    outdir = f"{outdir_root}/{patient}"
    os.makedirs(outdir, exist_ok=True)

    adata = load_patient(patient)

    counts = adata.obs["new_cell_type"].value_counts()
    cell_types = [ct for ct in CELL_TYPE_ORDER if counts.get(ct, 0) >= MIN_CELLS]
    print(f"Cell counts: {counts[counts > 0].to_dict()}")
    print(f"Cell types included (>= {MIN_CELLS} cells): {cell_types}")

    if len(cell_types) < 2:
        print(f"Skipping {patient}: fewer than 2 qualifying cell types.")
        return

    sub = adata[adata.obs["new_cell_type"].isin(cell_types)].copy()
    sub.obs["new_cell_type"] = sub.obs["new_cell_type"].astype(str)

    # ── Exclude mitochondrial and ribosomal genes ────────────────────────────
    keep_genes = ~(sub.var_names.str.match(MITO_RIBO_PATTERN) | sub.var_names.isin(EXTRA_RIBO_GENES))
    n_excluded = (~keep_genes).sum()
    print(f"Excluding {n_excluded} mitochondrial/ribosomal genes "
          f"(of {sub.n_vars} total).")
    sub = sub[:, keep_genes].copy()

    # ── Ensure log-normalised counts in .X ───────────────────────────────────
    if sub.raw is not None:
        sub_raw = sub.raw.to_adata()
        sub_raw = sub_raw[:, sub_raw.var_names.isin(sub.var_names)].copy()
        sc.pp.normalize_total(sub_raw, target_sum=1e4)
        sc.pp.log1p(sub_raw)
        sub.layers["lognorm"] = sub_raw[:, sub.var_names].X
        use_layer = "lognorm"
    else:
        use_layer = None

    # ── Determine comparator per cell type (mode-dependent) ───────────────────
    if reference_mode == "neighbor":
        print("Computing per-cell-type nearest neighbors (pseudobulk correlation)...")
        neighbor_of, corr_mat = compute_nearest_neighbors(sub, cell_types, layer=use_layer)
        comparator_of = neighbor_of
        groupby_col = {ct: "new_cell_type" for ct in cell_types}
        reference_of = neighbor_of
        testable_cell_types = list(cell_types)
        for ct in cell_types:
            print(f"  {ct}  ->  {neighbor_of[ct]}  (r={corr_mat.loc[ct, neighbor_of[ct]]:.3f})")
        pd.DataFrame({
            "cell_type": cell_types, "comparator": [neighbor_of[ct] for ct in cell_types],
            "method": "nearest_neighbor",
            "pearson_r": [corr_mat.loc[ct, neighbor_of[ct]] for ct in cell_types],
        }).to_csv(f"{outdir}/comparators.csv", index=False)
    elif reference_mode == "sclc_pool":
        print("Computing fixed SCLC-pool comparators...")
        comparator_of, groupby_col, reference_of, testable_cell_types = \
            compute_sclc_pool_comparators(sub, cell_types)
        for ct in testable_cell_types:
            print(f"  {ct}  ->  {comparator_of[ct]}")
        pd.DataFrame({
            "cell_type": testable_cell_types,
            "comparator": [comparator_of[ct] for ct in testable_cell_types],
            "method": "sclc_pool",
        }).to_csv(f"{outdir}/comparators.csv", index=False)
    else:
        raise ValueError(f"Unknown reference_mode: {reference_mode!r}")

    if not testable_cell_types:
        # neighbor mode always yields >=2 (every present cell type gets a
        # neighbor); sclc_pool mode CAN legitimately yield exactly 1 (e.g. one
        # non-SCLC type vs a single available SCLC subtype, no pairwise partner
        # needed since the reference is fixed/external, not one of the tested
        # types) — so the right guard is "at least 1 testable", not "at least 2".
        print(f"Skipping {patient}: no testable cell types under reference_mode={reference_mode}.")
        return
    all_cell_types_in_sub = cell_types  # keep for the dotplot, which needs sub's FULL category list
    cell_types = testable_cell_types  # everything else below only concerns cell types actually tested this run

    # ── DEG: one-vs-comparator Wilcoxon (not one-vs-rest, not balanced — see
    #    module docstring for why) ─────────────────────────────────────────────
    print(f"Running rank_genes_groups (vs {reference_mode} comparator)...")
    ranked_dfs, top_markers = {}, {}
    for ct in cell_types:
        sc.tl.rank_genes_groups(
            sub,
            groupby=groupby_col[ct],
            groups=[ct],
            reference=reference_of[ct],
            method="wilcoxon",
            layer=use_layer,
            use_raw=False,
            n_genes=sub.n_vars,
            key_added="rank_vs_comparator",
        )
        rgg = sc.get.rank_genes_groups_df(sub, group=ct, key="rank_vs_comparator")
        rgg["score"] = rgg["scores"]
        rgg = rgg.sort_values("score", ascending=False).drop_duplicates("names")
        ranked_dfs[ct] = rgg.set_index("names")["score"].dropna()
        sig = rgg[(rgg["logfoldchanges"] > 0) & (rgg["pvals_adj"] < DEG_FDR_CUTOFF)]
        top_markers[ct] = sig.head(30)["names"].tolist()

    # ── GSEA prerank with MSigDB Hallmarks ────────────────────────────────────
    print("Running GSEA prerank (Hallmarks)...")
    gsea_results = {}
    for ct in cell_types:
        print(f"  {ct}")
        rnk = ranked_dfs[ct].reset_index()
        rnk.columns = ["gene", "score"]
        res = gp.prerank(
            rnk=rnk,
            gene_sets="MSigDB_Hallmark_2020",
            outdir=None,
            min_size=10,
            max_size=500,
            permutation_num=1000,
            seed=42,
            verbose=False,
        )
        df = res.res2d.copy()
        if "Term" in df.columns:
            df = df.set_index("Term")
        gsea_results[ct] = df

    # ── Heatmap: NES across pathways for all cell types ──────────────────────
    print("Building NES heatmap...")
    sig_paths = set()
    for df in gsea_results.values():
        sig_paths.update(df[df["FDR q-val"] < GSEA_FDR_CUTOFF].index.tolist())
    if not sig_paths:
        for df in gsea_results.values():
            sig_paths.update(df.head(15).index.tolist())

    nes_mat = pd.DataFrame(
        {ct: gsea_results[ct].reindex(list(sig_paths))["NES"] for ct in cell_types},
        index=list(sig_paths),
    )
    nes_mat = nes_mat.dropna(how="all").fillna(0)
    nes_mat = nes_mat.loc[nes_mat.abs().max(axis=1).sort_values(ascending=False).index]

    xtick_labels = [f"{ct}\n(vs {comparator_of[ct]})" for ct in cell_types]
    mode_desc = "its nearest neighbor" if reference_mode == "neighbor" else "the fixed SCLC-pool reference"

    fig, ax = plt.subplots(figsize=(6, max(4, len(nes_mat) * 0.35)))
    im = ax.imshow(nes_mat.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(nes_mat)))
    ax.set_yticklabels(
        [p.replace("HALLMARK_", "").replace("_", " ").lower().title() for p in nes_mat.index],
        fontsize=7,
    )
    plt.colorbar(im, ax=ax, label="NES", shrink=0.6)
    ax.set_title(f"Hallmark pathway enrichment (NES), each vs {mode_desc}\n{patient}", fontsize=10)
    for tick, ct in zip(ax.get_xticklabels(), cell_types):
        tick.set_color(PALETTE[ct])
        tick.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(f"{outdir}/hallmark_NES_heatmap.pdf", bbox_inches="tight")
    plt.savefig(f"{outdir}/hallmark_NES_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {outdir}/hallmark_NES_heatmap.pdf")

    # ── Top enriched pathways per cell type (bar plots) ───────────────────────
    print("Building per-cell-type enrichment bar plots...")
    fig, axes = plt.subplots(1, len(cell_types), figsize=(4.5 * len(cell_types), 5), squeeze=False)
    axes = axes[0]
    for ax, ct in zip(axes, cell_types):
        df = gsea_results[ct].copy()
        df.index = df.index.astype(str)
        df["label"] = df.index.str.replace("HALLMARK_", "").str.replace("_", " ").str.title()
        sig = df[df["FDR q-val"] < GSEA_FDR_CUTOFF]
        pos = sig[sig["NES"] > 0].sort_values("NES", ascending=False).head(10)
        neg = sig[sig["NES"] < 0].sort_values("NES", ascending=True).head(5)
        plot_df = pd.concat([pos, neg]).sort_values("NES")

        ax.set_title(f"{ct}\n(vs {comparator_of[ct]})", fontsize=10, color=PALETTE[ct], fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if plot_df.empty:
            ax.text(0.5, 0.5, f"No significant pathways\n(FDR < {GSEA_FDR_CUTOFF})",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_xlabel("NES", fontsize=9)
            continue
        colors = [PALETTE[ct] if v > 0 else "lightgray" for v in plot_df["NES"]]
        ax.barh(range(len(plot_df)), plot_df["NES"], color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["label"], fontsize=7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("NES", fontsize=9)

    plt.suptitle(f"Significant Hallmark enrichments (FDR < {GSEA_FDR_CUTOFF}), each cell type vs {mode_desc} — {patient}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{outdir}/hallmark_barplots.pdf", bbox_inches="tight")
    plt.savefig(f"{outdir}/hallmark_barplots.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {outdir}/hallmark_barplots.pdf")

    # ── Top marker genes dotplot ───────────────────────────────────────────────
    print("Building top marker dotplot...")
    shown, final_markers = set(), {}
    for ct in cell_types:
        unique = [g for g in top_markers[ct] if g not in shown][:8]
        final_markers[ct] = unique
        shown.update(unique)
    gene_groups = {ct: v for ct, v in final_markers.items() if v}

    if gene_groups:
        dp = sc.pl.dotplot(
            sub,
            var_names=gene_groups,
            groupby="new_cell_type",
            categories_order=all_cell_types_in_sub,
            standard_scale="var",
            dendrogram=False,
            show=False,
            return_fig=True,
            dot_max=0.8,
            figsize=(2.7 * len(cell_types), 3),
        )
        dp.style(dot_edge_color="black", dot_edge_lw=0.5, cmap="Blues")
        main_ax = dp.get_axes()["mainplot_ax"]
        for lbl in main_ax.get_yticklabels():
            ct = lbl.get_text()
            lbl.set_color(PALETTE.get(ct, "black"))
            lbl.set_fontweight("bold")
        plt.savefig(f"{outdir}/top_markers_dotplot.pdf", bbox_inches="tight")
        plt.savefig(f"{outdir}/top_markers_dotplot.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {outdir}/top_markers_dotplot.pdf")

    # ── Summary + full tables ─────────────────────────────────────────────────
    print(f"\n=== GSEA Summary: Top enriched Hallmarks per cell type ({patient}) ===")
    for ct in cell_types:
        df = gsea_results[ct]
        top = df[df["NES"] > 0].sort_values(["FDR q-val", "NES"], ascending=[True, False]).head(10)
        print(f"\n{ct} (vs {comparator_of[ct]}):")
        for _, row in top.iterrows():
            path = str(row.name).replace("HALLMARK_", "").replace("_", " ")
            print(f"  {path:<45s}  NES={row['NES']:+.2f}  FDR={row['FDR q-val']:.3f}")

    print(f"\n=== Top marker genes per cell type (FDR < {DEG_FDR_CUTOFF}, {patient}) ===")
    for ct in cell_types:
        print(f"\n{ct} (vs {comparator_of[ct]}):  {', '.join(top_markers[ct][:15])}")

    for ct in cell_types:
        safe = ct.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "pos")
        gsea_results[ct].to_csv(f"{outdir}/gsea_hallmarks_{safe}.csv")
        print(f"Saved: {outdir}/gsea_hallmarks_{safe}.csv")

    print(f"\nDone: {patient}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patient", required=True,
        help=f"Patient ID to process, or 'all' to run the full cohort sequentially. "
             f"Choices: {PATIENTS}",
    )
    parser.add_argument(
        "--reference", choices=["neighbor", "sclc_pool"], default="neighbor",
        help="'neighbor' (default): one-vs-nearest-neighbor, data-driven per patient — "
             "best for marker discovery. 'sclc_pool': one-vs-fixed-SCLC-reference "
             "(SCLC-A+SCLC-N+FOXI1+ NonNE SCLC-N pooled) — best for co-divergence-from-SCLC "
             "questions and cross-patient-comparable results. See module docstring.",
    )
    args = parser.parse_args()

    outdir_root = OUTDIR_ROOT if args.reference == "neighbor" else f"{OUTDIR_ROOT}_sclc_pool"
    os.makedirs(outdir_root, exist_ok=True)

    if args.patient == "all":
        for patient in PATIENTS:
            run_patient(patient, reference_mode=args.reference, outdir_root=outdir_root)
    else:
        if args.patient not in PATIENTS:
            sys.exit(f"Unknown patient '{args.patient}'. Choices: {PATIENTS}")
        run_patient(args.patient, reference_mode=args.reference, outdir_root=outdir_root)


if __name__ == "__main__":
    main()
