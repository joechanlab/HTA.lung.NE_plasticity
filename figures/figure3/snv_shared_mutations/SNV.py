#!/usersoftware/chanj3/scdebussy-py311/bin/python3
"""UpSet plot of SNV overlaps between tumor cell types per patient.

Intersection bars are annotated with any driver-gene mutations found in
that group (EGFR, TP53, RB1, KRAS, STK11, KEAP1, PI3K/AKT pathway, etc.).
Bars containing driver genes are coloured gold; others remain dark grey.
"""

import os
import re
import glob
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import anndata as ad
import scanpy as sc

# ── Driver genes to highlight ─────────────────────────────────────────────────
DRIVER_GENES = {
    "EGFR", "TP53", "RB1", "KRAS", "STK11", "KEAP1",
    "CDKN2A", "CDKN2B", "MYC", "MYCN", "MYCL",
    "AKT1", "AKT2", "PIK3CA", "PIK3R2", "PTEN", "RICTOR", "AXIN1",
    "ASCL1", "NEUROD1",
    "BRAF", "NF1", "MET", "RET", "FGFR1", "ALK", "ROS1",
    "NOTCH1", "NOTCH2", "SMAD4", "ARID1A", "SMARCA4",
    "CREBBP", "EP300", "MDM2", "MDM4", "CDK4", "CDK6", "CCND1",
}

# Variant annotations that directly affect protein sequence or splicing.
# Any annotation NOT represented here (UTR, intron, synonymous, non-coding,
# intergenic, NMD transcript, etc.) is excluded from driver-gene labels.
PROTEIN_AFFECTING_ANNOT = {
    "missense_variant",
    "stop_gained",
    "frameshift_variant",
    "splice_donor_variant",
    "splice_region_variant",
    "inframe_insertion",
    "inframe_deletion",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
H5AD_PATH = (
    "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/"
    "scvi_latent150/adata.scvi.all_genes.final_subset.051826.h5ad"
)
GENOTYPE_DIR = (
    "/data1/chanj3/HTA.lung.NE_plasticity.120122/"
    "MSK_IMPACT/out.genotype.120122"
)
ANNOT_PATH = (
    "/data1/chanj3/HTA.lung.NE_plasticity.120122/"
    "MSK_IMPACT/HTA.MSK_IMPACT.nonsignedout.hg38.120122.txt"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Palette & category order (matches R pal_annot) ───────────────────────────
PALETTE = {
    "LUAD":                 "#ffd700",
    "LUSC":                 "#FFA500",
    "SCLC-A":               "#d62728",
    "SCLC-N":               "#17becf",
    "SCLC-P":               "#1f77b4",
    "NonNE SCLC":           "#4B0082",
    "EMT":                  "#673147",
    "FOXI1+ NonNE SCLC-N":  "#2ca02c",
    "SCLC-N (FOXI1+)":      "#2ca02c",
}
TYPE_ORDER = list(PALETTE.keys())   # canonical display order


# ── Load barcode → (patient, cell_type) map ───────────────────────────────────
print("Loading h5ad annotations …")
adata = ad.read_h5ad(H5AD_PATH, backed="r")

# Patient-specific cell-type corrections (must match violin/report scripts)
PATIENT_CT_REMAP = {
    "RU1518": {"FOXI1+ NonNE SCLC-N": "SCLC-N (FOXI1+)"},
}

seqc_to_meta: dict[str, tuple[str, str]] = {}
for bc, patient, ct in zip(
    adata.obs_names,
    adata.obs["patient"],
    adata.obs["cell_type_final2"],
):
    ct = PATIENT_CT_REMAP.get(str(patient), {}).get(str(ct), str(ct))
    seqc_to_meta[bc.rsplit("_", 1)[-1]] = (patient, ct)

print(f"  {len(seqc_to_meta):,} barcodes loaded")


# ── Discover genotype files ───────────────────────────────────────────────────
genotype_files = glob.glob(
    os.path.join(GENOTYPE_DIR, "*", "*",
                 "HTA.nonsignedout.*.genotype_info.060124.txt")
)
print(f"Found {len(genotype_files)} genotype files")

patient_files: dict[str, list[str]] = defaultdict(list)
for f in genotype_files:
    patient_files[f.split(os.sep)[-3]].append(f)


# ── Variant annotation lookup ─────────────────────────────────────────────────
def load_variant_annotations(annot_path: str) -> dict:
    """
    Parse the MSK-IMPACT annotation file.

    Returns {dmp_id: {(gene, pos_int): variant_annotation_str}}.
    Columns (0-indexed): 0=gene, 5=start_pos, 8=variant_annotation, 16=dmp_id.
    """
    lookup: dict[str, dict] = defaultdict(dict)
    with open(annot_path) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 17:
                continue
            gene   = cols[0]
            annot  = cols[8].strip()
            dmp_id = cols[16].strip()
            try:
                pos = int(cols[5])
            except ValueError:
                continue
            if dmp_id and annot:
                lookup[dmp_id][(gene, pos)] = annot
    return dict(lookup)


print("Loading variant annotations …")
DMP_ANNOTS = load_variant_annotations(ANNOT_PATH)
print(f"  {sum(len(v) for v in DMP_ANNOTS.values()):,} annotations for "
      f"{len(DMP_ANNOTS):,} DMP samples")


# ── Parsing ───────────────────────────────────────────────────────────────────
def _parse_genotype_file(fpath: str):
    """
    Yields (mut_id, gene, protein_change, variant_annot, seqc_bcs) per locus.

    Columns (0-indexed):
      0=gene, 1=pos, 2=ref, 3=major, 4=minor, 5=protein_change,
      6=read_count, 7=SNV_calls, 8=barcode_list (cellID:UMI,...)

    variant_annot is looked up from the MSK-IMPACT annotation file using the
    DMP ID embedded in the filename (e.g. HTA.nonsignedout.P-XXXX.genotype_info…).
    """
    dmp_match = re.search(r'(P-\d+-T\d+-IM\d+)', os.path.basename(fpath))
    patient_annots = DMP_ANNOTS.get(dmp_match.group(1), {}) if dmp_match else {}

    with open(fpath) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or not cols[8].strip():
                continue
            gene           = cols[0]
            protein_change = cols[5].strip() if len(cols) > 5 else ""
            mut_id         = f"{gene}_{cols[1]}_{cols[4]}"
            seqc_bcs       = [e.split(":")[0] for e in cols[8].split(",")]
            try:
                pos = int(cols[1])
            except ValueError:
                pos = None
            annot = patient_annots.get((gene, pos), "") if pos is not None else ""
            yield mut_id, gene, protein_change, annot, seqc_bcs


def get_mutation_sets(patient: str, files: list[str]) -> tuple[dict, dict, dict, dict]:
    """
    Returns:
      ct_muts      : {cell_type -> set(mut_id)}
      mut_genes    : {mut_id -> gene_symbol}
      mut_proteins : {mut_id -> protein_change string}
      mut_annots   : {mut_id -> variant_annotation string}
    """
    ct_muts:      dict[str, set[str]] = defaultdict(set)
    mut_genes:    dict[str, str]      = {}
    mut_proteins: dict[str, str]      = {}
    mut_annots:   dict[str, str]      = {}
    for fpath in files:
        for mut_id, gene, pchange, annot, seqc_bcs in _parse_genotype_file(fpath):
            if mut_id not in mut_genes:
                mut_genes[mut_id]    = gene
                mut_proteins[mut_id] = pchange
                mut_annots[mut_id]   = annot
            for seqc in seqc_bcs:
                meta = seqc_to_meta.get(seqc)
                if meta and meta[0] == patient and meta[1] in PALETTE:
                    ct_muts[meta[1]].add(mut_id)
    return dict(ct_muts), mut_genes, mut_proteins, mut_annots


# ── Custom UpSet plot ─────────────────────────────────────────────────────────
def upset_plot(
    ct_muts: dict[str, set[str]],
    patient: str,
    mut_genes: dict[str, str],
    mut_proteins: dict[str, str] | None = None,
    mut_annots: dict[str, str] | None = None,
) -> plt.Figure:
    """
    UpSet plot with per-category colors and driver-gene annotations.

    Intersections sorted by degree desc, then count desc.
    Driver-gene bars are coloured gold; a dedicated annotation strip below
    the dot matrix lists each intersection's driver genes with protein changes.
    """
    if mut_proteins is None:
        mut_proteins = {}
    if mut_annots is None:
        mut_annots = {}

    types = [t for t in TYPE_ORDER if t in ct_muts]
    n_sets = len(types)

    # ── Build intersection counts & per-intersection mutation sets ────
    all_muts: set[str] = set().union(*[ct_muts[t] for t in types])
    membership = {
        mut: frozenset(t for t in types if mut in ct_muts[t])
        for mut in all_muts
    }
    inter_counts: Counter = Counter(membership.values())

    inter_muts: dict[frozenset, list[str]] = defaultdict(list)
    for mut, inter_key in membership.items():
        inter_muts[inter_key].append(mut)

    sorted_inters = sorted(
        inter_counts.items(),
        key=lambda x: (-len(x[0]), -x[1]),
    )
    inter_sets   = [s for s, _ in sorted_inters]
    inter_values = [c for _, c in sorted_inters]
    n_inters = len(sorted_inters)

    # Driver gene labels per intersection: "GENE p.Change" (one per line)
    def driver_labels(s: frozenset) -> list[str]:
        seen = {}
        for m in inter_muts[s]:
            g = mut_genes.get(m, "")
            if g not in DRIVER_GENES:
                continue
            # Require at least one protein-affecting annotation term.
            # Compound annotations (e.g. "splice_region_variant,intron_variant")
            # are split on comma so the splice_region part still passes.
            # If annot is empty (not found in annotation file) we fall through
            # to the protein-change check below.
            annot = mut_annots.get(m, "")
            if annot and not any(
                a.strip() in PROTEIN_AFFECTING_ANNOT for a in annot.split(",")
            ):
                continue
            pc = mut_proteins.get(m, "").strip()
            # Exclude synonymous mutations (HGVS protein notation ends with "=")
            if pc.endswith("="):
                continue
            label = f"{g} {pc}".strip() if pc else g
            seen[g] = label   # deduplicate by gene, keep first protein change seen
        return sorted(seen.values())

    inter_driver_labels: list[list[str]] = [driver_labels(s) for s in inter_sets]

    # Max lines in annotation strip (cap at 5 to bound height)
    max_annot_lines = min(5, max((len(d) for d in inter_driver_labels), default=0))
    has_any_driver  = any(inter_driver_labels)

    # ── Figure layout ─────────────────────────────────────────────
    col_w   = max(0.50, 4.8 / n_inters)
    set_w   = 1.0
    label_w = 2.2
    bar_h   = 2.2
    mat_h   = n_sets * 0.65
    annot_h = max_annot_lines * 0.30 + 0.25 if has_any_driver else 0.0

    mat_w = n_inters * col_w
    fig_w = set_w + label_w + mat_w + 0.4
    fig_h = bar_h + mat_h + annot_h + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))

    n_rows       = 3 if has_any_driver else 2
    height_ratios = [bar_h, mat_h, annot_h] if has_any_driver else [bar_h, mat_h]

    gs = gridspec.GridSpec(
        n_rows, 3,
        width_ratios=[set_w, label_w, mat_w],
        height_ratios=height_ratios,
        hspace=0.18,
        wspace=0.0,
        left=0.04, right=0.97, top=0.89, bottom=0.05,
    )

    ax_tl     = fig.add_subplot(gs[0, 0])
    ax_tm     = fig.add_subplot(gs[0, 1])
    ax_bars   = fig.add_subplot(gs[0, 2])
    ax_sets   = fig.add_subplot(gs[1, 0])
    ax_labels = fig.add_subplot(gs[1, 1])
    ax_matrix = fig.add_subplot(gs[1, 2])
    ax_tl.axis("off")
    ax_tm.axis("off")

    # ── Intersection bars ─────────────────────────────────────────
    bar_colors = ["#c8a400" if d else "#444444" for d in inter_driver_labels]
    ax_bars.bar(range(n_inters), inter_values, color=bar_colors, width=0.6, zorder=2)

    y_max = max(inter_values, default=1)
    ax_bars.set_ylim(0, y_max * 1.25)

    for i, (v, dlabels) in enumerate(zip(inter_values, inter_driver_labels)):
        ax_bars.text(i, v + 0.015 * y_max, str(v),
                     ha="center", va="bottom", fontsize=16,
                     color="#c8a400" if dlabels else "#444444",
                     fontweight="bold" if dlabels else "normal")

    ax_bars.set_xlim(-0.5, n_inters - 0.5)
    ax_bars.set_ylabel("Mutation count", fontsize=16)
    ax_bars.tick_params(axis="y", labelsize=15)
    ax_bars.set_xticks([])
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)

    # ── Cell-type label panel ─────────────────────────────────────
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(-0.5, n_sets - 0.5)
    ax_labels.axis("off")
    for i, t in enumerate(types):
        if i % 2 == 0:
            ax_labels.axhspan(i - 0.5, i + 0.5, color="#f7f7f7", zorder=0)
        ax_labels.text(0.95, i, t,
                       ha="right", va="center",
                       fontsize=15, fontweight="bold",
                       color=PALETTE[t])

    # ── Dot matrix ───────────────────────────────────────────────
    for i in range(n_sets):
        if i % 2 == 0:
            ax_matrix.axhspan(i - 0.5, i + 0.5, color="#f7f7f7", zorder=0)

    dot_size = max(90, int(550 / n_sets))
    for col_idx, inter_set in enumerate(inter_sets):
        present_rows = [i for i, t in enumerate(types) if t in inter_set]
        if len(present_rows) > 1:
            ax_matrix.plot([col_idx, col_idx],
                           [min(present_rows), max(present_rows)],
                           color="#333333", linewidth=2.0, zorder=1,
                           solid_capstyle="round")
        for row_idx, t in enumerate(types):
            if t in inter_set:
                ax_matrix.scatter(col_idx, row_idx, s=dot_size,
                                  color=PALETTE[t], zorder=2, linewidths=0)
            else:
                ax_matrix.scatter(col_idx, row_idx, s=dot_size * 0.5,
                                  color="#d8d8d8", zorder=2, linewidths=0)

    ax_matrix.set_xlim(-0.5, n_inters - 0.5)
    ax_matrix.set_ylim(-0.5, n_sets - 0.5)
    ax_matrix.set_yticks([])
    ax_matrix.set_xticks([])
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    # ── Set-size bars ─────────────────────────────────────────────
    set_sizes = [len(ct_muts[t]) for t in types]
    for i, (t, sz) in enumerate(zip(types, set_sizes)):
        ax_sets.barh(i, sz, color=PALETTE[t], height=0.6,
                     edgecolor="white", linewidth=0.5, zorder=2)
    ax_sets.set_ylim(-0.5, n_sets - 0.5)
    ax_sets.set_yticks([])
    ax_sets.invert_xaxis()
    ax_sets.set_xlabel("Set size", fontsize=16)
    ax_sets.tick_params(axis="x", labelsize=15)
    for spine in ["top", "right", "left"]:
        ax_sets.spines[spine].set_visible(False)
    for i in range(n_sets):
        if i % 2 == 0:
            ax_sets.axhspan(i - 0.5, i + 0.5, color="#f7f7f7", zorder=0)

    # ── Driver gene annotation strip (below matrix) ───────────────
    if has_any_driver:
        ax_annot_l = fig.add_subplot(gs[2, 0])  # blank (left of annot)
        ax_annot_m = fig.add_subplot(gs[2, 1])  # blank (mid of annot)
        ax_annot   = fig.add_subplot(gs[2, 2])  # driver gene text
        ax_annot_l.axis("off")
        ax_annot_m.axis("off")

        ax_annot.set_xlim(-1.0, n_inters - 0.5)   # extra left padding so col-0 label clears the header
        ax_annot.set_ylim(0, 1)
        ax_annot.axis("off")

        # Divider line between matrix and annotation strip
        ax_annot.plot([0, 1], [1.0, 1.0], color="#cccccc", lw=0.8,
                      transform=ax_annot.transAxes, clip_on=False)

        for col_idx, dlabels in enumerate(inter_driver_labels):
            if not dlabels:
                continue
            # Stack labels top-to-bottom inside the strip
            n_lbl = len(dlabels[:5])
            for li, lbl in enumerate(dlabels[:5]):
                y_pos = 1.0 - (li + 0.5) / n_lbl
                ax_annot.text(
                    col_idx, y_pos, lbl,
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="#8B6914",
                    clip_on=True,
                )

        # Strip header on the label column
        ax_annot_m.set_xlim(0, 1)
        ax_annot_m.set_ylim(0, 1)
        ax_annot_m.axis("off")
        ax_annot_m.text(0.75, 0.5, "Driver genes\n detected",
                        ha="right", va="center",
                        fontsize=14, color="#8B6914",
                        fontstyle="italic")

    return fig


# ── Private-SNV UMAP helpers ──────────────────────────────────────────────────
INDIV_H5AD_DIR = (
    "/data1/chanj3/HTA.lung.NE_plasticity.120122/"
    "fresh.SCLC_NSCLC.062124/individual.021626"
)
EMBEDDING_KEY = "X_draw_graph_slalom_knn_15"
FOXI1_TYPE    = "SCLC-N (FOXI1+)"


def _private_genes_per_ct(ct_muts, mut_genes):
    """Return {cell_type: [gene, ...]} for mutations exclusive to each cell type."""
    types = list(ct_muts.keys())
    result = {}
    for ct in types:
        others = set().union(*[ct_muts[t] for t in types if t != ct])
        priv = ct_muts[ct] - others
        if priv:
            result[ct] = sorted({mut_genes[m] for m in priv})
    return result


def _annotate_cell_types(pat_adata, patient):
    """Add 'cell_type' to pat_adata.obs via the global seqc_to_meta barcode map.
    Patient is checked to avoid collisions where different patients share the same
    integer barcode key."""
    ct_vals = []
    for name in pat_adata.obs_names:
        key = name.rsplit("_", 1)[-1]
        meta = seqc_to_meta.get(key)
        ct_vals.append(meta[1] if (meta and str(meta[0]) == str(patient)) else "Unknown")
    pat_adata.obs["cell_type"] = ct_vals


def plot_private_gene_umaps(patient, ct_muts, mut_genes, out_dir):
    """
    Dotplot of private-mutation gene expression across cell types.

    Genes are grouped by the cell type they are private to (var_names groups).
    Expression is standard-scaled per gene (0–1 across cell types) so it is
    immediately visible whether the cell type that "owns" the private mutations
    simply expresses those genes at higher levels.

    Only runs when FOXI1+ SCLC-N is among the cell types.
    """
    if FOXI1_TYPE not in ct_muts:
        return

    private = _private_genes_per_ct(ct_muts, mut_genes)
    if not private:
        return

    # Find individual patient h5ad (most-recently dated file)
    candidates = sorted(glob.glob(
        os.path.join(INDIV_H5AD_DIR, f"{patient}*.no_cc.hvg_2000.*.h5ad")
    ))
    if not candidates:
        print(f"  [expr] No individual h5ad for {patient}, skipping")
        return

    h5ad_path = candidates[-1]
    print(f"  [expr] Loading {os.path.basename(h5ad_path)} …")
    pat_adata = sc.read_h5ad(h5ad_path)

    _annotate_cell_types(pat_adata, patient)
    pat_adata = pat_adata[pat_adata.obs["cell_type"].isin(PALETTE)].copy()

    # Build var_names grouped by private cell type (canonical order)
    var_names = {}
    for ct in TYPE_ORDER:
        if ct not in private:
            continue
        genes_in_var = [g for g in private[ct] if g in pat_adata.var_names]
        if genes_in_var:
            var_names[ct] = genes_in_var
            print(f"  [expr] {ct}: {len(genes_in_var)}/{len(private[ct])} private genes in var")

    if not var_names:
        print(f"  [expr] No private genes in var, skipping")
        return

    ct_order  = [t for t in TYPE_ORDER if t in pat_adata.obs["cell_type"].values]
    n_genes   = sum(len(g) for g in var_names.values())

    dp = sc.pl.dotplot(
        pat_adata,
        var_names=var_names,
        groupby="cell_type",
        categories_order=ct_order,
        standard_scale="var",
        show=False, return_fig=True,
        dot_max=0.8,
        figsize=(max(4, n_genes * 0.55), len(ct_order) * 0.55 + 1.5),
    )
    dp.style(dot_edge_color="black", dot_edge_lw=0.3, cmap="Reds")

    # Color y-tick labels by cell type
    main_ax = dp.get_axes()["mainplot_ax"]
    for lbl in main_ax.get_yticklabels():
        ct = lbl.get_text()
        lbl.set_color(PALETTE.get(ct, "black"))
        lbl.set_fontweight("bold")

    out_path = os.path.join(out_dir, f"{patient}_private_SNV_expression.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

for patient in sorted(patient_files.keys()):
    print(f"\n{patient} …")
    ct_muts, mut_genes, mut_proteins, mut_annots = get_mutation_sets(
        patient, patient_files[patient])

    if not ct_muts:
        print("  no tumor mutations found, skipping")
        continue

    for ct, muts in sorted(ct_muts.items()):
        print(f"  {ct}: {len(muts)} mutations")

    types = [t for t in TYPE_ORDER if t in ct_muts]
    if len(types) < 2:
        print("  < 2 tumor cell types, skipping")
        continue

    fig = upset_plot(ct_muts, patient, mut_genes, mut_proteins, mut_annots)
    outpath = os.path.join(OUT_DIR, f"{patient}_SNV_venn.pdf")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {outpath}")

    plot_private_gene_umaps(patient, ct_muts, mut_genes, OUT_DIR)

print("\nDone.")
