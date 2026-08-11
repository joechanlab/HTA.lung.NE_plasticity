#!/usersoftware/chanj3/scdebussy-py311/bin/python3
"""
SNV shared-mutation dot plots with VAF, statistical confidence, and driver-gene highlights.

For each patient with ≥2 tumor cell types:
  - Identifies mutations shared across cell types (public/partially-shared)
  - Computes per-cell-type VAF = mutant cells / total cells of that type
  - Wilson 95% CI on VAF; Fisher's exact test between all pairs of cell types
  - Driver gene mutations (EGFR, TP53, RB1, KRAS, STK11, KEAP1, PI3K/AKT pathway…)
    are highlighted with bold red labels and gold row backgrounds

Output (in same directory as this script):
  {patient}_SNV_shared.png  – per-patient dot plot
  SNV_driver_trunk_summary.png  – cross-patient heatmap of trunk driver genes
  SNV_shared_mutations.tsv  – full table of shared mutations with statistics

Column mapping (0-indexed):
  0: gene,  1: position,  2: ref,  3: major,  4: minor,  5: protein_change,
  6: read_count,  7: SNV_calls,  8: barcode_list (cellID:UMI,...)
"""

import os, glob, csv
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import anndata as ad

# ── Paths ─────────────────────────────────────────────────────────────────────
H5AD_PATH = (
    "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/"
    "scvi_latent150/adata.scvi.all_genes.final_subset.051826.h5ad"
)
GENOTYPE_DIR = (
    "/data1/chanj3/HTA.lung.NE_plasticity.120122/"
    "MSK_IMPACT/out.genotype.120122"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Driver genes to highlight ─────────────────────────────────────────────────
# Covers genes mentioned in the paper's genomics section and established lung cancer drivers
DRIVER_GENES = {
    # Core T-SCLC drivers (from paper)
    "EGFR", "TP53", "RB1", "KRAS", "STK11", "KEAP1",
    # Cell-cycle regulators
    "CDKN2A", "CDKN2B", "CDK4", "CDK6", "CCND1",
    # MYC family (NE plasticity)
    "MYC", "MYCN", "MYCL",
    # PI3K/AKT pathway (enriched in T-SCLC per paper)
    "AKT1", "AKT2", "PIK3CA", "PIK3R2", "PTEN", "RICTOR", "AXIN1",
    # NE lineage markers
    "ASCL1", "NEUROD1",
    # Common lung cancer drivers
    "BRAF", "NF1", "MET", "RET", "FGFR1", "ALK", "ROS1",
    "NOTCH1", "NOTCH2", "SMAD4", "ARID1A", "SMARCA4",
    "CREBBP", "EP300", "MDM2", "MDM4",
}

# ── Cell-type order and palette ───────────────────────────────────────────────
PALETTE = {
    "LUAD":                 "#ffd700",
    "LUSC":                 "#FFA500",
    "SCLC-A":               "#d62728",
    "SCLC-N":               "#17becf",
    "SCLC-P":               "#1f77b4",
    "NonNE SCLC":           "#4B0082",
    "EMT":                  "#673147",
    "FOXI1+ NonNE SCLC-N":  "#2ca02c",
}
TYPE_ORDER = list(PALETTE.keys())


# ── Load annotations ──────────────────────────────────────────────────────────
print("Loading h5ad …")
adata = ad.read_h5ad(H5AD_PATH, backed="r")

seqc_to_meta: dict[str, tuple[str, str]] = {}
for bc, patient, ct in zip(
    adata.obs_names,
    adata.obs["patient"],
    adata.obs["cell_type_final2"],
):
    seqc_to_meta[bc.rsplit("_", 1)[-1]] = (patient, ct)

# Total cells per (patient, cell_type) — used as denominator for VAF
total_cells: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
for seqc_id, (pat, ct) in seqc_to_meta.items():
    if ct in PALETTE:
        total_cells[pat][ct] += 1

print(f"  {len(seqc_to_meta):,} barcodes; {len(total_cells)} patients")


# ── Discover genotype files ───────────────────────────────────────────────────
genotype_files = glob.glob(
    os.path.join(GENOTYPE_DIR, "*", "*",
                 "HTA.nonsignedout.*.genotype_info.060124.txt")
)
patient_files: dict[str, list[str]] = defaultdict(list)
for f in genotype_files:
    patient_files[f.split(os.sep)[-3]].append(f)
print(f"Found {len(genotype_files)} genotype files across {len(patient_files)} patients")


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_genotype_file(fpath: str):
    """
    Yields records with gene, protein_change, mut_id, cell_ids.

    Columns (0-indexed):
      0=gene, 1=pos, 2=ref, 3=major, 4=minor, 5=protein_change,
      6=read_count, 7=SNV_calls, 8=barcodes (cellID:UMI,...)
    """
    with open(fpath) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or not cols[8].strip():
                continue
            gene = cols[0]
            pos = cols[1]
            minor = cols[4]
            protein_change = cols[5] if len(cols) > 5 else ""
            mut_id = f"{gene}_{pos}_{minor}"
            cell_ids = [e.split(":")[0] for e in cols[8].split(",")]
            yield gene, protein_change, mut_id, cell_ids


def get_mutation_data(patient: str, files: list[str]):
    """
    Returns:
      mut_info: {mut_id -> {"gene": str, "protein_change": str}}
      ct_mutcells: {mut_id -> {cell_type -> set(cell_ids)}}
    """
    mut_info: dict[str, dict] = {}
    ct_mutcells: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for fpath in files:
        for gene, pchange, mut_id, cell_ids in parse_genotype_file(fpath):
            if mut_id not in mut_info:
                mut_info[mut_id] = {"gene": gene, "protein_change": pchange}
            for cid in cell_ids:
                meta = seqc_to_meta.get(cid)
                if meta and meta[0] == patient and meta[1] in PALETTE:
                    ct_mutcells[mut_id][meta[1]].add(cid)

    return mut_info, dict(ct_mutcells)


# ── Statistics ────────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score 95% CI: returns (vaf, lower, upper)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def pairwise_fisher(k_a: int, n_a: int, k_b: int, n_b: int) -> float:
    """
    Fisher's exact test for equal mutation prevalence between two cell types.
    High p-value → prevalences are not significantly different → confidently shared.
    """
    if n_a == 0 or n_b == 0:
        return float("nan")
    table = [[k_a, n_a - k_a], [k_b, n_b - k_b]]
    _, p = stats.fisher_exact(table)
    return p


def min_pairwise_p(m: str, types: list[str], patient: str,
                   ct_mutcells: dict) -> float:
    """Minimum pairwise Fisher p across all cell-type pairs (most conservative)."""
    pvals = []
    for i, ta in enumerate(types):
        for tb in types[i + 1:]:
            n_ta = total_cells[patient].get(ta, 0)
            k_ta = len(ct_mutcells[m].get(ta, set()))
            n_tb = total_cells[patient].get(tb, 0)
            k_tb = len(ct_mutcells[m].get(tb, set()))
            p = pairwise_fisher(k_ta, n_ta, k_tb, n_tb)
            if not np.isnan(p):
                pvals.append(p)
    return min(pvals) if pvals else float("nan")


def pval_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── Per-patient dot plot ──────────────────────────────────────────────────────
def make_patient_dotplot(
    patient: str,
    mut_info: dict,
    ct_mutcells: dict,
    sharedness: str = "public",   # "public" | "partially" | "all"
) -> plt.Figure | None:
    """
    Dot plot of shared mutations vs cell types, sized by VAF.
    sharedness: which mutations to include.
    """
    types = [t for t in TYPE_ORDER
             if any(ct_mutcells[m].get(t) for m in ct_mutcells)]
    if len(types) < 2:
        return None

    n_types = len(types)

    # Classify mutations by how many types carry them
    def n_carrying(m):
        return sum(1 for t in types if ct_mutcells[m].get(t))

    shared_muts = [
        m for m in ct_mutcells
        if (sharedness == "all" and n_carrying(m) >= 1)
        or (sharedness == "public" and n_carrying(m) == n_types)
        or (sharedness == "partially" and n_carrying(m) >= 2)
    ]

    if not shared_muts:
        return None

    # Sort: drivers first, then by total mutant-cell count descending
    def sort_key(m):
        is_driver = mut_info[m]["gene"] in DRIVER_GENES
        total = sum(len(ct_mutcells[m].get(t, set())) for t in types)
        return (0 if is_driver else 1, -total)

    shared_muts.sort(key=sort_key)
    n_muts = len(shared_muts)

    # Compute VAF + CI and per-mutation stats
    vaf_data = {}   # mut -> type -> (vaf, lo, hi, n_mut, n_total)
    fisher_p = {}   # mut -> min pairwise Fisher p

    for m in shared_muts:
        vaf_data[m] = {}
        for t in types:
            n_total = total_cells[patient].get(t, 0)
            n_mut = len(ct_mutcells[m].get(t, set()))
            vaf_data[m][t] = (*wilson_ci(n_mut, n_total), n_mut, n_total)
        fisher_p[m] = min_pairwise_p(m, types, patient, ct_mutcells)

    # ── Figure layout ─────────────────────────────────────────────────────
    COL_W  = 1.1   # inches per cell-type column
    ROW_H  = 0.55  # inches per mutation row
    LEFT   = 3.8   # space for row labels
    RIGHT  = 1.0   # space for Fisher p annotation
    TOP    = 1.4   # space for column headers
    BOT    = 0.5
    MAX_S  = 400   # scatter marker size at VAF = 1.0

    fig_w = LEFT + n_types * COL_W + RIGHT
    fig_h = TOP + n_muts * ROW_H + BOT

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        LEFT / fig_w,
        BOT / fig_h,
        (n_types * COL_W) / fig_w,
        (n_muts * ROW_H) / fig_h,
    ])

    ax.set_xlim(-0.5, n_types - 0.5)
    ax.set_ylim(-0.5, n_muts - 0.5)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Grid
    for j in range(n_types + 1):
        ax.axvline(j - 0.5, color="#dddddd", linewidth=0.5, zorder=0)

    # Draw dots
    for i, m in enumerate(shared_muts):
        is_driver = mut_info[m]["gene"] in DRIVER_GENES

        # Row background
        bg = "#fff8dc" if is_driver else ("#f7f7f7" if i % 2 == 0 else "white")
        ax.axhspan(i - 0.5, i + 0.5, color=bg, zorder=0)

        for j, t in enumerate(types):
            vaf, lo, hi, n_mut, n_total = vaf_data[m][t]
            if n_mut > 0:
                s = max(30, vaf * MAX_S)
                ec = "#c0392b" if is_driver else "dimgray"
                lw = 2.0 if is_driver else 0.6
                ax.scatter(j, i, s=s, color=PALETTE[t], edgecolors=ec,
                           linewidths=lw, zorder=3, clip_on=False)
                # Mutant cell count label inside/below bubble
                txt_color = "white" if vaf > 0.25 else "black"
                ax.text(j, i, f"{n_mut}", ha="center", va="center",
                        fontsize=5.5, color=txt_color, fontweight="bold",
                        zorder=5, clip_on=False)
                # n_total below bubble in smaller gray text
                ax.text(j, i + 0.35, f"/{n_total}", ha="center", va="center",
                        fontsize=5, color="#888888", zorder=5, clip_on=False)
            else:
                # Not detected — ghost circle
                ax.scatter(j, i, s=55, color="none", edgecolors="#cccccc",
                           linewidths=0.8, zorder=2)

        # Row label (left of axes)
        gene = mut_info[m]["gene"]
        pchange = mut_info[m]["protein_change"]
        label = f"{gene} {pchange}".strip()
        fw = "bold" if is_driver else "normal"
        lc = "#c0392b" if is_driver else "black"
        ax.text(-0.55, i, label, ha="right", va="center",
                fontsize=8.5, fontweight=fw, color=lc,
                transform=ax.transData, clip_on=False)

        # Fisher p-value (right of axes)
        p = fisher_p[m]
        stars = pval_stars(p)
        p_str = f"p={p:.2f} {stars}" if not np.isnan(p) else ""
        ax.text(n_types - 0.38, i, p_str, ha="left", va="center",
                fontsize=7, color="#777777", transform=ax.transData,
                clip_on=False)

    # Column headers in the TOP margin axes
    hdr_ax = fig.add_axes([
        LEFT / fig_w,
        (BOT + n_muts * ROW_H) / fig_h,
        (n_types * COL_W) / fig_w,
        TOP / fig_h,
    ])
    hdr_ax.set_xlim(-0.5, n_types - 0.5)
    hdr_ax.set_ylim(0, 1)
    hdr_ax.axis("off")

    for j, t in enumerate(types):
        n_cells = total_cells[patient].get(t, 0)
        hdr_ax.text(j, 0.05, t, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=PALETTE[t],
                    rotation=30)
        hdr_ax.text(j, 0.0, f"(n={n_cells})", ha="center", va="top",
                    fontsize=6.5, color="gray")

    # Fisher p-value column header (top-right)
    hdr_ax.text(n_types - 0.38, 0.5, "Fisher p\n(pairwise)",
                ha="left", va="center", fontsize=7, color="#777777",
                transform=hdr_ax.transData, clip_on=False)

    # VAF size legend in the bottom margin
    leg_ax = fig.add_axes([
        LEFT / fig_w, 0.01,
        (n_types * COL_W) / fig_w, (BOT - 0.05) / fig_h,
    ])
    leg_ax.set_xlim(0, n_types)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")
    xleg = 0.1
    leg_ax.text(xleg, 0.5, "VAF:", ha="left", va="center", fontsize=7,
                transform=leg_ax.transData)
    xleg += 0.5
    for vaf_ex, lab_ex in [(0.01, "1%"), (0.05, "5%"), (0.15, "15%"), (0.50, "50%")]:
        s_ex = max(15, vaf_ex * MAX_S)
        leg_ax.scatter([xleg], [0.6], s=s_ex, color="#888888",
                       edgecolors="dimgray", linewidths=0.5)
        leg_ax.text(xleg, 0.0, lab_ex, ha="center", va="bottom",
                    fontsize=7, transform=leg_ax.transData)
        xleg += 0.7

    # Legend patches
    driver_patch = mpatches.Patch(facecolor="#fff8dc", edgecolor="#c0392b",
                                  linewidth=1.5, label="Driver gene (bold, gold bg)")
    fig.legend(handles=[driver_patch], loc="upper right",
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.99, 0.98))

    fig.suptitle(
        f"{patient} — Shared SNV mutations (bubble ∝ VAF; n_mut / n_total cells)",
        fontsize=10, fontweight="bold", y=0.995,
    )
    return fig


# ── Cross-patient driver-gene summary heatmap ─────────────────────────────────
def make_driver_summary(
    all_patient_data: list[dict],
    min_patients: int = 1,
) -> plt.Figure | None:
    """
    Heatmap: rows = patients (with ≥2 types), columns = driver genes that appear
    as trunk/shared mutations in ≥1 patient.
    Color: public (shared in ALL types) vs partially-shared vs private vs absent.
    """
    # Collect which driver genes are shared per patient
    # Value per (patient, gene): "public" | "partial" | "private" | None
    driver_status: dict[str, dict[str, str]] = {}   # patient -> gene -> status

    for rec in all_patient_data:
        patient = rec["patient"]
        types = rec["types"]
        n_types = len(types)
        mut_info = rec["mut_info"]
        ct_mutcells = rec["ct_mutcells"]

        driver_status[patient] = {}

        for mut_id, cells_by_type in ct_mutcells.items():
            gene = mut_info[mut_id]["gene"]
            if gene not in DRIVER_GENES:
                continue
            n_carry = sum(1 for t in types if cells_by_type.get(t))
            if n_carry == n_types and n_types >= 2:
                status = "public"
            elif n_carry >= 2:
                status = "partial"
            elif n_carry == 1:
                status = "private"
            else:
                continue
            # Keep most "public" status seen for this gene
            prev = driver_status[patient].get(gene)
            if prev is None or (prev == "private" and status != "private"):
                driver_status[patient][gene] = status

    # Filter to patients with ≥2 types and genes present in ≥min_patients
    multi_patients = [
        rec["patient"] for rec in all_patient_data
        if len(rec["types"]) >= 2
    ]
    gene_counts = defaultdict(int)
    for p in multi_patients:
        for gene in driver_status.get(p, {}):
            gene_counts[gene] += 1
    genes_to_show = sorted(
        [g for g, c in gene_counts.items() if c >= min_patients],
        key=lambda g: -gene_counts[g],
    )

    if not genes_to_show or not multi_patients:
        return None

    STATUS_COLORS = {
        "public":  "#2ecc71",
        "partial": "#f39c12",
        "private": "#bdc3c7",
    }

    n_p = len(multi_patients)
    n_g = len(genes_to_show)

    fig, ax = plt.subplots(figsize=(max(6, n_g * 0.7 + 2), max(4, n_p * 0.5 + 1.5)))

    for i, patient in enumerate(multi_patients):
        for j, gene in enumerate(genes_to_show):
            status = driver_status.get(patient, {}).get(gene)
            if status:
                color = STATUS_COLORS[status]
                ax.add_patch(mpatches.FancyBboxPatch(
                    (j - 0.4, i - 0.4), 0.8, 0.8,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor="white", linewidth=1.0, zorder=2,
                ))
            else:
                ax.scatter(j, i, s=20, color="#eeeeee", zorder=1)

    # Patient labels
    for i, patient in enumerate(multi_patients):
        types_str = " / ".join(all_patient_data[
            [r["patient"] for r in all_patient_data].index(patient)
        ]["types"])
        ax.text(-0.55, i, f"{patient} ({types_str})", ha="right", va="center",
                fontsize=8)

    # Gene labels
    for j, gene in enumerate(genes_to_show):
        ax.text(j, n_p - 0.4, gene, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", rotation=45)

    ax.set_xlim(-0.6, n_g - 0.3)
    ax.set_ylim(-0.6, n_p - 0.3)
    ax.invert_yaxis()
    ax.axis("off")

    # Legend
    legend_elems = [
        mpatches.Patch(facecolor=STATUS_COLORS["public"],  label="Public (all types)"),
        mpatches.Patch(facecolor=STATUS_COLORS["partial"], label="Partially shared"),
        mpatches.Patch(facecolor=STATUS_COLORS["private"], label="Private"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8,
              framealpha=0.9, bbox_to_anchor=(1.0, -0.08))

    fig.suptitle("Driver gene mutations: trunk vs private per patient",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ── TSV writer ────────────────────────────────────────────────────────────────
def write_tsv(all_rows: list[dict], outpath: str) -> None:
    if not all_rows:
        return
    fieldnames = [
        "patient", "cell_types", "mut_id", "gene", "protein_change",
        "is_driver", "sharing", "n_carrying_types",
    ]
    type_fields = [f"n_mut_{t}" for t in TYPE_ORDER] + [f"n_total_{t}" for t in TYPE_ORDER]
    fieldnames += type_fields + ["fisher_p_min", "fisher_stars"]
    with open(outpath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


# ── Main ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

all_patient_data = []
all_tsv_rows: list[dict] = []

for patient in sorted(patient_files.keys()):
    print(f"\n{patient} …")
    mut_info, ct_mutcells = get_mutation_data(patient, patient_files[patient])

    if not ct_mutcells:
        print("  no mutations, skipping")
        continue

    types = [t for t in TYPE_ORDER
             if any(ct_mutcells[m].get(t) for m in ct_mutcells)]
    n_types = len(types)
    print(f"  {len(ct_mutcells)} mutations, types: {types}")

    all_patient_data.append({
        "patient":    patient,
        "types":      types,
        "mut_info":   mut_info,
        "ct_mutcells": ct_mutcells,
    })

    if n_types < 2:
        continue

    # Classify each mutation
    for mut_id, cells_by_type in ct_mutcells.items():
        n_carry = sum(1 for t in types if cells_by_type.get(t))
        if n_carry == n_types:
            sharing = "public"
        elif n_carry >= 2:
            sharing = "partially_shared"
        else:
            sharing = "private"

        p_min = min_pairwise_p(mut_id, types, patient, ct_mutcells)

        row = {
            "patient":         patient,
            "cell_types":      ", ".join(types),
            "mut_id":          mut_id,
            "gene":            mut_info[mut_id]["gene"],
            "protein_change":  mut_info[mut_id]["protein_change"],
            "is_driver":       mut_info[mut_id]["gene"] in DRIVER_GENES,
            "sharing":         sharing,
            "n_carrying_types": n_carry,
            "fisher_p_min":    f"{p_min:.4f}" if not np.isnan(p_min) else "NA",
            "fisher_stars":    pval_stars(p_min),
        }
        for t in TYPE_ORDER:
            row[f"n_mut_{t}"]   = len(cells_by_type.get(t, set()))
            row[f"n_total_{t}"] = total_cells[patient].get(t, 0)
        all_tsv_rows.append(row)

    # Plot shared mutations (public + partially shared)
    fig = make_patient_dotplot(
        patient, mut_info, ct_mutcells, sharedness="partially"
    )
    if fig is not None:
        outpath = os.path.join(OUT_DIR, f"{patient}_SNV_shared.png")
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {outpath}")
    else:
        print("  no shared mutations to plot")

# Cross-patient driver summary
print("\nGenerating driver-gene summary …")
fig_sum = make_driver_summary(all_patient_data)
if fig_sum is not None:
    outpath = os.path.join(OUT_DIR, "SNV_driver_trunk_summary.png")
    fig_sum.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig_sum)
    print(f"→ {outpath}")

# TSV
tsv_path = os.path.join(OUT_DIR, "SNV_shared_mutations.tsv")
write_tsv(all_tsv_rows, tsv_path)
print(f"→ {tsv_path}  ({len(all_tsv_rows)} rows)")

print("\nDone.")
