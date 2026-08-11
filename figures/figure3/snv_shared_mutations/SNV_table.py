#!/usersoftware/chanj3/scdebussy-py311/bin/python3
"""
Tabulate public (trunk) and private (type-specific) SNVs per patient.

Output: SNV_public_private_table.tsv

Definitions
-----------
public          : mutation detected in ALL tumor cell types present → trunk / basal
partially shared: mutation detected in >1 but not all tumor cell types → subclonal shared
private         : mutation detected in exactly 1 tumor cell type → derived / distal

One row per patient that has ≥1 tumor cell type with SNVs.

Added columns (beyond raw counts):
  public_drivers          : driver genes found as trunk (public) mutations
  partial_drivers         : driver genes found as partially-shared mutations
  private_drivers         : driver genes found as private mutations (any type)
  n_public_drivers        : count of unique driver genes that are trunk
  n_partial_drivers       : count of unique driver genes that are partially shared
  public_mut_ids          : full mut_id strings for public mutations (gene_pos_minor)
"""

import os
import glob
import csv
from collections import defaultdict, Counter

import anndata as ad

# ── Driver genes of interest ──────────────────────────────────────────────────
# From paper genomics section: key T-SCLC drivers and enriched alterations
DRIVER_GENES = {
    "EGFR", "TP53", "RB1", "KRAS", "STK11", "KEAP1",
    "CDKN2A", "CDKN2B", "MYC", "MYCN", "MYCL",
    "AKT1", "AKT2", "PIK3CA", "PIK3R2", "PTEN", "RICTOR", "AXIN1",
    "ASCL1", "NEUROD1",
    "BRAF", "NF1", "MET", "RET", "FGFR1", "ALK", "ROS1",
    "NOTCH1", "NOTCH2", "SMAD4", "ARID1A", "SMARCA4",
    "CREBBP", "EP300", "MDM2", "MDM4", "CDK4", "CDK6", "CCND1",
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
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_TSV  = os.path.join(OUT_DIR, "SNV_public_private_table.tsv")

TYPE_ORDER = [
    "LUAD", "LUSC", "SCLC-A", "SCLC-N", "SCLC-P",
    "NonNE SCLC", "EMT", "FOXI1+ NonNE SCLC-N",
]

# ── Load barcode → (patient, cell_type) ──────────────────────────────────────
print("Loading h5ad annotations …")
adata = ad.read_h5ad(H5AD_PATH, backed="r")

seqc_to_meta: dict[str, tuple[str, str]] = {}
for bc, patient, ct in zip(
    adata.obs_names,
    adata.obs["patient"],
    adata.obs["cell_type_final2"],
):
    seqc_to_meta[bc.rsplit("_", 1)[-1]] = (patient, ct)
print(f"  {len(seqc_to_meta):,} barcodes loaded")

# ── Discover genotype files ───────────────────────────────────────────────────
genotype_files = glob.glob(
    os.path.join(GENOTYPE_DIR, "*", "*",
                 "HTA.nonsignedout.*.genotype_info.060124.txt")
)
patient_files: dict[str, list[str]] = defaultdict(list)
for f in genotype_files:
    patient_files[f.split(os.sep)[-3]].append(f)
print(f"Found {len(genotype_files)} genotype files across {len(patient_files)} patients")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_genotype_file(fpath: str):
    """
    Yields (mut_id, gene, protein_change, seqc_bcs) per mutated locus.

    Columns (0-indexed):
      0=gene, 1=pos, 2=ref, 3=major, 4=minor, 5=protein_change,
      6=read_count, 7=SNV_calls, 8=barcode_list (cellID:UMI,...)
    """
    with open(fpath) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or not cols[8].strip():
                continue
            gene    = cols[0]
            pos     = cols[1]
            minor   = cols[4]
            pchange = cols[5] if len(cols) > 5 else ""
            mut_id  = f"{gene}_{pos}_{minor}"
            seqc_bcs = [e.split(":")[0] for e in cols[8].split(",")]
            yield mut_id, gene, pchange, seqc_bcs


def get_mutation_sets(patient: str, files: list[str]) -> tuple[dict, dict]:
    """
    Returns:
      ct_muts   : {cell_type -> set(mut_id)}
      mut_genes : {mut_id -> (gene, protein_change)}
    """
    ct_muts:   dict[str, set[str]]         = defaultdict(set)
    mut_genes: dict[str, tuple[str, str]]  = {}
    for fpath in files:
        for mut_id, gene, pchange, seqc_bcs in _parse_genotype_file(fpath):
            if mut_id not in mut_genes:
                mut_genes[mut_id] = (gene, pchange)
            for seqc in seqc_bcs:
                meta = seqc_to_meta.get(seqc)
                if meta and meta[0] == patient and meta[1] in set(TYPE_ORDER):
                    ct_muts[meta[1]].add(mut_id)
    return dict(ct_muts), mut_genes


# ── Build table ───────────────────────────────────────────────────────────────
rows = []

for patient in sorted(patient_files.keys()):
    ct_muts, mut_genes = get_mutation_sets(patient, patient_files[patient])
    if not ct_muts:
        continue

    types = [t for t in TYPE_ORDER if t in ct_muts]   # canonical order
    n_types = len(types)

    # Map each mutation to the frozenset of types that carry it
    all_muts: set[str] = set().union(*ct_muts.values())
    membership = {
        mut: frozenset(t for t in types if mut in ct_muts[t])
        for mut in all_muts
    }
    degree_counter: Counter = Counter(len(s) for s in membership.values())

    n_total     = len(all_muts)
    n_public    = degree_counter.get(n_types, 0) if n_types >= 2 else 0
    n_private   = degree_counter.get(1, 0)
    n_partially = n_total - n_public - n_private

    # Private count per type
    private_per_type = {t: 0 for t in TYPE_ORDER}
    for mem in membership.values():
        if len(mem) == 1:
            private_per_type[next(iter(mem))] += 1

    # ── Driver gene annotation ────────────────────────────────────────────
    # For each sharing category, collect unique driver genes (sorted)
    def _drivers_in(muts_subset):
        genes = sorted({mut_genes[m][0] for m in muts_subset
                        if mut_genes.get(m) and mut_genes[m][0] in DRIVER_GENES})
        return "; ".join(genes)

    public_muts   = [m for m, mem in membership.items() if len(mem) == n_types and n_types >= 2]
    partial_muts  = [m for m, mem in membership.items() if 1 < len(mem) < n_types]
    private_muts  = [m for m, mem in membership.items() if len(mem) == 1]

    public_drivers  = _drivers_in(public_muts)
    partial_drivers = _drivers_in(partial_muts)
    private_drivers = _drivers_in(private_muts)

    # Count unique driver genes per category (by gene symbol, not per mutation)
    n_public_drivers  = len(set(mut_genes[m][0] for m in public_muts
                                if mut_genes.get(m) and mut_genes[m][0] in DRIVER_GENES))
    n_partial_drivers = len(set(mut_genes[m][0] for m in partial_muts
                                if mut_genes.get(m) and mut_genes[m][0] in DRIVER_GENES))

    # Full mut_id strings for public mutations (useful for cross-referencing)
    public_mut_ids = "; ".join(sorted(public_muts))

    row = {
        "patient":            patient,
        "cell_types":         ", ".join(types),
        "n_types":            n_types,
        "n_total_muts":       n_total,
        "n_public":           n_public,
        "n_partially_shared": n_partially,
        "n_private":          n_private,
        "n_public_drivers":   n_public_drivers,
        "n_partial_drivers":  n_partial_drivers,
        "public_drivers":     public_drivers,
        "partial_drivers":    partial_drivers,
        "private_drivers":    private_drivers,
        "public_mut_ids":     public_mut_ids,
    }
    for t in TYPE_ORDER:
        row[f"private_{t}"] = private_per_type[t]

    rows.append(row)
    driver_note = f"  drivers: public=[{public_drivers}]" if public_drivers else ""
    print(
        f"  {patient:<10}  types={n_types}  "
        f"total={n_total}  public={n_public}  "
        f"partial={n_partially}  private={n_private}"
        f"{driver_note}"
    )

# ── Write TSV ─────────────────────────────────────────────────────────────────
fieldnames = (
    ["patient", "cell_types", "n_types",
     "n_total_muts", "n_public", "n_partially_shared", "n_private",
     "n_public_drivers", "n_partial_drivers",
     "public_drivers", "partial_drivers", "private_drivers",
     "public_mut_ids"]
    + [f"private_{t}" for t in TYPE_ORDER]
)

with open(OUT_TSV, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved → {OUT_TSV}  ({len(rows)} patients)")
