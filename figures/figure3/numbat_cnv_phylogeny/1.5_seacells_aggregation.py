"""
SEACells metacell aggregation (joint, majority-vote cell type).

All cells in the sample are aggregated together in a single SEACells run.
Cell type of each metacell is determined by majority vote from constituent
cells. This preserves rare cell types (e.g. a few LUSC cells) that would
otherwise become passthrough singletons under a cell-type-stratified approach.

Target: ~25 cells per metacell (n_SEACells = n_total // 25).

Output format (identical to downstream expectations):
  metacell/barcodes.tsv           — metacell IDs (one per line)
  metacell/genes.tsv              — gene names (unchanged)
  metacell/{sample}.expression.mtx — summed raw counts (genes × metacells)
  metacell/{sample}.cell_types.tsv — metacell → majority-vote cell type
  metacell/{sample}.metacell_assignment.tsv — cell → metacell mapping
  BAM/metacell/{sample}_allele_counts.tsv.gz — aggregated allele counts

Usage:
  /usersoftware/chanj3/SEACells/bin/python 1.5_seacells_aggregation.py \
      --sample_id RU151_PDX
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp
import SEACells

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_CELLS_PER_METACELL = 25
MIN_METACELLS             = 3      # minimum metacells to request from SEACells
MIN_CELLS_FOR_SEACELLS    = 20     # skip sample entirely if fewer total cells

IN_DIR   = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/"
H5AD_ALL = ("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/"
            "scvi_latent150/adata.scvi.all_genes.final_subset.051826.h5ad")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--sample_id", required=True)
parser.add_argument("--target_cells_per_metacell", type=int,
                    default=TARGET_CELLS_PER_METACELL)
args = parser.parse_args()

SAMPLE_ID  = args.sample_id
PATIENT_ID = SAMPLE_ID.split("_")[0]
TARGET_CPM = args.target_cells_per_metacell

cell_type_col = "cell_type_final2"

GEX_DIR      = os.path.join(IN_DIR, PATIENT_ID, "GEX", SAMPLE_ID)
BAM_DIR      = os.path.join(IN_DIR, PATIENT_ID, "BAM")
META_GEX_DIR = os.path.join(GEX_DIR, "metacell")
META_BAM_DIR = os.path.join(BAM_DIR,  "metacell")
os.makedirs(META_GEX_DIR, exist_ok=True)
os.makedirs(META_BAM_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load h5ad — subset to this sample
# ---------------------------------------------------------------------------
print(f"[1/5] Loading h5ad for {SAMPLE_ID} ...")
adata_full = sc.read_h5ad(H5AD_ALL)
adata_full.obs["barcode_prefix"] = (
    adata_full.obs_names.str.split("_").str[:-1].str.join("_")
)
adata = adata_full[adata_full.obs["barcode_prefix"] == SAMPLE_ID].copy()
del adata_full
print(f"    Sample subset: {adata.n_obs} cells x {adata.n_vars} genes")

# Replace integer-encoded obs_names with decoded barcodes from barcodes.tsv
decoded_barcodes = pd.read_csv(
    os.path.join(GEX_DIR, "barcodes.tsv"), header=None
)[0].tolist()
assert len(decoded_barcodes) == adata.n_obs, (
    f"barcodes.tsv has {len(decoded_barcodes)} rows but adata subset has "
    f"{adata.n_obs} cells for {SAMPLE_ID}"
)
adata.obs_names = decoded_barcodes
print(f"    Replaced obs_names with decoded barcodes")

if adata.n_obs < MIN_CELLS_FOR_SEACELLS:
    print(f"    Too few cells ({adata.n_obs} < {MIN_CELLS_FOR_SEACELLS}). Skipping.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 2. Prepare AnnData for SEACells
# ---------------------------------------------------------------------------
print("[2/5] Preparing AnnData ...")
adata_sc = adata.copy()
if adata_sc.raw is not None:
    adata_sc.X = adata_sc.raw.X.copy()

sc.pp.normalize_total(adata_sc, target_sum=1e4)
sc.pp.log1p(adata_sc)

if "X_scVI" in adata_sc.obsm:
    build_kernel_on = "X_scVI"
    print("    Using X_scVI for SEACells kernel.")
elif "X_pca" in adata_sc.obsm:
    build_kernel_on = "X_pca"
    print("    Using X_pca for SEACells kernel.")
else:
    sc.pp.highly_variable_genes(adata_sc, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(adata_sc, use_highly_variable=True, n_comps=50)
    build_kernel_on = "X_pca"
    print("    Computed PCA on the fly.")

# ---------------------------------------------------------------------------
# 3. Run SEACells on all cells jointly
# ---------------------------------------------------------------------------
print("[3/5] Running SEACells (joint, all cells) ...")
np.random.seed(42)

n_total    = adata_sc.n_obs
n_SEACells = max(MIN_METACELLS, n_total // TARGET_CPM)
n_eigs     = min(10, n_SEACells)
n_nn       = min(15, n_total - 1)

sc.pp.neighbors(adata_sc, use_rep=build_kernel_on, n_neighbors=n_nn)

try:
    model = SEACells.core.SEACells(
        adata_sc,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_eigs,
        convergence_epsilon=1e-3,
        verbose=False,
    )
    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=100)

    assignments = adata_sc.obs[[cell_type_col, "SEACell"]].copy()
    print(f"    {n_total} cells -> {n_SEACells} metacells "
          f"(~{n_total / n_SEACells:.1f} cells/metacell)")

except Exception as e:
    print(f"    WARNING: SEACells failed ({e}). Using each cell as its own metacell.")
    assignments = adata_sc.obs[[cell_type_col]].copy()
    assignments["SEACell"] = [f"SEACell-{i}" for i in range(n_total)]

assignments.index.name = "cell"

# Cell type by majority vote within each metacell
metacell_celltypes = (
    assignments.groupby("SEACell")[cell_type_col]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"SEACell": "metacell"})
)

# Summary
ct_counts = metacell_celltypes[cell_type_col].value_counts()
print("    Metacells per cell type (majority vote):")
for ct, n in ct_counts.items():
    total_ct = (assignments[cell_type_col] == ct).sum()
    print(f"      {ct}: {n} metacells  (from {total_ct} cells, "
          f"~{total_ct / max(n, 1):.1f} cells/metacell)")

# ---------------------------------------------------------------------------
# 4. Aggregate expression counts (sum raw counts per metacell)
# ---------------------------------------------------------------------------
print("[4/5] Aggregating expression counts ...")

raw_mtx  = scipy.io.mmread(os.path.join(GEX_DIR, f"{SAMPLE_ID}.expression.mtx"))
barcodes = pd.read_csv(os.path.join(GEX_DIR, "barcodes.tsv"), header=None)[0].tolist()
genes    = pd.read_csv(os.path.join(GEX_DIR, "genes.tsv"),    header=None)[0].tolist()

raw_mtx = sp.csr_matrix(raw_mtx)   # genes x cells

bc_series  = pd.Series(barcodes, name="cell")
bc_to_meta = bc_series.map(assignments["SEACell"])

valid = bc_to_meta.notna()
if not valid.all():
    print(f"    Warning: {(~valid).sum()} barcodes not in SEACell assignments; dropping.")

metacell_ids = sorted(assignments["SEACell"].unique())
metacell_idx = {m: i for i, m in enumerate(metacell_ids)}

rows = bc_to_meta[valid].map(metacell_idx).values.astype(np.int32)
cols = np.where(valid)[0].astype(np.int32)
data = np.ones(valid.sum(), dtype=np.int32)
agg_mat = sp.csr_matrix(
    (data, (rows, cols)),
    shape=(len(metacell_ids), raw_mtx.shape[1]),
    dtype=np.int32,
)
raw_mtx_int   = sp.csr_matrix(raw_mtx, dtype=np.int32)
metacell_expr = (agg_mat @ raw_mtx_int.T).T.tocsc().astype(np.int32)

scipy.io.mmwrite(
    os.path.join(META_GEX_DIR, f"{SAMPLE_ID}.expression.mtx"),
    metacell_expr
)
pd.DataFrame(metacell_ids).to_csv(
    os.path.join(META_GEX_DIR, "barcodes.tsv"), header=False, index=False
)
pd.DataFrame(genes).to_csv(
    os.path.join(META_GEX_DIR, "genes.tsv"), header=False, index=False
)

celltypes_out = metacell_celltypes.set_index("metacell")[[cell_type_col]]
celltypes_out.index.name = None
celltypes_out.to_csv(
    os.path.join(META_GEX_DIR, f"{SAMPLE_ID}.cell_types.tsv"), sep="\t"
)

assignments[["SEACell"]].to_csv(
    os.path.join(META_GEX_DIR, f"{SAMPLE_ID}.metacell_assignment.tsv"), sep="\t"
)
print(f"    Saved metacell expression: {len(genes)} genes x {len(metacell_ids)} metacells")

# ---------------------------------------------------------------------------
# 5. Aggregate allele counts (sum AD, DP per metacell x SNP)
# ---------------------------------------------------------------------------
print("[5/5] Aggregating allele counts ...")

allele_path = os.path.join(BAM_DIR, f"{SAMPLE_ID}_allele_counts.tsv.gz")
if not os.path.exists(allele_path):
    print(f"    Allele counts not found at {allele_path}; skipping.")
else:
    df_allele = pd.read_csv(allele_path, sep="\t", compression="gzip")
    df_allele["metacell"] = df_allele["cell"].map(assignments["SEACell"])
    unmapped = df_allele["metacell"].isna().sum()
    if unmapped > 0:
        print(f"    Warning: {unmapped} allele rows have no metacell assignment; dropping.")
    df_allele = df_allele.dropna(subset=["metacell"])

    snp_meta = df_allele.drop_duplicates("snp_id").set_index("snp_id")[
        [c for c in ["CHROM", "POS", "cM", "REF", "ALT", "gene"] if c in df_allele.columns]
    ]

    agg = (
        df_allele.groupby(["metacell", "snp_id"])
        .agg(
            AD=("AD", "sum"),
            DP=("DP", "sum"),
            GT=("GT", lambda x: x.value_counts().index[0]),
        )
        .reset_index()
    )
    agg = agg.merge(snp_meta.reset_index(), on="snp_id", how="left")
    agg = agg.rename(columns={"metacell": "cell"})

    orig_cols = ["cell", "snp_id", "CHROM", "POS", "cM", "REF", "ALT", "AD", "DP", "GT", "gene"]
    agg = agg[[c for c in orig_cols if c in agg.columns]]

    out_allele = os.path.join(META_BAM_DIR, f"{SAMPLE_ID}_allele_counts.tsv.gz")
    agg.to_csv(out_allele, sep="\t", index=False, compression="gzip")
    print(f"    Saved metacell allele counts: {len(agg)} rows -> {out_allele}")

print("Done.")
