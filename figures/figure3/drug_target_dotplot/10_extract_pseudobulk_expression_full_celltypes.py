"""Extract per-patient pseudobulk mean expression of the drug-target gene panel,
across the FULL cell-type space (LUAD, LUSC, SCLC-A, SCLC-N, SCLC-P, NonNE SCLC,
EMT), for the expression-x-celltype dot plot.

layers['log'] is dense/unchunked in the source h5ad, so this reads per-patient
(row-slice) rather than the whole matrix, to bound peak memory to the largest
single patient's cell count x gene count.
"""

import h5py
import numpy as np
import pandas as pd

FULL_ADATA = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/combined/adata.SCLC_NSCLC.051226.h5ad"
IN_CSV = "cell_metadata.full_celltypes.csv"
OUT_CSV = "pseudobulk_gene_expression.full_celltypes.csv"

DRUG_TARGET_GENES = [
    "DLL3",
    "SEZ6",
    "CADM1",
    "SSTR2",
    "CD276",
    "SLFN11",
    "TACSTD2",
    "CEACAM5",
    "CEACAM6",
]

CELL_TYPES = ["LUAD", "LUSC", "SCLC-A", "SCLC-N", "SCLC-P", "NonNE SCLC", "EMT"]


def main():
    obs = pd.read_csv(IN_CSV, index_col=0)
    obs = obs[obs["cell_type"].isin(CELL_TYPES)].copy()
    print(f"Cells to extract (full cell-type set): {len(obs)} across {obs['patient'].nunique()} patients")

    f = h5py.File(FULL_ADATA, "r")

    var_names = [x.decode() for x in f["var"]["_index"][:]]
    var_pos = {g: i for i, g in enumerate(var_names)}
    gene_idx = np.array([var_pos[g] for g in DRUG_TARGET_GENES])

    full_barcodes = [x.decode() for x in f["obs"]["_index"][:]]
    barcode_to_pos = pd.Series(np.arange(len(full_barcodes)), index=full_barcodes)

    obs["pos"] = barcode_to_pos.loc[obs.index].values

    log_ds = f["layers"]["log"]

    rows = []
    for (patient, cell_type), sub in obs.groupby(["patient", "cell_type"], observed=True):
        positions = np.sort(sub["pos"].values)
        mat = log_ds[positions, :][:, gene_idx]  # (n_cells, n_genes)
        means = mat.mean(axis=0)
        group = sub["group"].iloc[0]
        for gene, mean_val in zip(DRUG_TARGET_GENES, means):
            rows.append(
                {
                    "patient": patient,
                    "group": group,
                    "cell_type": cell_type,
                    "gene": gene,
                    "mean_log_expr": float(mean_val),
                    "n_cells": len(positions),
                }
            )
        print(f"  {patient} / {cell_type}: n={len(positions)}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out)} rows)")


if __name__ == "__main__":
    main()
