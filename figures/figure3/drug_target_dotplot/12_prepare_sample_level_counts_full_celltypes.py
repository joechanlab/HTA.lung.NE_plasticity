"""Build a per-patient cell-type count table over the full cell_type_final2
label space (LUAD, LUSC, SCLC-A, SCLC-N, SCLC-P, NonNE SCLC, EMT), for the
Dirichlet regression in 13_dirichlet_regression_full_celltypes.R.

Input: cell_metadata.full_celltypes.csv, produced by 09_prepare_data_full_celltypes.py.
"""

import pandas as pd

IN_CSV = "cell_metadata.full_celltypes.csv"
OUT_CSV = "sample_level_counts.full_celltypes.csv"
CELL_TYPES = ["LUAD", "LUSC", "SCLC-A", "SCLC-N", "SCLC-P", "NonNE SCLC", "EMT"]


def main():
    obs = pd.read_csv(IN_CSV, index_col=0)

    counts = pd.crosstab([obs["patient"], obs["group"]], obs["cell_type"])
    counts = counts.reindex(columns=CELL_TYPES, fill_value=0)
    counts = counts.reset_index()[["patient", *CELL_TYPES, "group"]]

    print("Sample-level count table (patients x cell types):")
    print(counts)

    print("\nn patients with >0 cells, by group:")
    print(counts.groupby("group")[CELL_TYPES].apply(lambda d: (d > 0).sum()))

    counts.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
