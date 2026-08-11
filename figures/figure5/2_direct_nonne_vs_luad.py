"""
Step 2 of 3 — depends on step 1 (1_gsea_cell_type_all_patients.py) only for its
load_patient()/gene-filter helpers, not its per-patient CSV outputs.

Direct one-vs-one NonNE vs. LUAD comparison, in the two patients that have both
populations (RU1444, RU151). This is a genuine head-to-head test (NonNE SCLC as
`groups`, LUAD as `reference` in rank_genes_groups) — independent of which
comparator mode step 1 used for its OWN per-cell-type DEG (one-vs-nearest-neighbor
or one-vs-fixed-SCLC-pool), since this script always does a direct pairwise test
regardless. Step 3 (3_nonne_luad_hallmark_venn.py) should be run against step 1's
`--reference sclc_pool` output specifically, so its "co-divergent" pathways mean
"both diverge from the SAME actual SCLC reference in the same direction" — a
real, consistent baseline. But that still isn't the same claim as "NonNE and LUAD
resemble each other directly" (two populations can independently diverge from a
shared reference in a coincidentally similar direction without resembling each
other). This script answers that question directly: regardless of what step 3
finds, do NonNE and LUAD actually resemble each other when tested head-to-head?

Saves ALL Hallmark terms (not a hardcoded subset) per patient, so the output
stays useful regardless of which terms step 3's recurrence analysis flags as
"co-divergent" after any future threshold change.
"""
import os
import importlib.util
import warnings
import pandas as pd
import scanpy as sc
import gseapy as gp

warnings.filterwarnings("ignore", category=FutureWarning)

# 1_gsea_cell_type_all_patients.py starts with a digit, so it isn't a valid
# Python module name for a normal `from X import Y` — load it by file path.
_step1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1_gsea_cell_type_all_patients.py")
_spec = importlib.util.spec_from_file_location("gsea_cell_type_all_patients", _step1_path)
_step1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step1)
load_patient = _step1.load_patient
MITO_RIBO_PATTERN = _step1.MITO_RIBO_PATTERN
EXTRA_RIBO_GENES = _step1.EXTRA_RIBO_GENES

PATIENTS = ["RU1444", "RU151"]
OUTDIR = "figures/gsea_all_patients_sclc_pool/_summary"  # alongside step 3's output, the "live" summary dir


def direct_compare(patient):
    print(f"\n{'=' * 70}\n{patient}: NonNE SCLC vs. LUAD (direct, one-vs-one)\n{'=' * 70}")
    adata = load_patient(patient)
    sub = adata[adata.obs["new_cell_type"].isin(["NonNE SCLC", "LUAD"])].copy()
    sub.obs["new_cell_type"] = sub.obs["new_cell_type"].astype(str)
    cell_counts = sub.obs["new_cell_type"].value_counts().to_dict()
    print(f"Cell counts: {cell_counts}")
    n_cells = sub.n_obs

    keep_genes = ~(sub.var_names.str.match(MITO_RIBO_PATTERN) | sub.var_names.isin(EXTRA_RIBO_GENES))
    sub = sub[:, keep_genes].copy()

    if sub.raw is not None:
        sub_raw = sub.raw.to_adata()
        sub_raw = sub_raw[:, sub_raw.var_names.isin(sub.var_names)].copy()
        sc.pp.normalize_total(sub_raw, target_sum=1e4)
        sc.pp.log1p(sub_raw)
        sub.layers["lognorm"] = sub_raw[:, sub.var_names].X
        use_layer = "lognorm"
    else:
        use_layer = None

    sc.tl.rank_genes_groups(
        sub, groupby="new_cell_type", groups=["NonNE SCLC"], reference="LUAD",
        method="wilcoxon", layer=use_layer, use_raw=False, n_genes=sub.n_vars,
        key_added="direct",
    )
    rgg = sc.get.rank_genes_groups_df(sub, group="NonNE SCLC", key="direct")
    rgg["score"] = rgg["scores"]
    rgg = rgg.sort_values("score", ascending=False).drop_duplicates("names")
    ranked = rgg.set_index("names")["score"].dropna()

    rnk = ranked.reset_index()
    rnk.columns = ["gene", "score"]
    res = gp.prerank(
        rnk=rnk, gene_sets="MSigDB_Hallmark_2020", outdir=None,
        min_size=10, max_size=500, permutation_num=1000, seed=42, verbose=False,
    )
    df = res.res2d.copy()
    if "Term" in df.columns:
        df = df.set_index("Term")

    print("\nNES sign convention: positive = higher in NonNE SCLC than LUAD (direct);"
          " negative = higher in LUAD.")
    print("\nTop 10 terms overall (either direction):")
    top = df.reindex(df["NES"].abs().sort_values(ascending=False).index).head(10)
    for term, row in top.iterrows():
        print(f"  {term:<40s} NES={row['NES']:+.2f}  FDR={row['FDR q-val']:.3f}")

    return df, n_cells


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    results = {}
    n_cells = {}
    for patient in PATIENTS:
        results[patient], n_cells[patient] = direct_compare(patient)

    print(f"\n{'=' * 70}\nAll Hallmark terms, direct NonNE vs LUAD, both patients\n{'=' * 70}")
    long_rows = []
    for patient in PATIENTS:
        df = results[patient]
        for term, row in df.iterrows():
            long_rows.append({
                "Term": term, "Patient": patient,
                "NES": round(row["NES"], 3), "FDR": round(row["FDR q-val"], 3),
                "NOM_pval": round(row["NOM p-val"], 5), "n_cells": n_cells[patient],
            })

    long_df = pd.DataFrame(long_rows)
    long_path = f"{OUTDIR}/direct_nonne_vs_luad.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Saved: {long_path} ({len(long_df)} rows: {len(PATIENTS)} patients x "
          f"{len(long_df) // len(PATIENTS)} Hallmark terms)")


if __name__ == "__main__":
    main()
