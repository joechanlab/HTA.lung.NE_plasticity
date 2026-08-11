"""Build per-cell metadata table covering the FULL cell_type_final2 label space
(LUAD, LUSC, SCLC-A, SCLC-N, SCLC-P, NonNE SCLC, EMT), for the drug-target gene
dot plot (expression x cell type, colored by abundance shift).

RU151 is overridden with `new_cell_type` and RU325 is overridden to uniformly SCLC-A (per-patient
re-analyses that supersede the combined adata's cell_type_final2 for these two). No cell-type filter
is applied -- every cell keeps a label, unlike an SCLC-lineage-only subset. FOXI1+ NonNE SCLC-N is
recoded into SCLC-N per user request (a proliferative SCLC-N-adjacent subtype, not a separate category
for this analysis).
"""

import anndata as ad
import pandas as pd

FULL_ADATA = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/combined/adata.SCLC_NSCLC.051226.h5ad"
RU151_ADATA = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.021626/RU151.new_embedding.070326.h5ad"
RU325_ADATA = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/RU325.no_cc.hvg_2000.090124.SCLCA_relabeled.h5ad"

CONTROL_PATIENTS = [
    "RU1311", "RU1124", "RU1108", "RU1065", "RU871", "RU1322", "RU1144",
    "RU1066", "RU1195", "RU779", "RU1152", "RU426", "RU1231", "RU222",
    "RU1215", "RU1145", "RU1360", "RU1287",
]
TRANSFORMED_PATIENTS = [
    "RU325", "RU1676", "RU1181", "RU1304", "RU773", "RU1042", "RU1414",
    "RU1405", "RU1250", "RU1305", "RU226", "RU1444", "RU151", "RU942",
    "RU1083", "RU263", "RU1303", "RU1518", "RU581", "RU831", "RU1068",
    "RU1646", "RU1314",
]

RECODE = {"FOXI1+ NonNE SCLC-N": "SCLC-N"}

OUT_CSV = "cell_metadata.full_celltypes.csv"


def main():
    full = ad.read_h5ad(FULL_ADATA, backed="r")
    obs = full.obs[["patient", "cell_type_final2"]].copy()
    obs["patient"] = obs["patient"].astype(str)
    obs["cell_type_final2"] = obs["cell_type_final2"].astype(str)

    ru151 = ad.read_h5ad(RU151_ADATA, backed="r")
    new_ct_151 = ru151.obs["new_cell_type"].astype(str)
    shared_151 = obs.index.intersection(new_ct_151.index)
    assert len(shared_151) == (obs["patient"] == "RU151").sum(), "RU151 barcode mismatch"
    obs.loc[shared_151, "cell_type_final2"] = new_ct_151.loc[shared_151]

    ru325 = ad.read_h5ad(RU325_ADATA, backed="r")
    ct_325 = ru325.obs["cell_type_final2"].astype(str)
    assert (ct_325 == "SCLC-A").all(), "RU325 cell_type_final2 is not uniformly SCLC-A"
    shared_325 = obs.index.intersection(ct_325.index)
    assert len(shared_325) == (obs["patient"] == "RU325").sum(), "RU325 barcode mismatch"
    obs.loc[shared_325, "cell_type_final2"] = ct_325.loc[shared_325]

    obs["cell_type_final2"] = obs["cell_type_final2"].replace(RECODE)

    overlap = set(CONTROL_PATIENTS) & set(TRANSFORMED_PATIENTS)
    assert not overlap, f"Patients listed in both groups: {overlap}"

    group_map = {p: "Control" for p in CONTROL_PATIENTS}
    group_map.update({p: "Transformed" for p in TRANSFORMED_PATIENTS})
    obs["group"] = obs["patient"].map(group_map)

    all_patients = set(obs["patient"].unique())
    listed = set(CONTROL_PATIENTS) | set(TRANSFORMED_PATIENTS)
    missing = listed - all_patients
    assert not missing, f"Listed patients not found in adata: {missing}"
    unassigned = sorted(all_patients - listed)
    print(f"Patients present in adata but not in either group list (excluded): {unassigned}")

    n_before = len(obs)
    obs = obs[obs["group"].notna()].copy()
    n_after_group = len(obs)
    print(f"Cells: {n_before} total -> {n_after_group} after group filter (no cell-type filter this time)")

    obs = obs.rename(columns={"cell_type_final2": "cell_type"})
    obs.index.name = "barcode"
    obs.to_csv(OUT_CSV)

    print()
    print("Per-patient group assignment (n patients):")
    print(obs.groupby(["group", "patient"], observed=True).size().groupby(level=0).size())
    print()
    print("Cell type counts per group:")
    print(obs.groupby(["group", "cell_type"], observed=True).size().unstack(fill_value=0))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
