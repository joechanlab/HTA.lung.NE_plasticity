"""Stage 1 of the RU1518-style trajectory rerun: diffusion maps + Palantir pseudotime.

Runs under an env with `palantir` (e.g. scdebussy-py311). Automates the
manual "which diffusion component matches which cell type" step from
step2.CellRank_macrostates_individual.template.ipynb (RU1518 section) by
picking, per target cell type, the diffusion component most correlated with
that type's indicator, then taking its extreme cell. Saves QC plots
(highlight_cells_on_umap, plot_palantir_results) so picks can be sanity-checked.

Rule (per user spec): start = LUAD. Terminal = LUSC, NonNE SCLC, and
SCLC-A only if no SCLC-N is present for this patient (else SCLC-A is an
intermediate en route to SCLC-N, and SCLC-N is terminal instead).
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import palantir

FULL_ADATA_PATH = (
    "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/"
    "combined/adata.SCLC_NSCLC.051226.h5ad"
)


def assign_extreme_cells(eigvecs, ct_series, types):
    # Match cell types to diffusion components by global greedy max-|corr|
    # assignment, not by processing types in a fixed order and having each
    # one grab whatever's locally best. Sequential assignment lets an early
    # type (e.g. LUAD, processed first) steal the component/cell that would
    # have been another type's (e.g. LUSC's) much cleaner match, leaving that
    # type stuck on a weak, noisy component -- which can degenerate the whole
    # trajectory (a "terminal" type ending up with lower mean pseudotime than
    # the start type). Considering every (type, component) pair together and
    # claiming the strongest correlations first avoids that.
    candidates = []
    for t in types:
        indicator = (ct_series == t).astype(float).values
        for comp in eigvecs.columns:
            if comp == 0:
                continue
            corr = np.corrcoef(eigvecs[comp].values, indicator)[0, 1]
            candidates.append((abs(corr), t, comp, corr))
    candidates.sort(key=lambda c: c[0], reverse=True)

    assigned = {}
    used_components = set()
    used_cells = set()
    for _, t, comp, corr in candidates:
        if t in assigned or comp in used_components:
            continue
        vals = eigvecs[comp]
        cell = vals.idxmax() if corr > 0 else vals.idxmin()
        if cell in used_cells:
            # this component's single extreme cell is already claimed by
            # another type; leave the component available in case a later,
            # weaker-correlated type still wants it via a different cell
            continue
        assigned[t] = (cell, comp, corr)
        used_components.add(comp)
        used_cells.add(cell)
        if len(assigned) == len(types):
            break

    missing = [t for t in types if t not in assigned]
    if missing:
        raise RuntimeError(f"could not assign a distinct component/cell for: {missing}")
    return assigned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--fdl-key", required=True)
    parser.add_argument("--pca-key", required=True)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument(
        "--cell-type-source", choices=["combined", "own"], default="combined",
        help="'combined': overwrite with cell_type_final2 from the combined "
             "reference adata (default). 'own': keep the adata's own "
             "pre-existing new_cell_type column as-is.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--qc-dir", required=True)
    args = parser.parse_args()

    adata = sc.read_h5ad(args.path)

    if args.cell_type_source == "own":
        assert "new_cell_type" in adata.obs.columns, (
            f"{args.patient}: --cell-type-source=own requires an existing "
            "new_cell_type column in the input adata"
        )
        adata = adata[adata.obs["new_cell_type"].notna()].copy()
        adata.obs["new_cell_type"] = adata.obs["new_cell_type"].cat.remove_unused_categories()
    else:
        full_ct = sc.read_h5ad(FULL_ADATA_PATH, backed="r").obs["cell_type_final2"]
        new_ct = full_ct.reindex(adata.obs_names).cat.remove_unused_categories()
        if "FOXI1+ NonNE SCLC-N" in new_ct.cat.categories:
            new_ct = new_ct.cat.rename_categories({"FOXI1+ NonNE SCLC-N": "SCLC-N (FOXI1+)"})
        adata = adata[new_ct.notna()].copy()
        adata.obs["new_cell_type"] = new_ct[new_ct.notna()]

    present = set(adata.obs["new_cell_type"].unique())
    sclcn_label = "SCLC-N" if "SCLC-N" in present else ("SCLC-N (FOXI1+)" if "SCLC-N (FOXI1+)" in present else None)
    if sclcn_label is not None:
        end_candidates = ["LUSC", sclcn_label, "NonNE SCLC"]
    else:
        end_candidates = ["LUSC", "SCLC-A", "NonNE SCLC"]
    end_types = [t for t in end_candidates if t in present]

    assert "LUAD" in present, f"{args.patient}: LUAD not present, cannot pick start cell"
    assert len(end_types) >= 2, f"{args.patient}: fewer than 2 terminal types present ({end_types})"

    dm_res = palantir.utils.run_diffusion_maps(
        adata, n_components=args.n_components, pca_key=args.pca_key
    )
    palantir.utils.determine_multiscale_space(adata)

    eigvecs = dm_res["EigenVectors"]
    ct_series = adata.obs["new_cell_type"]

    assigned = assign_extreme_cells(eigvecs, ct_series, ["LUAD"] + end_types)

    picks = {}
    start_cell, comp, corr = assigned["LUAD"]
    picks["LUAD (start)"] = {"cell": start_cell, "component": int(comp), "corr": float(corr)}

    end_cells = {}
    for t in end_types:
        cell, comp, corr = assigned[t]
        end_cells[t] = cell
        picks[f"{t} (terminal)"] = {"cell": cell, "component": int(comp), "corr": float(corr)}

    with open(f"{args.qc_dir}/{args.patient}_picks.json", "w") as f:
        json.dump(picks, f, indent=2)
    print(json.dumps(picks, indent=2))

    highlight = pd.Series(
        ["LUAD (start)"] + list(end_cells.keys()),
        index=[start_cell] + list(end_cells.values()),
    )
    if highlight.index.duplicated().any():
        # two roles picked the same extreme cell (on different diffusion
        # components) -- palantir.plot.highlight_cells_on_umap chokes on a
        # duplicate index (annotation lookup returns a Series, not a scalar).
        # QC-only, so just note it and skip rather than block the actual
        # pseudotime computation below.
        dupes = highlight.index[highlight.index.duplicated()].tolist()
        print(f"WARNING: {args.patient}: cell(s) {dupes} picked for multiple roles {highlight.loc[dupes].tolist()}; skipping highlight plot")
    else:
        palantir.plot.highlight_cells_on_umap(adata, cells=highlight, embedding_basis=args.fdl_key)
        plt.savefig(f"{args.qc_dir}/{args.patient}_highlight_cells.png", dpi=150, bbox_inches="tight")
        plt.close()

    end_cells_series = pd.Series(list(end_cells.keys()), index=list(end_cells.values()))
    palantir.core.run_palantir(
        adata, start_cell, terminal_states=end_cells_series, knn=30,
        use_early_cell_as_start=True, num_waypoints=500,
        pseudo_time_key="palantir_pseudotime_slalom",
        entropy_key="palantir_entropy_slalom",
        fate_prob_key="palantir_fate_probabilities_slalom",
    )
    palantir.plot.plot_palantir_results(
        adata, s=3, embedding_basis=args.fdl_key,
        pseudo_time_key="palantir_pseudotime_slalom",
        entropy_key="palantir_entropy_slalom",
        fate_prob_key="palantir_fate_probabilities_slalom",
    )
    plt.savefig(f"{args.qc_dir}/{args.patient}_palantir_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(f"{args.qc_dir}/{args.patient}_end_types.json", "w") as f:
        json.dump({"start": "LUAD", "end_types": end_types}, f)

    adata.write_h5ad(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
