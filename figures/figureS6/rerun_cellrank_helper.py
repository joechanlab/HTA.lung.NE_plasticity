"""Stage 2 of the RU1518-style trajectory rerun: CellRank macrostates + fate probabilities.

Runs under the `cellrank` conda env. Takes the Stage 1 output (has the
freshly-recomputed palantir_pseudotime_slalom + new_cell_type) and follows
the RU1518 section of step2.CellRank_macrostates_individual.template.ipynb:
PseudotimeKernel -> GPCCA -> compute_macrostates -> pick terminal/initial
states matching the requested cell types -> compute_fate_probabilities.

Terminal macrostates are picked automatically: after fitting, macrostates
are grouped by their cell-type prefix (e.g. "LUSC_2" -> "LUSC") and, for
each target type, the largest (by cell count) matching macrostate is chosen
as that type's representative terminal state. Saves QC plots at each step.
"""
import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import cellrank as cr


def best_macrostate_for(categories_counts, target_type):
    # categories_counts: dict of macrostate name -> cell count
    candidates = []
    for name, count in categories_counts.items():
        m = re.match(r"^(.*?)(?:_(\d+))?$", name)
        prefix = m.group(1)
        if prefix == target_type:
            candidates.append((count, name))
    if not candidates:
        return None
    return max(candidates)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--fdl-key", required=True)
    parser.add_argument("--start-type", required=True)
    parser.add_argument("--end-types", required=True, help="comma-separated")
    parser.add_argument("--n-states", type=int, default=8)
    parser.add_argument("--n-cells", type=int, default=30)
    parser.add_argument("--out", required=True)
    parser.add_argument("--qc-dir", required=True)
    args = parser.parse_args()

    end_types = args.end_types.split(",")

    adata = sc.read_h5ad(args.path)
    adata.obsm["X_umap"] = adata.obsm[args.fdl_key].copy()
    time_key = "palantir_pseudotime_slalom"

    pk = cr.kernels.PseudotimeKernel(adata, time_key=time_key)
    pk.compute_transition_matrix(threshold_scheme="soft")

    with plt.rc_context({"figure.figsize": (6, 6)}):
        pk.plot_projection(basis=args.fdl_key, recompute=True)
    plt.savefig(f"{args.qc_dir}/{args.patient}_pk_projection.png", dpi=150, bbox_inches="tight")
    plt.close()

    g = cr.estimators.GPCCA(pk)
    if "macrostate_fwd_colors" in adata.uns:
        del adata.uns["macrostate_fwd_colors"]
    ct_key = "new_cell_type"

    g.compute_schur()
    g.compute_macrostates(n_states=args.n_states, n_cells=args.n_cells, cluster_key=ct_key)
    g.plot_macrostates(which="all", discrete=True, s=30, legend_loc="on data")
    plt.savefig(f"{args.qc_dir}/{args.patient}_macrostates_all.png", dpi=150, bbox_inches="tight")
    plt.close()

    counts = adata.obs["macrostates_fwd"].value_counts()
    counts = counts[counts > 0].to_dict()

    initial_state = best_macrostate_for(counts, args.start_type)
    terminal_states = []
    picked = {}
    for t in end_types:
        ms = best_macrostate_for(counts, t)
        if ms is not None:
            terminal_states.append(ms)
            picked[t] = ms
        else:
            picked[t] = None

    with open(f"{args.qc_dir}/{args.patient}_macrostate_picks.json", "w") as f:
        json.dump({"initial": initial_state, "terminal": picked}, f, indent=2)
    print(json.dumps({"initial": initial_state, "terminal": picked}, indent=2))

    assert len(terminal_states) == len(end_types), f"{args.patient}: missing macrostate for some end types: {picked}"

    g.set_terminal_states(states=terminal_states, allow_overlap=True)
    if initial_state is not None:
        g.set_initial_states(states=[initial_state], allow_overlap=True)
    else:
        # the start type (e.g. LUAD) is a transitional root, not a metastable state, so
        # GPCCA may not carve out a distinct macrostate for it even at high n_states.
        # set_initial_states is not required for compute_fate_probabilities (which only
        # depends on terminal_states) -- it's only used by other downstream analyses we
        # aren't running here, so it's safe to skip.
        print(f"NOTE: no macrostate found for start type {args.start_type}; skipping set_initial_states")
    g.plot_macrostates(which="terminal", s=100, title="", legend_loc="on data", legend_fontsize=14, time_key=time_key)
    plt.savefig(f"{args.qc_dir}/{args.patient}_macrostates_terminal.png", dpi=150, bbox_inches="tight")
    plt.close()

    # gmres (default) failed to converge for some patients without petsc4py installed.
    # use_petsc=True (the default) silently overrides solver="direct" back to gmres
    # when petsc4py is missing, so use_petsc must also be disabled explicitly.
    # These datasets are small enough for a direct scipy solve to be practical.
    g.compute_fate_probabilities(solver="direct", use_petsc=False)
    g.plot_fate_probabilities(legend_loc="right")
    plt.savefig(f"{args.qc_dir}/{args.patient}_fate_probabilities.png", dpi=150, bbox_inches="tight")
    plt.close()

    if len(terminal_states) >= 3:
        cr.pl.circular_projection(
            g.adata, keys=[ct_key], lineages=terminal_states,
            title="", size=10, legend_loc="none",
            dpi=150, figsize=(6, 6), lineage_order="default",
        )
        plt.savefig(f"{args.qc_dir}/{args.patient}_circular_projection.png", dpi=150, bbox_inches="tight")
        plt.close()

    g.adata.write_h5ad(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
