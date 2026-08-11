"""Dot plot: drug-target gene expression (dot size) x cell type, in Transformed
patients, colored by whether that gene's own expression is higher or lower in
Transformed vs Control (within the same cell type) -- with a top annotation
barplot showing the Transformed-vs-Control ABUNDANCE shift of each cell type.

Size = mean of each patient's own pseudobulk mean log-expression, averaged
within the Transformed group (patients missing a given cell type are excluded
from that cell type's average, not treated as zero).
Dot color = mean_log_expr(Transformed) - mean_log_expr(Control), per gene x
cell type -- a SEPARATE diverging scale from the bar's abundance-diff (same
red-up/blue-down-in-Transformed sign convention, different quantity/units, so
each gets its own colorbar). Gray dots = that cell type has 0 cells in
Control, so no expression difference can be computed there.
Bar = simple difference in mean per-patient relative abundance
(Transformed% - Control%), one value per cell type, aligned to the dot plot's
columns.

NOTE: abundance differences here are simple descriptive means, not all
statistically validated -- LUAD/SCLC-A/SCLC-N have the Dirichlet regression
(13_dirichlet_regression_full_celltypes.R, not significant at n=38);
LUSC/SCLC-P/NonNE SCLC/EMT do not.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CELL_META_CSV = "cell_metadata.full_celltypes.csv"
EXPR_CSV = "pseudobulk_gene_expression.full_celltypes.csv"
OUT_BASENAME = "dotplot_drug_targets_by_celltype"

GENES = [
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
CELL_TYPES = ["LUAD", "LUSC", "SCLC-A", "SCLC-N", "SCLC-P", "NonNE SCLC", "EMT"]  # final column order, left to right
GROUP = "Transformed"

# background color per drug-target category, matching the companion boxplot figure's panel colors
GENE_CATEGORY_COLOR = {
    "DLL3": "#FCF2C7",
    "SEZ6": "#FCF2C7",
    "CADM1": "#FCF2C7",
    "SSTR2": "#FCF2C7",
    "CD276": "#EBDDF0",
    "SLFN11": "#E1EEDA",
    "TACSTD2": "#DCE9F5",
    "CEACAM5": "#DCE9F5",
    "CEACAM6": "#DCE9F5",
}

SIZE_MIN, SIZE_MAX = 15, 220
# cap the expression-diff color scale well below the global max (driven by TACSTD2/CEACAM5/CEACAM6 in LUAD)
# so smaller-magnitude diffs (e.g. CADM1, SSTR2, whose own max is ~0.22) still show clearly saturated color
DOT_DIFF_CAP = 0.22
BAR_COLOR = "0.55"


def compute_abundance_diff():
    obs = pd.read_csv(CELL_META_CSV, index_col=0)
    counts = obs.groupby(["patient", "group", "cell_type"], observed=True).size().unstack(fill_value=0)
    for ct in CELL_TYPES:
        if ct not in counts.columns:
            counts[ct] = 0
    counts = counts[CELL_TYPES]
    frac = counts.div(counts.sum(axis=1), axis=0) * 100  # percent
    mean_pct = frac.groupby(level="group")[CELL_TYPES].mean()

    diff = mean_pct.loc["Transformed"] - mean_pct.loc["Control"]
    return diff, mean_pct


def compute_expression_means():
    expr = pd.read_csv(EXPR_CSV)
    # patient-level values already (one row per patient x group x cell_type x gene);
    # average across patients within each group
    group_mean = expr.groupby(["group", "cell_type", "gene"], observed=True)["mean_log_expr"].mean().reset_index()
    return group_mean


def compute_expression_diff(group_mean):
    wide = group_mean.pivot_table(index=["cell_type", "gene"], columns="group", values="mean_log_expr")
    wide["diff"] = wide.get("Transformed") - wide.get("Control")
    return wide["diff"]  # NaN where either group lacks that cell type


def main():
    abundance_diff, mean_pct = compute_abundance_diff()
    print("=== Mean relative abundance by group (%) ===")
    print(mean_pct.T)
    print("\n=== Abundance diff (Transformed - Control, pp) ===")
    print(abundance_diff)

    group_mean = compute_expression_means()
    sub = group_mean[group_mean["group"] == GROUP]
    expr_diff = compute_expression_diff(group_mean)
    print("\n=== Expression diff (Transformed - Control mean log-expr) ===")
    print(expr_diff.unstack("cell_type"))

    plot_cell_types = CELL_TYPES

    expr_max = sub["mean_log_expr"].max()
    expr_min = 0.0

    def size_for(value):
        frac = 0.0 if expr_max == expr_min else (value - expr_min) / (expr_max - expr_min)
        return SIZE_MIN + frac * (SIZE_MAX - SIZE_MIN)

    dot_cmap = plt.get_cmap("RdYlBu_r")  # blue-yellow-red diverging, so NA_COLOR gray reads distinctly
    dot_norm = plt.Normalize(vmin=-DOT_DIFF_CAP, vmax=DOT_DIFF_CAP, clip=True)
    NA_COLOR = "0.75"

    fig, (ax_bar, ax_dot) = plt.subplots(
        2, 1, figsize=(4.6, 5.0), sharex=True, gridspec_kw={"height_ratios": [1, 3.2], "hspace": 0.04}
    )

    ax_bar.bar(
        range(len(plot_cell_types)),
        [abundance_diff[ct] for ct in plot_cell_types],
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
        zorder=3,
    )
    ax_bar.axhline(0, color="0.4", linewidth=0.8, zorder=2)
    ax_bar.set_ylabel("Δ abundance\n(pp)", fontsize=9)
    ax_bar.grid(True, axis="y", color="0.9", linewidth=0.6, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(length=0, labelsize=9)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # Direction indicator (replaces the bar colorbar) -- bars have no color encoding now, so this
    # double-headed arrow just labels which way is up/down. Drawn directly on ax_bar in (axes-fraction
    # x, data y) so it's confined to the bar subplot's own vertical span and centered on its y=0 line.
    ymin, ymax = ax_bar.get_ylim()
    arrow_half = 0.7 * min(abs(ymin), ymax)
    dir_x = 1.06
    ax_bar.annotate(
        "",
        xy=(dir_x, arrow_half),
        xycoords=("axes fraction", "data"),
        xytext=(dir_x, -arrow_half),
        textcoords=("axes fraction", "data"),
        arrowprops=dict(arrowstyle="<->", lw=1.3, color="0.3", shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax_bar.annotate(
        "Transformed",
        xy=(dir_x, arrow_half),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        annotation_clip=False,
    )
    ax_bar.annotate(
        "Control",
        xy=(dir_x, -arrow_half),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        annotation_clip=False,
    )

    for gi, gene in enumerate(GENES):
        for ci, ct in enumerate(plot_cell_types):
            row = sub[(sub["gene"] == gene) & (sub["cell_type"] == ct)]
            value = row["mean_log_expr"].iloc[0] if len(row) else np.nan
            if np.isnan(value):
                continue
            d = expr_diff.get((ct, gene), np.nan)
            color = NA_COLOR if pd.isna(d) else dot_cmap(dot_norm(d))
            ax_dot.scatter(
                ci,
                gi,
                s=size_for(value),
                c=[color],
                edgecolors="black",
                linewidth=0.5,
                zorder=3,
            )

    ax_dot.set_xticks(range(len(plot_cell_types)))
    ax_dot.set_xticklabels(plot_cell_types, rotation=45, ha="right", fontsize=9.5)
    ax_dot.set_yticks(range(len(GENES)))
    ax_dot.set_yticklabels(GENES, fontsize=9.5)
    ax_dot.set_xlim(-0.5, len(plot_cell_types) - 0.5)
    ax_dot.set_ylim(-0.5, len(GENES) - 0.5)
    ax_dot.invert_yaxis()
    ax_dot.grid(True, color="0.9", linewidth=0.6, zorder=1)
    ax_dot.set_axisbelow(True)
    ax_dot.tick_params(length=0)
    for spine in ax_dot.spines.values():
        spine.set_visible(False)

    # shade each gene's y-tick label with its drug-target category color (matches companion boxplot figure)
    for tick_label in ax_dot.get_yticklabels():
        color = GENE_CATEGORY_COLOR[tick_label.get_text()]
        tick_label.set_bbox(dict(facecolor=color, edgecolor="none", pad=2, boxstyle="round,pad=0.15"))

    # Dot colorbar (expression diff) -- aligned with the dot subplot's vertical span; capped range means
    # TACSTD2/CEACAM5/CEACAM6's larger diffs are clipped to full saturation (no extend arrows shown)
    dot_cbar_ax = fig.add_axes([0.905, 0.38, 0.03, 0.22])
    dot_sm = plt.cm.ScalarMappable(norm=dot_norm, cmap=dot_cmap)
    dot_cbar = fig.colorbar(dot_sm, cax=dot_cbar_ax)
    dot_cbar.set_label("Δ mean log-expr\n(Transf. − Ctrl)", fontsize=8.5)
    dot_cbar.ax.tick_params(labelsize=8)

    legend_values = np.linspace(expr_min, expr_max, 4)
    legend_handles = [
        plt.scatter([], [], s=size_for(v), c="0.6", edgecolors="black", linewidth=0.5) for v in legend_values
    ]
    legend_labels = [f"{v:.2f}" for v in legend_values]
    fig.legend(
        legend_handles,
        legend_labels,
        title="mean log-norm.\nexpression\n(Transformed)",
        loc="center left",
        bbox_to_anchor=(0.90, 0.20),
        frameon=False,
        fontsize=8.5,
        title_fontsize=8.5,
        labelspacing=1.1,
    )

    fig.suptitle("Drug-target gene expression by cell type", fontsize=12, y=0.97)

    fig.subplots_adjust(left=0.24, right=0.87, top=0.90, bottom=0.16)

    fig.savefig(f"{OUT_BASENAME}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_BASENAME}.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT_BASENAME}.pdf / .png")


if __name__ == "__main__":
    main()
