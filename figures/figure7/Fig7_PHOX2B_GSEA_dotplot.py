"""
Fig 7 - PHOX2B-overexpression GSEA dotplot (H1975 vs PC9 engineered cell lines).

Reproduces main_plot_cleaned_narrow.png from PHOX2B_DEG_MAST_GSEA.ipynb: balanced
top +/-10 NES terms per cell line (curated_small gene sets, 500 permutations,
FDR < 0.1 for H1975 / FDR < 0.2 for PC9), terms shared between both cell lines
highlighted in red/bold, with the shared legend/colorbar rendered as a separate
figure so it can be placed independently in the panel layout.
"""
import os

import gseapy
import matplotlib.pyplot as plt

from phox2b_gsea_helper import (
    CELL_LINE_DOTPLOT_CUTOFF,
    CELL_LINE_QVAL_THRESH,
    format_and_color_labels,
    get_balanced_top_terms,
    prepare_subset_for_plotting,
    run_gsea_pipeline,
)

out_dir = "../../results/figures/figure7"
os.makedirs(out_dir, exist_ok=True)

results = run_gsea_pipeline()

h1975_top = get_balanced_top_terms(results["H1975"], q_val_thresh=CELL_LINE_QVAL_THRESH["H1975"])
pc9_top = get_balanced_top_terms(results["PC9"], q_val_thresh=CELL_LINE_QVAL_THRESH["PC9"])
common_terms = set(h1975_top.Term) & set(pc9_top.Term)
print(f"Found {len(common_terms)} common terms after collapsing.")

h1975_plot = prepare_subset_for_plotting(results["H1975"], h1975_top.Term)
pc9_plot = prepare_subset_for_plotting(results["PC9"], pc9_top.Term)

fig_main, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6),
                                     gridspec_kw={"width_ratios": [1, 1]})
fig_main.subplots_adjust(left=0.05, wspace=2)
ax1.set_box_aspect(2.5)
ax2.set_box_aspect(2.5)
gseapy.dotplot(h1975_plot, column="FDR q-val", top_term=h1975_plot.shape[0],
               cutoff=CELL_LINE_DOTPLOT_CUTOFF["H1975"],
               ax=ax1, title="H1975", show_ring=True)
format_and_color_labels(ax1, common_terms)

gseapy.dotplot(pc9_plot, column="FDR q-val", top_term=pc9_plot.shape[0],
               cutoff=CELL_LINE_DOTPLOT_CUTOFF["PC9"],
               ax=ax2, title="PC9", show_ring=True)
format_and_color_labels(ax2, common_terms)

original_legend = ax2.get_legend()
legend_handles = original_legend.legend_handles
legend_labels = [t.get_text() for t in original_legend.get_texts()]
mappable = ax1.collections[0]

fig_leg = plt.figure(figsize=(2.5, 2))
fig_leg.legend(handles=legend_handles, labels=legend_labels,
               title="% Genes in Set", loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 0.95))

cax = fig_leg.add_axes([0.3, 0.4, 0.3, 0.05])
cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
cbar.set_label("log10(1/FDR)")

fig_leg.savefig(os.path.join(out_dir, "Fig7_PHOX2B_GSEA_dotplot_legend.png"),
                 bbox_inches="tight", dpi=300, transparent=True)
plt.close(fig_leg)

if ax1.get_legend():
    ax1.get_legend().remove()
if ax2.get_legend():
    ax2.get_legend().remove()
extra_axes = [a for a in fig_main.axes if a not in [ax1, ax2]]
for extra_ax in extra_axes:
    extra_ax.remove()

fig_main.savefig(os.path.join(out_dir, "Fig7_PHOX2B_GSEA_dotplot_main.png"),
                  bbox_inches="tight", dpi=300)
plt.close(fig_main)

print(f"Saved main dotplot and legend to {out_dir}")
