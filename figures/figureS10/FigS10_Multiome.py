import os
from functools import reduce
import re
import scanpy as sc
import scanpy.external as sce
import pandas as pd
import numpy as np
import gseapy as gp
import muon as mu
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


sc.settings.figdir = "../../results/figures/figure7/"
sc.set_figure_params(
    dpi=120,
    dpi_save=600,
    figsize=(3, 2),
    fontsize=9,             # base font size for labels and ticks
    color_map='Spectral_r',
    frameon=False,
    vector_friendly=True,
)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.grid": False,
    "axes.titlesize": 10,       # titles like “PHOX2B expression”
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,         # embeds text as editable in Illustrator
    "ps.fonttype": 42,
})

def extract_tf(reg_name):
    return re.split(r'_', reg_name, maxsplit=1)[0]

gene_signature_path = "/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/external_data/phox2b/NE_MAPK_EGFR_gene_sets.xlsx"
def load_gene_sets_from_excel(
    excel_path,
    strip_whitespace = True,
):
    xls = pd.ExcelFile(excel_path)

    gene_sets = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        series = df.iloc[:, 0]
        genes = series.astype(str)
        if strip_whitespace:
            genes = genes.str.strip()
        genes = genes[genes != ""]
        gene_sets[sheet_name] = set(genes)
    return gene_sets

gene_sets = load_gene_sets_from_excel(gene_signature_path)

adata_RU151_downsampled = sc.read_h5ad(f'../../external_data/phox2b/RU151_PDX_downsampled.h5ad')
patients = ["RU581_LN", "RU151_PDX"]

# # RNA plot (original)
# adata_dict = {}
# for patient in patients:
#     adata = sc.read_h5ad(f'../../external_data/phox2b/{patient}.h5ad')
#     adata = adata[adata.obs['cell_type'] != 'Mixed', :].copy()
#     if patient == "RU151_PDX":
#         adata = adata[adata.obs.index.isin(adata_RU151_downsampled.obs.index),:].copy()
#     genes_of_interest = ["PHOX2B", "FOXA1", "SOX2", "REST", "EGFR"]
#     fig = sc.pl.umap(
#         adata,
#         color=genes_of_interest,
#         frameon=False,
#         legend_fontoutline=2,
#         wspace=0,
#         ncols=5,
#         use_raw=False,
#         return_fig=True,
#         s = 70 if patient == "RU151_PDX" else 30,
#         alpha = 1
#     )
#     fig.figure.text(
#         0.02, 0.5,  
#         f"{patient}".replace("_", " "), 
#         va='center',
#         ha='center',
#         rotation='vertical',
#         fontsize=18,
#         fontweight='bold'
#         )
#     for ax in fig.axes:
#         ax.title.set_fontsize(18)
#     fig.figure.savefig(os.path.join(sc.settings.figdir, 
#                                     f"umap_{patient}_fig7_genes.pdf"), dpi=300, bbox_inches="tight")
#     adata_dict[patient] = adata

# RNA plot (magic smoothed)
adata_dict = {}
for patient in patients:
    adata = sc.read_h5ad(f'../../external_data/phox2b/{patient}.h5ad')
    adata = adata[adata.obs['cell_type'] != 'Mixed', :].copy()
    genes_of_interest = ["PHOX2B", "FOXA1", "SOX2", "REST", "EGFR"]
    adata_magic = sce.pp.magic(
        adata,
        knn=3,
        t=1,
        name_list=genes_of_interest,
        copy=True
    )
    adata_magic.obsm['X_umap'] = adata.obsm["X_umap"].copy()
    if patient == "RU151_PDX":
        adata_magic = adata_magic[adata_magic.obs.index.isin(adata_RU151_downsampled.obs.index),:].copy()
    fig = sc.pl.umap(
        adata_magic,
        color=genes_of_interest,
        frameon=False,
        legend_fontoutline=2,
        wspace=0,
        ncols=5,
        use_raw=False,
        return_fig=True,
        s = 70 if patient == "RU151_PDX" else 30,
        alpha = 1
    )
    fig.figure.text(
        0.02, 0.5,  
        f"{patient}".replace("_", " "), 
        va='center',
        ha='center',
        rotation='vertical',
        fontsize=18,
        fontweight='bold'
        )
    for ax in fig.axes:
        ax.title.set_fontsize(18)
    fig.figure.savefig(os.path.join(sc.settings.figdir, 
                                    f"umap_{patient}_fig7_genes_magic.pdf"), dpi=300, bbox_inches="tight")
    adata_dict[patient] = adata

# # cell type plot
# adata_dict = {}
# cell_type_levels = {
#     "RU581_LN": ['SCLC-A', 'B-cell', 'T-cell'],
#     "RU151_PDX": ['NE-high', 'NE-intermediate', 'NonNE']
# }
# for patient in patients:
#     adata = sc.read_h5ad(f'../../external_data/phox2b/{patient}.h5ad')
#     if patient == "RU581_LN":
#         adata = adata[adata.obs['cell_type'] != 'Mixed', :].copy()
#     if patient == "RU151_PDX":
#         adata = adata[adata.obs.index.isin(adata_RU151_downsampled.obs.index),:].copy()
#     adata.obs['cell_type'] = pd.Categorical(adata.obs['cell_type'], 
#                                             ordered=True,
#                                             categories=cell_type_levels[patient])
    
#     fig = sc.pl.umap(
#         adata,
#         color="cell_type",
#         frameon=False,
#         legend_fontoutline=2,
#         wspace=0,
#         ncols=5,
#         use_raw=False,
#         return_fig=True,
#         legend_loc='on data',
#         title='cell type'
#     )
#     fig.figure.text(
#         0.02, 0.5,  
#         f"{patient}".replace("_", " "), 
#         va='center',
#         ha='center',
#         rotation='vertical',
#         fontsize=18,
#         fontweight='bold'
#         )
#     for ax in fig.axes:
#         ax.title.set_fontsize(18)
#     fig.figure.savefig(os.path.join(sc.settings.figdir, 
#                                     f"umap_{patient}_fig7_cell_type.pdf"), dpi=300, bbox_inches="tight")
#     adata_dict[patient] = adata

#----------------------------------
# Dot plot (multiome data)
# sc.set_figure_params(fontsize=14)
# patients = ["RU581_LN", "RU151_PDX"]
# cell_type_levels = {
#     "RU581_LN": ['SCLC-A', 'T-cell', 'B-cell'],
#     "RU151_PDX": ['NE-high', 'NE-intermediate', 'NonNE']
# }

# # Marker genes to validate the clusters
# for patient in patients:
#     # Load and subset as in your UMAP code
#     adata = sc.read_h5ad(f'../../external_data/phox2b/{patient}.h5ad')
    
#     if patient == "RU581_LN":
#         adata = adata[adata.obs['cell_type'] != 'Mixed', :].copy()
#     genes_of_interest = [ 
#                         "CHGA", "CHGB", "SYP", "ENO2", "NCAM1", "ASCL1", "NEUROD1", "POU2F3",
#                         "NKX2-1", "NAPSA", "KRT7",
#                         "YAP1", "WWTR1",
#                         "PTPRC", "CD3E", "MS4A1",
#                         ]
#     marker_groups = {
#         "TF": ["PHOX2B", "FOXA1", "SOX2", "REST", "EGFR"],
#         "Epithelial": ["EPCAM"],
#         "NE": ["CHGA", "CHGB", "SYP", "ENO2", "NCAM1", "ASCL1", "NEUROD1", "POU2F3"],
#         "NSCLC": ["NKX2-1", "NAPSA", "KRT7"],
#         "Non-NE": ["YAP1", "WWTR1"],
#         "Immune": ["PTPRC", "CD3E", "MS4A1"],
#         "Mesenchymal": ["VIM", "COL1A1"],
#         "Endothelial": ["PECAM1", "CLDN5"],
#         #"RBC": ["HBA1"],
#     }
#     # Ensure consistent cell type ordering
#     adata.obs['cell_type'] = pd.Categorical(
#         adata.obs['cell_type'],
#         ordered=True,
#         categories=cell_type_levels[patient]
#     )

#     grouped_var_names = {}
#     for group, genes in marker_groups.items():
#         present = [g for g in genes if g in adata.var_names]
#         if len(present) > 0:
#             grouped_var_names[group] = present

#     # generate dotplot – transposed, with group annotations
#     fig = sc.pl.dotplot(
#         adata,
#         var_names=grouped_var_names,   # dict = groups + genes
#         groupby="cell_type",
#         use_raw=False,
#         standard_scale="var",
#         swap_axes=False,                # cell types on y, genes on x (like your panel E)
#         var_group_rotation=90,         # rotate "Epithelial", "NE", ... labels
#         colorbar_title="Expression",
#         cmap="Reds",                   # to match the reference style
#         return_fig=True,
#         title=patient.replace("_", " "),
#     )

#     # optional cosmetic tweaks
#     outpath = os.path.join(
#         sc.settings.figdir,
#         f"dotplot_{patient}_figS_markers_by_celltype.pdf",
#     )
#     fig.savefig(outpath, dpi=300, bbox_inches="tight")

#----------------------------------
# NE and MAPK and EGFR gene set score (multiome data)
# sc.set_figure_params(
#     dpi=120,
#     dpi_save=600,
#     figsize=(3.5, 3.7),
#     fontsize=10,             # base font size for labels and ticks
#     color_map='Spectral_r',
#     frameon=False,
#     vector_friendly=True,
# )
# mpl.rcParams.update({
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
#     "axes.grid": False,
#     "axes.titlesize": 14,       # titles like “PHOX2B expression”
#     "axes.titleweight": "bold",
#     "axes.labelsize": 14,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "legend.fontsize": 10,
#     "pdf.fonttype": 42,         # embeds text as editable in Illustrator
#     "ps.fonttype": 42,
# })
# from scipy.stats import mannwhitneyu
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# import scanpy as sc
# import pandas as pd

# def p_to_stars(p):
#     if p < 1e-10:
#         return '***'
#     elif p < 1e-5:
#         return '**'
#     elif p < 0.05:
#         return '*'
#     else:
#         return 'ns'

# patients = ["RU581_LN", "RU151_PDX"]

# cell_type_levels = {
#     "RU581_LN": ['SCLC-A', 'T-cell', 'B-cell'],
#     "RU151_PDX": ['NE-high', 'NE-intermediate', 'NonNE'],
# }

# phox_high = {
#     "RU581_LN": "SCLC-A",
#     "RU151_PDX": "NE-high",
# }

# # which labels are pooled as control
# control_groups = {
#     "RU581_LN": ["T-cell", "B-cell"],      # adjust if you have an epithelial control
#     "RU151_PDX": ["NonNE", "NE-intermediate"],                # or ["NonNE", "NE-intermediate"]
# }

# for patient in patients:
#     # ---- load / subset ----
#     adata = sc.read_h5ad(f'/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/external_data/phox2b/{patient}.h5ad')

#     if patient == "RU581_LN":
#         adata = adata[adata.obs['cell_type'] != 'Mixed', :].copy()

#     adata.obs['cell_type'] = pd.Categorical(
#         adata.obs['cell_type'],
#         ordered=True,
#         categories=cell_type_levels[patient]
#     )

#     # ---- gene set intersection ----
#     ne_genes = list(gene_sets["NE_Zhang"])
#     mapk_genes = list(gene_sets["KEGG_MAPK_SIGNALING"])
#     egfr_genes = list(gene_sets["REACTOME_SIGNALING_BY_EGFR"])
#     ne_genes_use = [g for g in ne_genes if g in adata.var_names]
#     mapk_genes_use = [g for g in mapk_genes if g in adata.var_names]
#     egfr_genes_use = [g for g in egfr_genes if g in adata.var_names]

#     # ---- scores ----
#     sc.tl.score_genes(adata, ne_genes_use,   score_name="NE Score",   use_raw=False)
#     sc.tl.score_genes(adata, mapk_genes_use, score_name="MAPK Score", use_raw=False)
#     sc.tl.score_genes(adata, egfr_genes_use, score_name="EGFR Score", use_raw=False)
#     # ---- define high vs control ----
#     high_label = phox_high[patient]
#     ctrl_labels = control_groups[patient]

#     high_mask = adata.obs['cell_type'] == high_label
#     ctrl_mask = adata.obs['cell_type'].isin(ctrl_labels)

#     ne_high   = adata.obs.loc[high_mask, "NE Score"].values
#     ne_ctrl   = adata.obs.loc[ctrl_mask, "NE Score"].values
#     mapk_high = adata.obs.loc[high_mask, "MAPK Score"].values
#     mapk_ctrl = adata.obs.loc[ctrl_mask, "MAPK Score"].values
#     egfr_high = adata.obs.loc[high_mask, "EGFR Score"].values
#     egfr_ctrl = adata.obs.loc[ctrl_mask, "EGFR Score"].values

#     # ---- stats: one-sided Mann–Whitney ----
#     stat_ne,   p_ne   = mannwhitneyu(ne_high,   ne_ctrl,   alternative="greater")
#     stat_mapk, p_mapk = mannwhitneyu(mapk_high, mapk_ctrl, alternative="less")
#     stat_egfr, p_egfr = mannwhitneyu(egfr_high, egfr_ctrl, alternative="less")

#     ne_dmed   = np.median(ne_high)   - np.median(ne_ctrl)
#     mapk_dmed = np.median(mapk_high) - np.median(mapk_ctrl)
#     egfr_dmed = np.median(egfr_high) - np.median(egfr_ctrl)

#     print(f"\n{patient}")
#     print(f"  NE   Δmedian = {ne_dmed:.3f}, p = {p_ne:.2e}")
#     print(f"  MAPK Δmedian = {mapk_dmed:.3f}, p = {p_mapk:.2e}")
#     print(f"  EGFR Δmedian = {egfr_dmed:.3f}, p = {p_egfr:.2e}")

#     # ---- violin plot ----
#     sc.pl.violin(
#         adata,
#         keys=["NE Score", "MAPK Score", "EGFR Score"],
#         groupby="cell_type",
#         jitter=0.4,
#         rotation=45,
#         stripplot=False,
#         show=False,
#         alpha=0.5
#     )

#     fig = plt.gcf()
#     axes = fig.axes      # axes[0] = NE Score, axes[1] = MAPK Score, axes[2] = EGFR Score

#     # helper to add bracket + stars + Δmedian on one axis
#     def annotate_axis(ax, p, dmed, high_label, ctrl_labels):
#         # get x positions and labels from the current axis
#         xticks = ax.get_xticks()
#         xticklabels = [t.get_text() for t in ax.get_xticklabels()]

#         # positions of PHOX2B-high and pooled control
#         pos_high = xticks[xticklabels.index(high_label)]
#         pos_ctrl = np.mean([xticks[xticklabels.index(lab)] for lab in ctrl_labels])

#         # y position for bracket
#         y_min, y_max = ax.get_ylim()
#         y_span = y_max - y_min
#         y_bracket = y_max - 0.08 * y_span

#         # draw bracket
#         ax.plot(
#             [pos_high, pos_high, pos_ctrl, pos_ctrl],
#             [y_bracket - 0.01 * y_span, y_bracket, y_bracket, y_bracket - 0.01 * y_span],
#             linewidth=1
#         )

#         # Δmedian text in top-right corner of this panel
#         ax.text(
#             0.65, 0.97, 
#             f"Δmedian={dmed:.2f}",
#             transform=ax.transAxes,
#             ha="right",
#             va="center"
#         )

#         # stars above bracket
#         ax.text(
#             0.45,0.85,
#             p_to_stars(p),
#             transform=ax.transAxes,
#             ha="center",
#             va="bottom"
#         )

#     # annotate NE (axes[0]) and MAPK (axes[1])
#     annotate_axis(axes[0], p_ne,   ne_dmed,   high_label, ctrl_labels)
#     annotate_axis(axes[1], p_mapk, mapk_dmed, high_label, ctrl_labels)
#     annotate_axis(axes[2], p_egfr, egfr_dmed, high_label, ctrl_labels)

#     patient_title = patient.replace("_", " ")
#     fig.suptitle(patient_title, x=0.4, fontsize=14)
#     for ax in axes:
#         ax.yaxis.set_major_locator(MultipleLocator(0.2))
#         ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#     plt.tight_layout()
#     plt.show()

#----------------------------------
# To move to LCNEC (selecting PDX for IHC for Rishika)
# patients = ["RU1305", "RU1303", "RU1405"]
# color_map = {"NSCLC": "gold", "NonNE SCLC": "tab:purple", "SCLC-A": "tab:red", "SCLC-N": "tab:cyan", "SCLC-P": "tab:blue"}
# adatas = [
#     "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/RU1305.no_cc.hvg_2000.090124.h5ad",
#     "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/RU1303.no_cc.hvg_2000.090124.h5ad",
#     "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/RU1405.no_cc.hvg_2000.090124.h5ad"]
# for i, patient in enumerate(patients):
#     adata = sc.read_h5ad(adatas[i])
#     fig = sc.pl.umap(adata, color=["FOXA2", "HNF1A", "HNF4A", 'cell_type_final2'], frameon=False, use_raw=False, 
#                      s = 30, wspace=0, ncols=4, add_outline=True, return_fig=True, palette=color_map,
#                      vmax=0.5, vmin=0)
#     fig.suptitle(patient.replace("_", " "), x=0.45, y = 1.1)

#----------------------------------
# Dot plot (scRNAseq data)
# sc.set_figure_params(fontsize=14)
# patients = ["RU581_LN", "RU151_PDX"]
# adatas = [
#     "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/RU581.no_cc.hvg_2000.090124.h5ad",
#     "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/RU151.no_cc.hvg_2000.090124.h5ad",]
# cell_type_levels = ['NSCLC', 'SCLC-A', 'SCLC-N', 'SCLC-P', 'NonNE SCLC']

# # Marker genes to validate the clusters
# for i, patient in enumerate(patients):
#     # Load and subset as in your UMAP code
#     adata = sc.read_h5ad(adatas[i])
#     genes_of_interest = [ 
#                         "CHGA", "CHGB", "SYP", "ENO2", "NCAM1", "ASCL1", "NEUROD1", "POU2F3",
#                         "NKX2-1", "NAPSA", "KRT7",
#                         "YAP1", "WWTR1",
#                         "PTPRC", "CD3E", "MS4A1",
#                         ]
#     marker_groups = {
#         "TF": ["PHOX2B", "FOXA1", "SOX2", "REST", "EGFR"],
#         "Epithelial": ["EPCAM"],
#         "NE": ["CHGA", "CHGB", "SYP", "ENO2", "NCAM1", "ASCL1", "NEUROD1", "POU2F3"],
#         "NSCLC": ["NKX2-1", "NAPSA", "KRT7"],
#         "Non-NE": ["YAP1", "WWTR1"],
#         #"Immune": ["PTPRC", "CD3E", "MS4A1"],
#         "Mesenchymal": ["VIM", "COL1A1"],
#         "Endothelial": ["PECAM1", "CLDN5"],
#         #"RBC": ["HBA1"],
#     }
#     # Ensure consistent cell type ordering
#     adata.obs['cell_type'] = pd.Categorical(
#         adata.obs['cell_type_final2'],
#         ordered=True,
#         categories=[x for x in cell_type_levels if x in adata.obs['cell_type_final2'].unique()]
#     )

#     grouped_var_names = {}
#     for group, genes in marker_groups.items():
#         present = [g for g in genes if g in adata.var_names]
#         if len(present) > 0:
#             grouped_var_names[group] = present

#     # generate dotplot – transposed, with group annotations
#     fig = sc.pl.dotplot(
#         adata,
#         var_names=grouped_var_names,   # dict = groups + genes
#         groupby="cell_type",
#         use_raw=False,
#         standard_scale="var",
#         swap_axes=False,                # cell types on y, genes on x (like your panel E)
#         var_group_rotation=90,         # rotate "Epithelial", "NE", ... labels
#         colorbar_title="Expression",
#         cmap="Reds",                   # to match the reference style
#         return_fig=True,
#         title=patient.replace("_", " "),
#     )

#     # optional cosmetic tweaks
#     outpath = os.path.join(
#         sc.settings.figdir,
#         f"dotplot_{patient}_figS_markers_by_celltype_(scRNAseq).pdf",
#     )
#     fig.savefig(outpath, dpi=300, bbox_inches="tight")


#----------------------------------
# FDL for RU151 scRNAseq data
# color_map = {"NSCLC": "gold", "NonNE SCLC": "tab:purple", "SCLC-A": "tab:red", "SCLC-N": "tab:cyan", "SCLC-P": "tab:blue"}
# adata = sc.read_h5ad("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/RU151.no_cc.hvg_2000.090124.h5ad")

# fig = sc.pl.embedding(adata, basis = "X_draw_graph_knn30_curated", color=["PHOX2B", "FOXA1", "SOX2", "REST", "EGFR"], 
#                frameon=False, wspace=0, ncols=5, use_raw=False, s = 30,
#                 add_outline=True, palette = color_map, return_fig=True)
# fig.figure.text(
#     0.02, 0.5,  
#     "RU151 PDX", 
#     va='center',
#     ha='center',
#     rotation='vertical',
#     fontsize=18,
#     fontweight='bold'
#     )
# for ax in fig.axes:
#     ax.title.set_fontsize(18)

# fig = sc.pl.embedding(adata, color=["cell_type_final2"], basis = "X_draw_graph_knn30_curated", palette=color_map,
#                frameon=False, wspace=0, ncols=5, use_raw=False, s = 30, 
#                add_outline=True, return_fig=True, title = 'cell type')
# fig.figure.text(
#     0.02, 0.5,  
#     "RU151 PDX", 
#     va='center',
#     ha='center',
#     rotation='vertical',
#     fontsize=18,
#     fontweight='bold'
#     )
# for ax in fig.axes:
#     ax.title.set_fontsize(18)

#----------------------------------
# FDL for RU581 scRNAseq data
# color_map = {"NSCLC": "gold", "NonNE SCLC": "tab:purple", "SCLC-A": "tab:red", "SCLC-N": "tab:cyan", "SCLC-P": "tab:blue"}
# adata = sc.read_h5ad("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/RU581.no_cc.hvg_2000.090124.h5ad")
# fig = sc.pl.embedding(adata, basis = "X_draw_graph_slalom_harmony_knn_15", color=["PHOX2B", "FOXA1", "SOX2", "REST", "EGFR"], 
#                frameon=False, wspace=0, ncols=5, use_raw=False, s = 30, palette = color_map, return_fig=True, add_outline=True)
# fig.figure.text(
#     0.02, 0.5,  
#     "RU581 LN", 
#     va='center',
#     ha='center',
#     rotation='vertical',
#     fontsize=18,
#     fontweight='bold'
#     )
# for ax in fig.axes:
#     ax.title.set_fontsize(18)

# fig = sc.pl.embedding(adata, color=["cell_type_final2"], basis = "X_draw_graph_slalom_harmony_knn_15", palette=color_map,
#                frameon=False, wspace=0, ncols=5, use_raw=False, s = 30, return_fig=True, title = 'cell type', add_outline=True)
# fig.figure.text(
#     0.02, 0.5,  
#     "RU581 LN", 
#     va='center',
#     ha='center',
#     rotation='vertical',
#     fontsize=18,
#     fontweight='bold'
#     )
# for ax in fig.axes:
#     ax.title.set_fontsize(18)
    

# RU581
# TFs = ["PHOX2B", "FOXA1"]
# base_dir = "/data1/chanj3/HTA.multiome_plasticity.results/scenicplus/runs/RU581_LN_2/"
# direct_regulon = pd.read_csv(os.path.join(base_dir, "scplus/eRegulons_direct.tsv"), sep="\t")
# direct_auc = mu.read_h5mu(os.path.join(base_dir, "scplus/AUCell_direct.h5mu"))
# regulon_names = direct_auc.mod['Gene_based'].var_names
# TF_regulon_names = {}
# for TF in TFs:
#     contains_TF = [TF+"_" in x for x in regulon_names]
#     TF_regulon_names[TF] = regulon_names[contains_TF]
# regulon_names_of_interest = reduce(lambda a, b: a.append(b), TF_regulon_names.values())
# adata = sc.read_h5ad(os.path.join(base_dir, "rna/adata.h5ad"))
# adata = adata[direct_auc.mod['Gene_based'].obs_names,:].copy()
# for regulon_name in regulon_names_of_interest:
#     adata.obs[regulon_name] = pd.Series(direct_auc.mod['Gene_based'][:,regulon_name].X.flatten(), index=direct_auc.obs_names)
# labels = pd.read_csv('/data1/chanj3/HTA.multiome_plasticity.results/annotations/cell_types_v4_with_RU581_LN.csv', index_col=0)
# labels['sample'] = labels.index.str.split('_').str[:2].str.join('_')
# labels['barcode'] = labels.index.str.split('_').str[2]
# labels.index = labels.barcode + '___' + labels['sample']
# adata.obs['cell_type'] = labels['cell_type']
# adata = adata[~adata.obs.cell_type.str.contains('Mixed')].copy()
# tf_names = [extract_tf(x) for x in direct_auc.mod['Gene_based'].var_names]
# auc = direct_auc.mod['Gene_based'].to_df()
# corr_matrix = auc.corr(method='pearson')
# tf_series = pd.Series(tf_names, index=corr_matrix.index)
# mask_diff_tf = tf_series.values[:, None] != tf_series.values[None, :]
# background_corrs = (
#     corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#                .where(mask_diff_tf)
#                .stack()
# )
# PHOX2B_pattern = 'PHOX2B_direct_+/+'
# FOXA1_pattern = 'FOXA1_direct_+/+'
# PHOX2B_col = corr_matrix.columns[corr_matrix.columns.str.contains(PHOX2B_pattern)]
# FOXA1_col = ""
# fig = sc.pl.umap(adata, color = ['PHOX2B_direct_+/+_(50g)', 'FOXA1_direct_+/+_(360g)'], 
#                  title=['PHOX2B regulon score', "FOXA1 regulon score"],
#                  frameon=False, s = 30,
#                  return_fig=True)
# fig.figure.text(
#     0.02, 0.5,  
#     "RU581", 
#     va='center',
#     ha='center',
#     rotation='vertical',
#     fontsize=12,
#     fontweight='bold'
#     )
# for ax in fig.axes:
#     ax.title.set_fontsize(12)
# fig.figure.savefig(os.path.join(sc.settings.figdir, 
#                    f"umap_multiome_RU581.pdf"), dpi=300, bbox_inches="tight")

# # B
# corr = adata.obs[['PHOX2B_direct_+/+_(50g)',
#                  'FOXA1_direct_+/+_(360g)']].corr()
# interest_corrs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()

# plt.figure(figsize=(3, 2))
# sns.histplot(background_corrs, bins=50, kde=True, color='lightgray')
# colors = ['red', 'blue', 'green']
# for i, ((tf1, tf2), val) in enumerate(interest_corrs.items()):
#     plt.axvline(val, color=colors[i % len(colors)], linewidth=1.5)
#     plt.text(val + 0.05, 
#              plt.gca().get_ylim()[1] * 0.9,  # y position ~90% of max height
#              f'{tf1.split("_")[0]}–{tf2.split("_")[0]} \n({val:.2f})',
#              rotation=90, color=colors[i % len(colors)],
#              ha='left', va='top', fontsize=9)
# plt.axvline(np.quantile(background_corrs, 0.95), linestyle='--', color='black')
# plt.xlabel('Pairwise regulon Pearson correlation')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.savefig(os.path.join(sc.settings.figdir, "multiome_RU581_correlation_hist.pdf"))
# plt.show()

# # C
# target = 'PHOX2B_direct_+/+_(50g)'
# target_corrs = corr_matrix[target].drop(target)
# ranked_corrs = target_corrs.sort_values(ascending=False)

# queries = ['FOXA1_direct_+/+_(360g)', 'SOX2_direct_+/+_(97g)']
# for query in queries: 
#     if query in ranked_corrs.index:
#         tf = query.split("_")[0]
#         foxa1_rank = ranked_corrs.index.get_loc(query) + 1
#         foxa1_corr = ranked_corrs.loc[query]
#         print(f"{tf} regulon rank: {foxa1_rank} of {len(ranked_corrs)}")
#         print(f"Pearson correlation with {target}: {foxa1_corr:.3f}")
#     else:
#         print(f"⚠️ {query} not found among regulons.")

# target = 'PHOX2B_direct_+/+_(50g)'
# query  = 'FOXA1_direct_+/+_(360g)'   # regulon name; TF label is 'FOXA1'

# # --- Build correlation Series (vs target regulon) ---
# corr_s = corr_matrix[target].drop(target)
# corr_s = corr_s.reindex(tf_series.index).dropna()

# # --- Deduplicate by TF: keep the regulon with the largest |r| per TF ---
# df = pd.DataFrame({
#     'regulon': corr_s.index,
#     'tf': tf_series.loc[corr_s.index].values,
#     'corr': corr_s.values
# })
# keep_idx = df.groupby('tf')['corr'].apply(lambda x: x.abs().idxmax())
# df_dedup = df.loc[keep_idx].copy()

# # --- GLOBAL ranking (for emphasis text) ---
# df_rank_all = df_dedup.sort_values('corr', ascending=False).reset_index(drop=True)
# n_tfs = len(df_rank_all)
# foxa1_present = (df_rank_all['tf'] == 'FOXA1').any()
# foxa1_rank = None
# foxa1_corr = None
# if foxa1_present:
#     foxa1_rank = int(df_rank_all.index[df_rank_all['tf'] == 'FOXA1'][0]) + 1  # 1-based rank
#     foxa1_corr = float(df_rank_all.loc[df_rank_all['tf'] == 'FOXA1', 'corr'].iloc[0])

# # --- Zoom threshold (high positives only), but ensure FOXA1 is visible ---
# r_min = -1
# if foxa1_present and (0 < foxa1_corr < r_min):
#     r_min = foxa1_corr - 1e-6

# # --- Filter to zoomed region and sort by correlation for plotting ---
# df_zoom = df_dedup[df_dedup['corr'] >= r_min].sort_values('corr', ascending=False).reset_index(drop=True)
# if df_zoom.empty:
#     df_zoom = df_dedup.sort_values('corr', ascending=False).head(50).reset_index(drop=True)

# # --- Plot ---
# plt.figure(figsize=(3, 2))
# x = np.arange(len(df_zoom))
# y = df_zoom['corr'].values
# sns.scatterplot(x=x, y=y, s=5, alpha=1)

# # zero line
# plt.axhline(0, color='gray', linestyle='--', lw=0.7)

# # --- Highlight FOXA1 prominently ---
# if foxa1_present:
#     if (df_zoom['tf'] == 'FOXA1').any():
#         j = int(df_zoom.index[df_zoom['tf'] == 'FOXA1'][0])
#         plt.scatter(x[j], y[j],
#                     s=120, marker='*', color='red', edgecolor='black', linewidth=0.5,
#                     label=f"FOXA1 r={foxa1_corr:.3f}  (#{foxa1_rank}/{n_tfs})", zorder=4)
#         plt.axhline(y[j], color='red', linestyle='--', alpha=0.35)
#         plt.axvline(x[j], color='red', linestyle='--', alpha=0.35)
#         # Add a small badge on-plot
#         plt.text(x[j], y[j], f"  FOXA1  #{foxa1_rank}/{n_tfs}",
#                  fontsize=9, fontweight='bold', color='red',
#                  va='bottom', ha='left')
#     else:
#         # If outside zoom, still show global rank in a corner badge
#         plt.text(0.99, 0.02, f"FOXA1 r={foxa1_corr:.3f}  (#{foxa1_rank}/{n_tfs})",
#                  transform=plt.gca().transAxes, ha='right', va='bottom',
#                  fontsize=9, fontweight='bold', color='red')

# # --- Annotate top 20 TFs within the zoom (by correlation) ---
# topN = 0
# to_annotate = set(df_zoom.head(topN)['tf'])
# to_annotate.discard('FOXA1')

# texts = []
# for i, row in df_zoom.iterrows():
#     if row['tf'] in to_annotate:
#         texts.append(plt.text(i, row['corr'], row['tf'], fontsize=8))
# adjust_text(texts, arrowprops=dict(arrowstyle='-', lw=0.5, color='gray'))

# # --- Titles & labels with emphasis ---
# #plt.title("Highly correlated TFs with PHOX2B regulon")
# plt.xlabel(f"TF rank")
# plt.ylabel(f"Pearson r vs PHOX2B")

# # Tight y-limits around zoomed points
# pad = max(0.02, 0.1 * max(1e-6, np.nanmax(np.abs(y))))
# plt.ylim(y.min() - pad, min(1.0, y.max() + pad))
# plt.tight_layout()
# #plt.legend(loc='best', fontsize=8)
# plt.savefig(os.path.join(sc.settings.figdir, "multiome_RU581_PHOX2B_correlation.pdf"))
# plt.show()