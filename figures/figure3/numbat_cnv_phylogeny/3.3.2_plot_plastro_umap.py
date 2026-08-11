import re
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg
import seaborn as sns

STR2BINDICT = {
    65: 0b100, 67: 0b110, 71: 0b101, 84: 0b011, 78: 0b111,
    97: 0b100, 99: 0b110, 103: 0b101, 116: 0b011, 110: 0b111,
    'A': 0b100, 'C': 0b110, 'G': 0b101, 'T': 0b011, 'N': 0b111,
    'a': 0b100, 'c': 0b110, 'g': 0b101, 't': 0b011, 'n': 0b111
}


def encode(sequence) -> int:
    res = 0
    for nucleotide in sequence:
        res <<= 3
        res += STR2BINDICT[nucleotide]
    return res


def translate_lineage_barcode(barcode: str) -> str:
    patterns = [
        r"^(.*)_\[?([ACGTNacgtn]+)\]?-1$",
        r"^(.*)-\[?([ACGTNacgtn]+)\]?-1$",
    ]
    for pattern in patterns:
        match = re.match(pattern, barcode)
        if match:
            prefix, sequence = match.groups()
            return f"{prefix}_{encode(sequence)}"
    return barcode


paths = pd.read_excel("patient_adata.xlsx", sheet_name="Sheet1")
paths.set_index('patient', inplace=True)
combined_path = pd.read_excel("patient_adata.xlsx", sheet_name="Sheet2", header=None).iloc[0].tolist()[0]
obs = sc.read_h5ad(combined_path).obs
numeric_barcodes = obs.index.map(translate_lineage_barcode)
obs = pd.DataFrame({
    "cell": obs.index,
    "cell_type": obs.cell_type_final2
})
obs.to_csv("/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/cell_type_labels.csv", index=None)
samples_combined = [
    'RU1083',
    'RU263',
    'RU942',
    'RU1444',
    'RU151',
    'RU1303',
    'RU581',
    'RU831',
    'RU1518',
    'RU1646']

col_dict = {
    "LUAD": "gold",
    "LUSC": "orange", 
    "SCLC-A": "tab:red",
    "SCLC-N": "tab:cyan",
    "EMT": "#836953",
    "FOXI1+ NonNE SCLC-N": "tab:green",
    "NonNE SCLC": "tab:purple"
}

adata_dict = {}
for patient in samples_combined:    
    adata = sc.read_h5ad(paths.loc[patient, "path"])
    adata.obs['cell_type_final2'] = obs.loc[adata.obs_names, 'cell_type']
    cytotrace = pd.read_csv(paths.loc[patient, 'cytotrace_path'], sep='\t', index_col=0)
    adata.obs.loc[:,'CytoTRACE2_Score'] = cytotrace.CytoTRACE2_Score[adata.obs_names]
    plastro_path = paths.loc[patient, 'plastro_path']
    if pd.notna(plastro_path) and os.path.exists(plastro_path):
        plastro = pd.read_csv(plastro_path, index_col=0)
        barcodes = adata.obs_names.copy()
        numeric_barcodes = adata.obs_names.map(translate_lineage_barcode)
        mapping = dict(zip(numeric_barcodes, barcodes))
        plastro.index = plastro.index.map(mapping)
        adata.obs.loc[:, 'Plastro_Score'] = plastro.plastro_score[adata.obs_names]
    else:
        print(f"Warning: Plastro file not found for {patient}. Setting to NaN.")
        adata.obs.loc[:, 'Plastro_Score'] = np.nan
    is_tumor = adata.obs['cell_type_final2'].isin(col_dict.keys())
    adata = adata[is_tumor,:].copy()
    adata_dict[patient] = adata

# cell type
all_cell_types = col_dict.keys()
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for patient, ax in zip(samples_combined, np.ravel(axes)):
    layout = re.sub('X_draw_graph_','',paths.loc[patient, "fdl_key"])
    sc.pl.draw_graph(adata_dict[patient], color='cell_type_final2',add_outline=True, 
               ax=ax, show=False, layout = layout, title = patient, palette=col_dict,
               s=30, legend_loc = None, frameon=False)
patches = [mpatches.Patch(color=col_dict[ct], label=ct) for ct in all_cell_types]
fig.legend(handles=patches, title='Cell Type', loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.savefig("figures/cell_type.png", bbox_inches='tight', dpi=300)
plt.show()

# CytoTRACE
# fig, axes = plt.subplots(5, 2, figsize=(5, 12))
# for patient, ax in zip(samples_combined, np.ravel(axes)):
#     layout = re.sub('X_draw_graph_','',paths.loc[patient, "fdl_key"])
#     if patient == samples_combined[-1]:
#         legend_loc = 'upper left'
#     else:
#         legend_loc = None
#     sc.pl.draw_graph(adata_dict[patient], color='CytoTRACE2_Score',add_outline=True, 
#                ax=ax, show=False, layout = layout, title = patient, cmap='Spectral_r',
#                s=30, legend_loc = legend_loc, frameon=False, vmax='p99', vmin='p10')

# plt.savefig("figures/cytotrace2.png", bbox_inches='tight', dpi=300)


# # plastro
# fig, axes = plt.subplots(5, 2, figsize=(5, 12))
# for patient, ax in zip(samples_combined, np.ravel(axes)):
#     layout = re.sub('X_draw_graph_','',paths.loc[patient, "fdl_key"])
#     if patient == samples_combined[-1]:
#         legend_loc = 'upper left'
#     else:
#         legend_loc = None
#     sc.pl.draw_graph(adata_dict[patient], color='Plastro_Score',add_outline=True, 
#                ax=ax, show=False, layout = layout, title = patient, cmap='Spectral_r',
#                s=30, legend_loc = legend_loc, frameon=False, vmax='p99', vmin='p0')

# plt.savefig("figures/plastro.png", bbox_inches='tight', dpi=300)


# # tree
# fig, axes = plt.subplots(5, 2, figsize=(5, 12)) 
# axes_flat = axes.flatten()
# base_root = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
# for i, patient_id in enumerate(samples_combined):
#     img_path = os.path.join(base_root, patient_id, "numbat_output_FACETS_HISENS", f"{patient_id}_phylogeny.png")
#     ax = axes_flat[i]
#     if os.path.exists(img_path):
#         img = mpimg.imread(img_path)
#         ax.imshow(img)
#         ax.set_title(patient_id, fontsize=10)
#     else:
#         ax.text(0.5, 0.5, f"Missing:\n{patient_id}", ha='center', va='center')
#     ax.axis('off') 

# plt.tight_layout()
# plt.savefig("figures/combined_phylogeny_grid.png", dpi=300, bbox_inches='tight')


# # Barplots
# samples = ['RU1083', 'RU263', 'RU942', 'RU1444', 'RU151', 'RU1303', 'RU581', 'RU831', 'RU1518', 'RU1646']
# base_path = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"

# # summarized 
# all_summaries = []
# for sid in samples:
#     dist_file = os.path.join(base_path, sid, "numbat_output_FACETS_HISENS", f"{sid}_leaf_distances.csv")
#     if os.path.exists(dist_file):
#         df_dist = pd.read_csv(dist_file)
#         curr_obs = adata_dict[sid].obs[['CytoTRACE2_Score', 'Plastro_Score', 'cell_type_final2']].copy()
#         curr_obs['cell'] = curr_obs.index
#         merged_df = curr_obs.merge(df_dist, on='cell')
#         patient_ct_summary = merged_df.groupby('cell_type_final2').agg({
#             'CytoTRACE2_Score': 'mean',
#             'Plastro_Score': 'mean',
#             'distance_to_root': 'mean'
#         }).reset_index()
        
#         patient_ct_summary['patient'] = sid
#         all_summaries.append(patient_ct_summary)

# df_plot = pd.concat(all_summaries)
# df_plot.columns = ['Cell Type', 'CytoTRACE2 Score', 'Plastro Score', 'Distance to Root', 'Patient']
# # 2. Plotting: 1 row x 3 columns
# fig, axes = plt.subplots(1, 3, figsize=(8, 6))
# metrics = ['CytoTRACE2 Score', 'Plastro Score', 'Distance to Root']
# titles = ['Avg CytoTRACE2', 'Avg Plastro', 'Avg Distance to Normal Cells']

# for i, metric in enumerate(metrics):
#     # Plotting boxplot where N = number of patients
#     sns.boxplot(
#         data=df_plot,
#         x='Cell Type',
#         y=metric,
#         ax=axes[i],
#         palette=col_dict,
#         showfliers=False
#     )
    
#     # Optional: Add individual patient points on top of the boxplot
#     sns.stripplot(
#         data=df_plot,
#         x='Cell Type',
#         y=metric,
#         ax=axes[i],
#         color='black',
#         alpha=0.5,
#         size=4
#     )
    
#     axes[i].set_title(titles[i])
#     axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

# plt.tight_layout()
# plt.savefig("figures/celltype_average_boxplots.png")
# plt.show()
