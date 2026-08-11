import os
import pandas as pd
import scanpy as sc
PREFIX_OUT_DIR = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat/RU151/GEX/RU151_PDX"
adata = sc.read_h5ad(f"/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/RU151.no_cc.hvg_2000.h5ad")
adata_subset = adata[adata.obs.cell_type_final2 == 'NonNE SCLC'].copy() # 168 cells
markers = ['AGR2', 'ELF3', 'KRT5', 'KRT6A', 'TP63']
sc.pl.umap(adata_subset, color=markers)
sc.pp.neighbors(adata_subset, n_neighbors=10, n_pcs=20)
sc.tl.leiden(adata_subset, resolution=0.3)
sc.pl.umap(adata_subset, color=['leiden'] + markers)
# Create a mapping dictionary based on your marker observation
# Assuming Cluster 0 = KRT5/6A+ and Cluster 1 = AGR2+ (adjust if needed)
new_cluster_names = {
    '0': 'KRT5/6A+ NonNE SCLC',
    '1': 'AGR2+ NonNE SCLC',
    '2': 'KRT5/6A+ NonNE SCLC'
}

# Add the new names to a new observation column
adata_subset.obs['cell_type_sub'] = (
    adata_subset.obs['leiden']
    .map(new_cluster_names)
    .astype('category')
)

# Group markers by population for a cleaner plot
marker_dict = {
    'AGR2+ Markers': ['AGR2', 'ELF3'],
    'Basal-like Markers': ['KRT5', 'KRT6A', 'TP63']
}

# Generate the dotplot
sc.pl.dotplot(
    adata_subset, 
    marker_dict, 
    groupby='cell_type_sub', 
    standard_scale='var',
    save='_subclustering_markers.pdf'
)

adata.obs['cell_type_final3'] = adata.obs['cell_type_final2'].astype(str)
adata.obs.loc[adata_subset.obs.index, 'cell_type_final3'] = adata_subset.obs['cell_type_sub'].astype(str)
adata.obs['cell_type_final3'] = adata.obs['cell_type_final3'].astype('category')
print(adata.obs['cell_type_final3'].value_counts())
barcodes = pd.read_csv(os.path.join(PREFIX_OUT_DIR, "barcodes.tsv"), header=None)
adata.obs_names = barcodes.values.flatten()
adata.obs.loc[:, ['cell_type_final3']].to_csv(
    os.path.join(PREFIX_OUT_DIR, f"RU151_PDX.cell_types_fine.tsv"), 
    sep='\t', index=True, header=True
)

# load the clone information
clones = pd.read_csv("/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat/RU151/numbat_output_FACETS_HISENS/RU151_PDX/clone_post_2.tsv",
                    sep="\t")
cell_types = pd.read_csv(os.path.join(PREFIX_OUT_DIR, f"RU151_PDX.cell_types_fine.tsv"),
                    sep='\t')
cell_types.columns = ['cell', 'cell_type']
clones = clones.merge(cell_types, on='cell')
pd.crosstab(clones.clone_opt, clones.cell_type)

# color the tree by the 