import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import gseapy as gp

ADATA_BASE_DIR = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/"
FDL_KEY_PATH = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/ref/layouts.SCLC_NSCLC.individual.transformed.070125.csv"
fdl_key = pd.read_csv(FDL_KEY_PATH, index_col=None, header=None)
fdl_key_dict = dict(zip(fdl_key[0], fdl_key[1]))
patients = ['RU1083', 'RU1444', 'RU942', 'RU263', 'RU581', 'RU831', 'RU151']
marker = 'TM4SF1' #'KRT15'

# 1. Define your custom palette
custom_palette = {
    'NSCLC': 'gold',   # tab:blue
    'SCLC-A': '#d62728',  # tab:red
    'SCLC-N': '#17becf',  # tab:cyan
}
default_color = '#d3d3d3' # lightgray

fig, axes = plt.subplots(nrows=len(patients), ncols=2, figsize=(5, 2 * len(patients)))

for i, patient in enumerate(patients):
    file_path = os.path.join(ADATA_BASE_DIR, f"{patient}.no_cc.hvg_2000.090124.h5ad")
    adata_temp = sc.read_h5ad(file_path)
    
    # --- PALETTE SYNC LOGIC ---
    # Get all unique cell types in this specific patient
    current_types = adata_temp.obs['cell_type_final2'].cat.categories
    
    # Create a list of colors matching the order of the categories
    colors = [custom_palette.get(t, default_color) for t in current_types]
    
    # Assign the colors to the anndata object
    adata_temp.uns['cell_type_final2_colors'] = colors
    
    # 2. Plot Cell Type
    sc.pl.embedding(adata_temp, basis=fdl_key_dict[patient], color='cell_type_final2', ax=axes[i, 0], show=False, 
               frameon=False, title=f"{patient}: Cell Type", 
               add_outline=True, outline_width=(0.05, 0.05))
    
    # 3. Plot Marker (score or single marker)
    # Using 'magma' or 'Reds' for gene expression helps it pop against gray backgrounds
    sc.pl.embedding(adata_temp, basis=fdl_key_dict[patient], color=marker, ax=axes[i, 1], show=False, 
               frameon=False, title=f"{patient}: {marker}", 
               use_raw=False, cmap='Spectral_r', vmax='p99', add_outline=True) # use 99th percentile for contrast
plt.tight_layout()
plt.show()

# Use Gardner's signature
import numpy as np
marker = "Bottleneck Score"
GARDNER_PATH = "Gardner_list.xlsx"
gardner_signature = pd.read_excel("Gardner_list.xlsx", sheet_name='Macrostate Prob. Gene Corr.', header=0).loc[:,["Bottleneck_gene", "Bottleneck_corr"]].drop_duplicates()

def get_fisher_threshold(correlations, k=3):
    # 1. Fisher Z-transform: r -> z
    # We use arctanh because it's mathematically equivalent to the Fisher Z formula
    # clip values slightly to avoid infinity at exactly 1.0
    z_values = np.arctanh(np.clip(correlations, -0.999, 0.999))
    
    # 2. Calculate robust statistics in Z-space
    z_median = np.median(z_values)
    z_mad = np.median(np.abs(z_values - z_median))
    
    # Standard constant to make MAD comparable to SD
    z_threshold = z_median + (k * 1.4826 * z_mad)
    
    # 3. Inverse Fisher transform: z -> r
    # We use tanh to bring the threshold back to the [-1, 1] range
    r_threshold = np.tanh(z_threshold)
    
    return r_threshold

# Apply to your data
r_cutoff = get_fisher_threshold(gardner_signature['Bottleneck_corr'], k=5)
print(f"Statistically sound threshold (k=5): {r_cutoff:.4f}")

cutoff_bottleneck_genes = gardner_signature[gardner_signature['Bottleneck_corr'] >= r_cutoff]['Bottleneck_gene'].str.upper().tolist()
print(f"Using {len(cutoff_bottleneck_genes)} genes for Bottleneck Score calculation.")
signature = cutoff_bottleneck_genes
fig, axes = plt.subplots(nrows=len(patients), ncols=2, figsize=(5, 2 * len(patients)))
for i, patient in enumerate(patients):
    file_path = os.path.join(ADATA_BASE_DIR, f"{patient}.no_cc.hvg_2000.090124.h5ad")
    adata_temp = sc.read_h5ad(file_path)
    
    # --- PALETTE SYNC LOGIC ---
    # Get all unique cell types in this specific patient
    current_types = adata_temp.obs['cell_type_final2'].cat.categories
    
    # Create a list of colors matching the order of the categories
    colors = [custom_palette.get(t, default_color) for t in current_types]
    
    # Assign the colors to the anndata object
    adata_temp.uns['cell_type_final2_colors'] = colors
    # ... inside your patient loop ...
    sc.tl.score_genes(adata_temp, 
                  gene_list=signature, 
                  score_name=marker, 
                  use_raw=True)
    
    # Calculate a symmetric limit so that 0 is always white
    # Using the 99th percentile of absolute values prevents outliers from squishing the scale
    limit = np.percentile(np.abs(adata_temp.obs[marker]), 99)

    # 2. Plot Cell Type (same as before)
    sc.pl.embedding(adata_temp, basis=fdl_key_dict[patient], color='cell_type_final2', ax=axes[i, 0], show=False,  
               frameon=False, title=f"{patient}: Cell Type", 
               add_outline=True, outline_width=(0.05, 0.05))
    
    # 3. Plot Gardner Bottleneck Score with Symmetric Diverging Palette
    sc.pl.embedding(adata_temp, 
                    basis=fdl_key_dict[patient],
               color=marker, 
               ax=axes[i, 1], 
               show=False, 
               frameon=False, 
               title=f"{patient}: {marker}", 
               use_raw=False, 
               cmap='RdBu_r',      # Blue (low/neg) -> White (0) -> Red (high/pos)
               vmin=-limit,        # Set symmetric minimum
               vmax=limit,         # Set symmetric maximum
               add_outline=True)
plt.tight_layout()
plt.show()


# Use CellMarker's Basal signature
marker = "Basal Cell Lung "
library_name = 'CellMarker_2024'
full_library = gp.get_library(name=library_name)
gene_sets_to_analyze = [
    name for name in full_library.keys() 
    if any(term in name.lower() for term in ['lung', 'trachea', 'brain', 'prefrontal cortex'])
]
gene_set_dict = {name: full_library[name] for name in gene_sets_to_analyze}
fig, axes = plt.subplots(nrows=len(patients), ncols=2, figsize=(5, 2 * len(patients)))
for i, patient in enumerate(patients):
    file_path = os.path.join(ADATA_BASE_DIR, f"{patient}.no_cc.hvg_2000.090124.h5ad")
    adata_temp = sc.read_h5ad(file_path)
    
    # --- PALETTE SYNC LOGIC ---
    # Get all unique cell types in this specific patient
    current_types = adata_temp.obs['cell_type_final2'].cat.categories
    
    # Create a list of colors matching the order of the categories
    colors = [custom_palette.get(t, default_color) for t in current_types]
    
    # Assign the colors to the anndata object
    adata_temp.uns['cell_type_final2_colors'] = colors
    # ... inside your patient loop ...
    sc.tl.score_genes(adata_temp, 
                  gene_list=signature, 
                  score_name=marker, 
                  use_raw=True)
    
    # Calculate a symmetric limit so that 0 is always white
    # Using the 99th percentile of absolute values prevents outliers from squishing the scale
    limit = np.percentile(np.abs(adata_temp.obs[marker]), 99)

    # 2. Plot Cell Type (same as before)
    sc.pl.embedding(adata_temp, basis=fdl_key_dict[patient], color='cell_type_final2', ax=axes[i, 0], show=False,  
               frameon=False, title=f"{patient}: Cell Type", 
               add_outline=True, outline_width=(0.05, 0.05))
    
    # 3. Plot Gardner Bottleneck Score with Symmetric Diverging Palette
    sc.pl.embedding(adata_temp, 
                    basis=fdl_key_dict[patient],
               color=marker, 
               ax=axes[i, 1], 
               show=False, 
               frameon=False, 
               title=f"{patient}: {marker}", 
               use_raw=False, 
               cmap='RdBu_r',      # Blue (low/neg) -> White (0) -> Red (high/pos)
               vmin=-limit,        # Set symmetric minimum
               vmax=limit,         # Set symmetric maximum
               add_outline=True)
plt.tight_layout()
plt.show()
