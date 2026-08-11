import pandas as pd
import numpy as np
import scanpy as sc
import pickle

# 1) load binding scores and DA peaks
tf_binding = pd.read_csv("/data1/chanj3/HTA.multiome.lung.NE_plasticity.060124/results/atac_insilicochip/RU151_PDX3_tf_binding_scores.ISChIP.csv", sep="\t", index_col=0)
#pd.read_csv("/data1/chanj3/HTA.multiome.lung.NE_plasticity.060124/results/atac_insilicochip/RU581_LN_tf_binding_scores.ISChIP.csv", sep="\t", index_col=0)
# rows: TFs, cols: peaks "chr:start-end"

# Load SnapATAC2 DA peaks for PHOX2B high clusters (RU151_PDX3 cluster 7)
subject = "RU151_PDX3"
diff_peaks_csv_path = f"/data1/chanj3/HTA.multiome.lung.NE_plasticity.060124/results/atac_preprocess_snapatac2/{subject}/{subject}_diff_peaks.csv"
peak_mat = sc.read_h5ad(f"/data1/chanj3/HTA.multiome.lung.NE_plasticity.060124/results/atac_preprocess_snapatac2/{subject}/{subject}_peak_mtx.h5ad")
rna_adata = sc.read_h5ad(f"/data1/chanj3/HTA.multiome_plasticity.results/adatas/RU151_PDX123.h5ad")
NE_high_cells = rna_adata.obs.query(f"cell_type in ['NE-high'] & sample_name == '{subject}'").index.map(lambda x: x.split("_")[-1])
peak_mat.obs.leiden[peak_mat.obs_names.isin(NE_high_cells)].value_counts() # cluster 7 is the NE high cluster
motif_enrichment_path = "/data1/chanj3/HTA.multiome.lung.NE_plasticity.060124/results/atac_preprocess_snapatac2/RU151_PDX3/RU151_PDX3_motif_enrichment.pkl"
with open(motif_enrichment_path, "rb") as f:
    motif_enrichment = pickle.load(f)
motif_NE_high = motif_enrichment['7']
motif_NE_high_df = pd.DataFrame(motif_NE_high, columns=['id', 'name', 'family', 'log2(fold change)', 'p-value', 'adjusted p-value'])
motif_NE_high_df = motif_NE_high_df[motif_NE_high_df['adjusted p-value'] < 0.05]
motif_NE_high_df = motif_NE_high_df.sort_values(by='log2(fold change)', ascending=False)
top_motif = motif_NE_high_df.iloc[0]

# Load SnapATAC2 DA peaks for 
subject = "RU581_LN"
diff_peaks_csv_path = f"/data1/chanj3/HTA.multiome.lung.NE_plasticity.060124/results/atac_preprocess_snapatac2/{subject}/{subject}_diff_peaks.csv"
peak_mat = sc.read_h5ad(f"/data1/chanj3/HTA.multiome.lung.NE_plasticity.060124/results/atac_preprocess_snapatac2/{subject}/{subject}_peak_mtx.h5ad")
rna_adata = sc.read_h5ad(f"/data1/chanj3/HTA.multiome_plasticity.results/adatas/{subject}.h5ad")
NE_high_cells = rna_adata.obs.query(f"cell_type in ['SCLC-A'] & sample_name == '{subject}'").index.map(lambda x: x.split("_")[-1])
peak_mat.obs.leiden[peak_mat.obs_names.isin(NE_high_cells)].value_counts() # cluster 3 is the NE high cluster
motif_enrichment_path = f"/data1/chanj3/HTA.multiome.lung.NE_plasticity.060124/results/atac_preprocess_snapatac2/{subject}/{subject}_motif_enrichment.pkl"
with open(motif_enrichment_path, "rb") as f:
    motif_enrichment = pickle.load(f)
motif_NE_high = motif_enrichment['3']
motif_NE_high_df = pd.DataFrame(motif_NE_high, columns=['id', 'name', 'family', 'log2(fold change)', 'p-value', 'adjusted p-value'])
motif_NE_high_df = motif_NE_high_df[motif_NE_high_df['adjusted p-value'] < 0.05]
motif_NE_high_df = motif_NE_high_df.sort_values(by='log2(fold change)', ascending=False)
