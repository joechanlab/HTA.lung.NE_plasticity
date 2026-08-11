import scanpy as sc
import pandas as pd
import os

patients = pd.read_csv("patient_list.txt", header=None).loc[:,0].values
BASE_DIR = "/data1/chanj3/HTA.lung.NE_plasticity.120122/"
for patient in patients:
    GEX_DIR = os.path.join(BASE_DIR, "Numbat_All", patient, "GEX")
    patient_samples = os.listdir(GEX_DIR)
    for sample in patient_samples:
        print(sample)
        cell_types = pd.read_csv(os.path.join(BASE_DIR, "Numbat_All", patient, "GEX", sample, sample + ".cell_types.tsv"), sep='\t', index_col=0)
        cell_types.columns = ['cell_type']
        if sample == "RU1646_SP":
            cell_types.cell_type = cell_types.cell_type.replace('NSCLC','LUAD')
        else:
            cell_type_fine = pd.read_csv(os.path.join(BASE_DIR, "Numbat", patient, "GEX", sample, sample + ".cell_types.tsv"), sep='\t', index_col=0)
            cell_type_fine.columns = ['cell_type_fine']
            if sample == "RU151_PDX":
                cell_type_fine.cell_type_fine = cell_type_fine.cell_type_fine.replace('NSCLC','NonNE SCLC').replace('AGR2+ NonNE SCLC','LUAD').replace('KRT5/6A+ NonNE SCLC','LUSC')
            if sample == "RU1518_T":
                cell_type_fine.cell_type_fine = cell_type_fine.cell_type_fine.replace('NonNE SCLC','EMT')
            if sample == "RU1444_LN":
                cell_type_fine.cell_type_fine = cell_type_fine.cell_type_fine.replace('UBA52+ SCLC','NonNE SCLC')
            cell_types = cell_types.join(cell_type_fine)
            cell_types.cell_type_fine = cell_types.cell_type_fine.replace('NSCLC','LUAD')
            cell_types.loc[~cell_types.cell_type_fine.isna(), 'cell_type'] = cell_types.loc[~cell_types.cell_type_fine.isna(), 'cell_type_fine']
        print(cell_types.cell_type.value_counts())
        cell_types.loc[:,'cell_type'].to_csv(os.path.join(BASE_DIR, "Numbat_All", patient, "GEX", sample, sample + ".cell_types.tsv"), sep='\t')
