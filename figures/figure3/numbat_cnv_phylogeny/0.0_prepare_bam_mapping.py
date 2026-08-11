import pandas as pd

mapping_path = "/data1/chanj3/HTA.lung.NE_plasticity.120122/ref/HTA.fresh_plasticity_de_novo.manifest.120122.txt"
mapping_df = pd.read_csv(mapping_path, sep="\t", header=None)
mapping_df.columns = ['sample', 'file']
mapping_df.loc[:,'file'] = mapping_df['file'].str.split('/').str[-1]
mapping_df.loc[:,'file'] = mapping_df['file'].str.split('_dense.').str[0]
mapping_df.loc[:,'path'] = mapping_df.file.map(lambda x: f"/data1/chanj3/HTA.lung.NE_plasticity.120122/SEQC/bam/{x}_Aligned.out.bam")
mapping_df.loc[:,'patient'] = mapping_df['sample'].str.split('_').str[0]
mapping_df = mapping_df[['patient', 'sample', 'file', 'path']]
mapping_df = mapping_df.sort_values(by=['patient', 'sample'])


CHEMISTRY_MAP = pd.read_csv("/data1/chanj3/HTA.lung.NE_plasticity.120122/ref/HTA.fresh_plasticity.ten_x_v.txt", delimiter='\t', header=None)
CHEMISTRY_MAP.columns = ['file', 'chemistry']
mapping_df = mapping_df.merge(CHEMISTRY_MAP, on='file')
mapping_df = mapping_df.loc[:,['patient', 'sample', 'path', 'chemistry']]
mapping_df.to_csv("bam_mapping.txt", sep="\t", header=False, index=False)
