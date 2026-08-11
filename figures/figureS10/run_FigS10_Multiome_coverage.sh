# RU151_PDX
echo "Loading SCENIC+ results from RU151_PDX..."
python Fig8_Multiome_coverage.py \
  --scplus-res-dir /data1/chanj3/wangm10/work/b9/d05869b85096850f6ecc64f415fc5b/RU151_PDX/scplus \
  --gtf /data1/chanj3/shared/genome/GRCh38/MANE.GRCh38.v1.4.ensembl_genomic.gtf \
  --consensus-bed /data1/chanj3/wangm10/work/b9/d05869b85096850f6ecc64f415fc5b/RU151_PDX/peak_calling/consensus_regions.bed \
  --bw-dir /data1/chanj3/wangm10/work/b9/d05869b85096850f6ecc64f415fc5b/RU151_PDX/peak_calling/pseudobulk_bigwigs \
  --celltype-of-interest "NE-high" \
  --extend-left 50000 \
  --extend-right 50000 \
  --bw-ymax 25 \
  --min-fc 1.75 \
  --max-qval 0.05 \
  --min-pct 0.05 \
  --min-acc 300 \
  --prox-thresh 0.01 \
  --celltype-order ADC "NE-intermediate" "NE-high" \
  --top-da-genes 5

# RU581_LN
# echo "Loading SCENIC+ results from RU581_LN..."
# python Fig8_Multiome_coverage.py \
#   --scplus-res-dir /data1/chanj3/HTA.multiome_plasticity.results/scenicplus/runs/RU581_LN_2/scplus \
#   --gtf /data1/chanj3/shared/genome/GRCh38/MANE.GRCh38.v1.4.ensembl_genomic.gtf \
#   --consensus-bed /data1/chanj3/HTA.multiome_plasticity.results/scenicplus/runs/RU581_LN_2/peak_calling/consensus_regions.bed \
#   --bw-dir /data1/chanj3/HTA.multiome_plasticity.results/scenicplus/runs/RU581_LN_2/peak_calling/pseudobulk_bigwigs \
#   --celltype-of-interest "SCLC-A" \
#   --extend-left 50000 \
#   --extend-right 50000 \
#   --bw-ymax 25 \
#   --min-fc 1.75 \
#   --min-acc 300 \
#   --prox-thresh 0.01 \
#   --celltype-order "B-cell" "T-cell" "SCLC-A" \
#   --top-da-genes 5