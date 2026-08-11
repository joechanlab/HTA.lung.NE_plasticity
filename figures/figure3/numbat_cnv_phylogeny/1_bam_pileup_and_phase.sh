#!/bin/bash
#SBATCH --job-name=pileup
#SBATCH --output=logs/pileup_%A_%a.out
#SBATCH --error=logs/pileup_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --array=22-23
#SBATCH --partition=cpu,chanj3
#-----------------------------
# 1. Preparation: Pull and build numbat-rbase singularity image
# cd /usersoftware/chanj3/singularity
# singularity build numbat-rbase.sif docker://pkharchenkolab/numbat-rbase:latest
# singularity shell --bind /data1:/data1 /usersoftware/chanj3/singularity/numbat-rbase.sif
#-----------------------------
# 2. Prepare data
# Convert the bam file to Numbat compatible format using samtools
source activate /data1/chanj3/wangm10/conda_envs/htslib
PATIENT_LIST="patient_list.txt"
MAPPING_FILE="bam_mapping.txt"
SIF_PATH="/usersoftware/chanj3/singularity/numbat-rbase.sif"
SOFTWARE_DIR="/data1/chanj3/wangm10/git_repos/numbat/"
BASE_OUT_DIR="/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat/"

SAMPLE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PATIENT_LIST)
PATIENT=${SAMPLE%%_*}
OUT_DIR="${BASE_OUT_DIR}/${PATIENT}"
mkdir -p $OUT_DIR/BAM
OUT_DIR_ALL="/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/${PATIENT}/"
mkdir -p $OUT_DIR_ALL/BAM

echo "Processing Patient: ${PATIENT}"

# Create a bam file with CB and UB tags
#----------------------------
# 1. Process BAM files for this patient using the mapping file
# The mapping file columns are: patient, sample, bam_path
grep "^${PATIENT}" "$MAPPING_FILE" | while read -r P_ID SAMPLE_NAME BAM_PATH CHEMISTRY; do
    if [ -f "$BAM_PATH" ]; then
        echo "Processing Sample: ${SAMPLE_NAME}"
        echo "Input BAM: ${BAM_PATH}"
        
        OUTPUT_BAM="$OUT_DIR/BAM/${SAMPLE_NAME}.numbat_sorted.bam"
        
        samtools view -h "$BAM_PATH" | \
        awk -v sname="${SAMPLE_NAME}" 'BEGIN{FS="\t"; OFS="\t"} \
            # 1. Update Header (@SQ lines) for standard chromosomes only
            /^@SQ/ { 
                if ($2 ~ /SN:([0-9]{1,2}|[XY]|MT?)$/) { 
                    gsub("SN:", "SN:chr", $2); 
                } 
                print $0; next 
            } \
            # 2. Pass other header lines through
            /^@/ {print $0; next} \
            # 3. Update Alignments and Barcodes
            { 
                # Prepend "chr" to standard chromosomes in col 3 and col 7
                if ($3 ~ /^([0-9]{1,2}|[XY]|MT?)$/) $3 = "chr"$3; 
                if ($7 ~ /^([0-9]{1,2}|[XY]|MT?)$/) $7 = "chr"$7; 
                
                # Barcode logic: Split QNAME to extract CB and UB
                split($1, name_parts, ":");
                cb = sname "_" name_parts[2] "-1";
                ub = name_parts[3];
                
                print $0 "\tCB:Z:" cb "\tUB:Z:" ub 
            }' | \
        samtools sort -@ "${NCORES}" -o "$OUTPUT_BAM"
            
        samtools index "$OUTPUT_BAM"
    else
        echo "Warning: BAM file not found for ${SAMPLE_NAME} at ${BAM_PATH}"
    fi
done

# example code to check the bam file
# conda activate /usersoftware/chanj3/htslib; samtools view -h /data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat/RU1314/BAM/RU1314_T.numbat_sorted.bam | less
NCORES=32
SAMPLES=($(grep "^${PATIENT}[[:space:]]" "$MAPPING_FILE" | cut -f2))
BAM_LIST=$(IFS=,; for s in "${SAMPLES[@]}"; do echo -n "$OUT_DIR/BAM/${s}.numbat_sorted.bam,"; done | sed 's/,$//')
BARCODE_LIST=$(IFS=,; for s in "${SAMPLES[@]}"; do echo -n "$OUT_DIR_ALL/GEX/${s}/barcodes.tsv,"; done | sed 's/,$//')
SAMPLE_CSV_LIST=$(IFS=,; echo "${SAMPLES[*]}")

singularity exec --bind $OUT_DIR_ALL:/mnt/output,$SOFTWARE_DIR:/numbat,/data1:/data1 $SIF_PATH \
    Rscript /numbat/inst/bin/pileup_and_phase.R \
    --label $PATIENT \
    --samples "$SAMPLE_CSV_LIST" \
    --bams "$BAM_LIST" \
    --barcodes "$BARCODE_LIST" \
    --outdir /mnt/output/BAM \
    --gmap /Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz \
    --snpvcf /data1/chanj3/wangm10/1000G_phase1.snps.high_confidence.hg38.vcf.gz \
    --paneldir /data/1000G_hg38 \
    --ncores $NCORES
