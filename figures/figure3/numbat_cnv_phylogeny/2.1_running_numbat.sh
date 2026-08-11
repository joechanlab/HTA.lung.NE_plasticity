#!/bin/bash
#SBATCH --job-name=run_numbat
#SBATCH --array=1-36
#SBATCH --partition=cpu,chanj3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/numbat_%A_%a.out
#SBATCH --error=logs/numbat_%A_%a.err

echo "Start time: $(date)"
BULK_CNV_OPTION="FACETS_HISENS"
USE_METACELLS="TRUE"
SAMPLE_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sample_list.txt)
echo "Running Numbat with BULK_CNV_OPTION: $BULK_CNV_OPTION, SAMPLE_ID: $SAMPLE_ID, USE_METACELLS: $USE_METACELLS"

R_SCRIPT="/data1/chanj3/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/figure3/Numbat/2.1_running_numbat.R"
SIF_IMAGE="/usersoftware/chanj3/singularity/numbat-rbase.sif"

singularity exec --bind /data1:/data1 $SIF_IMAGE Rscript $R_SCRIPT $BULK_CNV_OPTION $SAMPLE_ID 5 0.5 $USE_METACELLS
echo "End time: $(date)"
