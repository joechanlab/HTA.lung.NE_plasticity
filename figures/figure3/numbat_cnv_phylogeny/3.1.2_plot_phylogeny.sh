#!/bin/bash
#SBATCH --job-name=plot_phylogeny
#SBATCH --output=logs/plot_phylogeny_%A_%a.out
#SBATCH --error=logs/plot_phylogeny_%A_%a.err
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-17
#SBATCH --partition=cpu,chanj3
set -euo pipefail

echo "Start time: $(date)"

SIF_IMAGE="/usersoftware/chanj3/singularity/numbat-rbase.sif"
BASE_ROOT="/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
USE_METACELLS="${USE_METACELLS:-TRUE}"
MODE="${MODE:-patient}"
if [ "$MODE" = "sample" ]; then
    LIST="sample_list.txt"
else
    LIST="patient_list.txt"
fi

ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
if [[ -z "${ID}" ]]; then
    echo "No entry found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${LIST}"
    exit 1
fi

echo "Mode: $MODE  Processing: $ID"

singularity exec --bind /data1:/data1 "$SIF_IMAGE" \
    Rscript 3.1.2_plot_phylogeny.R "$ID" "$BASE_ROOT" "$USE_METACELLS" "$MODE"

echo "End time: $(date)"
