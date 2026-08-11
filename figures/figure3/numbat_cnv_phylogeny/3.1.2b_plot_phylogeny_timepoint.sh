#!/bin/bash
#SBATCH --job-name=plot_phylo_tp
#SBATCH --output=logs/plot_phylo_tp_%A_%a.out
#SBATCH --error=logs/plot_phylo_tp_%A_%a.err
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-7
#SBATCH --partition=cpu,chanj3

set -euo pipefail

echo "Start time: $(date)"

SIF_IMAGE="/usersoftware/chanj3/singularity/numbat-rbase.sif"
BASE_ROOT="/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
USE_METACELLS="TRUE"

ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" patient_timepoint_list.txt)
if [[ -z "${ID}" ]]; then
    echo "No entry found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Processing: $ID"

singularity exec --bind /data1:/data1 "$SIF_IMAGE" \
    Rscript 3.1.2b_plot_phylogeny_timepoint.R "$ID" "$BASE_ROOT" "$USE_METACELLS"

echo "End time: $(date)"
