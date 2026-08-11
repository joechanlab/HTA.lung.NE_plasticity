#!/bin/bash
#SBATCH --job-name=scistreer
#SBATCH --output=logs/scistreer_%A_%a.out
#SBATCH --error=logs/scistreer_%A_%a.err
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-17
#SBATCH --partition=cpu,chanj3

echo "Start time: $(date)"

SIF_IMAGE="/usersoftware/chanj3/singularity/numbat-rbase.sif"
USE_METACELLS="TRUE"
MODE="patient"

PATIENT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" patient_list.txt)
echo "Mode: $MODE  PATIENT_ID: $PATIENT_ID  USE_METACELLS: $USE_METACELLS"

singularity exec --bind /data1:/data1 $SIF_IMAGE \
    Rscript 3.1.1_scistreer_cmd.R "$PATIENT_ID" "$USE_METACELLS" "$MODE"

echo "End time: $(date)"
