#!/bin/bash
#SBATCH --job-name=seacells_agg
#SBATCH --output=logs/seacells_%A_%a.out
#SBATCH --error=logs/seacells_%A_%a.err
#SBATCH --array=1-36
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=cpu,chanj3
#SBATCH --time=4:00:00

SAMPLE_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sample_list.txt)

echo "Start time: $(date)"
echo "Running SEACells aggregation for sample: $SAMPLE_ID"

/usersoftware/chanj3/SEACells/bin/python 1.5_seacells_aggregation.py \
    --sample_id "$SAMPLE_ID"

echo "End time: $(date)"
