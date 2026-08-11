#!/bin/bash
#SBATCH --job-name=metacell_leaf_dist
#SBATCH --partition=cpu,chanj3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/metacell_leaf_dist_%j.out
#SBATCH --error=logs/metacell_leaf_dist_%j.err

set -euo pipefail
echo "Start time: $(date)"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")"; pwd)}"
SIF_IMAGE="/usersoftware/chanj3/singularity/numbat-rbase.sif"

singularity exec --bind /data1:/data1 "$SIF_IMAGE" \
    Rscript "${SCRIPT_DIR}/7d_compute_metacell_leaf_distances.R"

echo "End time: $(date)"
