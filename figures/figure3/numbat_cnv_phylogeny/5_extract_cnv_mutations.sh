#!/bin/bash
#SBATCH --job-name=cnv_summary
#SBATCH --partition=cpu,chanj3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/cnv_summary_%j.out
#SBATCH --error=logs/cnv_summary_%j.err

set -euo pipefail

echo "Start time: $(date)"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")"; pwd)}"
SIF_IMAGE="/usersoftware/chanj3/singularity/numbat-rbase.sif"

singularity exec --bind /data1:/data1 "$SIF_IMAGE" \
    Rscript "${SCRIPT_DIR}/5_extract_cnv_mutations.R"

echo "End time: $(date)"
