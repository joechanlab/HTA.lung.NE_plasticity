#!/bin/bash
#SBATCH --job-name=gam_pathway_regen
#SBATCH --output=logs/gam_pathway_regen_%j.out
#SBATCH --error=logs/gam_pathway_regen_%j.err
#SBATCH --partition=chanj3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00

set -euo pipefail

PYTHON_BIN="/data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python"

mkdir -p logs

echo "Running regenerate_gam_and_pathway_enrichment.py on $(hostname) at $(date)"
"$PYTHON_BIN" regenerate_gam_and_pathway_enrichment.py
