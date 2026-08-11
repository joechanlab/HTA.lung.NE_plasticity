#!/bin/bash
#SBATCH --job-name=metacell_violin_data
#SBATCH --partition=cpu,chanj3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/metacell_violin_data_%j.out
#SBATCH --error=logs/metacell_violin_data_%j.err

set -euo pipefail
echo "Start time: $(date)"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")"; pwd)}"
PYTHON_BIN="/data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python"

"$PYTHON_BIN" "${SCRIPT_DIR}/7d_build_metacell_violin_data.py"

echo "End time: $(date)"
