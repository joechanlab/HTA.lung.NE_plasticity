#!/bin/bash
#SBATCH --job-name=cnv_burden_violin
#SBATCH --partition=cpu,chanj3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/cnv_burden_violin_%j.out
#SBATCH --error=logs/cnv_burden_violin_%j.err

set -euo pipefail
echo "Start time: $(date)"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")"; pwd)}"
PYTHON_BIN="/data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python"

"$PYTHON_BIN" "${SCRIPT_DIR}/5_cnv_burden_violin.py"

echo "End time: $(date)"
