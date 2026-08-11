#!/bin/bash
#SBATCH --job-name=cnv_burden
#SBATCH --partition=cpu,chanj3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/cnv_burden_%j.out
#SBATCH --error=logs/cnv_burden_%j.err

set -euo pipefail

echo "Start time: $(date)"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")"; pwd)}"
PY_SCRIPT="${SCRIPT_DIR}/7b_cnv_burden_per_cell.py"
PYTHON_BIN="${PYTHON_BIN:-/data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python}"

# Optional: path to GENCODE/Ensembl gene BED (3-column: chrom start end)
# Generate with: grep -v "^#" gencode.v43.annotation.gtf | awk '$3=="gene"' |
#   awk '{OFS="\t"; gsub("chr","",$1); print $1,$4-1,$5}' > hg38_genes.bed
GENE_BED="${GENE_BED:-}"

# Optional: restrict to specific patients (space-separated)
PATIENTS="${PATIENTS:-}"

CMD=("$PYTHON_BIN" "$PY_SCRIPT" --fga-threshold 0.5)

if [[ -n "$GENE_BED" ]]; then
    CMD+=(--gene-bed "$GENE_BED")
fi

if [[ -n "$PATIENTS" ]]; then
    CMD+=(--patients $PATIENTS)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "End time: $(date)"
