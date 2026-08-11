#!/bin/bash
#SBATCH --job-name=plastro_score
#SBATCH --partition=cpu,chanj3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/plastro_%A_%a.out
#SBATCH --error=logs/plastro_%A_%a.err
#SBATCH --array=2

set -euo pipefail

echo "Start time: $(date)"

PATIENT_LIST="patient_list.txt"
PY_SCRIPT="3.3.1_plastro_score.py"

# Optional overrides when submitting:
# sbatch --array=1-22 --export=ALL,BASE_DIR=/path/to/Numbat_All,RADII_STEPS=100 3.3_plastro_score.sh
BASE_DIR="${BASE_DIR:-/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/}"
PYTHON_BIN="/data1/chanj3/wangm10/software/plastro/bin/python"
RADII_STEPS="${RADII_STEPS:-100}"

if [[ ! -f "$PATIENT_LIST" ]]; then
    echo "ERROR: patient list not found: $PATIENT_LIST"
    exit 1
fi

# Read valid patient IDs (non-empty, non-comment)
mapfile -t PATIENTS < <(grep -v '^\s*#' "$PATIENT_LIST" | sed '/^\s*$/d')
TOTAL_PATIENTS="${#PATIENTS[@]}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit as an array job (sbatch --array=1-${TOTAL_PATIENTS} ...)."
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > TOTAL_PATIENTS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range 1-${TOTAL_PATIENTS}."
    exit 1
fi

PATIENT_ID="${PATIENTS[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "Running Plastro for patient: $PATIENT_ID (task ${SLURM_ARRAY_TASK_ID}/${TOTAL_PATIENTS})"
"$PYTHON_BIN" "$PY_SCRIPT" "$PATIENT_ID" --base-dir "$BASE_DIR" --radii-steps "$RADII_STEPS"

echo "End time: $(date)"
