#!/bin/bash
#SBATCH --job-name=phenotype_dist
#SBATCH --partition=cpu,chanj3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/phenotype_%A_%a.out
#SBATCH --error=logs/phenotype_%A_%a.err
#SBATCH --array=22

set -euo pipefail

echo "Start time: $(date)"

PATIENT_LIST="patient_list.txt"
PY_SCRIPT="3.2.2_compute_phenotypic_distance.py"

# Optional overrides when submitting:
# sbatch --array=1-22 --export=ALL,BASE_DATA_PATH=/path/to/h5ad,NUMBAT_BASE=/path/to/Numbat_All,EMBEDDING_KEY=X_pca_slalom,PYTHON_BIN=/path/to/python 3.2.2_compute_phenotypic_distance.sh
BASE_DATA_PATH="${BASE_DATA_PATH:-/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/}"
NUMBAT_BASE="${NUMBAT_BASE:-/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/}"
EMBEDDING_KEY="${EMBEDDING_KEY:-X_pca_slalom}"
PYTHON_BIN="${PYTHON_BIN:-/data1/chanj3/wangm10/software/plastro/bin/python}"

if [[ ! -f "$PATIENT_LIST" ]]; then
    echo "ERROR: patient list not found: $PATIENT_LIST"
    exit 1
fi

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

echo "Running phenotype distance for patient: $PATIENT_ID (task ${SLURM_ARRAY_TASK_ID}/${TOTAL_PATIENTS})"
"$PYTHON_BIN" "$PY_SCRIPT" "$PATIENT_ID" \
    --base-data-path "$BASE_DATA_PATH" \
    --numbat-base "$NUMBAT_BASE" \
    --embedding-key "$EMBEDDING_KEY" # "X_pca" used in RU1646

echo "End time: $(date)"
