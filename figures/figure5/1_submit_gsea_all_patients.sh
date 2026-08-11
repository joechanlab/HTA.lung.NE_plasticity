#!/bin/bash
#SBATCH --job-name=gsea_deg
#SBATCH --output=logs/gsea_deg_%A_%a.out
#SBATCH --error=logs/gsea_deg_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --array=0-11
#SBATCH --partition=cpu,chanj3
#-----------------------------
# Step 1: per-patient DEG + GSEA (MSigDB Hallmarks) job array. One SLURM task
# per patient; see 1_gsea_cell_type_all_patients.py for the analysis. Downstream
# steps (2_direct_nonne_vs_luad.py, 3_nonne_luad_hallmark_venn.py) depend on this
# step's per-patient CSV outputs and must be rerun after this array completes.
#
# Usage: sbatch 1_submit_gsea_all_patients.sh [neighbor|sclc_pool]
#   neighbor (default): one-vs-nearest-neighbor, data-driven per patient.
#   sclc_pool: one-vs-fixed-SCLC-reference — required for "co-divergence from
#              SCLC" / cross-patient-comparable analyses (e.g. the step-3 venn).
#   Outputs go to different directories per mode, so re-running one never
#   clobbers the other.
#-----------------------------
set -euo pipefail
mkdir -p logs

source activate /usersoftware/chanj3/scdebussy-py311

REFERENCE_MODE=${1:-neighbor}

PATIENTS=(RU1444 RU151 RU942 RU1083 RU263 RU1303 RU1518 RU581 RU831 RU1068 RU1646 RU1314)
PATIENT=${PATIENTS[$SLURM_ARRAY_TASK_ID]}

echo "Processing Patient: ${PATIENT} (reference=${REFERENCE_MODE})"
python 1_gsea_cell_type_all_patients.py --patient "${PATIENT}" --reference "${REFERENCE_MODE}"
