#!/bin/bash
#SBATCH --job-name=numbat_prep
#SBATCH --output=logs/prep_%A_%a.out
#SBATCH --error=logs/prep_%A_%a.err
#SBATCH --array=22-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=cpu,chanj3
#SBATCH --time=7-00:00:00

# 1. Get the patient ID for this specific array task
# This reads the Nth line of patients.txt where N is the array index
PATIENT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" patient_list.txt)

echo "Starting processing for Patient: $PATIENT_ID"

# 3. Run the python script with the new argument
/usersoftware/chanj3/scdebussy-py311/bin/python 0.1_prepare_expression_data.py --patient_id "$PATIENT_ID" --mode "All"