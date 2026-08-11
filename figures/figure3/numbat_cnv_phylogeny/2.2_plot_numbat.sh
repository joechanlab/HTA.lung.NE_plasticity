#!/bin/bash
#SBATCH --job-name=numbat_plot
#SBATCH --output=logs/numbat_plot_%A_%a.out
#SBATCH --error=logs/numbat_plot_%A_%a.err
#SBATCH --array=11
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu,chanj3  # Replace with your actual partition name

# 1. Define the base path and find all target directories
BASE_PATH="/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
USE_METACELLS="TRUE"  # set to FALSE to use single-cell data directly
SIF_IMAGE="/usersoftware/chanj3/singularity/numbat-rbase.sif"

OUTDIR_SUFFIX=""
METACELL_FLAG=""
if [ "$USE_METACELLS" = "TRUE" ]; then
    OUTDIR_SUFFIX="_metacell"
    METACELL_FLAG="--metacell"
fi

DIRECTORIES=($(ls -d ${BASE_PATH}/*/numbat_output_FACETS_HISENS${OUTDIR_SUFFIX}))

# 2. Get the specific directory for this task ID
TARGET_DIR=${DIRECTORIES[$SLURM_ARRAY_TASK_ID]}

# 3. Extract the Patient ID from the path
PATIENT_ID=$(echo $TARGET_DIR | rev | cut -d'/' -f2 | rev)

echo "Processing Patient: $PATIENT_ID"
echo "Target Directory: $TARGET_DIR"

# 4. Run the R script
singularity exec --bind /data1:/data1 $SIF_IMAGE Rscript 2.2_plot_numbat.R --indir "$TARGET_DIR" $METACELL_FLAG
echo "End time: $(date)"