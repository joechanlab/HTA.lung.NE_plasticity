#!/bin/bash
#SBATCH -p chanj3
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fig8
#SBATCH -o log/fig8_%A_%a.out
#SBATCH -e log/fig8_%A_%a.err
#SBATCH --array=0-1

# Define the cell lines indexed by array task ID
CELL_LINES=("H1975" "PC9")
CELL_LINE=${CELL_LINES[$SLURM_ARRAY_TASK_ID]}

echo "Running Figure 8 for cell line: $CELL_LINE"

# OPTIONAL: ensure script runs from the directory where it was submitted
cd $SLURM_SUBMIT_DIR

# Run your Python script
/usersoftware/chanj3/SnapATAC2/bin/python \
    Fig8_PHOX2B_Split_Cell_Line.py \
    --cell_line $CELL_LINE