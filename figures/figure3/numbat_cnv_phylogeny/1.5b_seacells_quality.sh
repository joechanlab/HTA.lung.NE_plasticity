#!/bin/bash
#SBATCH --job-name=seacells_qc
#SBATCH --output=logs/seacells_qc_%j.out
#SBATCH --error=logs/seacells_qc_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu,chanj3

echo "Start time: $(date)"
/usersoftware/chanj3/SEACells/bin/python 1.5b_seacells_quality.py
echo "End time: $(date)"
