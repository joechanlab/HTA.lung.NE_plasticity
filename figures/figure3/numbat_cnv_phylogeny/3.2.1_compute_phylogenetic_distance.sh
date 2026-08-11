#!/bin/bash
#SBATCH --job-name=phylo_dist
#SBATCH --output=logs/phylo_dist_%j.out
#SBATCH --error=logs/phylo_dist_%j.err
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu,chanj3

echo "Start time: $(date)"

SIF_IMAGE="/usersoftware/chanj3/singularity/numbat-rbase.sif"
USE_METACELLS="TRUE"

echo "Computing phylogenetic distances (USE_METACELLS=$USE_METACELLS) ..."

singularity exec --bind /data1:/data1 $SIF_IMAGE \
    Rscript 3.2.1_compute_phylogenetic_distance.R $USE_METACELLS

echo "End time: $(date)"
