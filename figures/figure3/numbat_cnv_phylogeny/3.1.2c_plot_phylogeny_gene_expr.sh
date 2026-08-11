#!/bin/bash
#SBATCH --job-name=phylogeny_gene_expr
#SBATCH --output=logs/phylogeny_gene_expr_%j.out
#SBATCH --error=logs/phylogeny_gene_expr_%j.err
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu,chanj3
set -euo pipefail

echo "Start time: $(date)"

SIF_IMAGE="/usersoftware/chanj3/singularity/numbat-rbase.sif"
BASE_ROOT="/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
ID="${ID:-RU1518}"
GENES="${GENES:-FOXI1,NEUROD1,YAP1}"

echo "Processing: $ID  Genes: $GENES"

singularity exec --bind /data1:/data1 "$SIF_IMAGE" \
    Rscript 3.1.2c_plot_phylogeny_gene_expr.R "$ID" "$BASE_ROOT" "$GENES"

echo "End time: $(date)"
