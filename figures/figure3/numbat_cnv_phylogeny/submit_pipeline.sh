#!/bin/bash
# Submit the full Numbat pipeline with SLURM dependency chaining.
# Each stage waits for all jobs in the prior stage to complete successfully.
#
# Stage 1: SEACells aggregation (cell-type-stratified) — 36 array jobs
# Stage 2: Numbat CNV inference                         — 36 array jobs
# Stage 3: Scistreer phylogeny construction             — 17 array jobs
# Stage 4: Phylogenetic distance computation            —  1 job
#
# Usage:  bash submit_pipeline.sh
#         (run from the figures/figure3/Numbat/ directory)

set -e
mkdir -p logs

echo "=== Submitting SEACells aggregation (36 samples) ==="
JOB1=$(sbatch --parsable 1.5_seacells_aggregation.sh)
echo "  SEACells job array: $JOB1"

echo "=== Submitting Numbat (36 samples, depends on $JOB1) ==="
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 2.1_running_numbat.sh)
echo "  Numbat job array: $JOB2"

echo "=== Submitting Scistreer phylogeny (17 patients, depends on $JOB2) ==="
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 3.1.1_scistreer_cmd.sh)
echo "  Scistreer job array: $JOB3"

echo "=== Submitting phylogenetic distance computation (depends on $JOB3) ==="
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 3.2.1_compute_phylogenetic_distance.sh)
echo "  Distance job: $JOB4"

echo ""
echo "Pipeline submitted:"
echo "  [1] SEACells  $JOB1  (36 samples)"
echo "  [2] Numbat    $JOB2  (36 samples, after $JOB1)"
echo "  [3] Scistreer $JOB3  (17 patients, after $JOB2)"
echo "  [4] Distances $JOB4  (all patients, after $JOB3)"
echo ""
echo "Monitor with:  squeue -u $USER"
