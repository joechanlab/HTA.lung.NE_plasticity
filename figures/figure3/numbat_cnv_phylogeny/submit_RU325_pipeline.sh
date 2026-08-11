#!/bin/bash
# Submit the Numbat pipeline for RU325 only (metacell level), with SLURM
# dependency chaining. RU325 is not part of the 17-patient LUAD->NE trajectory
# cohort (patient_list.txt / sample_list.txt) — it's a single, already-
# transformed pleural-fluid sample (SCLC-A + NonNE SCLC, no LUAD) that was
# never run through Numbat. These scripts are scoped to RU325 alone so they
# don't disturb the shared array-job list files.
#
# Stage 0: Prepare expression data (GEX matrices for RU325_PLE)
# Stage 1: BAM -> pileup -> phasing (allele counts)
# Stage 2: SEACells metacell aggregation (joint, majority-vote)
# Stage 3: Numbat CNV inference (metacell level, FACETS_HISENS prior)
#
# Usage:  bash submit_RU325_pipeline.sh
#         (run from the figures/figure3/Numbat/ directory)

set -e
mkdir -p logs

echo "=== Submitting RU325 expression prep ==="
JOB0=$(sbatch --parsable RU325_step0_prepare_expression.sh)
echo "  Job: $JOB0"

echo "=== Submitting RU325 pileup+phase (depends on $JOB0) ==="
JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 RU325_step1_pileup_phase.sh)
echo "  Job: $JOB1"

echo "=== Submitting RU325 SEACells metacell aggregation (depends on $JOB1) ==="
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 RU325_step2_seacells_metacell.sh)
echo "  Job: $JOB2"

echo "=== Submitting RU325 Numbat metacell run (depends on $JOB2) ==="
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 RU325_step3_numbat_metacell.sh)
echo "  Job: $JOB3"

echo ""
echo "Pipeline submitted:"
echo "  [0] Prep expr  $JOB0"
echo "  [1] Pileup     $JOB1  (after $JOB0)"
echo "  [2] SEACells   $JOB2  (after $JOB1)"
echo "  [3] Numbat     $JOB3  (after $JOB2)"
echo ""
echo "Monitor with:  squeue -u $USER"
