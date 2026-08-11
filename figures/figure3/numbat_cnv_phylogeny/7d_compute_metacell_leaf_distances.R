#!/usr/bin/env Rscript
# 7d_compute_metacell_leaf_distances.R
#
# Compute distance_to_root for each metacell leaf from the metacell treeML.rds,
# mirroring the per-cell leaf_distances.csv already in numbat_output_FACETS_HISENS.
# Output: {patient}_leaf_distances.csv in numbat_output_FACETS_HISENS_metacell/
#   Columns: cell, distance_to_root, cell_type

suppressPackageStartupMessages(library(ape))

BASE        <- "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
OUTDIR_NAME <- "numbat_output_FACETS_HISENS_metacell"

script_dir <- tryCatch({
    argv <- commandArgs(FALSE)
    f    <- argv[grep("^--file=", argv)]
    dirname(normalizePath(sub("^--file=", "", f)))
}, error = function(e) getwd())

patient_list <- readLines(file.path(script_dir, "patient_list.txt"))
patient_list <- patient_list[nzchar(patient_list)]

for (pid in patient_list) {
    patient <- strsplit(pid, "_")[[1]][1]
    outdir  <- file.path(BASE, patient, OUTDIR_NAME)

    tree_path <- file.path(outdir, paste0(patient, "_treeML.rds"))
    if (!file.exists(tree_path)) {
        message(sprintf("[%s] treeML not found — skipping", patient))
        next
    }

    ct_path <- file.path(outdir, paste0(patient, "_joint_post_filtered.csv"))
    if (!file.exists(ct_path)) {
        message(sprintf("[%s] joint_post_filtered not found — skipping", patient))
        next
    }

    treeML <- tryCatch(readRDS(tree_path),
                       error = function(e) { message(e$message); NULL })
    if (is.null(treeML)) next

    # Distances from root to each node (tips first in ape ordering)
    if (is.null(treeML$edge.length)) {
        message(sprintf("[%s] tree has no edge lengths — skipping", patient))
        next
    }
    all_dist <- ape::node.depth.edgelength(treeML)
    n_tips   <- ape::Ntip(treeML)
    tip_dist <- all_dist[seq_len(n_tips)]

    # Cell type from joint_post_filtered (one row per cell per segment → deduplicate)
    jp <- tryCatch(
        read.csv(ct_path)[, c("cell", "cell_type")],
        error = function(e) NULL
    )
    if (is.null(jp)) {
        cell_type_map <- setNames(rep(NA_character_, n_tips), treeML$tip.label)
    } else {
        jp_uniq       <- jp[!duplicated(jp$cell), ]
        cell_type_map <- setNames(jp_uniq$cell_type, jp_uniq$cell)
    }

    out_df <- data.frame(
        cell             = treeML$tip.label,
        distance_to_root = tip_dist,
        cell_type        = cell_type_map[treeML$tip.label],
        stringsAsFactors = FALSE
    )

    out_path <- file.path(outdir, paste0(patient, "_leaf_distances.csv"))
    write.csv(out_df, out_path, row.names = FALSE, quote = TRUE)
    message(sprintf("[%s] %d metacells → %s", patient, n_tips, out_path))
}
