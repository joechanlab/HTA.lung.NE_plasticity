#!/usr/bin/env Rscript
# 5_extract_cnv_mutations.R
#
# Classifies CNA segments by cell-type overlap, consistent with SNV classification:
#   trunk          : segment carried by cells of ALL tumor cell types in the patient
#   major_subclonal: carried by ≥2 but not all tumor cell types (partially shared)
#   sample_specific: carried by cells of only 1 tumor cell type (private)
#   absent         : not detected in any tumor cell type
#
# Cell-type assignment comes from cnv_burden_per_cell.tsv (metacell → cell_type).
# CNA genotype per metacell comes from clone_post_2.tsv (GT_opt column).
#
# Output: cnv_mutations.tsv  (one row per patient × cell_type × seg_id)
#         cnv_summary.tsv    (one row per patient × seg_id, with classification)

suppressPackageStartupMessages({
    library(dplyr)
})

TUMOR_CTS <- c("LUAD", "LUSC", "SCLC-A", "SCLC-N", "SCLC-P",
                "NonNE SCLC", "EMT", "FOXI1+ NonNE SCLC-N")

# ── GRCh38 centromere midpoints (Mb) for arm assignment ──────────────────────
CENTROMERE <- c(
    "1"=123.4, "2"=93.9,  "3"=90.9,  "4"=50.4,  "5"=48.8,
    "6"=59.8,  "7"=60.6,  "8"=45.2,  "9"=43.0,  "10"=39.8,
    "11"=53.4, "12"=35.5, "13"=17.7, "14"=17.2, "15"=19.0,
    "16"=36.8, "17"=25.1, "18"=18.5, "19"=26.2, "20"=28.1,
    "21"=12.0, "22"=15.0, "X"=60.6,  "Y"=10.4
)

seg_label_fn <- function(chrom, seg_start, seg_end, cnv_state) {
    chrom_str <- as.character(chrom)
    cen_mb    <- CENTROMERE[chrom_str]
    mid_mb    <- (as.numeric(seg_start) + as.numeric(seg_end)) / 2e6
    arm       <- if (!is.na(cen_mb) && mid_mb < cen_mb) "p" else "q"
    state_str <- switch(cnv_state,
        "amp"  = "gain",
        "del"  = "loss",
        "loh"  = "LOH",
        "bamp" = "bamp",
        "bdel" = "homdel",
        cnv_state
    )
    paste0("chr", chrom_str, arm, " ", state_str)
}

# ── Configuration ─────────────────────────────────────────────────────────────
BASE        <- "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
OUTDIR_NAME <- "numbat_output_FACETS_HISENS_metacell"

script_dir <- tryCatch({
    argv <- commandArgs(FALSE)
    f    <- argv[grep("^--file=", argv)]
    dirname(normalizePath(sub("^--file=", "", f)))
}, error = function(e) getwd())

# ── Load per-cell CNA data with cell-type labels ──────────────────────────────
per_cell_file <- file.path(script_dir, "cnv_burden_per_cell.tsv")
per_cell_all  <- read.table(per_cell_file, header=TRUE, sep="\t",
                             stringsAsFactors=FALSE)
per_cell_all  <- per_cell_all[per_cell_all$cell_type %in% TUMOR_CTS, ]

patient_list_file <- file.path(script_dir, "patient_list.txt")
patients <- readLines(patient_list_file)
patients <- patients[nzchar(patients)]

detail_rows  <- list()
summary_rows <- list()

for (pid in patients) {
    patient_id <- strsplit(pid, "_")[[1]][1]
    outdir_pat <- file.path(BASE, patient_id, OUTDIR_NAME)

    # Metacell → cell_type for this patient (from pre-computed table)
    pat_cells <- per_cell_all[per_cell_all$patient_id == patient_id,
                               c("cell", "cell_type", "sample_id")]
    if (nrow(pat_cells) == 0) {
        message("  [", patient_id, "] No cell-type data — skipping")
        next
    }
    n_types_total <- length(unique(pat_cells$cell_type))

    # Discover per-sample subdirectories with clone_post_2.tsv
    sample_dirs   <- list.dirs(outdir_pat, full.names=FALSE, recursive=FALSE)
    valid_samples <- sample_dirs[sapply(sample_dirs, function(s) {
        file.exists(file.path(outdir_pat, s, "clone_post_2.tsv"))
    })]

    if (length(valid_samples) == 0) {
        message("  [", patient_id, "] No clone_post_2.tsv found — skipping")
        next
    }
    message("  [", patient_id, "] ", length(valid_samples), " sample(s), ",
            n_types_total, " tumor cell type(s): ",
            paste(unique(pat_cells$cell_type), collapse=", "))

    # ── Load consensus segments ────────────────────────────────────────────────
    segs_file <- file.path(outdir_pat, valid_samples[1], "segs_consensus_2.tsv")
    segs_all  <- tryCatch(
        read.table(segs_file, header=TRUE, sep="\t"),
        error = function(e) NULL
    )
    if (is.null(segs_all)) { message("    Cannot read segs — skipping"); next }

    segs_nonneu <- segs_all[segs_all$cnv_state_post != "neu",
                            c("seg","CHROM","seg_start","seg_end","cnv_state_post")]
    colnames(segs_nonneu)[5] <- "cnv_state"
    segs_nonneu$seg_label <- mapply(seg_label_fn,
        segs_nonneu$CHROM, segs_nonneu$seg_start,
        segs_nonneu$seg_end, segs_nonneu$cnv_state)

    if (nrow(segs_nonneu) == 0) {
        message("    No non-neutral segments — skipping")
        next
    }

    # ── Build seg → carrying cell types map ───────────────────────────────────
    # cell_id in pat_cells: "{sample_id}_SEACell-N"
    # cell_id in clone_post_2.tsv: "SEACell-N" — prepend "{sid}_" to match
    ct_lookup <- setNames(pat_cells$cell_type, pat_cells$cell)

    seg_carrying_cts <- list()   # seg_id → character vector of cell types

    for (sid in valid_samples) {
        cp_file <- file.path(outdir_pat, sid, "clone_post_2.tsv")
        cp <- tryCatch(
            read.table(cp_file, header=TRUE, sep="\t", stringsAsFactors=FALSE),
            error = function(e) NULL
        )
        if (is.null(cp) || nrow(cp) == 0) next

        # Only tumor compartment cells
        if ("compartment_opt" %in% colnames(cp)) {
            cp <- cp[cp$compartment_opt == "tumor", ]
        }
        if (nrow(cp) == 0) next

        full_ids <- paste0(sid, "_", cp$cell)
        cell_types_vec <- ct_lookup[full_ids]   # NA for non-tumor or missing

        for (i in seq_len(nrow(cp))) {
            ct <- cell_types_vec[i]
            if (is.na(ct)) next

            gt_str <- as.character(cp$GT_opt[i])
            if (is.na(gt_str) || gt_str == "") next
            segs_in_cell <- strsplit(gt_str, ",")[[1]]
            segs_in_cell <- segs_in_cell[segs_in_cell != ""]

            for (seg_id in segs_in_cell) {
                if (is.null(seg_carrying_cts[[seg_id]])) {
                    seg_carrying_cts[[seg_id]] <- character(0)
                }
                seg_carrying_cts[[seg_id]] <- union(seg_carrying_cts[[seg_id]], ct)
            }
        }

        # Detail rows: one per cell type × seg for this sample
        for (seg_id in segs_nonneu$seg) {
            sr <- segs_nonneu[segs_nonneu$seg == seg_id, ]
            if (nrow(sr) == 0) next
            cts_carrying <- seg_carrying_cts[[seg_id]]
            detail_rows[[length(detail_rows)+1]] <- data.frame(
                patient_id    = patient_id,
                sample_id     = sid,
                seg_id        = seg_id,
                CHROM         = sr$CHROM[1],
                seg_start     = sr$seg_start[1],
                seg_end       = sr$seg_end[1],
                cnv_state     = sr$cnv_state[1],
                seg_label     = sr$seg_label[1],
                cell_types    = paste(sort(cts_carrying), collapse=","),
                n_cell_types  = length(cts_carrying),
                stringsAsFactors = FALSE
            )
        }
    }

    # ── Classify each segment by cell-type overlap ────────────────────────────
    for (seg_id in segs_nonneu$seg) {
        sr  <- segs_nonneu[segs_nonneu$seg == seg_id, ]
        cts <- seg_carrying_cts[[seg_id]]
        if (is.null(cts)) cts <- character(0)
        n_carrying <- length(cts)

        cls <- if (n_carrying == 0) {
            "absent"
        } else if (n_types_total >= 2 && n_carrying == n_types_total) {
            "trunk"
        } else if (n_carrying >= 2) {
            "major_subclonal"   # partially shared (≥2 but not all cell types)
        } else {
            "sample_specific"   # private (only 1 cell type)
        }

        summary_rows[[length(summary_rows)+1]] <- data.frame(
            patient_id         = patient_id,
            seg_id             = seg_id,
            CHROM              = sr$CHROM[1],
            seg_start          = sr$seg_start[1],
            seg_end            = sr$seg_end[1],
            cnv_state          = sr$cnv_state[1],
            seg_label          = sr$seg_label[1],
            n_cell_types_total = n_types_total,
            n_cell_types_carry = n_carrying,
            cell_types_carry   = paste(sort(cts), collapse=","),
            classification     = cls,
            stringsAsFactors   = FALSE
        )
    }

    n_trunk <- sum(sapply(
        summary_rows[sapply(summary_rows, function(r) r$patient_id == patient_id)],
        function(r) r$classification == "trunk"
    ))
    message("    trunk=", n_trunk, "  total_nonneu=", nrow(segs_nonneu))
}

# ── Write outputs ─────────────────────────────────────────────────────────────
if (length(detail_rows) > 0) {
    detail_df <- bind_rows(detail_rows)
    detail_df <- detail_df[order(detail_df$patient_id, detail_df$sample_id,
                                 detail_df$CHROM, detail_df$seg_start), ]
    out_detail <- file.path(script_dir, "cnv_mutations.tsv")
    write.table(detail_df, out_detail, sep="\t", row.names=FALSE, quote=FALSE)
    message("\nDetail TSV: ", out_detail, "  (", nrow(detail_df), " rows)")
}

if (length(summary_rows) > 0) {
    summary_df <- bind_rows(summary_rows)
    summary_df <- summary_df[order(summary_df$patient_id,
                                   summary_df$CHROM, summary_df$seg_start), ]
    out_summary <- file.path(script_dir, "cnv_summary.tsv")
    write.table(summary_df, out_summary, sep="\t", row.names=FALSE, quote=FALSE)
    message("Summary TSV: ", out_summary, "  (", nrow(summary_df), " rows, ",
            length(unique(summary_df$patient_id)), " patients)")

    cat("\nClassification breakdown:\n")
    print(table(summary_df$classification))
}
