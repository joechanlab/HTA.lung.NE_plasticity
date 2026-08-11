#!/usr/bin/env Rscript
# 7_extract_phylo_celltype.R
#
# For each patient × sample, extracts:
#   - Clone phylogenetic depth (from mut_graph_2.rds)
#   - Branch-specific CNV events (last_mut) per clone
#   - Tumor cell-type composition per clone (from clone_post_2.tsv + joint_post)
#   - Segment arm-labels (from segs_consensus_2.tsv)
#
# Output: phylo_celltype.tsv
#   Columns: patient_id, sample_id, clone_id, depth, n_cells,
#            cell_type, ct_frac,       <- one row per clone × cell_type
#            branch_segs,              <- comma-sep segment IDs on this branch
#            branch_labels,            <- comma-sep arm labels on this branch
#            cum_segs                  <- cumulative segments in this clone (GT)

suppressPackageStartupMessages({
    library(igraph)
    library(dplyr)
})

# ── GRCh38 centromere midpoints (Mb) ─────────────────────────────────────────
CENTROMERE <- c(
    "1"=123.4,"2"=93.9,"3"=90.9,"4"=50.4,"5"=48.8,
    "6"=59.8,"7"=60.6,"8"=45.2,"9"=43.0,"10"=39.8,
    "11"=53.4,"12"=35.5,"13"=17.7,"14"=17.2,"15"=19.0,
    "16"=36.8,"17"=25.1,"18"=18.5,"19"=26.2,"20"=28.1,
    "21"=12.0,"22"=15.0,"X"=60.6,"Y"=10.4
)

arm_label <- function(chrom, seg_start, seg_end, cnv_state) {
    c   <- as.character(chrom)
    cen <- CENTROMERE[c]
    mid <- (as.numeric(seg_start) + as.numeric(seg_end)) / 2e6
    arm <- if (!is.na(cen) && mid < cen) "p" else "q"
    st  <- switch(cnv_state,
        "amp"="gain","del"="loss","loh"="LOH","bamp"="bamp","bdel"="homdel",cnv_state)
    paste0("chr", c, arm, " ", st)
}

depth_from_parent_map <- function(node, parent_map) {
    d <- 0L; cur <- node
    while (!is.na(parent_map[cur])) { d <- d + 1L; cur <- parent_map[cur] }
    d
}

TUMOR_TYPES <- c("LUAD","LUSC","SCLC-A","SCLC-N","SCLC-P",
                 "NonNE SCLC","EMT","FOXI1+ NonNE SCLC-N")

BASE        <- "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
OUTDIR_NAME <- "numbat_output_FACETS_HISENS_metacell"

script_dir <- tryCatch({
    argv <- commandArgs(FALSE)
    f    <- argv[grep("^--file=", argv)]
    dirname(normalizePath(sub("^--file=", "", f)))
}, error = function(e) getwd())

patient_list <- readLines(file.path(script_dir, "patient_list.txt"))
patient_list <- patient_list[nzchar(patient_list)]

all_rows <- list()

for (pid_entry in patient_list) {
    patient_id <- strsplit(pid_entry, "_")[[1]][1]
    outdir_pat <- file.path(BASE, patient_id, OUTDIR_NAME)

    # Patient-level joint_post → cell → cell_type
    jp_path <- file.path(outdir_pat, paste0(patient_id, "_joint_post_filtered.csv"))
    if (!file.exists(jp_path)) {
        message("  [", patient_id, "] No joint_post — skipping")
        next
    }
    jp <- read.csv(jp_path, stringsAsFactors=FALSE)
    ct_map <- jp[!duplicated(jp$cell), c("cell","cell_type")]

    # Discover samples
    sample_dirs <- list.dirs(outdir_pat, full.names=FALSE, recursive=FALSE)
    valid_samples <- sample_dirs[sapply(sample_dirs, function(s) {
        file.exists(file.path(outdir_pat, s, "mut_graph_2.rds")) &&
        file.exists(file.path(outdir_pat, s, "clone_post_2.tsv"))
    })]

    if (length(valid_samples) == 0) {
        message("  [", patient_id, "] No valid samples — skipping")
        next
    }
    message("  [", patient_id, "] ", length(valid_samples), " sample(s)")

    for (sid in valid_samples) {
        sdir <- file.path(outdir_pat, sid)

        # Segment arm labels
        segs <- tryCatch(
            read.table(file.path(sdir, "segs_consensus_2.tsv"), header=TRUE, sep="\t"),
            error=function(e) NULL)
        if (is.null(segs)) next

        seg_lbl_map <- setNames(
            mapply(arm_label, segs$CHROM, segs$seg_start, segs$seg_end, segs$cnv_state_post),
            segs$seg)

        # Clone assignments
        cp <- tryCatch(
            read.table(file.path(sdir, "clone_post_2.tsv"), header=TRUE, sep="\t"),
            error=function(e) NULL)
        if (is.null(cp)) next

        cp_tumor <- cp[cp$compartment_opt == "tumor", c("cell","clone_opt")]
        cp_tumor$cell_full <- paste0(sid, "_", cp_tumor$cell)

        merged <- merge(cp_tumor, ct_map, by.x="cell_full", by.y="cell")
        if (nrow(merged) == 0) next

        # Clone cell-type composition
        clone_comp <- merged %>%
            group_by(clone_id=clone_opt, cell_type) %>%
            summarise(n_ct=n(), .groups="drop") %>%
            group_by(clone_id) %>%
            mutate(ct_frac=n_ct/sum(n_ct), n_cells=sum(n_ct)) %>%
            ungroup() %>%
            filter(cell_type %in% TUMOR_TYPES)    # keep only tumor types

        if (nrow(clone_comp) == 0) next

        # Mut-graph depths and branch CNVs
        mg  <- tryCatch(readRDS(file.path(sdir, "mut_graph_2.rds")), error=function(e) NULL)
        if (is.null(mg)) next

        vdf <- igraph::as_data_frame(mg, what="vertices")
        edf <- igraph::as_data_frame(mg, what="edges")
        parent_map <- setNames(edf$from_node, edf$to_node)

        vdf$depth <- sapply(vdf$node,
            function(n) if (is.na(n)) 0L else depth_from_parent_map(n, parent_map))
        clone_tree <- vdf[!is.na(vdf$clone), c("clone","depth","last_mut","GT")]
        clone_tree$clone <- as.integer(clone_tree$clone)

        # Join
        full <- merge(clone_comp, clone_tree,
                      by.x="clone_id", by.y="clone", all.x=TRUE)

        # Expand branch_segs → arm labels
        full$branch_labels <- sapply(full$last_mut, function(s) {
            if (is.na(s) || s == "") return("")
            segs_ids <- strsplit(s, ",")[[1]]
            paste(na.omit(seg_lbl_map[segs_ids]), collapse="; ")
        })
        full$cum_labels <- sapply(full$GT, function(s) {
            if (is.na(s) || s == "") return("")
            segs_ids <- strsplit(s, ",")[[1]]
            paste(na.omit(seg_lbl_map[segs_ids]), collapse="; ")
        })

        full$patient_id  <- patient_id
        full$sample_id   <- sid
        full$branch_segs <- full$last_mut
        full$cum_segs    <- full$GT

        out <- full[, c("patient_id","sample_id","clone_id","depth","n_cells",
                        "cell_type","ct_frac","branch_segs","branch_labels",
                        "cum_segs","cum_labels")]

        all_rows[[length(all_rows)+1]] <- out
        message("    ", sid, ": ", nrow(clone_comp), " tumor clone×ct rows")
    }
}

if (length(all_rows) == 0) {
    message("No data extracted.")
    quit(save="no", status=0)
}

result <- bind_rows(all_rows)
result  <- result[order(result$patient_id, result$sample_id,
                        result$depth, result$clone_id), ]

out_tsv <- file.path(script_dir, "phylo_celltype.tsv")
write.table(result, out_tsv, sep="\t", row.names=FALSE, quote=FALSE)
message("\nSaved: ", out_tsv, "  (", nrow(result), " rows, ",
        length(unique(result$patient_id)), " patients)")
