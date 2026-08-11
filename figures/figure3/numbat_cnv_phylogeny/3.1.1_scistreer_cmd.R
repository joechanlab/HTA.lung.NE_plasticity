library(numbat)
library(dplyr)
library(ggplot2)
library(ggraph)
library(tidygraph)
library(RColorBrewer)
library(igraph)
library(tidyr)
library(ape)
library(ggtree)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Usage: Rscript 3.1.1_scistreer_cmd.R <ID> [USE_METACELLS] [MODE]\n  MODE: 'patient' (default) or 'sample'")
}
id <- args[1]
use_metacells <- if (length(args) >= 2) as.logical(args[2]) else FALSE
mode <- if (length(args) >= 3) args[3] else "patient"   # "patient" or "sample"
outdir_name <- paste0("numbat_output_FACETS_HISENS", if (use_metacells) "_metacell" else "")

# Derive patient_id and the list of samples to process
if (mode == "sample") {
    sample_id  <- id
    patient_id <- strsplit(sample_id, "_")[[1]][1]
    samples_to_run <- sample_id
} else {
    patient_id <- id
    samples_to_run <- NULL   # filled below from GEX dir
}
print(paste("Mode:", mode))
print(paste("Processing:", id))
print(paste("USE_METACELLS:", use_metacells))
print(paste("outdir_name:", outdir_name))

extract_joint_post <- function(sample_name) {
    patient = strsplit(sample_name, split="_")[[1]][1]
    gex_dir <- if (use_metacells) {
        file.path("/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/", patient, "GEX", sample_name, "metacell")
    } else {
        file.path("/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/", patient, "GEX", sample_name)
    }
    jp_path <- file.path("/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/", patient, outdir_name, sample_name, "joint_post_2.tsv")
    if (!file.exists(jp_path)) {
        message(paste("  Skipping", sample_name, "— joint_post_2.tsv not found (Numbat not yet complete?)"))
        return(NULL)
    }
    annot_df_path <- file.path(gex_dir, paste0(sample_name, ".cell_types.tsv"))
    annot_df <- read.table(annot_df_path, header=TRUE, sep="\t")
    # normalise column names regardless of what the TSV header says
    colnames(annot_df)[1] <- "cell"
    colnames(annot_df)[2] <- "cell_type"
    barcodes <- read.csv(file.path(gex_dir, "barcodes.tsv"), header=FALSE, col.names="cell")
    # Prefix metacell IDs with sample name so they stay unique when combining
    # multiple samples (SEACells reuses SEACell-0, SEACell-1, ... per sample)
    annot_df$cell <- paste0(sample_name, "_", barcodes$cell)
    joint_post <- read.table(jp_path, sep="\t", header=TRUE)
    colnames(joint_post)[1] <- "cell"
    joint_post$cell <- paste0(sample_name, "_", joint_post$cell)
    joint_post %>%
      left_join(annot_df, by="cell")
}

filter_cnv <- function(joint_post) {
    joint_post_filtered = joint_post %>%
        filter(cnv_state %in% c('del', 'loh'))
    n_cnv = length(unique(joint_post_filtered$seg))
    print(paste0('Filtered to ', n_cnv, ' CNVs to construct phylogeny'))
    joint_post_filtered
}

construct_phylogeny <- function(joint_post_filtered) {
    P_MIN = 1e-10
    P = joint_post_filtered %>%
        mutate(p_cnv = pmax(pmin(p_cnv, 1-P_MIN), P_MIN)) %>%
        data.table::as.data.table() %>%
        data.table::dcast(cell ~ seg, value.var = 'p_cnv', fill = 0.5) %>%
        tibble::column_to_rownames('cell') %>%
        as.matrix
    dist_mat = parallelDist::parDist(rbind(P, 'outgroup' = 0), threads = 2)

    treeUPGMA = phangorn::upgma(dist_mat) %>%
        ape::root(outgroup = 'outgroup') %>%
        ape::drop.tip('outgroup') %>%
        reorder(order = 'postorder') 
    UPGMA_score = scistreer::score_tree(treeUPGMA, as.matrix(P))$l_tree
    treeNJ = ape::nj(dist_mat) %>%
        ape::root(outgroup = 'outgroup') %>%
        ape::drop.tip('outgroup') %>%
        reorder(order = 'postorder')
    NJ_score = scistreer::score_tree(treeNJ, as.matrix(P))$l_tree
    if (UPGMA_score > NJ_score) {
        tree_init = treeUPGMA
        print('Using UPGMA tree as seed..')
    } else {
        tree_init = treeNJ
        print('Using NJ tree as seed..')
    }

    NCORES = 2
    EPS = 1e-5
    MAX_ITER = 100
    tree_list = scistreer::perform_nni(tree_init, P, ncores = NCORES, eps = EPS, max_iter = MAX_ITER)
    treeML = tree_list[[length(tree_list)]]
    treeML
}

plot_phylogeny <- function(treeML, joint_post, out_file, pal_annot) {
    tree_data_final =  as_tibble(treeML) %>%
        left_join(joint_post[,c('cell', 'cell_type')] %>% distinct(), by = c("label" = "cell"))
    print(table(tree_data_final$cell_type))
    other_cells <- setdiff(unique(tree_data_final$cell_type), names(pal_annot))
    for (other_cell in other_cells) {
        pal_annot[other_cell] <- "#808080" 
    }
    png(out_file, width = 2000, height = 3000, res = 150)
    p <- treeML %>%
        ggtree(ladderize = TRUE, right = FALSE) %<+% tree_data_final + 
        geom_tippoint(aes(color = cell_type), size = 3, alpha = 0.5) +
        scale_color_manual(values = pal_annot) +
        theme(
            axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.text.x = element_blank(),
            axis.line.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank(),
            panel.background = element_rect(fill = "transparent",colour = NA),
            plot.background = element_rect(fill = "transparent", color = NA)
        ) +
        guides(color = 'none') +
        theme(legend.position = "right")
    print(p)
    dev.off()
}

pal_annot = c(
    "LUAD" = "#ffd700",
    "LUSC" = "#FFA500",
    "SCLC-A" = "#d62728",  
    "SCLC-N" = "#17becf",
    "SCLC-P" = "#1f77b4",
    "NonNE SCLC" = "#4B0082",
    "EMT" = "#673147"
)
base_root <- "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"

if (mode == "patient") {
    samples_to_run <- list.dirs(file.path(base_root, patient_id, "GEX"), full=FALSE, recursive=FALSE)
    out_label   <- patient_id
    out_dir     <- file.path(base_root, patient_id, outdir_name)
} else {
    out_label   <- sample_id
    out_dir     <- file.path(base_root, patient_id, outdir_name, sample_id)
}

joint_post_total <- data.frame()
for (sample_name in samples_to_run) {
    print(sample_name)
    joint_post_sample <- extract_joint_post(sample_name)
    joint_post_total  <- dplyr::bind_rows(joint_post_total, joint_post_sample)
}
if (nrow(joint_post_total) == 0) {
    stop(paste("No Numbat output found for", id,
               "— ensure Numbat has completed before running scistreer."))
}
joint_post_filtered <- filter_cnv(joint_post_total)
table(distinct(joint_post_filtered, cell, cell_type)$cell_type)
treeML <- construct_phylogeny(joint_post_filtered)
plot_phylogeny(treeML, joint_post_filtered, paste0(out_label, "_phylogeny.png"), pal_annot)
write.csv(joint_post_filtered, file.path(out_dir, paste0(out_label, "_joint_post_filtered.csv")))
saveRDS(treeML, file=file.path(out_dir, paste0(out_label, "_treeML.rds")))
