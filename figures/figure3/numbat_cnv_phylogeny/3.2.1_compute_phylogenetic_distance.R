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

STR2BINDICT <- c(
    "A" = 4, "C" = 6, "G" = 5, "T" = 3, "N" = 7,
    "a" = 4, "c" = 6, "g" = 5, "t" = 3, "n" = 7
)

# 2. Encode function
encode <- function(sequence) {
    # Split string into individual characters
    chars <- strsplit(sequence, "")[[1]]
    res <- 0
    
    for (nucleotide in chars) {
        # res <<= 3
        res <- res * 2^3
        # res += STR2BINDICT[nucleotide]
        res <- res + STR2BINDICT[nucleotide]
    }
    
    # Format as a string to avoid scientific notation for very large numbers
    return(format(res, scientific = FALSE, trim = TRUE))
}

# 3. Translate Lineage Barcode function
translate_lineage_barcode <- function(barcode) {
    patterns <- c(
        "^(.*)_\\[?([ACGTNacgtn]+)\\]?-1$",
        "^(.*)-\\[?([ACGTNacgtn]+)\\]?-1$"
    )
    
    for (pat in patterns) {
        # Check for match
        if (grepl(pat, barcode)) {
            # Extract capture groups
            matches <- regmatches(barcode, regexec(pat, barcode))[[1]]
            prefix <- matches[2]
            sequence <- matches[3]
            
            return(paste0(prefix, "_", encode(sequence)))
        }
    }
    
    # Return original if no match
    return(barcode)
}

get_pairwise_distances <- function(tree_path) {
    tr <- readRDS(tree_path)
    phylo_tree <- as.phylo(tr)
    dist_matrix <- cophenetic(phylo_tree)
    dist_matrix <- as.matrix(dist_matrix)
    return(dist_matrix)
}

run_sample_distance <- function(tr, annot) {
    tree_data <- as_tibble(tr) %>%
        left_join(
            annot, by = c("label" = "cell")
        )
    phylo_tree <- as.phylo(tr)
    all_distances <- node.depth.edgelength(phylo_tree)
    n_tips <- length(phylo_tree$tip.label)
    
    dist_df <- data.frame(
        cell = phylo_tree$tip.label,
        distance_to_root = all_distances[1:n_tips]
    ) %>%
        left_join(
            annot, by = "cell"
        )
    return(dist_df)
}

args         <- commandArgs(trailingOnly = TRUE)
use_metacells <- if (length(args) >= 1) as.logical(args[1]) else TRUE
outdir_name  <- paste0("numbat_output_FACETS_HISENS", if (use_metacells) "_metacell" else "")
message(paste("USE_METACELLS:", use_metacells, "→ reading from", outdir_name))

samples = read.csv("patient_list.txt", header=FALSE, stringsAsFactors = FALSE)$V1
base_path <- "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/"
# pairwise distance
for (sid in samples) {
    message(paste0("--- Processing Sample: ", sid, " ---"))
    tree_path <- file.path(base_path, sid, outdir_name, paste0(sid, "_treeML.rds"))
    result <- tryCatch({
        if (!file.exists(tree_path)) {
            stop(paste("Files missing for", sid))
        }
        df <- get_pairwise_distances(tree_path)
        write.csv(df, file.path(base_path, sid, outdir_name, paste0(sid, "_pairwise_distances.csv")), row.names = FALSE)
        
    }, error = function(e) {
        message(paste0("Error in sample ", sid, ": ", e$message))
        return(NULL) 
    })
}

# distance to root
# --- Distance to Root ---
base_root = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
obs_path <- file.path(base_root, "cell_type_labels.csv")
cell_types = read.csv(obs_path)
colnames(cell_types) = c('cell', 'cell_type')

for (sid in samples) {
    sid = strsplit(sid, "_")[[1]][1]
    message(paste0("--- Calculating Distance to Root: ", sid, " ---"))
    tree_path <- file.path(base_path, sid, outdir_name, paste0(sid, "_treeML.rds"))
    output_path <- file.path(base_path, sid, outdir_name, paste0(sid, "_leaf_distances.csv"))
    
    result <- tryCatch({
        tr <- readRDS(tree_path)
        nodes = tr$tip.label
        nodes_encoded = sapply(nodes, translate_lineage_barcode)
        tr$tip.label = nodes_encoded
        annot = cell_types[cell_types$cell %in% nodes_encoded,]

        df_root <- run_sample_distance(tr, annot)
        
        write.csv(df_root, output_path, row.names = FALSE)
        message(paste0("Successfully saved root distances for ", sid))
        
    }, error = function(e) {
        message(paste0("Error calculating root distance for ", sid, ": ", e$message))
        return(NULL)
    })
}
