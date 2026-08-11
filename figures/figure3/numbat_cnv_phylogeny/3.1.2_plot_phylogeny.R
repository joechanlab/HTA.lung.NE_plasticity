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
suppressPackageStartupMessages({
    library(phangorn)
    library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript 3.1.2_plot_phylogeny.R <ID> [BASE_ROOT] [USE_METACELLS] [MODE]\n  MODE: 'patient' (default) or 'sample'")
}

id        <- args[1]
base_root <- ifelse(length(args) >= 2, args[2], "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All")
use_metacells <- if (length(args) >= 3) as.logical(args[3]) else FALSE
mode      <- if (length(args) >= 4) args[4] else "patient"   # "patient" or "sample"
outdir_name <- paste0("numbat_output_FACETS_HISENS", if (use_metacells) "_metacell" else "")

if (mode == "sample") {
    sample_id  <- id
    patient_id <- strsplit(sample_id, "_")[[1]][1]
} else {
    patient_id <- strsplit(id, "_")[[1]][1]
    sample_id  <- NULL
}

message(paste("Mode:", mode))
message(paste("Processing:", id))
message(paste("Base root:", base_root))

# collapse_tree_by_cell_type <- function(treeML, joint_post, pal_annot, min_cells_normal = 2, min_cells_other = Inf) {
#     annot_map <- joint_post[, c("cell", "cell_type")] %>%
#         distinct() %>%
#         filter(cell %in% treeML$tip.label)

#     tip_type <- setNames(annot_map$cell_type, annot_map$cell)
#     palette_types <- names(pal_annot)
#     n_tip <- length(treeML$tip.label)
#     internal_nodes <- (n_tip + 1):(n_tip + treeML$Nnode)
#     root_node <- n_tip + 1

#     children_map <- split(treeML$edge[, 2], treeML$edge[, 1])
#     parent_map <- setNames(treeML$edge[, 1], treeML$edge[, 2])

#     node_hom_type <- rep(NA_character_, n_tip + treeML$Nnode)
#     node_n_tips <- integer(n_tip + treeML$Nnode)

#     for (i in seq_len(n_tip)) {
#         lbl <- treeML$tip.label[i]
#         node_hom_type[i] <- ifelse(is.na(tip_type[lbl]), NA_character_, as.character(tip_type[lbl]))
#         node_n_tips[i] <- 1L
#     }

#     get_postorder <- function(node) {
#         kids <- children_map[[as.character(node)]]
#         if (is.null(kids) || length(kids) == 0) {
#             return(node)
#         }
#         c(unlist(lapply(kids, get_postorder), use.names = FALSE), node)
#     }
#     postorder_nodes <- get_postorder(root_node)
#     postorder_internal <- postorder_nodes[postorder_nodes > n_tip]

#     for (node in postorder_internal) {
#         kids <- children_map[[as.character(node)]]
#         if (is.null(kids) || length(kids) == 0) {
#             next
#         }
#         child_types <- node_hom_type[kids]
#         child_sizes <- node_n_tips[kids]
#         node_n_tips[node] <- sum(child_sizes)

#         if (any(is.na(child_types))) {
#             node_hom_type[node] <- NA_character_
#         } else if (length(unique(child_types)) == 1) {
#             node_hom_type[node] <- child_types[1]
#         } else {
#             node_hom_type[node] <- NA_character_
#         }
#     }

#     node_candidates <- lapply(internal_nodes, function(node) {
#         this_type <- node_hom_type[node]
#         if (is.na(this_type)) {
#             return(NULL)
#         }
#         is_normal <- !(this_type %in% palette_types)
#         threshold <- ifelse(is_normal, min_cells_normal, min_cells_other)
#         if (node_n_tips[node] < threshold) {
#             return(NULL)
#         }
#         data.frame(node = node, cell_type = this_type, n_cells = node_n_tips[node], is_normal = is_normal, stringsAsFactors = FALSE)
#     })

#     node_candidates <- dplyr::bind_rows(node_candidates) %>%
#         dplyr::filter(is_normal)

#     if (nrow(node_candidates) == 0) {
#         annot_df <- data.frame(label = treeML$tip.label, cell_type = tip_type[treeML$tip.label], n_cells = 1, stringsAsFactors = FALSE)
#         return(list(tree = treeML, annot = annot_df, collapsed = data.frame()))
#     }

#     get_ancestors <- function(node) {
#         ancestors <- integer(0)
#         current <- as.character(node)
#         while (!is.na(parent_map[current])) {
#             p <- parent_map[current]
#             ancestors <- c(ancestors, p)
#             current <- as.character(p)
#             if (p == root_node) {
#                 break
#             }
#         }
#         ancestors
#     }

#     candidate_nodes <- node_candidates$node
#     keep_nodes <- candidate_nodes[sapply(candidate_nodes, function(node) {
#         anc <- get_ancestors(node)
#         !any(anc %in% candidate_nodes)
#     })]
#     node_candidates <- node_candidates %>% filter(node %in% keep_nodes)

#     if (nrow(node_candidates) == 0) {
#         annot_df <- data.frame(label = treeML$tip.label, cell_type = tip_type[treeML$tip.label], n_cells = 1, stringsAsFactors = FALSE)
#         return(list(tree = treeML, annot = annot_df, collapsed = data.frame()))
#     }

#     drop_tips <- character(0)
#     rep_map <- c()
#     rep_cell_type <- c()
#     rep_n_cells <- c()

#     for (i in seq_len(nrow(node_candidates))) {
#         node_i <- node_candidates$node[i]
#         type_i <- node_candidates$cell_type[i]
#         tip_idx <- phangorn::Descendants(treeML, node = node_i, type = "tips")[[1]]
#         tips <- treeML$tip.label[tip_idx]
#         rep_tip <- tips[1]
#         new_label <- paste0(type_i, " (n=", length(tips), ") [", node_i, "]")
#         rep_map[rep_tip] <- new_label
#         rep_cell_type[new_label] <- type_i
#         rep_n_cells[new_label] <- length(tips)
#         drop_tips <- c(drop_tips, setdiff(tips, rep_tip))
#     }

#     tree_collapsed <- ape::drop.tip(treeML, unique(drop_tips))
#     tree_collapsed$tip.label <- ifelse(tree_collapsed$tip.label %in% names(rep_map), rep_map[tree_collapsed$tip.label], tree_collapsed$tip.label)
#     tree_collapsed <- ape::collapse.singles(tree_collapsed)

#     annot_df <- data.frame(label = tree_collapsed$tip.label, stringsAsFactors = FALSE)
#     annot_df$cell_type <- ifelse(
#         annot_df$label %in% names(rep_cell_type),
#         rep_cell_type[annot_df$label],
#         tip_type[annot_df$label]
#     )
#     annot_df$n_cells <- ifelse(
#         annot_df$label %in% names(rep_n_cells),
#         as.numeric(rep_n_cells[annot_df$label]),
#         1
#     )

#     list(tree = tree_collapsed, annot = annot_df, collapsed = node_candidates)
# }

# plot_phylogeny <- function(treeML, joint_post, out_file, pal_annot) {
#     collapsed_obj <- collapse_tree_by_cell_type(treeML, joint_post, pal_annot = pal_annot)
#     tree_plot <- collapsed_obj$tree
#     annot_plot <- collapsed_obj$annot

#     tree_data_final = as_tibble(tree_plot) %>%
#         left_join(annot_plot, by = "label")

#     print(table(tree_data_final$cell_type))
#     message(paste("Collapsed clades:", nrow(collapsed_obj$collapsed)))

#     other_cells <- setdiff(unique(tree_data_final$cell_type), names(pal_annot))
#     for (other_cell in other_cells) {
#         pal_annot[other_cell] <- "#808080" 
#     }

#     png(out_file, width = 2000, height = 3000, res = 150)
#     p <- tree_plot %>%
#         ggtree(ladderize = TRUE, right = FALSE) %<+% tree_data_final + 
#         geom_tippoint(aes(color = cell_type, size = n_cells), alpha = 0.7) +
#         scale_color_manual(values = pal_annot) +
#         scale_size_continuous(range = c(1.5, 6)) +
#         theme(
#             axis.title.x = element_blank(),
#             axis.ticks.x = element_blank(),
#             axis.text.x = element_blank(),
#             axis.line.y = element_blank(),
#             axis.ticks.y = element_blank(),
#             axis.text.y = element_blank(),
#             panel.background = element_rect(fill = "transparent",colour = NA),
#             plot.background = element_rect(fill = "transparent", color = NA)
#         ) +
#         guides(color = 'none', size = 'none') +
#         theme(legend.position = "right")
#     print(p)
#     dev.off()
# }
# 1. Define the mapping dictionary (using a named vector)
# Note: R doesn't have a native bit literal like 0b100, so we use decimals
# 100=4, 110=6, 101=5, 011=3, 111=7
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

pal_annot = c(
    "LUAD" = "#ffd700",
    "LUSC" = "#FFA500",
    "SCLC-A" = "#d62728",
    "SCLC-N" = "#17becf",
    "SCLC-P" = "#1f77b4",
    "NonNE SCLC" = "#4B0082",
    "EMT" = "#673147",
    "FOXI1+ NonNE SCLC-N" = "#2ca02c"
)

# Find maximal subtrees whose tips are ALL non-tumor, drop their individual
# tips, and replace each clade with one pseudo-tip labeled ".Normal_n<k>_<i>".
# Returns list(tree, pseudo) where pseudo has columns (label, n).
collapse_normal_clades <- function(treeML, cell_types_df, tumor_types) {
    type_map   <- setNames(cell_types_df$cell_type, cell_types_df$cell)
    n_tips     <- length(treeML$tip.label)
    is_tumor   <- sapply(treeML$tip.label, function(lbl) {
        ct <- type_map[lbl]; !is.na(ct) && ct %in% tumor_types
    })
    normal_set <- treeML$tip.label[!is_tumor]

    empty <- list(tree  = treeML,
                  pseudo = data.frame(label=character(), n=integer(),
                                      stringsAsFactors=FALSE))
    if (length(normal_set) == 0) return(empty)

    parent_of <- setNames(treeML$edge[, 1], as.character(treeML$edge[, 2]))

    # Identify internal nodes whose every tip-descendant is normal
    all_normal_internal <- integer(0)
    for (node in (n_tips + 1):(n_tips + treeML$Nnode)) {
        desc <- treeML$tip.label[phangorn::Descendants(treeML, node, "tips")[[1]]]
        if (length(desc) >= 2 && all(desc %in% normal_set))
            all_normal_internal <- c(all_normal_internal, node)
    }

    # Keep only the maximal ones (parent is not itself all-normal)
    maximal <- all_normal_internal[vapply(all_normal_internal, function(nd) {
        par <- parent_of[as.character(nd)]
        is.na(par) || !(as.integer(par) %in% all_normal_internal)
    }, logical(1))]

    drop_tips  <- character(0)
    pseudo_old <- character(0)
    pseudo_n   <- integer(0)

    for (node in maximal) {
        lbls       <- treeML$tip.label[phangorn::Descendants(treeML, node, "tips")[[1]]]
        pseudo_old <- c(pseudo_old, lbls[1])
        pseudo_n   <- c(pseudo_n,   length(lbls))
        drop_tips  <- c(drop_tips,  lbls[-1])
    }

    # Isolated normal tips not captured by any all-normal internal node
    isolated   <- setdiff(normal_set, c(drop_tips, pseudo_old))
    pseudo_old <- c(pseudo_old, isolated)
    pseudo_n   <- c(pseudo_n,   rep(1L, length(isolated)))

    tree_new  <- ape::drop.tip(treeML, unique(drop_tips))
    pseudo_df <- data.frame(old=pseudo_old, n=pseudo_n, stringsAsFactors=FALSE)
    pseudo_df$label <- paste0(".Normal_n", pseudo_df$n, "_", seq_len(nrow(pseudo_df)))

    for (i in seq_len(nrow(pseudo_df))) {
        idx <- which(tree_new$tip.label == pseudo_df$old[i])
        if (length(idx) > 0) tree_new$tip.label[idx] <- pseudo_df$label[i]
    }

    list(tree=tree_new, pseudo=pseudo_df[, c("label", "n")])
}

numbat_out_dir <- file.path(base_root, patient_id, outdir_name)
gex_root       <- file.path(base_root, patient_id, "GEX")

if (mode == "sample") {
    out_label     <- sample_id
    tree_rds_path <- file.path(numbat_out_dir, sample_id, paste0(sample_id, "_treeML.rds"))
    out_png_path  <- file.path(numbat_out_dir, sample_id, paste0(sample_id, "_phylogeny.png"))
} else {
    out_label     <- patient_id
    tree_rds_path <- file.path(numbat_out_dir, paste0(patient_id, "_treeML.rds"))
    out_png_path  <- file.path(numbat_out_dir, paste0(patient_id, "_phylogeny.png"))
    # Fallback for patients with a single sample where only sample-level scistreer
    # output exists (e.g. id = "RU1518_T", patient_id = "RU1518").
    # Load the sample-level treeML but keep patient_id as the plot title and
    # save to the patient-level PNG path so the compile script finds it directly.
    if (!file.exists(tree_rds_path) && id != patient_id) {
        fallback <- file.path(numbat_out_dir, id, paste0(id, "_treeML.rds"))
        if (file.exists(fallback)) {
            message(paste("Patient-level treeML not found; using sample-level fallback:", fallback))
            tree_rds_path <- fallback
        }
    }
}

if (!file.exists(tree_rds_path)) {
    stop(paste("treeML RDS not found:", tree_rds_path))
}
if (!dir.exists(gex_root)) {
    stop(paste("GEX directory not found:", gex_root))
}

treeML = readRDS(tree_rds_path)

if (use_metacells) {
    # Cell types come from per-sample metacell cell_types.tsv files
    if (mode == "sample") {
        sample_names <- sample_id  # only the one sample
    } else {
        sample_names <- list.dirs(gex_root, full.names = FALSE, recursive = FALSE)
    }
    cell_types <- dplyr::bind_rows(lapply(sample_names, function(s) {
        ct_path <- file.path(gex_root, s, "metacell", paste0(s, ".cell_types.tsv"))
        if (!file.exists(ct_path)) return(NULL)
        df <- read.table(ct_path, header = TRUE, sep = "\t")
        colnames(df) <- c("cell", "cell_type")
        df$cell <- paste0(s, "_", df$cell)
        df
    }))
    nodes = treeML$tip.label
    # Metacell IDs (SEACell-XX) need no barcode translation
    cell_types = cell_types[cell_types$cell %in% nodes, ]
} else {
    obs_path <- file.path(base_root, "cell_type_labels.csv")
    cell_types = read.csv(obs_path)
    colnames(cell_types) = c('cell', 'cell_type')
    nodes = treeML$tip.label
    nodes_encoded = sapply(nodes, translate_lineage_barcode)
    treeML$tip.label = nodes_encoded
    cell_types = cell_types[cell_types$cell %in% nodes_encoded, ]
}

# ── Collapse normal clades then build tumor-only tree ─────────────────────────
coll      <- collapse_normal_clades(treeML, cell_types, names(pal_annot))
pseudo_df <- coll$pseudo
message(paste("Collapsed", nrow(pseudo_df), "normal clade(s),",
              sum(pseudo_df$n), "normal cells total"))

tumor_tips <- setdiff(coll$tree$tip.label, pseudo_df$label)
if (length(pseudo_df$label) > 0 && length(tumor_tips) >= 2) {
    tumor_tree <- ape::drop.tip(coll$tree, pseudo_df$label)
} else {
    tumor_tree <- coll$tree
}

# ── Add "Normal" as an outgroup leaf at the root ──────────────────────────────
# The root explicitly forks into Normal (single leaf) and the tumor subtree.
# Normal's branch length = max tumor tip depth so it sits at leaf level.
if (nrow(pseudo_df) > 0 && length(tumor_tips) >= 1) {
    n_total <- sum(pseudo_df$n)
    if (!is.null(tumor_tree$edge.length)) {
        depths    <- ape::node.depth.edgelength(tumor_tree)
        max_depth <- max(depths[seq_len(ape::Ntip(tumor_tree))], na.rm = TRUE)
    } else {
        max_depth <- max(ape::node.depth(tumor_tree)[seq_len(ape::Ntip(tumor_tree))])
    }
    nwk       <- sub(";$", "", ape::write.tree(tumor_tree))
    plot_tree <- ape::read.tree(text = sprintf("(Normal:%g,%s):0;", max_depth, nwk))
    # Edge from new root → old tumor root has no branch length in Newick → NA.
    # Set those NA edges to 0 so ggtree uses the phylogram layout correctly.
    if (!is.null(plot_tree$edge.length))
        plot_tree$edge.length[is.na(plot_tree$edge.length)] <- 0
} else {
    plot_tree <- tumor_tree
    n_total   <- 0L
}

# ── Shared palette setup ──────────────────────────────────────────────────────
tip_ct_named <- setNames(
    cell_types$cell_type[match(plot_tree$tip.label, cell_types$cell)],
    plot_tree$tip.label
)
tip_ct_named["Normal"] <- "Normal"
tip_ct_named[is.na(tip_ct_named)] <- "Unknown"

unknown_types <- setdiff(unique(tip_ct_named), names(pal_annot))
if (length(unknown_types) > 0)
    pal_annot <- c(pal_annot,
                   setNames(rep("#AAAAAA", length(unknown_types)), unknown_types))
pal_annot["Normal"]  <- "#888888"
pal_annot["0"]       <- "#AAAAAA"
pal_annot["Unknown"] <- "#AAAAAA"

options(ignore.negative.edge = TRUE)

# ── Shared plot constructor ───────────────────────────────────────────────────
# node_ct: character vector indexed 1..n_nodes; each value is a cell_type label.
# The edge parent→child is colored by the child node's value.
make_tree_plot <- function(tree, node_ct, subtitle, pal, n_normal = 0) {
    p <- ggtree(tree, layout = "rectangular")
    p$data$edge_ct <- node_ct[p$data$node]
    # Always pin Normal tip to its own color regardless of method
    p$data$edge_ct[!is.na(p$data$label) & p$data$label == "Normal"] <- "Normal"
    p <- p +
        aes(color = edge_ct) +
        scale_color_manual(values = pal, na.value = "#AAAAAA") +
        ggtitle(subtitle) +
        theme(
            legend.position  = "none",
            plot.title       = element_text(hjust = 0.5, size = 7),
            plot.background  = element_blank(),
            panel.background = element_blank()
        )
    if (n_normal > 0 && "Normal" %in% tree$tip.label) {
        circle_size <- pmax(1.0, pmin(5.0, 1.0 + log10(n_normal + 1)))
        norm_pt <- p$data %>% dplyr::filter(!is.na(label) & label == "Normal")
        if (nrow(norm_pt) > 0) {
            p <- p + geom_point(
                data = norm_pt, aes(x = x, y = y),
                size = circle_size, color = "#888888",
                alpha = 0.65, inherit.aes = FALSE, show.legend = FALSE
            )
        }
    }
    p
}

# ── Method 0: groupOTU (LCA path, original) ───────────────────────────────────
grp <- list()
for (ct in unique(cell_types$cell_type)) {
    in_tree <- cell_types$cell[cell_types$cell_type == ct]
    in_tree <- in_tree[in_tree %in% plot_tree$tip.label]
    if (length(in_tree) > 0) grp[[ct]] <- in_tree
}
if ("Normal" %in% plot_tree$tip.label) grp[["Normal"]] <- "Normal"

p0_raw <- ggtree(plot_tree) %>% groupOTU(., grp, "cell_type")
node_ct_otu <- setNames(as.character(p0_raw$data$cell_type),
                        as.character(p0_raw$data$node))
n_nodes <- length(node_ct_otu)
node_ct_otu_vec <- node_ct_otu[as.character(seq_len(n_nodes))]
p_otu <- make_tree_plot(plot_tree, node_ct_otu_vec,
                        "LCA path (groupOTU)", pal_annot, n_total)

# ── Method 1: Post-order majority vote ───────────────────────────────────────
# Each internal node gets the plurality cell type of all its descendant tips.
color_by_majority <- function(tree, tip_ct) {
    n_tips  <- length(tree$tip.label)
    n_nodes <- n_tips + tree$Nnode
    node_ct <- character(n_nodes)
    node_ct[seq_len(n_tips)] <- tip_ct[tree$tip.label]

    children <- split(tree$edge[, 2], tree$edge[, 1])

    get_postorder <- function(nd) {
        kids <- children[[as.character(nd)]]
        if (is.null(kids)) return(nd)
        c(unlist(lapply(kids, get_postorder)), nd)
    }
    po_internal <- get_postorder(n_tips + 1L)
    po_internal <- po_internal[po_internal > n_tips]

    for (nd in po_internal) {
        desc_idx <- phangorn::Descendants(tree, nd, type = "tips")[[1]]
        cts <- node_ct[desc_idx]
        cts <- cts[cts != "" & !is.na(cts) & cts != "Normal"]
        if (length(cts) == 0) { node_ct[nd] <- "0"; next }
        node_ct[nd] <- names(sort(table(cts), decreasing = TRUE))[1]
    }
    node_ct[node_ct == "" | is.na(node_ct)] <- "0"
    node_ct
}

node_ct_maj <- tryCatch(
    color_by_majority(plot_tree, tip_ct_named),
    error = function(e) { message("Majority vote failed: ", e$message); NULL }
)
p_maj <- if (!is.null(node_ct_maj)) {
    make_tree_plot(plot_tree, node_ct_maj, "Majority vote", pal_annot, n_total)
} else NULL

# ── Method 2: Fitch parsimony ─────────────────────────────────────────────────
# Post-order: each node's state = intersection of children states (or union if
# intersection is empty — i.e. a genuine state change occurs here).
# Ties within the union are resolved by descendant-tip majority.
color_by_parsimony <- function(tree, tip_ct) {
    n_tips  <- length(tree$tip.label)
    n_nodes <- n_tips + tree$Nnode
    ct_tips <- tip_ct[tree$tip.label]
    lvls    <- sort(unique(ct_tips[!is.na(ct_tips) & ct_tips != "Normal"]))
    if (length(lvls) < 2) {
        v <- rep(if (length(lvls) == 1) lvls else "0", n_nodes); return(v)
    }

    # State sets as lists (each element = valid states for that node)
    state_sets        <- vector("list", n_nodes)
    state_sets[seq_len(n_tips)] <- as.list(ct_tips)
    state_sets[sapply(state_sets, function(x) is.null(x) || is.na(x))] <- list("0")

    children <- split(tree$edge[, 2], tree$edge[, 1])
    get_postorder <- function(nd) {
        kids <- children[[as.character(nd)]]
        if (is.null(kids)) return(nd)
        c(unlist(lapply(kids, get_postorder)), nd)
    }
    po_internal <- get_postorder(n_tips + 1L)
    po_internal <- po_internal[po_internal > n_tips]

    node_ct <- character(n_nodes)
    node_ct[seq_len(n_tips)] <- ct_tips
    node_ct[is.na(node_ct) | node_ct == ""] <- "0"

    for (nd in po_internal) {
        kids    <- children[[as.character(nd)]]
        k_sets  <- lapply(kids, function(k) state_sets[[k]])
        # Remove Normal from state sets so it doesn't pollute internal nodes
        k_sets  <- lapply(k_sets, function(s) setdiff(s, "Normal"))
        inter   <- Reduce(intersect, k_sets)
        chosen  <- if (length(inter) > 0) inter else Reduce(union, k_sets)
        chosen  <- chosen[chosen != "0" & chosen != ""]
        if (length(chosen) == 0) { state_sets[[nd]] <- list("0"); node_ct[nd] <- "0"; next }
        state_sets[[nd]] <- chosen
        if (length(chosen) == 1) {
            node_ct[nd] <- chosen
        } else {
            # Tie: resolve by descendant-tip majority restricted to valid states
            desc_idx <- phangorn::Descendants(tree, nd, type = "tips")[[1]]
            cts      <- node_ct[desc_idx]
            cts      <- cts[cts %in% chosen]
            if (length(cts) > 0)
                node_ct[nd] <- names(sort(table(cts), decreasing = TRUE))[1]
            else
                node_ct[nd] <- chosen[1]
        }
    }
    node_ct[node_ct == "" | is.na(node_ct)] <- "0"
    node_ct
}

node_ct_pars <- tryCatch(
    color_by_parsimony(plot_tree, tip_ct_named),
    error = function(e) { message("Parsimony failed: ", e$message); NULL }
)
p_pars <- if (!is.null(node_ct_pars)) {
    make_tree_plot(plot_tree, node_ct_pars, "Fitch parsimony", pal_annot, n_total)
} else NULL

# ── Method 3: ML ancestral state (ape::ace) ───────────────────────────────────
# Skipped for trees > MAX_TIPS_ML to keep runtime reasonable.
MAX_TIPS_ML <- 500L
color_by_ml <- function(tree, tip_ct, max_tips = MAX_TIPS_ML) {
    n_tips <- length(tree$tip.label)
    if (n_tips > max_tips) {
        message(sprintf("ML ancestral state skipped (%d tips > %d)", n_tips, max_tips))
        return(NULL)
    }
    ct_clean <- tip_ct[tree$tip.label]
    lvls     <- sort(unique(ct_clean[!is.na(ct_clean) & ct_clean != "Normal"]))
    if (length(lvls) < 2) return(NULL)
    ct_clean[is.na(ct_clean) | ct_clean == "Normal"] <- lvls[1]

    fit <- tryCatch(
        ape::ace(ct_clean, tree, type = "discrete", method = "ML"),
        error = function(e) { message("ace() failed: ", e$message); NULL }
    )
    if (is.null(fit)) return(NULL)

    # fit$lik.anc: matrix[n_internal × n_states], rows ordered n_tips+1..n_nodes
    internal_states <- lvls[apply(fit$lik.anc, 1, which.max)]
    c(ct_clean, internal_states)   # tip states + internal states → length n_nodes
}

node_ct_ml <- tryCatch(
    color_by_ml(plot_tree, tip_ct_named),
    error = function(e) { message("ML failed: ", e$message); NULL }
)
p_ml <- if (!is.null(node_ct_ml)) {
    make_tree_plot(plot_tree, node_ct_ml,
                   sprintf("ML ancestral state (ace, n<=%d)", MAX_TIPS_ML),
                   pal_annot, n_total)
} else NULL

# ── Save individual PNGs ──────────────────────────────────────────────────────
method_plots <- list(
    groupotu  = list(plot = p_otu,  suffix = ""),          # original path
    majority  = list(plot = p_maj,  suffix = "_majority"),
    parsimony = list(plot = p_pars, suffix = "_parsimony"),
    ml        = list(plot = p_ml,   suffix = "_ml")
)

for (nm in names(method_plots)) {
    mp <- method_plots[[nm]]
    if (is.null(mp$plot)) next
    png_path <- sub("\\.png$", paste0(mp$suffix, ".png"), out_png_path)
    # Strip embedded method subtitle for individual PNGs (title lives in the report)
    p_save <- mp$plot + theme(plot.title = element_blank())
    ggsave(p_save, file = png_path, width = 2, height = 3, units = "in", dpi = 150)
    message(paste("Saved", nm, "phylogeny:", png_path))
}

# ── Combined 4-panel figure ───────────────────────────────────────────────────
valid_plots <- Filter(Negate(is.null), lapply(method_plots, `[[`, "plot"))
if (length(valid_plots) >= 2) {
    combined <- Reduce(`+`, valid_plots) +
        patchwork::plot_layout(ncol = length(valid_plots))
    combined_path <- sub("\\.png$", "_methods.png", out_png_path)
    ggsave(combined, file = combined_path,
           width = 2 * length(valid_plots), height = 3, units = "in", dpi = 150)
    message(paste("Saved combined methods:", combined_path))
}
