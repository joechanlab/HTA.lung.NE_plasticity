library(Matrix)
library(ape)
library(ggtree)
library(ggplot2)
library(dplyr)
library(patchwork)
suppressPackageStartupMessages(library(phangorn))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript 3.1.2c_plot_phylogeny_gene_expr.R <ID> [BASE_ROOT] [GENES]")
}

id        <- args[1]
base_root <- ifelse(length(args) >= 2, args[2],
                    "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All")
genes_arg <- if (length(args) >= 3) args[3] else "FOXI1,NEUROD1,YAP1"
genes_of_interest <- strsplit(genes_arg, ",")[[1]]

patient_id <- strsplit(id, "_")[[1]][1]

message(paste("Processing:", id))
message(paste("Genes:", paste(genes_of_interest, collapse = ", ")))

# ── Paths ─────────────────────────────────────────────────────────────────────
numbat_out_dir <- file.path(base_root, patient_id,
                            "numbat_output_FACETS_HISENS_metacell")
gex_root       <- file.path(base_root, patient_id, "GEX")

tree_rds_path <- file.path(numbat_out_dir, paste0(patient_id, "_treeML.rds"))
if (!file.exists(tree_rds_path) && id != patient_id) {
    fallback <- file.path(numbat_out_dir, id, paste0(id, "_treeML.rds"))
    if (file.exists(fallback)) {
        message("Using sample-level treeML fallback: ", fallback)
        tree_rds_path <- fallback
    }
}
if (!file.exists(tree_rds_path)) stop("treeML RDS not found: ", tree_rds_path)

# Find all samples for this patient and collect expression matrices
sample_names <- list.dirs(gex_root, full.names = FALSE, recursive = FALSE)

# ── Load and normalize expression ─────────────────────────────────────────────
read_metacell_expr <- function(sample_name, gex_root, genes_of_interest) {
    mc_dir  <- file.path(gex_root, sample_name, "metacell")
    bc_path <- file.path(mc_dir, "barcodes.tsv")
    gn_path <- file.path(mc_dir, "genes.tsv")
    mt_path <- file.path(mc_dir, paste0(sample_name, ".expression.mtx"))
    if (!file.exists(bc_path) || !file.exists(gn_path) || !file.exists(mt_path)) {
        message("Missing metacell files for: ", sample_name)
        return(NULL)
    }
    barcodes <- readLines(bc_path)
    genes    <- readLines(gn_path)
    mat      <- readMM(mt_path)   # genes × cells
    rownames(mat) <- genes
    colnames(mat) <- paste0(sample_name, "_", barcodes)

    # log1p(CPM) normalisation
    col_sums <- Matrix::colSums(mat)
    col_sums[col_sums == 0] <- 1
    mat_norm <- Matrix::t(Matrix::t(mat) / col_sums) * 1e4
    mat_log  <- log1p(mat_norm)

    present <- intersect(genes_of_interest, rownames(mat_log))
    missing <- setdiff(genes_of_interest, rownames(mat_log))
    if (length(missing) > 0) message("Genes not found in ", sample_name, ": ",
                                     paste(missing, collapse = ", "))
    as.data.frame(t(as.matrix(mat_log[present, , drop = FALSE])))
}

expr_list <- lapply(sample_names, read_metacell_expr,
                    gex_root = gex_root, genes_of_interest = genes_of_interest)
names(expr_list) <- sample_names
expr_list <- Filter(Negate(is.null), expr_list)

if (length(expr_list) == 0) stop("No expression data loaded.")
expr_df <- dplyr::bind_rows(expr_list)

# Ensure all requested genes are present as columns (fill missing with 0)
for (g in genes_of_interest) {
    if (!g %in% colnames(expr_df)) expr_df[[g]] <- 0
}
message(paste("Loaded expression for", nrow(expr_df), "metacells"))

# ── Palette (defines which cell types are "tumor" vs normal) ──────────────────
pal_annot <- c(
    "LUAD"                = "#ffd700",
    "LUSC"                = "#FFA500",
    "SCLC-A"              = "#d62728",
    "SCLC-N"              = "#17becf",
    "SCLC-P"              = "#1f77b4",
    "NonNE SCLC"          = "#4B0082",
    "EMT"                 = "#673147",
    "FOXI1+ NonNE SCLC-N" = "#2ca02c",
    "Normal"              = "#888888",
    "Unknown"             = "#AAAAAA",
    "0"                   = "#AAAAAA"
)

# ── Load tree ─────────────────────────────────────────────────────────────────
treeML <- readRDS(tree_rds_path)

# ── Collapse normal clades → single Normal outgroup (mirrors 3.1.2_plot_phylogeny.R) ──
collapse_normal_clades <- function(treeML, cell_types_df, tumor_types) {
    type_map <- setNames(cell_types_df$cell_type, cell_types_df$cell)
    n_tips   <- length(treeML$tip.label)
    is_tumor <- sapply(treeML$tip.label, function(lbl) {
        ct <- type_map[lbl]; !is.na(ct) && ct %in% tumor_types
    })
    normal_set <- treeML$tip.label[!is_tumor]

    empty <- list(tree   = treeML,
                  pseudo = data.frame(label = character(), n = integer(),
                                      stringsAsFactors = FALSE))
    if (length(normal_set) == 0) return(empty)

    parent_of <- setNames(treeML$edge[, 1], as.character(treeML$edge[, 2]))

    all_normal_internal <- integer(0)
    for (node in (n_tips + 1):(n_tips + treeML$Nnode)) {
        desc <- treeML$tip.label[phangorn::Descendants(treeML, node, "tips")[[1]]]
        if (length(desc) >= 2 && all(desc %in% normal_set))
            all_normal_internal <- c(all_normal_internal, node)
    }
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
    isolated   <- setdiff(normal_set, c(drop_tips, pseudo_old))
    pseudo_old <- c(pseudo_old, isolated)
    pseudo_n   <- c(pseudo_n,   rep(1L, length(isolated)))

    tree_new  <- ape::drop.tip(treeML, unique(drop_tips))
    pseudo_df <- data.frame(old = pseudo_old, n = pseudo_n, stringsAsFactors = FALSE)
    pseudo_df$label <- paste0(".Normal_n", pseudo_df$n, "_", seq_len(nrow(pseudo_df)))

    for (i in seq_len(nrow(pseudo_df))) {
        idx <- which(tree_new$tip.label == pseudo_df$old[i])
        if (length(idx) > 0) tree_new$tip.label[idx] <- pseudo_df$label[i]
    }
    list(tree = tree_new, pseudo = pseudo_df[, c("label", "n")])
}

# Load cell type annotations (needed for both collapse and cell-type panel)
cell_types <- dplyr::bind_rows(lapply(sample_names, function(s) {
    ct_path <- file.path(gex_root, s, "metacell", paste0(s, ".cell_types.tsv"))
    if (!file.exists(ct_path)) return(NULL)
    df <- read.table(ct_path, header = TRUE, sep = "\t")
    colnames(df) <- c("cell", "cell_type")
    df$cell <- paste0(s, "_", df$cell)
    df
}))
cell_types <- cell_types[cell_types$cell %in% treeML$tip.label, ]

tumor_types <- names(pal_annot)[!names(pal_annot) %in% c("Normal", "Unknown", "0")]
coll      <- collapse_normal_clades(treeML, cell_types, tumor_types)
pseudo_df <- coll$pseudo
message(paste("Collapsed", nrow(pseudo_df), "normal clade(s),",
              sum(pseudo_df$n), "normal cells total"))

tumor_tips <- setdiff(coll$tree$tip.label, pseudo_df$label)
if (length(pseudo_df$label) > 0 && length(tumor_tips) >= 2) {
    tumor_tree <- ape::drop.tip(coll$tree, pseudo_df$label)
} else {
    tumor_tree <- coll$tree
}

# Add single "Normal" outgroup leaf at the root (same branch length as original)
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
    if (!is.null(plot_tree$edge.length))
        plot_tree$edge.length[is.na(plot_tree$edge.length)] <- 0
} else {
    plot_tree <- tumor_tree
    n_total   <- 0L
}

# ── Propagate tip expression to internal nodes (post-order mean) ───────────────
compute_node_expr <- function(tree, expr_df, gene) {
    n_tips_local <- length(tree$tip.label)
    node_expr    <- rep(NA_real_, n_tips_local + tree$Nnode)

    tip_vals <- expr_df[[gene]][match(tree$tip.label, rownames(expr_df))]
    node_expr[seq_len(n_tips_local)] <- tip_vals

    children <- split(tree$edge[, 2], tree$edge[, 1])

    get_postorder <- function(nd) {
        kids <- children[[as.character(nd)]]
        if (is.null(kids)) return(nd)
        c(unlist(lapply(kids, get_postorder)), nd)
    }
    po_internal <- get_postorder(n_tips_local + 1L)
    po_internal <- po_internal[po_internal > n_tips_local]

    for (nd in po_internal) {
        desc_idx <- phangorn::Descendants(tree, nd, type = "tips")[[1]]
        vals <- node_expr[desc_idx]
        vals <- vals[!is.na(vals)]
        node_expr[nd] <- if (length(vals) > 0) mean(vals) else NA_real_
    }
    node_expr
}

# ── Build one plot per gene ────────────────────────────────────────────────────
make_expr_plot <- function(tree, node_expr, gene, show_legend = FALSE) {
    p <- ggtree(tree, layout = "rectangular")
    p$data$expr_val <- node_expr[p$data$node]

    p <- p +
        aes(color = expr_val) +
        scale_color_gradientn(
            colours = c("#d3d3d3", "#fee08b", "#f46d43", "#a50026"),
            na.value = "#d3d3d3",
            name = "log1p(CPM)"
        ) +
        ggtitle(gene) +
        theme(
            plot.title       = element_text(hjust = 0.5, size = 8, face = "italic"),
            legend.position  = if (show_legend) "right" else "none",
            legend.title     = element_text(size = 6),
            legend.text      = element_text(size = 5),
            legend.key.height = unit(0.4, "cm"),
            plot.background  = element_blank(),
            panel.background = element_blank()
        )
    p
}

plots <- lapply(seq_along(genes_of_interest), function(i) {
    gene <- genes_of_interest[i]
    node_expr <- compute_node_expr(plot_tree, expr_df, gene)
    is_last <- (i == length(genes_of_interest))
    make_expr_plot(plot_tree, node_expr, gene, show_legend = is_last)
})

# ── Cell-type reference panel (same tree as gene expression panels) ────────────
tip_ct <- setNames(
    cell_types$cell_type[match(plot_tree$tip.label, cell_types$cell)],
    plot_tree$tip.label
)
tip_ct["Normal"]          <- "Normal"
tip_ct[is.na(tip_ct)]    <- "Unknown"

unknown_types <- setdiff(unique(tip_ct), names(pal_annot))
for (ut in unknown_types) pal_annot[ut] <- "#AAAAAA"

# Majority vote coloring for cell type reference panel
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
        cts <- cts[cts != "" & !is.na(cts) & cts != "Normal" & cts != "Unknown"]
        if (length(cts) == 0) { node_ct[nd] <- "0"; next }
        node_ct[nd] <- names(sort(table(cts), decreasing = TRUE))[1]
    }
    node_ct[node_ct == "" | is.na(node_ct)] <- "0"
    node_ct
}

node_ct_maj <- color_by_majority(plot_tree, tip_ct)
p_ct <- ggtree(plot_tree, layout = "rectangular")
p_ct$data$edge_ct <- node_ct_maj[p_ct$data$node]
p_ct <- p_ct +
    aes(color = edge_ct) +
    scale_color_manual(values = pal_annot, na.value = "#AAAAAA",
                       name = "Cell type") +
    ggtitle("Cell type") +
    theme(
        plot.title       = element_text(hjust = 0.5, size = 8),
        legend.position  = "right",
        legend.title     = element_text(size = 6),
        legend.text      = element_text(size = 5),
        legend.key.size  = unit(0.35, "cm"),
        plot.background  = element_blank(),
        panel.background = element_blank()
    )

# ── Save combined figure ───────────────────────────────────────────────────────
all_plots <- c(list(p_ct), plots)
combined  <- Reduce(`+`, all_plots) +
    patchwork::plot_layout(ncol = length(all_plots))

out_dir  <- numbat_out_dir
out_path <- file.path(out_dir, paste0(patient_id, "_phylogeny_gene_expr.png"))

ggsave(combined, file = out_path,
       width = 2.2 * length(all_plots), height = 3.5,
       units = "in", dpi = 150)
message("Saved: ", out_path)

# ── Save each gene as a standalone PNG ────────────────────────────────────────
for (i in seq_along(genes_of_interest)) {
    gene <- genes_of_interest[i]
    p_solo <- plots[[i]] +
        theme(plot.title = element_blank(), legend.position = "right",
              legend.title = element_text(size = 6),
              legend.text  = element_text(size = 5),
              legend.key.height = unit(0.4, "cm"))
    solo_path <- file.path(out_dir,
                           paste0(patient_id, "_phylogeny_", gene, ".png"))
    ggsave(p_solo, file = solo_path,
           width = 2.5, height = 3, units = "in", dpi = 150)
    message("Saved: ", solo_path)
}
