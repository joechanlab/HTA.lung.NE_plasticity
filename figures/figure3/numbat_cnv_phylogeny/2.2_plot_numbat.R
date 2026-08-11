library(optparse)
library(numbat)
library(dplyr)
library(ggplot2)
library(ggraph)
library(tidygraph)
library(RColorBrewer)

option_list <- list(
    make_option(c("-i", "--indir"), type="character", default=NULL,
                help="Base Numbat directory containing subfolders", metavar="character"),
    make_option(c("-m", "--metacell"), action="store_true", default=FALSE,
                help="Read cell type annotations from metacell/ subdirectories")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$indir)){
    print_help(opt_parser)
    stop("Input Directory must be supplied", call.=FALSE)
}

numbat_root  <- opt$indir
use_metacells <- isTRUE(opt$metacell)
plot_annot_stratified_heatmap_v2 <- function(gtree, joint_post, clone_post, segs_consensus, annot, pal_annot = NULL, annot_title = 'Annotation') {
    if (is.null(pal_annot) || length(pal_annot) < length(unique(annot[[2]]))) {
        message("Expanding palette to accommodate more categories...")
        pal_annot <- scales::hue_pal()(length(unique(annot[[2]])))
    }
    # 1. Get the natural phylogenetic order from the tree
    # This represents the "Phylo" info for the finest level of sorting
    phylo_order = numbat:::get_ordered_tips(gtree)
    # 2. Build a hierarchy: Cell Type > Clone > Phylo Order
    annot_ordered = annot %>%
        # Add clone information to the annotation dataframe
        dplyr::left_join(
            clone_post %>% dplyr::select(cell, clone_opt) %>% dplyr::distinct(), 
            by = "cell"
        ) %>%
        # Add a numerical rank based on the phylogeny
        dplyr::mutate(phylo_rank = match(cell, phylo_order)) %>%
        # Multi-level sort: Primary (Type), Secondary (Clone), Tertiary (Tree Position)
        dplyr::arrange(
            dplyr::across(2), # Your cell type column
            clone_opt,        # Group by clone within types
            phylo_rank        # Sort by tree position within clones
        )
    cell_order = annot_ordered$cell
    # 3. Re-factor joint_post to follow this nested hierarchy
    joint_post = joint_post %>%
        dplyr::mutate(cell = factor(cell, levels = cell_order)) %>%
        dplyr::filter(!is.na(cell)) %>%
        dplyr::mutate(cell_index = as.integer(droplevels(cell)))
    # 4. Build the heatmap segments (with aligned Y-axis)
    p_segs = ggplot(joint_post %>% dplyr::mutate(cnv_state = ifelse(cnv_state == 'neu', NA, cnv_state))) +
        geom_segment(aes(x = seg_start, xend = seg_end, y = cell_index, yend = cell_index, color = cnv_state, alpha = p_cnv), size = 0.1) +
        scale_y_continuous(expand = c(0, 0)) + 
        theme_classic() +
        theme(
            panel.spacing = unit(0, 'mm'),
            axis.text = element_blank(),
            axis.ticks = element_blank(),
            strip.background = element_blank(),
            panel.border = element_rect(linewidth = 0.5, color = 'gray', fill = NA),
            plot.margin = margin(0, 0, 0, 0)
        ) +
        facet_grid(.~CHROM, space = 'free', scales = 'free') +
        scale_color_manual(values = c('amp' = 'darkred', 'del' = 'darkblue', 'loh' = 'darkgreen', 'bamp' = 'salmon', 'bdel' = 'blue'))
    # 5. Create the sidebar
    p_annot = numbat:::annot_bar(
        D = annot_ordered %>% dplyr::rename(annot = 2) %>% dplyr::mutate(cell = factor(cell, levels = cell_order)),
        transpose = TRUE,
        pal_annot = pal_annot,
        legend_title = annot_title
    ) + 
    scale_y_discrete(expand = c(0, 0)) +
    theme(plot.margin = margin(0, 0, 0, 0))
    # 6. Combine
    library(patchwork)
    final_plot = (p_annot | p_segs) + 
        plot_layout(widths = c(1, 15), guides = 'collect') & 
        theme(plot.margin = margin(5, 5, 5, 5))
    
    return(final_plot)
}

sample_folders <- list.dirs(path = numbat_root, full.names = TRUE, recursive = FALSE)
BULK_CNV_OPTION = "FACETS_HISENS"
i = 2
for (sample_path in sample_folders) {
    tryCatch({
        message(paste("--- Processing sample:", basename(sample_path), "---"))
        sample_id = sample_path
        print(paste0("Loading ", sample_path))
        print(paste0("Processing BULK_CNV_OPTION: ", BULK_CNV_OPTION))
        nb = Numbat$new(out_dir = sample_path)

        png_output = file.path(sample_path, paste0("mutation_history_", BULK_CNV_OPTION, "_", i, ".png"))
        png(png_output, width = 1200, height = 800, res = 150)
        print(nb$plot_mut_history(node_label=FALSE))
        dev.off()

        print(paste0("Saving the mutation history to: ", png_output))
        sample_name <- basename(sample_path)
        gex_root <- file.path(dirname(sample_path), "../GEX")
        if (use_metacells) {
            ct_path <- file.path(gex_root, sample_name, "metacell",
                                 paste0(sample_name, ".cell_types.tsv"))
            file_list <- if (file.exists(ct_path)) ct_path else character(0)
        } else {
            file_list <- list.files(
                path = gex_root,
                pattern = "\\.cell_types\\.tsv$",
                full.names = TRUE,
                recursive = FALSE
            )
        }
        df_anno <- do.call(rbind, lapply(file_list, function(f) {
            df <- read.table(f, sep="\t", header=TRUE, stringsAsFactors=FALSE, fill=TRUE)
            colnames(df) <- c("cell", "cell_type")
            if (use_metacells)
                df$cell <- paste0(sample_name, "_", df$cell)
            return(df)
        }))
        df_anno <- unique(df_anno)
        all_cells <- unique(df_anno$cell_type)
        sclc_cells <- all_cells[grep("SCLC", all_cells)]
        other_cells <- setdiff(all_cells, sclc_cells)
        df_anno$cell_type <- factor(df_anno$cell_type, levels = c(sclc_cells, other_cells))

        # Plot cell type annotation with clones
        png_output = file.path(sample_path, paste0("custom_phylo_heatmap_cell_type_", BULK_CNV_OPTION, "_", i, ".png"))
        print(paste0("Saving phylo heatmap to: ", png_output))
        png(png_output, 
            width = 1200, height = 800, res = 150)
        manual_colors = c(
            "SCLC-A" = "#d62728",      # tab:red
            "SCLC-N" = "#17becf",      # tab:blue
            # "KRT5/6A+ NonNE SCLC" = "#800080",  # purple
            # "AGR2+ NonNE SCLC" = "#9370db",  # mediumpurple
            "NonNE SCLC" = "#800080",  # tab:orange
            "NSCLC" = "#ffd700"        # gold
        )
        pal_annot_full <- setNames(rep("lightgray", length(all_cells)), all_cells)
        common_names <- intersect(names(manual_colors), names(pal_annot_full))
        pal_annot_full[common_names] <- manual_colors[common_names]
        print(nb$plot_phylo_heatmap(
            annot = df_anno, 
            pal_annot = pal_annot_full,
            annot_title = "cell type",
            clone_bar = TRUE,
            clone_stack = FALSE,
            clone_title = "Clone"
        ))
        dev.off()

        #----
        # plot only the cell type
        df_anno_tumor <- df_anno %>% 
            dplyr::filter(grepl("SCLC", cell_type))
        df_anno_tumor$cell_type <- factor(as.character(df_anno_tumor$cell_type), levels = c(sclc_cells))
        stratified_out = file.path(sample_path, paste0("stratified_heatmap_", BULK_CNV_OPTION, "_", i, ".png"))
        print(paste0("Saving stratified heatmap to: ", stratified_out))
        png(stratified_out, width = 1200, height = 800, res = 150)
        print(plot_annot_stratified_heatmap_v2(
            gtree = nb$gtree,
            joint_post = nb$joint_post,
            segs_consensus = nb$segs_consensus,
            annot = df_anno_tumor,
            pal_annot = pal_annot_full,
            clone_post = nb$clone_post
        ))
        dev.off()
    
    message(paste("Successfully processed:", basename(sample_path)))

    }, error = function(e) {
        message(paste("Error in sample", basename(sample_path), ":", e$message))
    })
}
