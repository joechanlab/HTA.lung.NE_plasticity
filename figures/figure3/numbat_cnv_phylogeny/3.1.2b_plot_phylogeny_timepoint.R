library(numbat)
library(dplyr)
library(ggplot2)
library(ape)
library(ggtree)
library(phangorn)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript 3.1.2b_plot_phylogeny_timepoint.R <PATIENT_ID> [BASE_ROOT] [USE_METACELLS]")
}

patient_id    <- args[1]
base_root     <- ifelse(length(args) >= 2, args[2],
                        "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All")
use_metacells <- if (length(args) >= 3) as.logical(args[3]) else TRUE

outdir_name <- paste0("numbat_output_FACETS_HISENS", if (use_metacells) "_metacell" else "")
numbat_out_dir <- file.path(base_root, patient_id, outdir_name)
gex_root       <- file.path(base_root, patient_id, "GEX")

message("Timepoint phylogeny  patient: ", patient_id)

# ── Tumor-type palette (for normal-cell classification only) ─────────────────
TUMOR_TYPES <- c("LUAD","LUSC","SCLC-A","SCLC-N","SCLC-P",
                 "NonNE SCLC","EMT","FOXI1+ NonNE SCLC-N")

# ── collapse_normal_clades (same as 3.1.2_plot_phylogeny.R) ─────────────────
collapse_normal_clades <- function(treeML, cell_types_df, tumor_types) {
    type_map <- setNames(cell_types_df$cell_type, cell_types_df$cell)
    n_tips   <- length(treeML$tip.label)
    is_tumor <- sapply(treeML$tip.label, function(lbl) {
        ct <- type_map[lbl]; !is.na(ct) && ct %in% tumor_types
    })
    normal_set <- treeML$tip.label[!is_tumor]

    empty <- list(tree  = treeML,
                  pseudo = data.frame(label=character(), n=integer(),
                                      stringsAsFactors=FALSE))
    if (length(normal_set) == 0) return(empty)

    parent_of <- setNames(treeML$edge[,1], as.character(treeML$edge[,2]))

    all_normal_internal <- integer(0)
    for (node in (n_tips+1):(n_tips+treeML$Nnode)) {
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
    pseudo_df <- data.frame(old=pseudo_old, n=pseudo_n, stringsAsFactors=FALSE)
    pseudo_df$label <- paste0(".Normal_n", pseudo_df$n, "_", seq_len(nrow(pseudo_df)))
    for (i in seq_len(nrow(pseudo_df))) {
        idx <- which(tree_new$tip.label == pseudo_df$old[i])
        if (length(idx) > 0) tree_new$tip.label[idx] <- pseudo_df$label[i]
    }
    list(tree=tree_new, pseudo=pseudo_df[, c("label","n")])
}

# ── Extract sample_id from a metacell tip label ───────────────────────────────
# Tip labels have the form  {sample_id}_SEACell-{n}
extract_sample_id <- function(lbl) sub("_SEACell-[0-9]+$", "", lbl)

# ── Load timepoint mapping ────────────────────────────────────────────────────
script_dir <- tryCatch({
    argv <- commandArgs(FALSE)
    f    <- argv[grep("^--file=", argv)]
    dirname(normalizePath(sub("^--file=", "", f)))
}, error = function(e) getwd())
tp_file <- file.path(script_dir, "sample_timepoints.tsv")
if (!file.exists(tp_file)) stop("sample_timepoints.tsv not found at: ", tp_file)

tp_all <- read.table(tp_file, header=TRUE, sep="\t", stringsAsFactors=FALSE)
tp_pat <- tp_all[tp_all$patient_id == patient_id, ]
if (nrow(tp_pat) == 0) stop("No timepoint entries for patient: ", patient_id)

n_distinct_tp <- max(tp_pat$date_rank)
if (n_distinct_tp < 2) {
    message("Patient ", patient_id, " has only 1 distinct timepoint — skipping.")
    quit(save="no", status=0)
}
message("  Distinct timepoints: ", n_distinct_tp,
        "  (", paste(tp_pat$sample_id, collapse=", "), ")")

# ── Fixed discrete palette shared across all patients (rank 1 = earliest) ────
# Pink → magenta → dark purple-pink (RdPu family); avoids cell-type colors
# (blue/cyan = SCLC-P/N, red/orange/gold = LUAD/LUSC/SCLC-A, green = FOXI1+,
#  indigo/mauve = NonNE SCLC/EMT).
FIXED_TP_COLORS <- c(
    "1" = "#FDE0EF",   # pale pink      (earliest)
    "2" = "#FA9FB5",   # rose pink
    "3" = "#C51B8A",   # deep magenta
    "4" = "#7A0177"    # dark purple-pink (latest)
)
# Always anchor: rank 1 = pale pink, last rank = dark purple-pink; interpolate middle indices
max_tp   <- length(FIXED_TP_COLORS)   # 4
if (n_distinct_tp == 1) {
    idx <- 1L
} else {
    idx <- round(seq(1, max_tp, length.out = n_distinct_tp))
}
tp_palette <- FIXED_TP_COLORS[idx]
names(tp_palette) <- as.character(seq_len(n_distinct_tp))

# ── Load tree ─────────────────────────────────────────────────────────────────
tree_rds <- file.path(numbat_out_dir, paste0(patient_id, "_treeML.rds"))
if (!file.exists(tree_rds)) stop("treeML RDS not found: ", tree_rds)
treeML <- readRDS(tree_rds)

# ── Load cell types for normal-cell classification ────────────────────────────
if (use_metacells) {
    sample_names <- list.dirs(gex_root, full.names=FALSE, recursive=FALSE)
    cell_types <- dplyr::bind_rows(lapply(sample_names, function(s) {
        ct_path <- file.path(gex_root, s, "metacell", paste0(s, ".cell_types.tsv"))
        if (!file.exists(ct_path)) return(NULL)
        df <- read.table(ct_path, header=TRUE, sep="\t")
        colnames(df) <- c("cell","cell_type")
        df$cell <- paste0(s, "_", df$cell)
        df
    }))
} else {
    stop("Non-metacell mode not supported for timepoint plot")
}
cell_types <- cell_types[cell_types$cell %in% treeML$tip.label, ]

# ── Collapse normal clades then build tumor-only tree ─────────────────────────
coll      <- collapse_normal_clades(treeML, cell_types, TUMOR_TYPES)
pseudo_df <- coll$pseudo
message("  Collapsed ", nrow(pseudo_df), " normal clade(s), ",
        sum(pseudo_df$n), " normal cells total")

tumor_tips <- setdiff(coll$tree$tip.label, pseudo_df$label)
if (length(pseudo_df$label) > 0 && length(tumor_tips) >= 2) {
    tumor_tree <- ape::drop.tip(coll$tree, pseudo_df$label)
} else {
    tumor_tree <- coll$tree
}

# ── Add "Normal" as an outgroup leaf at the root ──────────────────────────────
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
    if (!is.null(plot_tree$edge.length))
        plot_tree$edge.length[is.na(plot_tree$edge.length)] <- 0
} else {
    plot_tree <- tumor_tree
    n_total   <- 0L
}

# ── Build groupOTU groups by timepoint rank; Normal gets its own group ─────────
grp_tp <- list()
for (rank_i in seq_len(n_distinct_tp)) {
    samples_at_rank <- tp_pat$sample_id[tp_pat$date_rank == rank_i]
    tips_at_rank <- plot_tree$tip.label[
        sapply(plot_tree$tip.label, function(lbl) {
            extract_sample_id(lbl) %in% samples_at_rank
        })
    ]
    if (length(tips_at_rank) > 0)
        grp_tp[[as.character(rank_i)]] <- tips_at_rank
}

tp_palette["0"]      <- "#AAAAAA"
tp_palette["Normal"] <- "#888888"

# Normal leaf → its own group
if ("Normal" %in% plot_tree$tip.label) {
    grp_tp[["Normal"]] <- "Normal"
}

# Tips not matched to any timepoint (and not Normal) → gray
all_assigned <- unique(unlist(grp_tp))
unassigned   <- setdiff(plot_tree$tip.label, all_assigned)
if (length(unassigned) > 0) {
    message("  Unassigned tips (will be gray): ", length(unassigned))
    grp_tp[["unassigned"]] <- unassigned
    tp_palette["unassigned"] <- "#AAAAAA"
}

# ── Plot tree ─────────────────────────────────────────────────────────────────
options(ignore.negative.edge = TRUE)

tp_labels_short <- tapply(
    sub(paste0("^", patient_id, "_"), "", tp_pat$sample_id),
    tp_pat$date_rank, paste, collapse="/"
)
entries  <- paste(names(tp_labels_short), tp_labels_short, sep="=")
lines    <- sapply(seq(1, length(entries), by = 2), function(i)
                   paste(entries[i:min(i+1, length(entries))], collapse=" | "))
subtitle <- paste(lines, collapse="\n")

p_tree <- ggtree(plot_tree, layout="rectangular") %>%
    groupOTU(., grp_tp, "timepoint") +
    aes(color = timepoint) +
    scale_color_manual(values=tp_palette, na.value="#AAAAAA") +
    ggtitle(patient_id, subtitle=subtitle) +
    theme(
        legend.position    = "none",
        plot.title         = element_text(hjust=0.5, size=9),
        plot.subtitle      = element_text(hjust=0.5, size=5, color="grey40")
    )

# ── Sized circle at the Normal leaf ──────────────────────────────────────────
if (n_total > 0 && "Normal" %in% plot_tree$tip.label) {
    circle_size <- pmax(1.0, pmin(5.0, 1.0 + log10(n_total + 1)))
    norm_pt <- p_tree$data %>% dplyr::filter(label == "Normal")
    if (nrow(norm_pt) > 0) {
        p_tree <- p_tree +
            geom_point(
                data        = norm_pt,
                mapping     = aes(x = x, y = y),
                size        = circle_size,
                color       = "#888888",
                alpha       = 0.65,
                inherit.aes = FALSE,
                show.legend = FALSE
            )
    }
}

out_png <- file.path(numbat_out_dir, paste0(patient_id, "_phylogeny_timepoint.png"))
ggsave(p_tree, file=out_png, width=2, height=3, units="in", dpi=150)
message("Saved: ", out_png)
