library(numbat)
library(Matrix)
library(dplyr)

# Take argument for bulk CNV option
args = commandArgs(trailingOnly=TRUE)
BULK_CNV_OPTION = args[1]
SAMPLE_ID = args[2]
PATIENT_ID = strsplit(SAMPLE_ID, "_")[[1]][1]
MIN_LLR      = if(!is.na(args[3])) as.numeric(args[3]) else 5
MAX_ENTROPY  = if(!is.na(args[4])) as.numeric(args[4]) else 0.5
USE_METACELLS = if(length(args) >= 5 && !is.na(args[5])) as.logical(args[5]) else FALSE
print(paste0("BULK_CNV_OPTION: ", BULK_CNV_OPTION))
print(paste0("PATIENT_ID: ", PATIENT_ID))
print(paste0("SAMPLE_ID:", SAMPLE_ID))
print(paste0("MIN_LLR: ", MIN_LLR))
print(paste0("MAX_ENTROPY: ", MAX_ENTROPY))
print(paste0("USE_METACELLS: ", USE_METACELLS))

load_count_mat = function(mtx_path, genes_path, barcodes_path) {
    count_mat = readMM(mtx_path)
    count_mat = as(count_mat, "CsparseMatrix")
    barcodes = read.table(barcodes_path, header = FALSE, stringsAsFactors = FALSE)$V1
    genes = read.table(genes_path, header = FALSE, stringsAsFactors = FALSE)$V1
    rownames(count_mat) = genes
    colnames(count_mat) = barcodes
    return(count_mat)
}

lookup_facet_file = function(facets_dir, manifest_path, sample_id, bulk_cnv_option) {
    manifest = read.csv(manifest_path, header=FALSE, col.names=c("sample_id", "facets_path"), stringsAsFactors=FALSE)
    row = manifest[manifest$sample_id == sample_id, ]
    if (nrow(row) == 0) {
        # Fall back to any sample from the same patient (e.g. RU222_T2 → RU222_T1)
        patient_prefix = strsplit(sample_id, "_")[[1]][1]
        row = manifest[startsWith(manifest$sample_id, patient_prefix), ]
        if (nrow(row) > 0) {
            message(paste0("  FACETS: no exact match for ", sample_id,
                           "; using ", row$sample_id[1], " as patient-level fallback"))
        }
    }
    if (nrow(row) == 0) {
        stop(paste0("No matching FACETS file found in manifest for sample ID: ", sample_id))
    }
    facets_file = basename(row$facets_path[1])
    facets_path = file.path(facets_dir, facets_file)
    if (bulk_cnv_option == "FACETS_PURITY") {
        facets_path = gsub("\\.hisens\\.csv$", ".purity.csv", facets_path)
    }
    return(facets_path)
}

IN_DIR="/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/"
OUT_DIR=file.path(IN_DIR, PATIENT_ID)
MANIFEST_PATH = "/data1/chanj3/HTA.lung.NE_plasticity.120122/MSK_IMPACT/facets.final_optimized_output.per_sample/manifest.sc-seq_to_facets.csv"
FACETS_DIR = "/data1/chanj3/HTA.lung.NE_plasticity.120122/MSK_IMPACT/facets.final_optimized_output.per_sample/"
outdir_name = paste0("numbat_output_", BULK_CNV_OPTION, if(USE_METACELLS) "_metacell" else "")

normal_count_mat = load_count_mat(paste0(IN_DIR, "Healthy_Lung/GEX/healthy_lung.expression.mtx"),
                                    paste0(IN_DIR, "Healthy_Lung/GEX/genes.tsv"),
                                    paste0(IN_DIR, "Healthy_Lung/GEX/barcodes.tsv"))
cell_annot = read.table(paste0(IN_DIR, "Healthy_Lung/GEX/healthy_lung.cell_annotations.tsv"), header=TRUE, sep="\t", stringsAsFactors=FALSE)
cell_annot = cell_annot %>%
    dplyr::select(cell = X, group=cell_type_final)
ref_internal = aggregate_counts(normal_count_mat, cell_annot)
rm(normal_count_mat)
gc()

if (grepl("FACETS", BULK_CNV_OPTION)) {
    facet_path = lookup_facet_file(FACETS_DIR, MANIFEST_PATH, SAMPLE_ID, BULK_CNV_OPTION)
    print(paste0("Using FACETS file: ", facet_path))
    if (grepl("PURITY", BULK_CNV_OPTION)) {
        facets = read.csv(facet_path)
    } else if (grepl("HISENS", BULK_CNV_OPTION)) {
        print("Using FACETS CNV data hisens file")
        facets = read.csv(facet_path)
    }
    ploidy_baseline = round(facets$tcn.em[1] - facets$seg.mean[1])
    lcn_baseline = ploidy_baseline / 2
    bulk_cnv = facets %>%
        filter(chrom %in% 1:22) %>%
        dplyr::mutate(
            cnv_state = dplyr::case_when(
                tcn.em > ploidy_baseline ~ "amp",              # Total exceeds baseline
                tcn.em < ploidy_baseline ~ "del",              # Total is below baseline
                lcn.em < lcn_baseline ~ "loh",                 # Lesser allele is below its baseline
                TRUE ~ "neu"
            ),
            CHROM = as.character(chrom),
            seg = paste0(CHROM, "_", seq_len(dplyr::n())),
            seg_start = loc.start,
            seg_end = loc.end
            ) %>%
        dplyr::select(CHROM, seg, seg_start, seg_end, cnv_state) %>%
        as.data.frame()
    print(head(bulk_cnv))
} else if (BULK_CNV_OPTION == "cbioportal") {
    print("Using cBioPortal CNV data")
    bulk_cnv = read.table("/data1/chanj3/HTA.lung.NE_plasticity.120122/MSK_IMPACT/mskimpact_segments.cnv.seg.032824.txt", header=TRUE, sep="\t", stringsAsFactors=FALSE) 
    margin = sd(bulk_cnv$seg.mean, na.rm=TRUE)
    png(file.path(OUT_DIR, outdir_name, "seg.mean.dist.png"))
    hist(bulk_cnv$seg.mean, breaks=50, main="Seg.Mean Distribution with 1 SD Margin")
    abline(v = c(-margin, margin), col="red", lty=2)
    dev.off()
    bulk_cnv = bulk_cnv %>%
        filter((chrom %in% 1:22) & (ID == "MSK_LX151")) %>%
        dplyr::mutate(
            CHROM = as.character(chrom),
            seg = paste0(CHROM, "_", seq_len(dplyr::n())),
            seg_start = loc.start,
            seg_end = loc.end,
            cnv_state = dplyr::case_when(
                seg.mean > margin ~ "amp",
                seg.mean < -margin ~ "del",
                TRUE ~ "neu"
            )
        ) %>%
        dplyr::filter(cnv_state != "neu") %>%
        dplyr::select(CHROM, seg, seg_start, seg_end, cnv_state) %>%
        as.data.frame()
    print(head(bulk_cnv))
} else {
    bulk_cnv = NULL
}

# Tumor (take into account that there are multiple samples per patients)
SAMPLES = list.dirs(paste0(IN_DIR, PATIENT_ID, "/GEX/"), recursive=FALSE, full.names=FALSE)
METHOD = "samplewise" # "patientwise"
filter_passthrough <- function(count_mat, df_allele, gex_dir, sample) {
    # Passthrough metacells: single cells that couldn't be aggregated.
    # Their IDs follow {ct_safe}__{0000} (no "SEACell-" substring).
    # Aggregated metacells contain "SEACell-" in their ID.
    asgn_path = file.path(gex_dir, paste0(sample, ".metacell_assignment.tsv"))
    if (!file.exists(asgn_path)) {
        print("  metacell_assignment.tsv not found; skipping passthrough filter")
        return(list(count_mat=count_mat, df_allele=df_allele))
    }
    asgn = read.table(asgn_path, header=TRUE, sep="\t", stringsAsFactors=FALSE)
    aggregated_ids = unique(asgn$SEACell[grepl("SEACell-", asgn$SEACell)])
    n_before = ncol(count_mat)
    count_mat = count_mat[, colnames(count_mat) %in% aggregated_ids, drop=FALSE]
    df_allele = df_allele[df_allele$cell %in% aggregated_ids, ]
    print(paste0("  Passthrough filter: ", n_before, " → ", ncol(count_mat),
                 " aggregated metacells (removed ", n_before - ncol(count_mat), " passthrough)"))
    return(list(count_mat=count_mat, df_allele=df_allele))
}

run_sample_numbat <- function(sample) {
    gex_dir    = if(USE_METACELLS) file.path(IN_DIR, PATIENT_ID, "GEX", sample, "metacell") else file.path(IN_DIR, PATIENT_ID, "GEX", sample)
    allele_dir = if(USE_METACELLS) file.path(IN_DIR, PATIENT_ID, "BAM", "metacell")          else file.path(IN_DIR, PATIENT_ID, "BAM")
    count_mat = load_count_mat(file.path(gex_dir, paste0(sample, ".expression.mtx")),
                                    file.path(gex_dir, "genes.tsv"),
                                    file.path(gex_dir, "barcodes.tsv"))
    df_allele = read.table(file.path(allele_dir, paste0(sample, "_allele_counts.tsv.gz")), header=TRUE, sep="\t", stringsAsFactors=FALSE)
    print(paste0("Saving output to: ", file.path(OUT_DIR, outdir_name, sample)))
    min_cells_val = if(USE_METACELLS) 3 else 10
    out = run_numbat(
        count_mat,
        ref_internal,
        df_allele,
        genome = "hg38",
        t = 1e-5,
        ncores = 4,
        plot = TRUE,
        segs_consensus_fix = bulk_cnv,
        min_cells = min_cells_val,
        min_LLR = MIN_LLR,
        max_entropy = MAX_ENTROPY,
        out_dir = file.path(OUT_DIR, outdir_name, sample),
        init_k = 10
    )
    out
}

if ((METHOD == "samplewise") | (length(SAMPLES) == 1)) {
    if (SAMPLE_ID == "RU1518_PDX") {
        next
    }
    print("Running numbat sample-wise on:")
    print(SAMPLE_ID)
    out = run_sample_numbat(SAMPLE_ID)
} else {
    print("Running numbat on combined samples")
    outdir_path = file.path(OUT_DIR, outdir_name, "combined")
    count_mats_list = list()
    df_alleles_list = list()
    for (SAMPLE in SAMPLES) {
        expr_path  = if(USE_METACELLS) file.path(IN_DIR, PATIENT_ID, "GEX", SAMPLE, "metacell") else file.path(IN_DIR, PATIENT_ID, "GEX", SAMPLE)
        allele_dir = if(USE_METACELLS) file.path(IN_DIR, PATIENT_ID, "BAM", "metacell")          else file.path(IN_DIR, PATIENT_ID, "BAM")
        curr_mat = load_count_mat(
            file.path(expr_path, paste0(SAMPLE, ".expression.mtx")),
            file.path(expr_path, "genes.tsv"),
            file.path(expr_path, "barcodes.tsv"))
        curr_df_allele = read.table(
                file.path(allele_dir, paste0(SAMPLE, "_allele_counts.tsv.gz")),
                header = TRUE, sep = "\t", stringsAsFactors = FALSE)
        count_mats_list[[SAMPLE]] = curr_mat
        df_alleles_list[[SAMPLE]] = curr_df_allele
    }
    count_mat_combined = do.call(cbind, count_mats_list)
    df_allele_combined = do.call(rbind, df_alleles_list)
    print(paste0("Saving output to: ", outdir_path))
    min_cells_val = if(USE_METACELLS) 3 else 10
    out = run_numbat(
        count_mat_combined,
        ref_internal,
        df_allele_combined,
        genome = "hg38",
        t = 1e-5,
        ncores = 8,
        plot = TRUE,
        segs_consensus_fix = bulk_cnv,
        min_cells = min_cells_val,
        min_LLR = MIN_LLR,
        max_entropy = MAX_ENTROPY,
        out_dir = outdir_path
    )
}
print("Numbat run completed.")
