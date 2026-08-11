"""
Shared pipeline for the PHOX2B-overexpression GSEA analysis (H1975 / PC9
engineered cell lines), reproducing the wilcoxon-ranked GSEA (curated_small
gene sets, 500 permutations, per-cell-line FDR thresholds) behind the current
main_plot_cleaned_narrow.png in PHOX2B_DEG_MAST_GSEA.ipynb.

Used by Fig7_PHOX2B_GSEA_dotplot.py (figure), Table_S17_PHOX2B_differential_expression.py
and Table_S18_PHOX2B_differential_pathway.py (supplementary tables) so all three draw
from the same cached, deterministic wilcoxon DE + GSEA run.
"""
import os

import gseapy
import pandas as pd
import scanpy as sc

CELL_LINES = ["H1975", "PC9"]
ADATA_PATHS = {
    "H1975": "/data1/chanj3/PHOX2B.scRNAseq.2024/processed/nf-scrnaseq_split/rna_postprocessing/H1975_outlier_filtered_scran_scvi_postprocessing_scvi.h5ad",
    "PC9": "/data1/chanj3/PHOX2B.scRNAseq.2024/processed/nf-scrnaseq_split/rna_postprocessing/PC9_outlier_filtered_scran_scvi_postprocessing_scvi.h5ad",
}
MITO_RIBO_PATTERN = "^MT-|^MTMR|^MTND|^RPS|^RPL|^MRP|^FAU$|^UBA52$|^MALAT1$"

CURATED_SMALL_PATH = "/data1/chanj3/HTA.lung.NE_plasticity.120122/ref/curated.small.100124.gmt"
GSEA_PERMUTATION_NUM = 500

# Per-cell-line term-selection FDR threshold (get_balanced_top_terms) and
# dotplot display cutoff, matched to the notebook's current parameters.
CELL_LINE_QVAL_THRESH = {"H1975": 0.1, "PC9": 0.2}
CELL_LINE_DOTPLOT_CUTOFF = {"H1975": 0.01, "PC9": 0.2}

CACHE_DIR = "../../results/tables/figure7/cache"


def get_balanced_top_terms(df, n_per_side=10, q_val_thresh=0.2):
    """Top N terms per NES direction, ranked by FDR then |NES|."""
    df_sig = df[df["FDR q-val"] < q_val_thresh].copy()
    df_sig["abs_NES"] = df_sig["NES"].abs()

    pos_res = df_sig[df_sig["NES"] > 0].sort_values(
        ["FDR q-val", "abs_NES"], ascending=[True, False]
    ).head(n_per_side)
    neg_res = df_sig[df_sig["NES"] < 0].sort_values(
        ["FDR q-val", "abs_NES"], ascending=[True, False]
    ).head(n_per_side)

    return pd.concat([pos_res, neg_res])


def clean_term_name(term):
    """Strip database prefixes and title-case, preserving known acronyms."""
    prefixes = ["GO_", "HALLMARK_", "REACTOME_", "KEGG_", "BIOCARTA_", "WP_"]
    new_name = term
    for p in prefixes:
        if new_name.startswith(p):
            new_name = new_name.replace(p, "", 1)
            break

    acronyms = [
        "SCLC", "NSCLC", "BRCA", "CCLE", "DNA", "RNA", "E2F",
        "E2F3", "MRNA", "MYC", "EGFR", "TGFB1", "TLX", "YBX1",
    ]

    words = new_name.replace("_", " ").split()
    processed_words = []
    for word in words:
        if word.upper() in acronyms:
            processed_words.append(word.upper())
        else:
            processed_words.append(word.capitalize())

    return " ".join(processed_words)


def format_and_color_labels(ax, common_list):
    """Clean y-tick term labels and bold/red the ones shared across cell lines."""
    yticklabels = ax.get_yticklabels()
    new_labels = []

    for label in yticklabels:
        original_text = label.get_text()
        cleaned_text = clean_term_name(original_text)

        if original_text in common_list:
            label.set_color("red")
            label.set_weight("bold")

        label.set_text(cleaned_text)
        new_labels.append(cleaned_text)

    ax.set_yticklabels(new_labels)


def prepare_subset_for_plotting(df, selected_terms):
    """Subset to selected terms and floor FDR q-val == 0 for plotting."""
    plot_df = df[df["Term"].isin(selected_terms)].copy()

    min_fdr = df.loc[df["FDR q-val"] > 0, "FDR q-val"].min() if (df["FDR q-val"] > 0).any() else 1e-4
    plot_df["FDR q-val"] = plot_df["FDR q-val"].replace(0, min_fdr)

    return plot_df.sort_values("NES", ascending=False)


def get_wilcoxon_de(cell_line, use_cache=True):
    """
    Full wilcoxon rank_genes_groups (PHOX2B vs LV105) result for one cell line,
    on the same mito/ribo-filtered gene set used for the GSEA ranking. Caches
    to CACHE_DIR so this exact DE table (not the unfiltered MAST run) can be
    reused as the DEG table underlying the GSEA in Table S18 / Fig 7.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cell_line}_full_wilcoxon_de.csv")
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached wilcoxon DE for {cell_line}: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Running wilcoxon DE for {cell_line} ...")
    adata = sc.read_h5ad(ADATA_PATHS[cell_line])
    keep_genes = ~adata.var_names.str.contains(MITO_RIBO_PATTERN, case=False, regex=True)
    adata = adata[:, keep_genes].copy()
    sc.tl.rank_genes_groups(adata, groupby="transduction", method="wilcoxon")
    de = sc.get.rank_genes_groups_df(adata, group="PHOX2B")
    de.to_csv(cache_path, index=False)
    return de


def run_gsea_pipeline(use_cache=True, threads=None):
    """
    Wilcoxon rank_genes_groups (PHOX2B vs LV105) -> gseapy.prerank against the
    curated_small gene sets (500 permutations), per cell line.

    Returns {cell_line: full res2d DataFrame}. Caches per-cell-line results to
    CACHE_DIR so the figure script and the supplementary table script agree
    without recomputing (gseapy.prerank's default seed is fixed, so re-running
    reproduces identical numbers regardless).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    threads = threads or min(16, os.cpu_count() or 1)

    results = {}
    for cell_line in CELL_LINES:
        cache_path = os.path.join(CACHE_DIR, f"{cell_line}_full_gsea_res2d.csv")
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached GSEA results for {cell_line}: {cache_path}")
            results[cell_line] = pd.read_csv(cache_path)
            continue

        de = get_wilcoxon_de(cell_line, use_cache=use_cache)
        rnk = de[["names", "scores"]].dropna()

        print(f"Running GSEA prerank for {cell_line} ...")
        res = gseapy.prerank(
            rnk=rnk,
            gene_sets=CURATED_SMALL_PATH,
            permutation_num=GSEA_PERMUTATION_NUM,
            threads=threads,
            outdir=None,
        )
        results[cell_line] = res.res2d
        res.res2d.to_csv(cache_path, index=False)

    return results
