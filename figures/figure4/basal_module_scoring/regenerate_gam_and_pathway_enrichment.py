#!/usr/bin/env python
"""
Regenerate the two deleted scratch cache files consumed by figure_4.py and
basal_ridge_plot.py:

    /scratch/chanj3/wangm10/adata_aligned_NSCLC-SCLC-A-SCLC-N.h5ad
    /scratch/chanj3/wangm10/gam_results_NSCLC-SCLC-A-SCLC-N.pkl

and additionally run the ordered-gene / Spectra pathway "running enrichment"
analysis (as in basal_ridge_plot.py) extended to include ROS-response and
ROS-detoxification gene sets, writing peak statistics to:

    results/tables/figure4/Table_S8_scDeBussy_trajectory_pathways_RAW.csv

This script reproduces, verbatim, the pipeline in figure_4.ipynb (the
canonical/current notebook -- NOT figure_4_030426.ipynb) and the downstream
gene-ordering / enrichment helpers duplicated in figure_4.py and
basal_ridge_plot.py. Run in the scdebussy-py311 conda env on the cluster
(see the matching .sbatch file in this directory).

Expected wall time: ~30-60 min for per-patient loading + HVG, plus ~20 min
for scDeBussy alignment, plus ~20-30 min for GAM fitting over ~5000 genes
(per the notebook's own comment: "about 20 minutes for three cell type
transitions").
"""

import gc
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pybiomart import Dataset
from scipy.sparse import diags
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from pygam import LinearGAM, f, s
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_on_synthetic import align_pseudotime_pipeline  # noqa: E402

# ============================================================================
# Config (verbatim from figure_4.ipynb / figure_4.py / basal_ridge_plot.py)
# ============================================================================

BASE_DIR = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/"
CELL_TYPE_COL = "cell_type_final2"
# NOTE: figure_4.ipynb's *live* uncommented line at the time of conversion was
# CELL_TYPES = ["^NSCLC$", "^SCLC"] (a regex-looking variant). That value is
# used only with pandas .isin()/== (no regex matching happens anywhere in the
# pipeline), so it would silently produce an empty selection and a
# cell_type_suffix that does not match what figure_4.py / basal_ridge_plot.py
# expect ("NSCLC-SCLC-A-SCLC-N"). figure_4.py and basal_ridge_plot.py (the
# scripts that CONSUME these caches) both hardcode the exact-match list below,
# so we use that -- it is also the form the notebook itself has commented out
# as the evident intended value.
CELL_TYPES = ["NSCLC", "SCLC-A", "SCLC-N"]
cell_type_suffix = "-".join(CELL_TYPES)  # "NSCLC-SCLC-A-SCLC-N"
PSEUDOTIME_COL = "palantir_pseudotime_slalom"
SAMPLES = ['RU1083', 'RU1293', 'RU151', 'RU1181', 'RU1215', 'RU1303', 'RU1304', 'RU1444', 'RU1518', 'RU1646',
           'RU1676', 'RU263', 'RU581', 'RU831', 'RU942']
PERCENTILE_CLIP = [1, 99]
BATCH_KEY = "patient"
N_HVG = 5000

SCRATCH_DIR = "/scratch/chanj3/wangm10"
ADATA_ALIGNED_PATH = os.path.join(SCRATCH_DIR, f"adata_aligned_{cell_type_suffix}.h5ad")
GAM_RESULTS_PATH = os.path.join(SCRATCH_DIR, f"gam_results_{cell_type_suffix}.pkl")

REPO_ROOT = "/data1/chanj3/wangm10/HTA.lung.NE_plasticity_scDeBussy"
OUT_TABLE_PATH = os.path.join(
    REPO_ROOT, "results/tables/figure4/Table_S8_scDeBussy_trajectory_pathways_RAW.csv"
)
GENE_SET_PATH = "/data1/chanj3/HTA.lung.NE_plasticity.120122/ref/Spectra.NE_NonNE.gene_sets.120524.p"

# Cell-cycle gene lists used by clean_gene_sets() in basal_ridge_plot.py
S_GENES_TIROSH = [
    'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6',
    'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP',
    'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2',
    'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2',
    'USP1', 'CLSPN', 'CHAF1B', 'BRIP1', 'E2F8'
]
G2M_GENES_TIROSH = [
    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2',
    'NUF2', 'CKS1B', 'PCLAF', 'TACC3', 'CENPA', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2',
    'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP',
    'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5',
    'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR',
    'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPF'
]

# Gene sets to include in the running-enrichment / peak-statistics analysis:
# the 5 originally plotted in basal_ridge_plot.py, plus IL6-JAK-STAT3 and
# mTORC1 (thematically related), plus the two ROS sets requested. All 9 keys
# were confirmed present in full_library['global'] of the Spectra pickle
# (152 gene sets total) prior to writing this script.
GENE_SETS_OF_INTEREST = [
    'all_TNF-via-NFkB_signaling',
    'all_JAK-STAT_signaling',
    'all_IL6-JAK-STAT3_signaling',
    'all_PI3K-AKT-mTOR_signaling',
    'all_mTORC1_signaling',
    'all_hypoxia-response',
    'all_ros_response',
    'all_ROS-detoxification',
    'all_DNA-repair',
]

ENRICHMENT_WINDOW_SIZE = 200
ENRICHMENT_STEP_SIZE = 40


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# STEP 1: Load per-patient adatas, filter to CELL_TYPES, build s_local, concat
# (verbatim from figure_4.ipynb section "1. Combine adata files into one object")
# ============================================================================

def load_and_concat_adata():
    adata_list = []
    for sample in SAMPLES:
        log(f"Loading sample: {sample}")
        adata_path = os.path.join(BASE_DIR, sample + ".no_cc.hvg_2000.090124.h5ad")
        adata = ad.read_h5ad(adata_path)
        cluster_missing = not all(np.isin(CELL_TYPES, adata.obs[CELL_TYPE_COL].unique()))
        if (not cluster_missing) and (PSEUDOTIME_COL in adata.obs.columns):
            log(f"  {sample}: OK (has all cell types + pseudotime column)")
            adata = adata[adata.obs[CELL_TYPE_COL].isin(CELL_TYPES)].copy()
            raw_pt = adata.obs[PSEUDOTIME_COL]
            vmin, vmax = np.percentile(raw_pt, PERCENTILE_CLIP)
            rescaled_pt = (np.clip(raw_pt, vmin, vmax) - vmin) / (vmax - vmin)
            adata.obs['s_local'] = rescaled_pt
            adata_list.append(adata)
        else:
            if cluster_missing:
                log(f"  {sample}: SKIPPED, present cell types = "
                    f"{adata.obs[CELL_TYPE_COL].astype(str).unique().tolist()}")
            if PSEUDOTIME_COL not in adata.obs.columns:
                log(f"  {sample}: SKIPPED, missing pseudotime column '{PSEUDOTIME_COL}'")

    log(f"Concatenating {len(adata_list)} filtered patient datasets...")
    adata = ad.concat(adata_list)
    log(f"Concatenation complete: {adata.shape[0]} cells x {adata.shape[1]} genes")
    del adata_list
    gc.collect()
    log("Patient cell counts:\n" + adata.obs['patient'].value_counts().to_string())
    return adata


# ============================================================================
# STEP 2: Gene filtering -- biomart protein-coding filter + HVG selection
# (verbatim from figure_4.ipynb section "2. Gene filtering")
# ============================================================================

def filter_genes(adata):
    log("Querying Ensembl biomart for protein-coding gene annotations...")
    dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
    gene_annotations = dataset.query(attributes=['external_gene_name', 'gene_biotype'])
    protein_coding = gene_annotations[gene_annotations['Gene type'] == 'protein_coding'].dropna()
    to_exclude = (
        protein_coding['Gene name'].str.upper().str.startswith(('MT-', 'RPS', 'RPL'))
        | protein_coding['Gene name'].str.upper().str.contains("ORF")
    )
    protein_coding = protein_coding[~to_exclude]
    to_keep = adata.var_names.isin(protein_coding['Gene name'].values)
    adata = adata[:, to_keep].copy()
    log(f"After protein-coding filter: {adata.n_vars} genes")

    log(f"Selecting top {N_HVG} HVGs (seurat_v3, batch_key='{BATCH_KEY}')... "
        "(this step can take several minutes)")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=N_HVG, layer="counts", flavor="seurat_v3",
        batch_key=BATCH_KEY, span=1
    )
    hvg_genes = pd.DataFrame(
        adata.var[['highly_variable_rank', 'highly_variable_nbatches']]
    )[adata.var['highly_variable']]
    hvg_genes = hvg_genes[hvg_genes['highly_variable_nbatches'] > hvg_genes['highly_variable_nbatches'].median()]
    hvg_genes.sort_values(['highly_variable_rank'], inplace=True)
    adata = adata[:, hvg_genes.index].copy()
    log(f"After HVG filtering, number of genes: {adata.n_vars}")
    return adata


# ============================================================================
# STEP 3: Cell-type-aware segmented scDeBussy alignment
# (verbatim from figure_4.ipynb, function align_pseudotime_pipeline_celltype_aware)
# ============================================================================

def align_pseudotime_pipeline_celltype_aware(
    adata,
    pseudotime_key="s_local",
    batch_key="patient",
    cell_type_key="cell_type",
    align_order=None,
    win_sz=0.3,
    num_pts=30,
    gamma=10.0,
    pca_components=10,
    use_windowed_matching=False,
    window_fraction=0.5,
    verbose=True,
    adaptive=False,
    segment_mode="sequential",  # "sequential" or "independent"
):
    """Cell type aware segmented alignment pipeline (copied verbatim from figure_4.ipynb)."""
    if verbose:
        print(f"\n{'='*80}")
        print("CELL TYPE AWARE SEGMENTED ALIGNMENT")
        print(f"{'='*80}\n")

    if align_order is None:
        align_order = list(adata.obs[cell_type_key].unique())
        if verbose:
            print(f"No align_order provided. Using cell types in order: {align_order}")
    else:
        available_types = set(adata.obs[cell_type_key].unique())
        missing_types = set(align_order) - available_types
        if missing_types:
            raise ValueError(f"Cell types in align_order not found in data: {missing_types}")
        extra_types = available_types - set(align_order)
        if extra_types and verbose:
            print(f"Warning: Cell types in data but not in align_order will be excluded: {extra_types}")

    if verbose:
        print(f"Alignment order: {' -> '.join(align_order)}")
        print(f"Number of segments: {len(align_order)}\n")

    aligned_pseudotime = np.full(len(adata), np.nan, dtype=float)
    segment_metadata = {
        "align_order": align_order,
        "segment_ranges": {},
        "segment_stats": {},
        "barycenters": {},
    }

    for segment_idx, cell_type in enumerate(align_order):
        if verbose:
            print(f"\n{'-'*80}")
            print(f"Segment {segment_idx + 1}/{len(align_order)}: {cell_type}")
            print(f"{'-'*80}")

        cell_mask = adata.obs[cell_type_key] == cell_type
        n_cells = cell_mask.sum()

        if n_cells == 0:
            if verbose:
                print(f"  No cells found for {cell_type}. Skipping.")
            continue

        if verbose:
            print(f"  Cells in segment: {n_cells}")
            n_batches = adata.obs.loc[cell_mask, batch_key].nunique()
            print(f"  Batches in segment: {n_batches}")

        adata_segment = adata[cell_mask].copy()

        try:
            adata_segment = align_pseudotime_pipeline(
                adata_segment,
                pseudotime_key=pseudotime_key,
                batch_key=batch_key,
                win_sz=win_sz,
                num_pts=num_pts,
                gamma=gamma,
                pca_components=min(pca_components, n_cells - 1),
                use_windowed_matching=use_windowed_matching,
                window_fraction=window_fraction,
                verbose=verbose,
                adaptive=adaptive,
            )

            segment_aligned = adata_segment.obs["aligned_pseudotime"].values

            valid_mask = ~np.isnan(segment_aligned)
            if valid_mask.sum() > 0:
                segment_min = np.nanmin(segment_aligned)
                segment_max = np.nanmax(segment_aligned)
                segment_range = segment_max - segment_min

                segment_metadata["segment_stats"][cell_type] = {
                    "n_cells": n_cells,
                    "n_batches": n_batches,
                    "original_min": segment_min,
                    "original_max": segment_max,
                    "original_range": segment_range,
                }

                if "barycenter" in adata_segment.uns:
                    segment_metadata["barycenters"][cell_type] = adata_segment.uns["barycenter"]

                if segment_mode == "sequential":
                    segment_aligned_norm = (segment_aligned - segment_min) / (segment_range + 1e-12)
                    segment_aligned_final = segment_idx + segment_aligned_norm
                    segment_metadata["segment_ranges"][cell_type] = (segment_idx, segment_idx + 1)
                    if verbose:
                        print(f"  Aligned pseudotime range: [{segment_idx:.2f}, {segment_idx + 1:.2f}]")
                else:
                    segment_aligned_final = segment_aligned
                    segment_metadata["segment_ranges"][cell_type] = (segment_min, segment_max)

                segment_indices_in_adata = adata.obs.index.get_indexer(adata_segment.obs.index)
                aligned_pseudotime[segment_indices_in_adata] = segment_aligned_final
            else:
                if verbose:
                    print(f"  Warning: All aligned pseudotime values are NaN for {cell_type}")

        except Exception as e:
            if verbose:
                print(f"  Error aligning segment {cell_type}: {e}")
            continue

    adata.obs["aligned_pseudotime"] = aligned_pseudotime
    adata.uns["celltype_alignment"] = segment_metadata

    if verbose:
        print(f"\n{'='*80}")
        print("SEGMENTED ALIGNMENT COMPLETE")
        print(f"{'='*80}")
        n_aligned = (~np.isnan(aligned_pseudotime)).sum()
        print(f"Successfully aligned: {n_aligned}/{len(adata)} cells")
        if segment_mode == "sequential":
            print(f"Final pseudotime range: [0, {len(align_order)}]")
            print("Each cell type occupies unit interval")
        print()

    return adata


def run_alignment(adata):
    log("Running align_pseudotime_pipeline_celltype_aware "
        f"(align_order={CELL_TYPES}, gamma=0.01, num_pts=100)...")
    adata_aligned = align_pseudotime_pipeline_celltype_aware(
        adata,
        pseudotime_key="s_local",
        batch_key=BATCH_KEY,
        cell_type_key=CELL_TYPE_COL,
        align_order=CELL_TYPES,
        gamma=0.01,
        num_pts=100,
        verbose=True,
    )

    # Convert tuples to lists before saving (h5ad doesn't support tuples)
    if 'celltype_alignment' in adata_aligned.uns and 'segment_ranges' in adata_aligned.uns['celltype_alignment']:
        for key in adata_aligned.uns['celltype_alignment']['segment_ranges']:
            ranges = adata_aligned.uns['celltype_alignment']['segment_ranges'][key]
            if isinstance(ranges, tuple):
                adata_aligned.uns['celltype_alignment']['segment_ranges'][key] = list(ranges)
            elif isinstance(ranges, list):
                adata_aligned.uns['celltype_alignment']['segment_ranges'][key] = [
                    list(item) if isinstance(item, tuple) else item for item in ranges
                ]

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    adata_aligned.write_h5ad(ADATA_ALIGNED_PATH)
    log(f"Saved adata_aligned to {ADATA_ALIGNED_PATH}")
    return adata_aligned


# ============================================================================
# STEP 4: GAM fitting -> summary_df, gene_curve, scores_df
# (verbatim from figure_4.ipynb section "GLM fitting on the aligned pseudotime")
# ============================================================================

def _remove_outliers(df, column, method="zscore", threshold=3):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]


def compute_mutual_information(df, aligned_col, aggregated_col, bins=10):
    aligned_discrete = pd.cut(df[aligned_col], bins=bins, labels=False)
    aggregated_discrete = pd.cut(df[aggregated_col], bins=bins, labels=False)
    mi = mutual_info_score(aligned_discrete, aggregated_discrete)
    return mi


def maximum_point(expression_data):
    return np.argmax(expression_data) / len(expression_data)


def _gam_smooth_expression_anndata(adata, gene, subject_col, pseudotime_col, n_splines=6, lam=3):
    gene_idx = adata.var_names.get_loc(gene)
    if hasattr(adata.X, 'toarray'):
        expression = adata.X[:, gene_idx].toarray().flatten()
    else:
        expression = np.asarray(adata.X[:, gene_idx]).flatten()

    df = pd.DataFrame({
        subject_col: adata.obs[subject_col].values,
        pseudotime_col: adata.obs[pseudotime_col].values,
        "expression": expression
    })

    df = _remove_outliers(df.copy(), pseudotime_col, method="zscore")
    df["expression"] = df.groupby(subject_col)["expression"].transform(
        lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
    )
    df.sort_values(by=pseudotime_col, inplace=True)

    x = df[[pseudotime_col, subject_col]].copy()
    x[subject_col] = x[subject_col].astype("category").cat.codes
    y = df["expression"].values

    gam = LinearGAM(s(0, n_splines=n_splines, lam=lam) + f(1)).fit(x, y)

    df["smoothed"] = gam.predict(x)
    df["deviance_residue"] = gam.deviance_residuals(x, y)
    subject_deviance_agg = df.groupby(subject_col)["deviance_residue"].mean().reset_index()

    x_grid = np.linspace(df[pseudotime_col].min(), df[pseudotime_col].max(), 1000)
    prediction_dfs = []
    for subject in x[subject_col].unique():
        grid_df = pd.DataFrame({pseudotime_col: x_grid})
        grid_df[subject_col] = subject
        grid_df["smoothed"] = gam.predict(grid_df)
        prediction_dfs.append(grid_df)
    predictions_combined = pd.concat(prediction_dfs)
    aggregated_curve = predictions_combined.groupby(pseudotime_col)["smoothed"].mean().reset_index()

    mi_score = compute_mutual_information(aggregated_curve, pseudotime_col, "smoothed")
    max_point = maximum_point(aggregated_curve["smoothed"])
    total_deviance_mean = subject_deviance_agg["deviance_residue"].mean()

    scores = {
        "gene": gene,
        "GCV": gam.statistics_["GCV"],
        "AIC": gam.statistics_["AIC"],
        "MI": mi_score,
        "Max": max_point * 100,
        "Deviance": total_deviance_mean,
    }

    return df, aggregated_curve, scores


def gam_smooth_expression_anndata(
    adata, genes, subject_col="subject", pseudotime_col="aligned_pseudotime", n_splines=6, lam=3
):
    gene_curve = {}
    scores_list = []
    summary_df_list = []

    for i, gene in enumerate(tqdm(genes, desc="Processing genes")):
        if gene not in adata.var_names:
            print(f"Warning: gene {gene} not found in adata.var_names, skipping")
            continue

        summary_df, aggregated_curve, scores = _gam_smooth_expression_anndata(
            adata, gene, subject_col=subject_col, pseudotime_col=pseudotime_col,
            n_splines=n_splines, lam=lam
        )
        if i == 0 or pseudotime_col not in gene_curve:
            gene_curve[pseudotime_col] = aggregated_curve[pseudotime_col]
        gene_curve[gene] = aggregated_curve.smoothed
        summary_df["gene"] = gene
        summary_df_list.append(summary_df)
        scores_list.append(scores)
        del summary_df, aggregated_curve
        gc.collect()

    summary_df = pd.concat(summary_df_list, ignore_index=True)
    scores_df = pd.DataFrame(scores_list)
    scores_df = scores_df.sort_values(
        by=["Deviance", "GCV", "AIC", "MI", "Max"], ascending=[True, True, True, False, True]
    )
    gene_curve = pd.DataFrame(gene_curve)
    return summary_df, gene_curve, scores_df


def run_gam_fitting(adata_aligned):
    log(f"Fitting GAMs for {adata_aligned.n_vars} genes "
        "(expect ~20-30 min for this 3-cell-type transition)...")
    summary_df, gene_curve, scores_df = gam_smooth_expression_anndata(
        adata_aligned,
        genes=adata_aligned.var_names.tolist(),
        pseudotime_col="aligned_pseudotime",
        subject_col=BATCH_KEY,
    )
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    with open(GAM_RESULTS_PATH, "wb") as fh:
        pickle.dump({
            "summary_df": summary_df,
            "gene_curve": gene_curve,
            "scores_df": scores_df,
        }, fh)
    log(f"Saved summary_df, gene_curve, scores_df to {GAM_RESULTS_PATH}")
    return summary_df, gene_curve, scores_df


# ============================================================================
# STEP 5: Order/cluster genes by peak pseudotime position
# (verbatim from figure_4.py / basal_ridge_plot.py, process_gene_data & helpers)
# ============================================================================

def order_categories_by_average_position(categories):
    position_data = defaultdict(lambda: {"sum": 0, "count": 0})
    for index, category in enumerate(categories):
        position_data[category]["sum"] += index
        position_data[category]["count"] += 1
    averages = {category: data["sum"] / data["count"] for category, data in position_data.items()}
    sorted_categories = sorted(averages, key=averages.get)
    return sorted_categories


def bin_and_average_time_series(time_series, window_size, stride):
    binned_averages = []
    length = len(time_series)
    for start in range(0, length - window_size + 1, stride):
        b = time_series[start:start + window_size]
        binned_averages.append(np.mean(b))
    return np.array(binned_averages)


def argmax_with_largest_index_tie_breaking(array):
    """Returns the largest index of the maximum value in the array."""
    max_value = np.max(array)
    max_indices = np.flatnonzero(array == max_value)
    largest_index = max_indices[-1]
    return largest_index


def mean_around_max(time_series, num_elements):
    """Computes the mean of `num_elements` surrounding the maximum value in a
    time series (copied verbatim from figure_4.py / basal_ridge_plot.py)."""
    argmax_position = argmax_with_largest_index_tie_breaking(time_series)
    half_window = num_elements // 2

    if argmax_position < half_window:
        start_index = 0
        end_index = num_elements
    elif argmax_position > len(time_series) - half_window:
        start_index = len(time_series) - num_elements
        end_index = len(time_series)
    else:
        start_index = argmax_position - half_window
        end_index = argmax_position + half_window

    surrounding_elements = time_series[start_index:end_index]
    mean = np.mean(surrounding_elements)
    return argmax_position, mean


def find_max(time_series, weight):
    max_position = len(time_series)
    argmax_position, mean = mean_around_max(time_series, max_position // 2)
    composite_score = argmax_position * (1 - weight) + mean * weight * np.sign(0.5 - argmax_position / max_position)
    return composite_score


def order_genes(df, weight, window_size=10, stride=10):
    max_indices = []
    for row_idx, expression in df.iterrows():
        binned_averages = bin_and_average_time_series(expression.values, window_size, stride)
        composite_score = find_max(binned_averages, weight=weight)
        max_indices.append((row_idx, composite_score))
    sorted_indices = sorted(max_indices, key=lambda x: x[1], reverse=False)
    sorted_row_numbers = [df.index.get_loc(row_idx) for row_idx, _ in sorted_indices]
    return sorted_row_numbers


def filter_and_normalize_genes(scores_df, gene_curve, MI_threshold=0, GCV_threshold=1, AIC_threshold=np.inf):
    is_filtered_genes = scores_df.gene[
        (scores_df.MI > MI_threshold) & (scores_df.GCV < GCV_threshold) & (scores_df.AIC < AIC_threshold)
    ]
    aligned_score = gene_curve.iloc[:, 0]
    gene_curve_filtered = gene_curve.loc[:, is_filtered_genes]
    zscore_scaler = StandardScaler()
    normalized_gene_curve = pd.DataFrame(
        zscore_scaler.fit_transform(gene_curve_filtered),
        columns=gene_curve_filtered.columns,
        index=gene_curve_filtered.index
    )
    return normalized_gene_curve, aligned_score


def order_genes_by_peak(normalized_gene_curve, weight=0.5):
    combined_order = order_genes(normalized_gene_curve.T, weight=weight)
    sorted_gene_curve = normalized_gene_curve.T.iloc[combined_order, :]
    return sorted_gene_curve, np.array(combined_order)


def cluster_sorted_genes(sorted_gene_curve, n_clusters=3, hierarchical=True, n_init=5,
                          band_width=1, enforce_contiguity=True):
    gene_data = sorted_gene_curve.values
    n_genes = gene_data.shape[0]

    if hierarchical:
        if enforce_contiguity:
            connectivity = diags(
                [1] * 3, offsets=[-1, 0, 1], shape=(n_genes, n_genes)
            ).toarray()
        else:
            connectivity = diags(
                [1] * (2 * band_width + 1),
                offsets=range(-band_width, band_width + 1),
                shape=(n_genes, n_genes)
            ).toarray()
        clusters = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", connectivity=connectivity
        ).fit_predict(gene_data)
    else:
        from tslearn.clustering import KShape
        if enforce_contiguity:
            print("Warning: enforce_contiguity requires hierarchical=True, ignoring...")
        clusters = KShape(
            n_clusters=n_clusters, random_state=0, verbose=True, n_init=n_init
        ).fit(gene_data).labels_

    return clusters


def assign_temporal_categories(clusters, label_names=None):
    label_orders = order_categories_by_average_position(clusters)
    labels = pd.Series(clusters).map({old_label: new_label for new_label, old_label in enumerate(label_orders)})
    if label_names is None:
        category = labels.tolist()
    else:
        category = [label_names[x] for x in labels]
    row_colors = pd.Series(category)
    return row_colors, category


def categorize_values(X, Y):
    result = []
    for x in X:
        category = 0
        for i, cutoff in enumerate(Y):
            if x < cutoff:
                break
            category = i + 1
        result.append(category)
    return pd.Series(result)


def process_gene_data(
    scores_df,
    gene_curve,
    cell_colors,
    cutoff_points,
    MI_threshold=0,
    GCV_threshold=1,
    AIC_threshold=np.inf,
    n_clusters=3,
    label_names=None,
    n_init=5,
    hierarchical=True,
    weight=0.5,
    enforce_contiguity=True,
    band_width=4
):
    normalized_gene_curve, aligned_score = filter_and_normalize_genes(
        scores_df, gene_curve, MI_threshold, GCV_threshold, AIC_threshold
    )
    sorted_gene_curve, gene_order = order_genes_by_peak(normalized_gene_curve, weight=weight)
    clusters = cluster_sorted_genes(
        sorted_gene_curve,
        n_clusters=n_clusters,
        hierarchical=hierarchical,
        n_init=n_init,
        enforce_contiguity=enforce_contiguity,
        band_width=band_width
    )
    row_colors, category = assign_temporal_categories(clusters, label_names)
    col_cols = categorize_values(aligned_score, cutoff_points)
    return sorted_gene_curve, row_colors, col_cols, category, gene_order


def run_gene_ordering(scores_df, gene_curve):
    log("Ordering genes by peak pseudotime position (process_gene_data)...")
    # Parameters copied verbatim from figure_4.py's call site.
    color_map = {'NSCLC': 'gold', 'SCLC-A': 'tab:red', 'SCLC-N': 'tab:cyan'}
    clusters_of_interest = ['NSCLC', 'SCLC-A']  # , 'SCLC-N'  (commented out upstream too)
    colors = [color_map[x] for x in clusters_of_interest]
    sorted_gene_curve, row_colors, col_colors, categories, gene_order = process_gene_data(
        scores_df, gene_curve, colors, [1, 2],
        n_clusters=4,
        label_names=['Early 1', 'Early 2', 'Middle', 'Late'],
        hierarchical=True,
        enforce_contiguity=False,
        band_width=10
    )
    log(f"Ordered {len(sorted_gene_curve)} genes into {len(set(categories))} temporal clusters.")
    return sorted_gene_curve


# ============================================================================
# STEP 6: Spectra pathway "running enrichment" analysis
# (verbatim from basal_ridge_plot.py, clean_gene_sets & compute_running_enrichment)
# ============================================================================

def clean_gene_sets(gene_set_dict, extra_genes_to_remove=None):
    """Removes proliferation and cell-cycle genes from the gene sets."""
    cell_cycle_genes = set(S_GENES_TIROSH + G2M_GENES_TIROSH)
    if extra_genes_to_remove:
        cell_cycle_genes.update(extra_genes_to_remove)

    cleaned_dict = {}
    for set_name, genes in gene_set_dict.items():
        cleaned_genes = [g for g in genes if g.upper() not in cell_cycle_genes]
        if len(cleaned_genes) >= 5:
            cleaned_dict[set_name] = cleaned_genes
    return cleaned_dict


def compute_running_enrichment(ordered_genes, gene_set_dict, window_size=100, step_size=20):
    """Compute running enrichment scores for gene sets along ordered genes
    using a hypergeometric test (copied verbatim from basal_ridge_plot.py)."""
    from scipy.stats import hypergeom

    ordered_genes = list(ordered_genes)
    total_genes = len(ordered_genes)

    results = []

    for gene_set_name, gene_set in gene_set_dict.items():
        gene_set_present = [g for g in gene_set if g in ordered_genes]
        n_gene_set = len(gene_set_present)

        if n_gene_set < 5:
            continue

        for start in range(0, total_genes - window_size + 1, step_size):
            end = start + window_size
            window_genes = ordered_genes[start:end]

            overlap = len(set(window_genes) & set(gene_set_present))

            pval = hypergeom.sf(overlap - 1, total_genes, n_gene_set, window_size)

            expected = (window_size * n_gene_set) / total_genes
            enrichment_ratio = overlap / expected if expected > 0 else 0

            if pval < 1e-10:
                score = 10 * np.sign(overlap - expected)
            else:
                score = -np.log10(pval + 1e-10) * np.sign(overlap - expected)

            results.append({
                'gene_set': gene_set_name,
                'position': (start + end) / 2,
                'start': start,
                'end': end,
                'overlap': overlap,
                'expected': expected,
                'enrichment_ratio': enrichment_ratio,
                'pval': pval,
                'score': score,
                'window_genes': window_genes,
                'gene_set_present': gene_set_present,
            })

    return pd.DataFrame(results)


def run_pathway_enrichment(sorted_gene_curve):
    log(f"Loading Spectra gene set pickle from {GENE_SET_PATH} ...")
    with open(GENE_SET_PATH, "rb") as fh:
        full_library = pickle.load(fh)
    gene_set_dict = full_library['global']
    log(f"Loaded {len(gene_set_dict)} gene sets from full_library['global'].")

    gene_set_dict = clean_gene_sets(gene_set_dict)
    log(f"{len(gene_set_dict)} gene sets remain after removing cell-cycle genes "
        "(clean_gene_sets, min 5 genes).")

    missing = [gs for gs in GENE_SETS_OF_INTEREST if gs not in gene_set_dict]
    if missing:
        log(f"WARNING: requested gene sets not found after cleaning: {missing}")

    ordered_genes = list(sorted_gene_curve.index)
    log(f"Running compute_running_enrichment over {len(ordered_genes)} ordered genes "
        f"(window_size={ENRICHMENT_WINDOW_SIZE}, step_size={ENRICHMENT_STEP_SIZE})...")
    enrichment_df = compute_running_enrichment(
        ordered_genes,
        gene_set_dict,
        window_size=ENRICHMENT_WINDOW_SIZE,
        step_size=ENRICHMENT_STEP_SIZE,
    )
    log(f"Running enrichment computed for {enrichment_df['gene_set'].nunique()} gene sets "
        f"({len(enrichment_df)} windows total).")

    # Restrict to gene sets of interest that survived cleaning + had >=5 genes present
    available_sets = [gs for gs in GENE_SETS_OF_INTEREST if gs in enrichment_df['gene_set'].unique()]
    log(f"Extracting peak statistics for {len(available_sets)} gene sets of interest: {available_sets}")

    total_genes = len(ordered_genes)
    records = []
    for gs_name in available_sets:
        sub = enrichment_df[enrichment_df['gene_set'] == gs_name]
        peak_row = sub.loc[sub['score'].idxmax()]
        overlap_genes = sorted(set(peak_row['window_genes']) & set(peak_row['gene_set_present']))
        records.append({
            'gene_set': gs_name,
            'n_genes_in_set_present': len(peak_row['gene_set_present']),
            'peak_position_pct': 100.0 * peak_row['position'] / total_genes,
            'peak_position_index': peak_row['position'],
            'peak_window_start': peak_row['start'],
            'peak_window_end': peak_row['end'],
            'peak_score': peak_row['score'],
            'peak_pval': peak_row['pval'],
            'peak_enrichment_ratio': peak_row['enrichment_ratio'],
            'peak_overlap': peak_row['overlap'],
            'window_size': ENRICHMENT_WINDOW_SIZE,
            'step_size': ENRICHMENT_STEP_SIZE,
            'total_ordered_genes': total_genes,
            'peak_overlap_genes': ";".join(overlap_genes),
        })

    peak_stats = pd.DataFrame(records).sort_values('peak_score', ascending=False)
    return peak_stats


# ============================================================================
# Main
# ============================================================================

def main():
    log("=" * 80)
    log("STEP 1/6: Load per-patient adatas and concatenate")
    log("=" * 80)
    adata = load_and_concat_adata()

    log("=" * 80)
    log("STEP 2/6: Gene filtering (protein-coding + HVG)")
    log("=" * 80)
    adata = filter_genes(adata)

    log("=" * 80)
    log("STEP 3/6: scDeBussy cell-type-aware segmented alignment")
    log("=" * 80)
    adata_aligned = run_alignment(adata)
    del adata
    gc.collect()

    log("=" * 80)
    log("STEP 4/6: GAM fitting -> summary_df, gene_curve, scores_df")
    log("=" * 80)
    summary_df, gene_curve, scores_df = run_gam_fitting(adata_aligned)
    del adata_aligned
    gc.collect()

    log("=" * 80)
    log("STEP 5/6: Order genes by peak pseudotime position")
    log("=" * 80)
    sorted_gene_curve = run_gene_ordering(scores_df, gene_curve)

    log("=" * 80)
    log("STEP 6/6: Spectra pathway running-enrichment analysis")
    log("=" * 80)
    peak_stats = run_pathway_enrichment(sorted_gene_curve)

    os.makedirs(os.path.dirname(OUT_TABLE_PATH), exist_ok=True)
    peak_stats.to_csv(OUT_TABLE_PATH, index=False)
    log(f"Wrote pathway peak-statistics table to {OUT_TABLE_PATH}")

    log("ALL DONE.")


if __name__ == "__main__":
    main()
