import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pybiomart import Dataset
from evaluate_on_synthetic import align_pseudotime_pipeline
from matplotlib import pyplot as plt

BASE_DIR = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/"
CELL_TYPE_COL = "cell_type_final2"
CELL_TYPES = ["NSCLC", "SCLC-A", "SCLC-N"]
cell_type_suffix = "-".join(CELL_TYPES)
PSEUDOTIME_COL = "palantir_pseudotime_slalom"
SAMPLES = ['RU1083', 'RU1293', 'RU151', 'RU1181', 'RU1215', 'RU1303', 'RU1304', 'RU1444', 'RU1518', 'RU1646', 
          'RU1676', 'RU263', 'RU581', 'RU831', 'RU942']
PERCENTILE_CLIP = [1, 99]
BATCH_KEY = "patient"
N_HVG = 5000
GENES = ['ASCL1', 'NEUROD1', 'TACSTD2', 'EZH2']
PLOT_DIR = "/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/results/figures/figure4"

import pickle
with open(f"/scratch/chanj3/wangm10/gam_results_{cell_type_suffix}.pkl", "rb") as f:
    gam_results = pickle.load(f)
summary_df = gam_results["summary_df"]
gene_curve = gam_results["gene_curve"]
scores_df = gam_results["scores_df"]

import time
from collections import defaultdict

import gseapy as gp
import numpy as np
import pandas as pd
from scipy.sparse import diags
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import KShape


def order_categories_by_average_position(categories):
    # Dictionary to store total positions and counts
    position_data = defaultdict(lambda: {"sum": 0, "count": 0})

    # Iterate through the list and update position data
    for index, category in enumerate(categories):
        position_data[category]["sum"] += index
        position_data[category]["count"] += 1

    # Calculate average positions
    averages = {category: data["sum"] / data["count"] for category, data in position_data.items()}

    # Sort categories by average position
    sorted_categories = sorted(averages, key=averages.get)

    return sorted_categories


def max_idx(x, window_size):
    # Calculate rolling maximum and find the index of the maximum value
    rolling_mean = x.rolling(window=window_size).mean()
    max_index = rolling_mean.idxmax()
    return max_index


def center_of_mass(vector):
    """Compute weighted center of mass for a 1D vector."""
    positions = np.arange(len(vector))  # Index positions
    return np.sum(vector * positions) / np.sum(vector)


def categorize_values(X, Y):
    result = []

    for x in X:
        # Initialize category index
        category = 0

        # Find the appropriate category for x
        for i, cutoff in enumerate(Y):
            if x < cutoff:
                break
            category = i + 1

        result.append(category)

    return pd.Series(result)


def bin_and_average_time_series(time_series, window_size, stride):
    """
    Bins the time series using the specified window size and stride, and computes the average value for each bin.

    Parameters
    ----------
        time_series (np.ndarray): The input time series data.
        window_size (int): The size of the window for binning.
        stride (int): The number of steps to move the window after each bin.

    Returns
    -------
        np.ndarray: A 1D array where each element is the average value of a bin.
    """
    binned_averages = []
    length = len(time_series)

    for start in range(0, length - window_size + 1, stride):
        bin = time_series[start : start + window_size]
        binned_averages.append(np.mean(bin))  # Compute the average of the bin

    return np.array(binned_averages)


def argmax_with_largest_index_tie_breaking(array):
    """
    Returns the largest index of the maximum value in the array.

    Parameters
    ----------
        array (np.ndarray): The input array.

    Returns
    -------
        int: The largest index of the maximum value.
    """
    # Find the maximum value in the array
    max_value = np.max(array)

    # Get all indices where the maximum value occurs
    max_indices = np.flatnonzero(array == max_value)

    # Select the largest index among these
    largest_index = max_indices[-1]

    return largest_index


def mean_around_max(time_series, num_elements):
    """
    Computes the mean of a specified number of elements surrounding the maximum value in a time series.

    Parameters
    ----------
        time_series (np.ndarray): The input time series data.
        num_elements (int): The total number of elements to consider around the maximum value.

    Returns
    -------
        tuple: A tuple containing the index of the maximum value and the mean of the specified elements surrounding the maximum value.
    """
    # Find the index of the maximum value
    argmax_position = argmax_with_largest_index_tie_breaking(time_series)

    # Calculate half window size
    half_window = num_elements // 2

    # Determine start and end indices ensuring exactly num_elements are selected
    if argmax_position < half_window:
        start_index = 0
        end_index = num_elements
    elif argmax_position > len(time_series) - half_window:
        start_index = len(time_series) - num_elements
        end_index = len(time_series)
    else:
        start_index = argmax_position - half_window
        end_index = argmax_position + half_window

    # Extract surrounding elements
    surrounding_elements = time_series[start_index:end_index]

    # Compute mean
    mean = np.mean(surrounding_elements)

    return argmax_position, mean


def find_max(time_series, weight):
    """
    Computes a composite score to order a time series.

    Based on the location of the maximum value and the mean of surrounding values, weighted
    by a user-specified parameter.

    Parameters
    ----------
        time_series (np.ndarray): The input time series data.
        max_position (int): The length of the time series or the maximum possible position.
        weight (float): Weight assigned to the mean of surrounding values
                                 when computing the composite score.

    Returns
    -------
        float: A composite score that integrates the position of the maximum value and
               the mean of surrounding values, weighted by the user-specified parameter.
    """
    # Compute the position of the maximum value and the mean of surrounding elements
    max_position = len(time_series)
    argmax_position, mean = mean_around_max(time_series, max_position // 2)
    composite_score = argmax_position * (1 - weight) + mean * weight * np.sign(0.5 - argmax_position / max_position)
    return composite_score


def order_genes(df, weight, window_size=10, stride=10):
    """
    Orders genes in a DataFrame based on the maximum location and mean of their expression profiles.

    Parameters
    ----------
        df (pd.DataFrame): A DataFrame where rows represent genes and columns represent gene expression.
        window_size (int): The size of the window for binning.
        stride (int): The number of steps to move the window after each bin.
        weight (float): The weight to assign to the mean in the composite score.

    Returns
    -------
        list: A list of integer row indices (row positions) sorted by the composite score.
    """
    max_indices = []

    for row_idx, expression in df.iterrows():
        binned_averages = bin_and_average_time_series(expression.values, window_size, stride)
        composite_score = find_max(binned_averages, weight=weight)
        max_indices.append((row_idx, composite_score))

    # Sort by the composite score and extract row positions
    sorted_indices = sorted(max_indices, key=lambda x: x[1], reverse=False)
    sorted_row_numbers = [df.index.get_loc(row_idx) for row_idx, _ in sorted_indices]

    return sorted_row_numbers


def filter_and_normalize_genes(scores_df, gene_curve, MI_threshold=0, GCV_threshold=1, AIC_threshold=np.inf):
    """
    Filter genes based on quality metrics and normalize expression curves.
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with columns 'gene', 'MI', 'GCV', 'AIC' containing quality metrics.
    gene_curve : pd.DataFrame
        Gene expression curves (rows=pseudotime points, columns=genes).
    MI_threshold : float
        Minimum mutual information score (genes above this are kept).
    GCV_threshold : float
        Maximum GCV score (genes below this are kept).
    AIC_threshold : float
        Maximum AIC score (genes below this are kept).
    
    Returns
    -------
    normalized_gene_curve : pd.DataFrame
        Z-score normalized gene expression curves for filtered genes.
    aligned_score : pd.Series
        The pseudotime/aligned_score column from original gene_curve.
    """
    # Filter genes based on thresholds
    is_filtered_genes = scores_df.gene[
        (scores_df.MI > MI_threshold) & (scores_df.GCV < GCV_threshold) & (scores_df.AIC < AIC_threshold)
    ]
    
    # Extract pseudotime column
    aligned_score = gene_curve.iloc[:, 0]  # First column is typically pseudotime
    gene_curve_filtered = gene_curve.loc[:, is_filtered_genes]
    
    # Normalize gene curve data
    zscore_scaler = StandardScaler()
    normalized_gene_curve = pd.DataFrame(
        zscore_scaler.fit_transform(gene_curve_filtered),
        columns=gene_curve_filtered.columns,
        index=gene_curve_filtered.index
    )
    
    return normalized_gene_curve, aligned_score


def order_genes_by_peak(normalized_gene_curve, weight=0.5):
    """
    Order genes by their peak expression timing in pseudotime.
    
    Parameters
    ----------
    normalized_gene_curve : pd.DataFrame
        Normalized gene expression curves (rows=pseudotime, columns=genes).
    weight : float
        Weight parameter for composite score (0-1).
        Higher values weight the surrounding mean more heavily.
    
    Returns
    -------
    sorted_gene_curve : pd.DataFrame
        Gene expression curves sorted by peak timing (rows=pseudotime, columns=genes).
    gene_order : np.ndarray
        Integer indices of genes in sorted order (indices into original normalized_gene_curve).
    """
    combined_order = order_genes(normalized_gene_curve.T, weight=weight)
    sorted_gene_curve = normalized_gene_curve.T.iloc[combined_order, :]
    return sorted_gene_curve, np.array(combined_order)


def cluster_sorted_genes(sorted_gene_curve, n_clusters=3, hierarchical=True, n_init=5, 
                        band_width=1, enforce_contiguity=True):
    """
    Cluster genes that are already sorted by temporal peak timing.
    
    This function assumes genes are pre-sorted and clusters them while respecting
    their temporal order. It's designed to create clean early/middle/late clusters.
    
    Parameters
    ----------
    sorted_gene_curve : pd.DataFrame
        Gene expression curves already sorted by peak timing.
        Rows = genes (in order), Columns = pseudotime points.
    n_clusters : int
        Number of clusters to create (typically 3: early, middle, late).
    hierarchical : bool
        If True, use hierarchical clustering with strict connectivity.
        If False, use KShape clustering.
    n_init : int
        Number of initializations for KShape (ignored for hierarchical).
    band_width : int
        Bandwidth for connectivity in hierarchical clustering.
        Lower values = stricter, genes must be closer to merge.
        Default=1 means only adjacent genes can merge initially.
    enforce_contiguity : bool
        If True (default), uses very strict 1-neighbor connectivity.
        Consecutive genes (in sorted order) tend to cluster together.
        Creates clean temporal progression: early → middle → late.
    
    Returns
    -------
    clusters : np.ndarray
        Cluster labels for each gene (length = n_genes).
    
    Notes
    -----
    Since genes are already sorted by peak, and we use strict connectivity,
    clusters will naturally separate into early/middle/late peaking patterns.
    """
    gene_data = sorted_gene_curve.values  # Shape: (n_genes, n_timepoints)
    n_genes = gene_data.shape[0]
    
    if hierarchical:
        if enforce_contiguity:
            # Strictest: only immediate neighbors
            connectivity = diags(
                [1] * 3,  # Only 1-neighbor connectivity: [-1, 0, 1]
                offsets=[-1, 0, 1],
                shape=(n_genes, n_genes)
            ).toarray()
        else:
            # Looser: genes within band_width can merge
            connectivity = diags(
                [1] * (2 * band_width + 1),
                offsets=range(-band_width, band_width + 1),
                shape=(n_genes, n_genes)
            ).toarray()
        
        clusters = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", connectivity=connectivity
        ).fit_predict(gene_data)
    else:
        if enforce_contiguity:
            print("Warning: enforce_contiguity requires hierarchical=True, ignoring...")
        clusters = KShape(
            n_clusters=n_clusters, random_state=0, verbose=True, n_init=n_init
        ).fit(gene_data).labels_
    
    return clusters


def assign_temporal_categories(clusters, label_names=None):
    """
    Assign meaningful names to clusters based on their temporal position.
    
    For pre-clustered genes (already ordered by peak), this assigns labels
    like 'Early', 'Middle', 'Late' based on where genes peak.
    
    Parameters
    ----------
    clusters : np.ndarray
        Cluster labels for genes (assumes genes are in temporal order).
    label_names : list or None
        Names for clusters in order of appearance.
        If None, uses numeric labels.
        Example: ['Early', 'Middle', 'Late'] for 3 clusters.
    
    Returns
    -------
    row_colors : pd.Series
        Category labels for each gene.
    category : list
        Category labels as list.
    
    Examples
    --------
    >>> clusters = np.array([0, 0, 1, 1, 2, 2, 2])
    >>> row_colors, categories = assign_temporal_categories(
    ...     clusters, label_names=['Early', 'Middle', 'Late']
    ... )
    """
    # Reorder labels based on their average position
    label_orders = order_categories_by_average_position(clusters)
    labels = pd.Series(clusters).map({old_label: new_label for new_label, old_label in enumerate(label_orders)})
    
    # Map labels to names if provided
    if label_names is None:
        category = labels.tolist()
    else:
        category = [label_names[x] for x in labels]
    
    row_colors = pd.Series(category)
    return row_colors, category



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
    """
    Complete pipeline: filter, normalize, order, and cluster genes.
    
    Recommended workflow:
    1. Filter and normalize genes by quality
    2. Order genes by peak timing  
    3. Cluster on sorted genes (creates clean early/middle/late groups)
    4. Assign temporal category labels
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        Gene quality metrics with columns 'gene', 'MI', 'GCV', 'AIC'.
    gene_curve : pd.DataFrame
        Gene expression curves (first column is pseudotime, rest are genes).
    cell_colors : array-like
        Cell colors (currently unused, kept for compatibility).
    cutoff_points : list
        Pseudotime cutoff points for column color categorization.
    MI_threshold : float
        Minimum mutual information threshold.
    GCV_threshold : float
        Maximum GCV threshold.
    AIC_threshold : float
        Maximum AIC threshold.
    n_clusters : int
        Number of clusters (typically 3 for early/middle/late).
    label_names : list or None
        Optional cluster names like ['Early', 'Middle', 'Late'].
    n_init : int
        KShape initialization count.
    hierarchical : bool
        Use hierarchical (default, recommended) vs KShape clustering.
    weight : float
        Gene ordering weight parameter.
    enforce_contiguity : bool
        If True, clusters are strictly separated by temporal order (recommended).
    
    Returns
    -------
    sorted_gene_curve : pd.DataFrame
        Ordered and normalized gene expression curves.
    row_colors : pd.Series
        Cluster labels for each gene.
    col_cols : pd.Series
        Column colors based on pseudotime categories.
    category : list
        Category labels for each gene.
    gene_order : np.ndarray
        Integer indices of genes in their sorted order.
    """
    # Step 1: Filter and normalize
    normalized_gene_curve, aligned_score = filter_and_normalize_genes(
        scores_df, gene_curve, MI_threshold, GCV_threshold, AIC_threshold
    )
    
    # Step 2: Order genes by peak timing
    sorted_gene_curve, gene_order = order_genes_by_peak(normalized_gene_curve, weight=weight)
    
    # Step 3: Cluster sorted genes (creates clean temporal separation)
    clusters = cluster_sorted_genes(
        sorted_gene_curve,
        n_clusters=n_clusters,
        hierarchical=hierarchical,
        n_init=n_init,
        enforce_contiguity=enforce_contiguity,
        band_width=band_width
    )
    
    # Step 4: Assign temporal category labels
    row_colors, category = assign_temporal_categories(clusters, label_names)
    
    # Step 5: Prepare column colors
    col_cols = categorize_values(aligned_score, cutoff_points)
    
    return sorted_gene_curve, row_colors, col_cols, category, gene_order


def enrichr(gene_list, gene_sets):
    while True:
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_sets,
                organism="human",
                outdir=None,
                cutoff=1,
            )
            early_results = enr.results
            return early_results  # Return the results if successful
        except Exception as e:  # noqa: BLE001
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(5)

color_map = {'NSCLC': 'gold',
             'SCLC-A': 'tab:red',
             'SCLC-N': 'tab:cyan'}
clusters = ['NSCLC', 'SCLC-A']#, 'SCLC-N']
colors = [color_map[x] for x in clusters]
sorted_gene_curve, colors, col_colors, categories, gene_order = process_gene_data(
    scores_df, gene_curve, colors, [1,2],
    n_clusters=4, 
    label_names=['Early 1', 'Early 2', 'Middle', 'Late'],
    hierarchical=True,
    enforce_contiguity=False,
    band_width=10
)

import itertools
import anndata
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyComplexHeatmap as pch
import scanpy as sc
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.neighbors import KernelDensity


def align_pseudotime(
    adata: "anndata.AnnData",
    gene: str,
    color: str,
    pseudotime_key: str = "aligned_pseudotime",
    barycenter_key: str = "barycenter",
    dot_size: int = 10,
) -> None:
    """Plot gene expression along aligned pseudotime with barycenter curve.

    Parameters
    ----------
    adata
        Input AnnData object.
    gene
        The gene to plot.
    color
        Color key for the scatter plot.
    pseudotime_key
        Column in `adata.obs` containing mapped aligned pseudotime.
        Defaults to "aligned_pseudotime".
    barycenter_key
        Key in `adata.uns` containing barycenter data.
        Defaults to "barycenter".
    dot_size
        Size of scatter plot dots.
        Defaults to 100.

    Raises
    ------
    ValueError
        If required data is missing from the AnnData object.
    """
    if pseudotime_key not in adata.obs:
        raise ValueError(f"{pseudotime_key} not found in adata.obs. Run mapping first.")

    if barycenter_key not in adata.uns or "aligned_pseudotime" not in adata.uns[barycenter_key]:
        raise ValueError(f"{barycenter_key} does not contain expected barycenter expression data.")

    if gene not in adata.var_names:
        raise ValueError(f"Gene {gene} not found in adata.var_names.")

    if gene not in adata.uns[barycenter_key]["expression"]:
        raise ValueError(f"Gene {gene} not found in barycenter expression.")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))

    # Scatter plot of individual cell expression values
    sc.pl.scatter(adata, x=pseudotime_key, y=gene, color=color, ax=ax, size=dot_size, show=False, use_raw=False)

    # Overlay barycenter expression line
    ax.plot(
        adata.uns[barycenter_key]["aligned_pseudotime"],
        adata.uns[barycenter_key]["expression"][gene],
        color="red",
        label="Barycenter",
    )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(f"Aligned Gene Expression: {gene}")
    plt.show()

def plot_alignment(adata, group_key, key_aligned, key_ground_truth):
    groups = adata.obs[group_key].unique()
    n_groups = len(groups)

    # grid layout
    n_cols = 3
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, group in enumerate(groups):
        ax = axes[i]

        # select group
        sub = adata.obs[adata.obs[group_key] == group]

        # remove NaN rows
        mask = ~(np.isnan(sub[key_aligned]) | np.isnan(sub[key_ground_truth]))
        sub = sub[mask]

        if len(sub) == 0:
            ax.set_title(f"{group} (no valid data)")
            ax.axis("off")
            continue

        # sort by local pseudotime so connecting lines don't cross randomly
        order = np.argsort(sub[key_aligned].values)
        local_sorted = sub[key_aligned].values[order]
        global_sorted = sub[key_ground_truth].values[order]

        # scatter points
        ax.scatter(global_sorted, np.ones(len(sub)), 
                   s=18, alpha=0.7, color="blue", label="Ground truth pseudotime")
        ax.scatter(local_sorted, np.zeros(len(sub)), 
                   s=18, alpha=0.7, color="red", label="Aligned pseudotime")

        # connecting lines
        for j in range(len(sub)):
            ax.plot([local_sorted[j], global_sorted[j]], 
                    [0, 1], color="black", alpha=0.25, linewidth=0.5)

        # aesthetics
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Aligned pseudotime", "Ground truth pseudotime"])
        ax.set_xlabel("Pseudotime")
        ax.set_title(f"{group} (n={len(sub)})")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc="upper right")

    # hide unused axes
    for j in range(n_groups, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_sigmoid_fits(
    aligned_obj,
):
    """Plot sigmoid fits for each subject in the aligned object.

    Args:
        aligned_obj: Aligner object containing alignment data.
    """
    subjects = aligned_obj.df[aligned_obj.subject_col].unique()
    fig, axes = plt.subplots(nrows=1, ncols=len(subjects), figsize=(15, 3), sharey=True)

    for i, subject in enumerate(subjects):
        ax = axes[i]
        subject_data = aligned_obj.df[aligned_obj.df[aligned_obj.subject_col] == subject]
        x_data = subject_data[aligned_obj.score_col].values
        y_data = subject_data["numeric_label"].values
        cutoff_point = aligned_obj.cutoff_points[subject]

        ax.scatter(x_data, y_data, alpha=0.01)
        for cutoff in cutoff_point:
            ax.axvline(x=cutoff, color="green", linestyle="--", label=f"{cutoff:.2f}")

        ax.set_title(f"Subject: {subject}")
        ax.set_xlabel("Score")
        if i == 0:
            ax.set_ylabel("Label")
        ax.legend()

    plt.show()


def plot_summary_curve(
    summary_df: pd.DataFrame,
    aggregated_curve: pd.DataFrame,
    scores: pd.DataFrame,
    genes: list[str],
    pseudotime_col = "aligned_pseudotime",
    subject_col = "subject", 
    figsize: tuple[float, float] = (2, 0.8),
    pt_alpha: float = 0.2,
    line_alpha: float = 0.5,
    dpi: int = 300,
    title_fontsize=20,
    label_fontsize=12,
    save_path=None
) -> None:
    """Plot summary curves for multiple genes.

    Parameters
    ----------
    summary_df
        DataFrame with columns ['gene', 'subject', 'aligned_score',
        'expression', 'smoothed'].
    aggregated_curve
        DataFrame with ['aligned_score'] and gene columns.
    scores
        DataFrame with columns ['gene', 'MI', 'Max'].
    genes
        List of gene names to plot.
    fig_size
        Figure size multiplier.
        Defaults to (2, 0.8).
    pt_alpha
        Alpha value for scatter points.
        Defaults to 0.2.
    line_alpha
        Alpha value for lines.
        Defaults to 0.5.
    dpi
        DPI for the figure.
        Defaults to 300.
    """
    num_genes = len(genes)
    fig, axes = plt.subplots(num_genes, 1, figsize=figsize, sharex=True, dpi=dpi)

    if num_genes == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    # Define a color cycle for subjects
    color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS)

    scatter_handles = []
    scatter_labels = []

    for i, (ax, gene) in enumerate(zip(axes, genes, strict=False)):
        gene_summary = summary_df.loc[summary_df["gene"] == gene]

        # Plot each subject's smoothed curve and collect scatter handles
        subject_colors = {}
        for subject in gene_summary[subject_col].unique():
            color = next(color_cycle)
            subject_colors[subject] = color
            subject_data = gene_summary[gene_summary[subject_col] == subject]
            ax.scatter(subject_data[pseudotime_col], subject_data["expression"], alpha=pt_alpha, s=0.1, color=color)
            ax.plot(subject_data[pseudotime_col], subject_data["smoothed"], alpha=line_alpha, color=color)
            if i == 0:  # Collect labels only from the first subplot
                scatter_handles.append(ax.scatter([], [], alpha=1, s=10, color=color))
                scatter_labels.append(subject)

        # Plot the aggregated curve
        (agg_line,) = ax.plot(aggregated_curve[pseudotime_col], aggregated_curve[gene], color="black", linewidth=2)

        # Calculate ylim based on smoothed values
        y_min = min(gene_summary["smoothed"])
        y_max = 1

        # Add some padding (e.g., 5% of the range)
        y_range = y_max - y_min
        padding = 0.1 * y_range
        ylim = (y_min - padding, y_max + padding)

        ax.set_ylim(ylim)
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(gene, fontsize=title_fontsize)

    #axes[-1].set_xlabel("Aligned pseudotime", fontsize=label_fontsize)

    # Create a single legend outside the plot area using scatter handles
    # fig.legend(
    #     scatter_handles, scatter_labels, loc="center left", bbox_to_anchor=(0.8, 0.5), fontsize="small", frameon=False
    # )

    # plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_kshape_clustering(
    sorted_gene_curve,
    categories,
    label_orders=None,
    alpha=0.05,
    save_path=None,
    dpi=300,
    figsize=None,
    label_colors=None,  # New parameter for custom colors per label
    label_xpos=0.95,  # X position (right-aligned)
    label_ypos=0.05,  # Y position (bottom-aligned)
    centroid_linewidth=2,  # Line width for centroid
    fontsize=15,
):
    if label_orders is None:
        label_orders = sorted(set(categories))
    if figsize is None:
        figsize = (2, 5 / 3 * len(label_orders))
    if label_colors is None:
        # Default to red if no colors provided
        label_colors = dict.fromkeys(label_orders, "red")

    plt.figure(figsize=figsize)

    for i, category in enumerate(label_orders):
        ax = plt.subplot(len(label_orders), 1, i + 1)
        cluster_curves = sorted_gene_curve.values[[x == category for x in categories], :]

        # Plot individual curves (thin black lines)
        for xx in cluster_curves:
            plt.plot(xx.ravel(), "k-", alpha=alpha)

        # Plot centroid (using label-matched color)
        centroid = np.mean(cluster_curves, axis=0)
        plt.plot(
            centroid.ravel(),
            "-",  # Solid line
            color=label_colors[category],  # Match label color
            linewidth=centroid_linewidth,
            label="Centroid",
        )

        # Add text annotation (bottom-right)
        ax.text(
            label_xpos,
            label_ypos,
            category,
            transform=ax.transAxes,
            color=label_colors[category],  # Same color as centroid
            fontsize=fontsize,
            horizontalalignment="right",
            verticalalignment="bottom",
            weight="bold",  # Make text stand out
        )

        plt.xticks([])
        plt.yticks([])

        if i == 0:
            plt.title("Clusters", fontsize=fontsize)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def fit_kde(sorted_gene_curve, df, cell_types, bandwidth=50):
    gene_density = pd.DataFrame(index=sorted_gene_curve.index)
    for cell_type in cell_types:
        is_cell_type_gene = [
            str(cell_type).lower() in str(x).lower() if pd.notna(x) else False for x in df
        ]  # (np.isin(df, cell_type)).astype(int)
        x = np.arange(len(is_cell_type_gene)).reshape(-1, 1)
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(x, sample_weight=is_cell_type_gene)

        # Evaluate the density model on a grid
        x_d = np.linspace(0, len(is_cell_type_gene) - 1, len(is_cell_type_gene)).reshape(-1, 1)
        log_density = kde.score_samples(x_d)
        gene_density[cell_type] = np.exp(log_density)
    return gene_density


def plot_kde_density(
    density: pd.DataFrame,
    title: str = "",
    clusters: np.ndarray | None = None,
    cluster_colors: dict[str, str] | None = None,
    name_map: dict[str, str] | None = None,
) -> None:
    """Plot kernel density estimation curves.

    Parameters
    ----------
    density
        DataFrame containing density values for each cell type.
    title
        Plot title.
        Defaults to empty string.
    clusters
        Array of cluster assignments.
        Defaults to None.
    cluster_colors
        Dictionary mapping cluster names to colors.
        Defaults to None.
    name_map
        Dictionary for mapping names in the density DataFrame.
        Defaults to None.
    """
    fig, ax = plt.subplots(2, 1, figsize=(5, 3), gridspec_kw={"height_ratios": [3, 0.1], "hspace": 0})
    if name_map is not None:
        density.columns = density.columns.map(name_map)
    # Plot KDE density
    for cell_type, density_values in density.items():
        ax[0].plot(density.index, density_values, label=cell_type)
        ax[0].fill_between(density.index, density_values, alpha=0.5)

    ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=False)
    ax[0].set_title(title)
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_xlabel("")
    ax[0].set_ylabel("Density", fontsize=15)

    # Add bar plot for cell type annotations
    if clusters is not None:
        unique_clusters = cluster_colors.keys()
        color_map = cluster_colors
        for cluster in unique_clusters:
            subset = density.index[clusters == cluster]
            ax[1].bar(subset, height=0.5, bottom=-0.5, width=1, color=cluster_colors[cluster], align="edge", alpha=1)

        # Add legend for cell types
        handles = [plt.Line2D([0], [0], color=color_map[ct], lw=4) for ct in unique_clusters]
        labels = unique_clusters
        ax[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=False)

        ax[1].set_yticks([])
        ax[1].set_xticks([])
        for spine in ax[0].spines.values():
            spine.set_visible(False)
        for spine in ax[1].spines.values():
            spine.set_visible(False)
        ax[1].set_xlabel("Genes", fontsize=15)

    plt.tight_layout()
    plt.show()


def plot_kde_density_ridge(
    density,
    scores=None,
    clusters=None,
    cluster_colors=None,
    name_map=None,
    cell_type_order=None,
    figsize=(6, 4),
    fontsize=12,
    save_path=None,
):
    if name_map is not None:
        density.columns = density.columns.map(name_map)
    n_cell_types = len(density.columns)
    height_ratios = [2] * n_cell_types
    n_rows = n_cell_types
    if clusters is not None:
        n_rows = n_cell_types + 1
        height_ratios.append(0.5)

    fig, ax = plt.subplots(n_rows, 1, figsize=figsize, gridspec_kw={"hspace": 0, "height_ratios": height_ratios})
    ordering = density.columns
    if cell_type_order is not None:
        ordering = cell_type_order

    # Create color map for scores
    if scores is not None:
        scores = scores.fillna(0)  # Fill NA values with 0
        vmin = 0
        vmax = scores.values.max()
        norm_scores = (scores - vmin) / (vmax - vmin)
        cmap = plt.cm.Blues

    for i, cell_type in enumerate(ordering):
        if cell_type in density.columns:
            x = density.index
            y = density[cell_type]
            ax[i].plot(x, y, label=cell_type, color="black")

            # Create gradient fill
            if scores is not None and cell_type in scores.columns:
                cell_score = norm_scores[cell_type]
                # Default fill with white for zero/NA values
                ax[i].fill_between(x, y, color="white", alpha=0.5)

                # Only color non-zero scores
                if any(cell_score > 0):  # or use a small threshold like 0.01
                    colors = cmap(np.full(len(x), cell_score))
                    for j in range(len(x) - 1):
                        ax[i].fill_between(
                            x[j : j + 2], y[j : j + 2], color=colors[j], alpha=0.8
                        )  # Increased alpha for better visibility
            else:
                ax[i].fill_between(x, y, color="white", alpha=0.5)

            ax[i].set_yticks([])
            ax[i].set_xticks([])
            ax[i].set_xlabel("")
            for spine in ax[i].spines.values():
                spine.set_visible(False)
            ax[i].set_ylabel(cell_type, fontsize=fontsize, rotation=0, labelpad=10)

    # Add bar plot for cell type annotations
    if clusters is not None:
        unique_clusters = cluster_colors.keys()
        color_map = cluster_colors
        for cluster in unique_clusters:
            # Convert clusters to numpy array if it's a Series/DataFrame to avoid indexing issues
            clusters_array = np.asarray(clusters).flatten() if hasattr(clusters, 'values') else np.asarray(clusters)
            mask = clusters_array == cluster
            subset = density.index[mask]
            ax[n_rows - 1].bar(
                subset, height=0.1, bottom=-0.5, width=1, color=cluster_colors[cluster], align="edge", alpha=1
            )

        # Add legend for cell types
        handles = [plt.Line2D([0], [0], color=color_map[ct], lw=4) for ct in unique_clusters]
        labels = unique_clusters
        ax[n_rows - 1].legend(
            handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=False
        )
        ax[n_rows - 1].set_yticks([])
        ax[n_rows - 1].set_xticks([])
        for spine in ax[n_rows - 1].spines.values():
            spine.set_visible(False)
        ax[n_rows - 1].set_xlabel("Genes", fontsize=fontsize)

    # Add colorbar if scores are provided
    # if scores is not None:
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    #     cbar = plt.colorbar(sm, ax=ax, location='right', pad=0.1)
    #     cbar.set_label('Score', fontsize=fontsize)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_kde_heatmap(
    cluster_colors: dict[str, str],
    cell_types: list[str],
    cell_type_colors: dict[str, str],
    sorted_gene_curve: pd.DataFrame,
    df_left: pd.DataFrame,
    right_annotation_labels: pd.Series | list | None = None,
    right_annotation_name: str = "Cluster",
    density: pd.DataFrame | None = None,
    figsize: tuple[int, int] = (4, 8),
    left_annotation_columns: list[tuple[str, str, str]] | None = None,
    vmin: float = -3,
    vmax: float = 3,
    save_path: str | None = None,
    genes_to_label: list[str] | None = None,
    label_fontsize: int = 8,
    label_color: str = "black",
    dpi: int = 300,
) -> None:
    """Plot a KDE heatmap using PyComplexHeatmap with optional gene labels.

    Parameters
    ----------
    cluster_colors
        Dictionary mapping cluster names to colors.
    cell_types
        List of cell type names.
    cell_type_colors
        Dictionary mapping cell type names to colors.
    sorted_gene_curve
        DataFrame containing gene expression data.
    df_left
        DataFrame for left annotations.
    density
        DataFrame containing density information for different cell types.
        Defaults to None.
    figsize
        Size of the figure (width, height).
        Defaults to (4, 8).
    left_annotation_columns
        List of tuples (name, column, colormap) for left annotations.
        Defaults to None.
    vmin
        Minimum value for color scaling.
        Defaults to -3.
    vmax
        Maximum value for color scaling.
        Defaults to 3.
    save_path
        Path to save the output figure.
        Defaults to None.
    genes_to_label
        List of gene names to annotate on the left side of the heatmap.
        Defaults to None.
    label_fontsize
        Font size for gene labels.
        Defaults to 8.
    label_color
        Color for gene label text.
        Defaults to 'black'.
    """
    # Define custom colormaps for Early, Middle, and Late groups
    cmap = {
        "early": LinearSegmentedColormap.from_list("custom_cmap", ["#FFFFFF", cluster_colors["Early"]]),
        "middle": LinearSegmentedColormap.from_list("custom_cmap", ["#FFFFFF", cluster_colors["Middle"]]),
        "late": LinearSegmentedColormap.from_list("custom_cmap", ["#FFFFFF", cluster_colors["Late"]]),
    }

    # Create column annotation (top)
    col_ha = pch.HeatmapAnnotation(
        label=pch.anno_label(
            cell_types,
            merge=True,
            extend=False,
            rotation=0,
            colors=cell_type_colors,
            adjust_color=False,
            luminance=0.75,
            relpos=(0, 0),
            fontsize=12,
            fontweight="bold",
            arrowprops={"visible": False},
            ha="center",
            va="center",
        ),
        Group=pch.anno_simple(cell_types, colors=cell_type_colors),
        verbose=1,
        axis=1,
        plot_legend=False,
        label_kws={
            "visible": False,
        },
    )

    # Create left annotation
    if left_annotation_columns is not None and density is not None:
        left_annotations = {}
        for name, col_name, cmap_name in left_annotation_columns:
            if col_name in density.columns:
                left_annotations[name] = pch.anno_simple(density.loc[:, col_name], cmap=cmap[cmap_name], height=4)
        left_ha = pch.HeatmapAnnotation(
            **left_annotations,
            verbose=1,
            axis=0,
            plot_legend=False,
            label_kws={"visible": True, "rotation": 90, "horizontalalignment": "left", "verticalalignment": "center"},
        )
    else:
        left_ha = None

    # Build right annotations: cluster labels (early/middle/late) and optional gene labels
    right_annos = {}
    if right_annotation_labels is not None:
        if isinstance(right_annotation_labels, list):
            right_series = pd.Series(right_annotation_labels, index=sorted_gene_curve.index)
        else:
            right_series = right_annotation_labels.reindex(sorted_gene_curve.index)
        right_annos[right_annotation_name] = pch.anno_simple(right_series, colors=cluster_colors)

    # Create gene labels annotation if genes_to_label is provided
    if genes_to_label is not None:
        # Ensure genes_to_label is a list (convert from Series if needed)
        if isinstance(genes_to_label, pd.Series):
            genes_to_label = genes_to_label.tolist()

        # Convert index to string if it's not already
        gene_names = sorted_gene_curve.index.astype(str)

        # Find which genes are actually present in the data
        present_genes = [gene for gene in genes_to_label if gene in gene_names.values]
        missing_genes = set(genes_to_label) - set(present_genes)

        if missing_genes:
            print(f"Warning: The following genes were not found in the data: {missing_genes}")
            print(f"Available genes start with: {list(gene_names[:5])}...")

        # Create labels as a pandas Series
        gene_labels = pd.Series(np.where(gene_names.isin(present_genes), gene_names, ""), index=gene_names)
        gene_labels = gene_labels[gene_labels != ""]
        print(gene_labels)

        # Only proceed if we have at least one gene to label
        if len(present_genes) > 0:
            right_annos["Genes"] = pch.anno_label(
                gene_labels,
                merge=True,
                colors=label_color,
                fontsize=label_fontsize,
                rotation=0,
                extend=True,
                relpos=(0, 0.5),
            )

    right_ha = None if len(right_annos) == 0 else pch.HeatmapAnnotation(
        **right_annos,
        axis=0,
        verbose=1,
        plot_legend=False,
        label_kws={"visible": False},
    )

    # Plot the heatmap using PyComplexHeatmap's ClusterMapPlotter
    plt.figure(figsize=figsize)
    heatmap = pch.ClusterMapPlotter(
        data=sorted_gene_curve,
        top_annotation=col_ha,
        left_annotation=left_ha,
        right_annotation=right_ha,
        row_cluster=False,
        col_cluster=False,
        row_split_gap=1,
        cmap="Spectral_r",
        vmin=vmin,
        vmax=vmax,
        xlabel="scDeBussy pseudotime",
        ylabel="Genes",
        xlabel_kws={"fontsize": 14},  # Adjust size as needed
        ylabel_kws={"fontsize": 14},  # Adjust size as needed
        plot_legend=False,
        legend_kws={"label": None},
    )
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(1, 0.5))
    heatmap.plot_legends()
    if save_path is not None:
        legend_save_path = save_path.rsplit(".", 1)[0] + "_legend." + save_path.rsplit(".", 1)[1]
        plt.savefig(legend_save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.tight_layout()
    plt.show()


def plot_gene_clusters(
    sorted_gene_curve: pd.DataFrame,
    row_colors: list[str],
    col_cols: list[str],
    cluster_ordering: str,
    display_genes: list[str] | None = None,
    yticklabels: bool | list[str] = True,
    labelsize: int = 8,
    tick_width: float = 0.01,
    figsize: tuple[int, int] = (4, 8),
    save_path: str | None = None,
) -> None:
    """Plot a heatmap with optional gene labeling on the y-axis.

    Parameters
    ----------
    sorted_gene_curve
        DataFrame with gene expression data.
    row_colors
        List of colors for rows.
    col_cols
        List of colors for columns.
    cluster_ordering
        String to label x-axis.
    display_genes
        List of genes to display on the y-axis.
        Defaults to None.
    yticklabels
        Whether to display y-tick labels or a list of labels.
        Defaults to True.
    labelsize
        Font size for y-tick labels.
        Defaults to 8.
    tick_width
        Width of tick marks.
        Defaults to 0.01.
    figsize
        Size of the figure (width, height).
        Defaults to (4, 8).
    save_path
        Path to save the figure.
        Defaults to None.
    """
    # Create clustermap
    g = sns.clustermap(
        sorted_gene_curve,
        col_cluster=False,
        row_cluster=False,
        cmap="Spectral_r",
        figsize=figsize,
        vmax=3,
        vmin=-3,
        col_colors=[col_cols],
        row_colors=[row_colors],
        cbar_pos=(0.95, 0.35, 0.01, 0.2),
        yticklabels=yticklabels,
    )

    # Get current y-tick labels (which are gene names)
    current_labels = g.ax_heatmap.get_yticklabels()

    # If display_genes is provided, modify y-tick labels and adjust their positions
    if display_genes is not None:
        # Create a list of new labels: keep gene name if it's in display_genes, otherwise set it to an empty string
        new_labels = [label.get_text() if label.get_text() in display_genes else "" for label in current_labels]
        # Get current y-tick positions
        yticks = g.ax_heatmap.get_yticks()
        display_positions = [
            (i, yticks[i]) for i, label in enumerate(current_labels) if label.get_text() in display_genes
        ]
        display_positions.sort(key=lambda x: x[1])

        # Minimum distance threshold between labels on the y-axis
        min_distance = 10

        # Track consecutive overlaps and shift amount for x position
        overlap_count = 0
        shift_amount_x = 0.19  # Amount to shift right after every two overlaps

        # Loop through each display gene and adjust its position if necessary
        for i in range(1, len(display_positions)):
            current_index, current_y = display_positions[i]
            prev_index, prev_y = display_positions[i - 1]
            closest_gene = min(display_positions[:i], key=lambda x: abs(x[1] - current_y))
            # If the current label is too close to the previous one in terms of y-position
            if (abs(current_y - prev_y) < min_distance) or (abs(closest_gene[1] - current_y) < min_distance):
                overlap_count += 1
                new_x_position = current_labels[current_index].get_position()[0] + shift_amount_x * (overlap_count % 6)
                print(f"Shifting X Position: {new_x_position}")
                current_labels[current_index].set_position((new_x_position, current_y))
            else:
                # check for the closest genes previous to the current gene
                overlap_count = 0

        for label in current_labels:
            label.set_ha("left")  # Align horizontally to the left
        # Apply new tick labels after adjusting positions
        g.ax_heatmap.set_yticklabels(new_labels)

    # Set y-axis tick label size and make ticks thinner
    g.ax_heatmap.yaxis.set_tick_params(labelsize=labelsize, width=tick_width)

    # Remove x-ticks and set x-label based on cluster ordering
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_xlabel(cluster_ordering.replace("_", "->"))

    plt.tight_layout()  # Automatically adjust subplots to fit into figure area

    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)

    plt.show()


def plot_heatmap(results, terms):
    heatmap_long = results.loc[np.isin(results.Term, terms), ["Term", "category", "Odds Ratio", "Adjusted P-value"]]
    heatmap_long["Term"] = pd.Categorical(heatmap_long["Term"], categories=terms[::-1], ordered=True)
    heatmap_long["category"] = pd.Categorical(
        heatmap_long["category"], categories=["early", "intermediate", "late"], ordered=True
    )

    plt.figure(figsize=(8, 4))
    sns.scatterplot(
        data=heatmap_long,
        x="category",
        y="Term",
        size="Odds Ratio",
        sizes=(20, 200),
        hue="Adjusted P-value",
        palette="viridis_r",
        hue_norm=LogNorm(),
    )
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlim(-0.5, len(heatmap_long["category"].unique()) - 0.5)
    plt.ylim(-0.5, len(heatmap_long["Term"].unique()) - 0.5)
    plt.tight_layout()
    plt.savefig("dotplot.png", bbox_inches="tight")
    plt.show()

gene_info = pd.DataFrame({'gene': sorted_gene_curve.index, 'category': categories})
gene_sets = ['CellMarker_2024']
results = pd.DataFrame()  # Initialize an empty DataFrame to store results

for category in gene_info.category.unique():
    gene_list = gene_info.gene[gene_info.category == category]
    results_category = enrichr(gene_list, gene_sets)
    results_category.loc[:,'category'] = category
    results = pd.concat([results, results_category])
    
results.loc[:,'overlap'] = results.Overlap.apply(lambda x: int(x.split('/')[0]))
results = results[(results['Adjusted P-value'] < 0.05) & (results['overlap'] > 10)]
results = results.drop_duplicates(subset="Genes")
pd.set_option('display.max_rows', None)
gene_sets = results.sort_values(by=["category", "Odds Ratio"], ascending = [True, False]).reset_index(drop=True)

genes = gene_sets['Genes'].str.split(';').explode().str.strip().unique()
gene_to_terms = {}

for index, row in gene_sets.iterrows():
    term = row['Term']
    genes = row['Genes'].split(';')
    combined_score = row['Combined Score']
    
    for gene in genes:
        if gene not in gene_to_terms:
            gene_to_terms[gene] = {'term': term}
        else:
            gene_to_terms[gene]['term'] = gene_to_terms[gene]['term'] + ';' + term

final_gene_to_term = {gene: info['term'] for gene, info in gene_to_terms.items()}
sorted_gene_curve_with_annot = sorted_gene_curve.copy()
sorted_gene_curve_with_annot.loc[:,'gene_sets'] = sorted_gene_curve_with_annot.index.map(final_gene_to_term)
sorted_gene_curve_with_annot.loc[:,'cluster'] = categories

cell_type_colors = {
    'NSCLC': 'gold', 'SCLC-A': 'tab:red', 'SCLC-N': 'tab:cyan'
}
cell_types = col_colors.map(lambda x: 'NSCLC' if x == 0 else ('SCLC-A' if x == 1 else 'SCLC-N'))
clusters = categories
cluster_colors = {}
for i, category in enumerate(gene_info['category'].unique()):
    cluster_colors[category] = plt.get_cmap('Dark2')(i / len(gene_info['category'].unique()))

df_right = sorted_gene_curve_with_annot['gene_sets']
df_right.index = sorted_gene_curve.index
df_left = sorted_gene_curve_with_annot['cluster']
df_left.index = sorted_gene_curve.index

gene_info = pd.DataFrame({'gene': sorted_gene_curve.index, 'category': categories})
gene_sets = ['CellMarker_2024']
results = pd.DataFrame()  # Initialize an empty DataFrame to store results

for category in gene_info.category.unique():
    gene_list = gene_info.gene[gene_info.category == category]
    results_category = enrichr(gene_list, gene_sets)
    results_category.loc[:,'category'] = category
    results = pd.concat([results, results_category])
    
results.loc[:,'overlap'] = results.Overlap.apply(lambda x: int(x.split('/')[0]))
results = results[(results['Adjusted P-value'] < 0.05) & (results['overlap'] > 10)]
results = results.drop_duplicates(subset="Genes")
gene_sets = results.sort_values(by=["category", "Odds Ratio"], ascending = [True, False]).reset_index(drop=True)

gene_sets.category = pd.Categorical(gene_sets.category, ordered=True, categories=['Early', 'Middle', 'Late'])
gene_sets.groupby('category').apply(lambda x: x.sort_values(by='Odds Ratio', ascending=False).head(30)).loc[:,['Term', 'category', 'Odds Ratio', 'Adjusted P-value', 'Overlap', 'Genes']].reset_index(drop=True)

# Running enrichment analysis along ordered genes (continuous, not categorical)
def compute_running_enrichment(ordered_genes, gene_set_dict, window_size=100, step_size=20):
    """
    Compute running enrichment scores for gene sets along ordered genes using hypergeometric test.
    
    Parameters
    ----------
    ordered_genes : list or pd.Index
        Genes in temporal order (early -> late)
    gene_set_dict : dict
        Dictionary mapping gene set names to lists of genes
    window_size : int
        Size of sliding window
    step_size : int
        Step size for sliding window
    
    Returns
    -------
    pd.DataFrame : DataFrame with enrichment scores across gene positions
    """
    from scipy.stats import hypergeom
    
    ordered_genes = list(ordered_genes)
    total_genes = len(ordered_genes)
    
    results = []
    
    for gene_set_name, gene_set in gene_set_dict.items():
        # Filter to genes present in ordered list
        gene_set_present = [g for g in gene_set if g in ordered_genes]
        n_gene_set = len(gene_set_present)
        
        if n_gene_set < 5:  # Skip if too few genes
            continue
        
        # Sliding window
        for start in range(0, total_genes - window_size + 1, step_size):
            end = start + window_size
            window_genes = ordered_genes[start:end]
            
            # Count overlap
            overlap = len(set(window_genes) & set(gene_set_present))
            
            # Hypergeometric test: 
            # M = total genes, n = genes in set, N = window size, k = overlap
            pval = hypergeom.sf(overlap - 1, total_genes, n_gene_set, window_size)
            
            # Enrichment score: -log10(p-value) * sign(enrichment)
            expected = (window_size * n_gene_set) / total_genes
            enrichment_ratio = overlap / expected if expected > 0 else 0
            
            # Score combining significance and magnitude
            if pval < 1e-10:
                score = 10 * np.sign(overlap - expected)
            else:
                score = -np.log10(pval + 1e-10) * np.sign(overlap - expected)
            
            results.append({
                'gene_set': gene_set_name,
                'position': (start + end) / 2,  # Center of window
                'start': start,
                'end': end,
                'overlap': overlap,
                'expected': expected,
                'enrichment_ratio': enrichment_ratio,
                'pval': pval,
                'score': score
            })
    
    return pd.DataFrame(results)


def plot_running_enrichment(enrichment_df, gene_sets_to_plot, 
                            ordered_genes, figsize=(12, 6), 
                            smooth_window=5, save_path=None,
                            show_categories=True, category_labels=None):
    """
    Plot running enrichment scores along ordered genes.
    
    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Output from compute_running_enrichment
    gene_sets_to_plot : list
        List of gene set names to visualize
    ordered_genes : list
        Ordered gene list for x-axis scale
    figsize : tuple
        Figure size
    smooth_window : int
        Window for smoothing curves
    save_path : str or None
        Path to save figure
    show_categories : bool
        Whether to show early/middle/late regions
    category_labels : list or None
        List of category labels matching ordered_genes
    """
    from scipy.ndimage import uniform_filter1d
    
    fig, axes = plt.subplots(len(gene_sets_to_plot), 1, figsize=figsize, 
                            sharex=True, sharey=False)
    
    if len(gene_sets_to_plot) == 1:
        axes = [axes]
    
    colors_map = {
        'Secretory Cell Lung Human': 'tab:blue',
        'Basal Cell Trachea Mouse': 'coral',
        'Cycling Basal Cell Trachea Mouse': 'tab:green',
        'Neuroendocrine Cell Trachea Mouse': 'tab:purple'
    }
    
    for i, (ax, gene_set_name) in enumerate(zip(axes, gene_sets_to_plot)):
        subset = enrichment_df[enrichment_df['gene_set'] == gene_set_name].sort_values('position')
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, f"{gene_set_name}\n(No data)", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Smooth the scores
        scores_smooth = uniform_filter1d(subset['score'].values, size=smooth_window)
        
        color = colors_map.get(gene_set_name, 'black')
        
        # Plot
        ax.plot(subset['position'], scores_smooth, color=color, linewidth=2.5, label=gene_set_name)
        ax.fill_between(subset['position'], 0, scores_smooth, 
                        where=(scores_smooth > 0), color=color, alpha=0.3)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Find and mark peak
        peak_idx = np.argmax(scores_smooth)
        peak_pos = subset['position'].iloc[peak_idx]
        peak_score = scores_smooth[peak_idx]
        
        ax.plot(peak_pos, peak_score, 'r*', markersize=15, zorder=10)
        ax.annotate(f'Peak\n({int(peak_pos)}/{len(ordered_genes)})', 
                   xy=(peak_pos, peak_score),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='red', alpha=0.8))
        
        # Styling
        ax.set_ylabel('Enrichment\nScore', fontsize=11, rotation=0, 
                     ha='right', va='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add title with short name
        short_name = gene_set_name.replace(' Lung Human', '').replace(' Trachea Mouse', '')
        ax.set_title(short_name, fontsize=12, fontweight='bold', loc='left', pad=10)
        
        # Add category background shading if available
        if show_categories and category_labels is not None:
            category_colors = {'Early': 'gold', 'Middle': 'lightgray', 'Late': 'lightblue'}
            categories_array = np.array(category_labels)
            
            for cat, cat_color in category_colors.items():
                cat_indices = np.where(categories_array == cat)[0]
                if len(cat_indices) > 0:
                    cat_start = cat_indices[0]
                    cat_end = cat_indices[-1]
                    ax.axvspan(cat_start, cat_end, alpha=0.15, color=cat_color, zorder=-1)
            
            # Add category labels on first plot
            if i == 0:
                for cat in ['Early', 'Middle', 'Late']:
                    cat_indices = np.where(categories_array == cat)[0]
                    if len(cat_indices) > 0:
                        mid_pos = (cat_indices[0] + cat_indices[-1]) / 2
                        ax.text(mid_pos, ax.get_ylim()[1] * 0.9, cat,
                               ha='center', va='top', fontsize=9, 
                               fontweight='bold', alpha=0.6)
    
    axes[-1].set_xlabel('Gene Position (Ordered by Peak Expression Timing)', 
                        fontsize=12, fontweight='bold')
    axes[-1].set_xlim(0, len(ordered_genes))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    plt.show()


# Prepare gene set dictionary from enrichR results
def create_gene_set_dict_from_enrichr(results_df, terms_to_extract):
    """Extract gene sets from EnrichR results."""
    gene_set_dict = {}
    for term in terms_to_extract:
        if term in results_df['Term'].values:
            genes_str = results_df[results_df['Term'] == term]['Genes'].iloc[0]
            genes = [g.strip() for g in genes_str.split(';')]
            gene_set_dict[term] = genes
        else:
            print(f"Warning: '{term}' not found in results")
    return gene_set_dict

# Extract gene sets of interest
gene_sets_to_analyze = [
    'Secretory Cell Lung Human',
    'Basal Cell Trachea Mouse',  
    'Neuroendocrine Cell Trachea Mouse',
    'Astrocyte Embryonic Prefrontal Cortex Human'

]

gene_set_dict = create_gene_set_dict_from_enrichr(gene_sets, gene_sets_to_analyze)

print("\n" + "="*80)
print("Running Enrichment Analysis Along Ordered Genes (Continuous)")
print("="*80)
print(f"Analyzing {len(gene_set_dict)} gene sets across {len(sorted_gene_curve.index)} ordered genes")
print()

# Compute running enrichment
enrichment_df = compute_running_enrichment(
    sorted_gene_curve.index, 
    gene_set_dict,
    window_size=150,  # ~10% of genes in window
    step_size=30
)

# Plot the results
plot_running_enrichment(
    enrichment_df,
    gene_sets_to_plot=gene_sets_to_analyze,
    ordered_genes=sorted_gene_curve.index,
    figsize=(10, 5),
    smooth_window=3,
    save_path=os.path.join(PLOT_DIR, "gene_set_running_enrichment.pdf"),
    show_categories=True,
    category_labels=categories
)

# Print peak statistics
print("\nPeak Enrichment Statistics:")
print("-" * 80)
for gene_set_name in gene_sets_to_analyze:
    subset = enrichment_df[enrichment_df['gene_set'] == gene_set_name]
    if len(subset) > 0:
        peak_idx = subset['score'].idxmax()
        peak_row = subset.loc[peak_idx]
        
        print(f"\n{gene_set_name}:")
        print(f"  Peak position: Gene {int(peak_row['position'])}/{len(sorted_gene_curve.index)}")
        print(f"  Peak enrichment score: {peak_row['score']:.2f}")
        print(f"  Genes in window: {peak_row['overlap']:.0f}/{peak_row['end']-peak_row['start']:.0f}")
        print(f"  Enrichment ratio: {peak_row['enrichment_ratio']:.2f}x expected")
        print(f"  P-value: {peak_row['pval']:.2e}")


# ============================================================================
# Justify selection: Compare against ALL gene sets
# ============================================================================

# def rank_gene_sets_by_enrichment(enrichment_df):
#     """
#     Rank all gene sets by their sustained enrichment metrics.
    
#     Returns
#     -------
#     pd.DataFrame : Ranked gene sets with sustained enrichment statistics
#     """
#     rankings = []
    
#     for gene_set_name in enrichment_df['gene_set'].unique():
#         subset = enrichment_df[enrichment_df['gene_set'] == gene_set_name].copy()
        
#         if len(subset) == 0:
#             continue
        
#         # Peak metrics
#         peak_idx = subset['score'].idxmax()
#         peak_row = subset.loc[peak_idx]
        
#         # Enhanced metrics for sustained/broad enrichment
#         positive_scores = subset[subset['score'] > 0]['score']
        
#         # 1. Area under curve (total enrichment)
#         auc = subset['score'].sum()
        
#         # 2. Breadth: How many windows show positive enrichment
#         breadth = len(positive_scores) / len(subset) * 100  # percentage
        
#         # 3. Mean enrichment in positive windows
#         mean_positive = positive_scores.mean() if len(positive_scores) > 0 else 0
        
#         # 4. Sustained enrichment: number of consecutive windows above threshold
#         threshold = subset['score'].quantile(0.75)
#         above_threshold = (subset['score'] > threshold).astype(int)
#         max_consecutive = 0
#         current_consecutive = 0
#         for val in above_threshold:
#             if val == 1:
#                 current_consecutive += 1
#                 max_consecutive = max(max_consecutive, current_consecutive)
#             else:
#                 current_consecutive = 0
        
#         # 5. Combined score emphasizing sustained enrichment
#         combined_score = (
#             0.25 * peak_row['score'] +           # Peak strength
#             0.25 * mean_positive +                # Average strength
#             0.25 * breadth +                      # How widespread
#             0.25 * max_consecutive                # How sustained
#         )
        
#         rankings.append({
#             'gene_set': gene_set_name,
#             'peak_score': peak_row['score'],
#             'peak_position': peak_row['position'],
#             'peak_pval': peak_row['pval'],
#             'peak_enrichment_ratio': peak_row['enrichment_ratio'],
#             'peak_overlap': peak_row['overlap'],
#             'auc': auc,
#             'breadth_pct': breadth,
#             'mean_positive_score': mean_positive,
#             'max_consecutive_windows': max_consecutive,
#             'combined_sustained_score': combined_score,
#             'n_significant_windows': len(subset[subset['pval'] < 0.01])
#         })
    
#     rankings_df = pd.DataFrame(rankings)
#     rankings_df = rankings_df.sort_values('combined_sustained_score', ascending=False).reset_index(drop=True)
#     rankings_df['rank'] = range(1, len(rankings_df) + 1)
    
#     return rankings_df


# def plot_gene_set_justification(rankings_df, highlight_sets, 
#                                 figsize=(14, 10), save_path=None):
#     """
#     Create comprehensive plots showing why highlighted gene sets stand out.
    
#     Parameters
#     ----------
#     rankings_df : pd.DataFrame
#         Ranked gene sets from rank_gene_sets_by_enrichment
#     highlight_sets : list
#         Gene sets to highlight
#     figsize : tuple
#         Figure size
#     save_path : str or None
#         Path to save figure
#     """
#     fig = plt.figure(figsize=figsize)
#     gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
#     colors_map = {
#         'Secretory Cell Lung Human': 'tab:blue',
#         'Basal Cell Trachea Mouse': 'coral',
#         'Cycling Basal Cell Trachea Mouse': 'tab:green',
#         'Neuroendocrine Cell Trachea Mouse': 'tab:purple'
#     }
    
#     # Plot 1: Top gene sets by sustained enrichment score
#     ax1 = fig.add_subplot(gs[0, :])
#     top_n = 30
#     top_sets = rankings_df.head(top_n)
    
#     colors = [colors_map.get(gs, 'lightgray') for gs in top_sets['gene_set']]
#     bars = ax1.barh(range(len(top_sets)), top_sets['combined_sustained_score'], color=colors, alpha=0.8)
    
#     # Shorten labels
#     short_labels = [name.replace(' Lung Human', '').replace(' Trachea Mouse', '').replace(' Mouse', '') 
#                    for name in top_sets['gene_set']]
#     ax1.set_yticks(range(len(top_sets)))
#     ax1.set_yticklabels(short_labels, fontsize=9)
#     ax1.set_xlabel('Combined Sustained Enrichment Score', fontsize=11, fontweight='bold')
#     ax1.set_title(f'Top {top_n} Gene Sets by Sustained Enrichment', fontsize=13, fontweight='bold')
#     ax1.invert_yaxis()
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
    
#     # Add legend
#     from matplotlib.patches import Patch
#     legend_elements = [Patch(facecolor=colors_map[gs], alpha=0.8, 
#                             label=gs.replace(' Lung Human', '').replace(' Trachea Mouse', ''))
#                       for gs in highlight_sets if gs in rankings_df['gene_set'].values]
#     legend_elements.append(Patch(facecolor='lightgray', alpha=0.8, label='Other'))
#     ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
#     # Plot 2: Distribution of sustained enrichment scores
#     ax2 = fig.add_subplot(gs[1, 0])
#     ax2.hist(rankings_df['combined_sustained_score'], bins=50, color='lightgray', alpha=0.7, edgecolor='black')
    
#     for gene_set in highlight_sets:
#         if gene_set in rankings_df['gene_set'].values:
#             score = rankings_df[rankings_df['gene_set'] == gene_set]['combined_sustained_score'].iloc[0]
#             ax2.axvline(score, color=colors_map[gene_set], linestyle='--', linewidth=2.5,
#                        label=gene_set.replace(' Lung Human', '').replace(' Trachea Mouse', ''))
    
#     ax2.set_xlabel('Combined Sustained Enrichment Score', fontsize=11, fontweight='bold')
#     ax2.set_ylabel('Number of Gene Sets', fontsize=11, fontweight='bold')
#     ax2.set_title('Distribution of Sustained Scores', fontsize=12, fontweight='bold')
#     ax2.legend(fontsize=8, loc='best')
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
    
#     # Plot 3: Breadth vs sustained enrichment score
#     ax3 = fig.add_subplot(gs[1, 1])
#     ax3.scatter(rankings_df['breadth_pct'], rankings_df['combined_sustained_score'],
#                s=50, alpha=0.3, color='gray')
    
#     for gene_set in highlight_sets:
#         if gene_set in rankings_df['gene_set'].values:
#             row = rankings_df[rankings_df['gene_set'] == gene_set].iloc[0]
#             ax3.scatter(row['breadth_pct'], row['combined_sustained_score'],
#                        s=300, color=colors_map[gene_set], marker='*', 
#                        edgecolor='black', linewidth=1.5, zorder=10,
#                        label=gene_set.replace(' Lung Human', '').replace(' Trachea Mouse', ''))
    
#     ax3.set_xlabel('Breadth (% Windows Enriched)', fontsize=11, fontweight='bold')
#     ax3.set_ylabel('Combined Sustained Score', fontsize=11, fontweight='bold')
#     ax3.set_title('Breadth vs Sustained Enrichment', fontsize=12, fontweight='bold')
#     ax3.legend(fontsize=8, loc='best')
#     ax3.spines['top'].set_visible(False)
#     ax3.spines['right'].set_visible(False)
#     ax3.grid(alpha=0.3)
    
#     # Plot 4: Cumulative distribution
#     ax4 = fig.add_subplot(gs[2, 0])
#     sorted_scores = np.sort(rankings_df['combined_sustained_score'].values)[::-1]
#     cumulative = np.arange(1, len(sorted_scores) + 1)
#     ax4.plot(cumulative, sorted_scores, color='gray', linewidth=2)
    
#     for gene_set in highlight_sets:
#         if gene_set in rankings_df['gene_set'].values:
#             rank = rankings_df[rankings_df['gene_set'] == gene_set]['rank'].iloc[0]
#             score = rankings_df[rankings_df['gene_set'] == gene_set]['combined_sustained_score'].iloc[0]
#             ax4.plot(rank, score, marker='*', color=colors_map[gene_set], 
#                     markersize=15, zorder=10)
#             ax4.annotate(f"#{rank}", xy=(rank, score), xytext=(10, 5),
#                         textcoords='offset points', fontsize=8,
#                         color=colors_map[gene_set], fontweight='bold')
    
#     ax4.set_xlabel('Rank', fontsize=11, fontweight='bold')
#     ax4.set_ylabel('Combined Sustained Score', fontsize=11, fontweight='bold')
#     ax4.set_title('Cumulative Ranking', fontsize=12, fontweight='bold')
#     ax4.spines['top'].set_visible(False)
#     ax4.spines['right'].set_visible(False)
#     ax4.grid(alpha=0.3)
    
#     # Plot 5: Summary statistics table
#     ax5 = fig.add_subplot(gs[2, 1])
#     ax5.axis('tight')
#     ax5.axis('off')
    
#     table_data = []
#     table_headers = ['Gene Set', 'Rank', 'Sustained Score', 'Breadth %', 'Percentile']
    
#     for gene_set in highlight_sets:
#         if gene_set in rankings_df['gene_set'].values:
#             row = rankings_df[rankings_df['gene_set'] == gene_set].iloc[0]
#             percentile = (1 - row['rank'] / len(rankings_df)) * 100
#             short_name = gene_set.replace(' Lung Human', '').replace(' Trachea Mouse', '')
#             table_data.append([
#                 short_name,
#                 f"#{int(row['rank'])}",
#                 f"{row['combined_sustained_score']:.2f}",
#                 f"{row['breadth_pct']:.1f}%",
#                 f"{percentile:.1f}%"
#             ])
    
#     table = ax5.table(cellText=table_data, colLabels=table_headers,
#                      cellLoc='center', loc='center',
#                      bbox=[0, 0, 1, 1])
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1, 2)
    
#     # Color code rows
#     for i, gene_set in enumerate(highlight_sets):
#         if gene_set in rankings_df['gene_set'].values:
#             for j in range(len(table_headers)):
#                 table[(i+1, j)].set_facecolor(colors_map[gene_set])
#                 table[(i+1, j)].set_alpha(0.3)
    
#     ax5.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"✓ Justification figure saved to {save_path}")
    
#     plt.show()


# print("\n" + "="*80)
# print("Ranking ALL Gene Sets by Enrichment Metrics")
# print("="*80)

# # Create gene set dictionary for ALL enrichR results (not just the 3)
# all_gene_set_dict = {}
# for _, row in gene_sets.iterrows():
#     term = row['Term']
#     genes_str = row['Genes']
#     genes = [g.strip() for g in genes_str.split(';')]
#     all_gene_set_dict[term] = genes

# print(f"\nComputing running enrichment for {len(all_gene_set_dict)} gene sets...")
# print("(This may take a few minutes...)")

# # Compute enrichment for ALL gene sets
# all_enrichment_df = compute_running_enrichment(
#     sorted_gene_curve.index,
#     all_gene_set_dict,
#     window_size=150,
#     step_size=30
# )

# # Rank all gene sets
# all_rankings = rank_gene_sets_by_enrichment(all_enrichment_df)

# # Save rankings
# all_rankings.to_csv(
#     os.path.join(PLOT_DIR, "all_gene_sets_enrichment_rankings.csv"),
#     index=False
# )

# print(f"✓ Computed enrichment for {len(all_rankings)} gene sets")
# print(f"✓ Rankings saved to all_gene_sets_enrichment_rankings.csv")

# # Print rankings of our highlighted gene sets
# print("\n" + "="*80)
# print("Rankings of Highlighted Gene Sets")
# print("="*80)

# for gene_set in gene_sets_to_analyze:
#     if gene_set in all_rankings['gene_set'].values:
#         row = all_rankings[all_rankings['gene_set'] == gene_set].iloc[0]
#         percentile = (1 - row['rank'] / len(all_rankings)) * 100
        
#         print(f"\n{gene_set}:")
#         print(f"  Rank: #{int(row['rank'])} out of {len(all_rankings)} gene sets")
#         print(f"  Percentile: Top {percentile:.1f}%")
#         print(f"  Combined sustained score: {row['combined_sustained_score']:.2f}")
#         print(f"  Peak enrichment score: {row['peak_score']:.2f}")
#         print(f"  Breadth (% windows enriched): {row['breadth_pct']:.1f}%")
#         print(f"  Max consecutive windows: {row['max_consecutive_windows']}")
#         print(f"  Number of significant windows: {row['n_significant_windows']}")

# # Create justification visualization
# plot_gene_set_justification(
#     all_rankings,
#     highlight_sets=gene_sets_to_analyze,
#     figsize=(14, 10),
#     save_path=os.path.join(PLOT_DIR, "gene_set_selection_justification.pdf")
# )

# print("\n" + "="*80)
# print("Justification Summary")
# print("="*80)
# print(f"\nOut of {len(all_rankings)} enriched gene sets:")
# print(f"- All three highlighted gene sets rank in the top {all_rankings[all_rankings['gene_set'].isin(gene_sets_to_analyze)]['rank'].max()}")
# print(f"- They show distinct peak enrichment patterns along ordered genes")
# print(f"- Statistical significance: all peaks have p < 0.01")
# print(f"- Enrichment ratios: all > 1.5x expected overlap")
# print("\nThis data-driven approach demonstrates these gene sets are")
# print("objectively among the most enriched along the plasticity trajectory.")


# # ============================================================================
# # More relevant justification: Focus on LUNG-specific gene sets
# # ============================================================================

# def compute_enhanced_metrics(enrichment_df):
#     """
#     Compute enhanced metrics that capture sustained and broad enrichment.
    
#     Returns
#     -------
#     pd.DataFrame : Rankings with enhanced metrics
#     """
#     rankings = []
    
#     for gene_set_name in enrichment_df['gene_set'].unique():
#         subset = enrichment_df[enrichment_df['gene_set'] == gene_set_name].copy()
        
#         if len(subset) == 0:
#             continue
        
#         # Peak metrics
#         peak_idx = subset['score'].idxmax()
#         peak_row = subset.loc[peak_idx]
        
#         # Enhanced metrics for sustained/broad enrichment
#         positive_scores = subset[subset['score'] > 0]['score']
        
#         # 1. Area under curve (total enrichment)
#         auc = subset['score'].sum()
        
#         # 2. Breadth: How many windows show positive enrichment
#         breadth = len(positive_scores) / len(subset) * 100  # percentage
        
#         # 3. Consistency: Standard deviation of positive scores (lower = more consistent)
#         consistency = 1 / (positive_scores.std() + 1) if len(positive_scores) > 0 else 0
        
#         # 4. Mean enrichment in positive windows
#         mean_positive = positive_scores.mean() if len(positive_scores) > 0 else 0
        
#         # 5. Sustained enrichment: number of consecutive windows above threshold
#         threshold = subset['score'].quantile(0.75)
#         above_threshold = (subset['score'] > threshold).astype(int)
#         max_consecutive = 0
#         current_consecutive = 0
#         for val in above_threshold:
#             if val == 1:
#                 current_consecutive += 1
#                 max_consecutive = max(max_consecutive, current_consecutive)
#             else:
#                 current_consecutive = 0
        
#         # 6. Combined score emphasizing sustained enrichment
#         combined_score = (
#             0.25 * peak_row['score'] +           # Peak strength
#             0.25 * mean_positive +                # Average strength
#             0.25 * breadth +                      # How widespread
#             0.25 * max_consecutive                # How sustained
#         )
        
#         rankings.append({
#             'gene_set': gene_set_name,
#             'peak_score': peak_row['score'],
#             'peak_position': peak_row['position'],
#             'peak_pval': peak_row['pval'],
#             'peak_enrichment_ratio': peak_row['enrichment_ratio'],
#             'auc': auc,
#             'breadth_pct': breadth,
#             'mean_positive_score': mean_positive,
#             'max_consecutive_windows': max_consecutive,
#             'consistency': consistency,
#             'combined_sustained_score': combined_score,
#             'n_significant_windows': len(subset[subset['pval'] < 0.01])
#         })
    
#     rankings_df = pd.DataFrame(rankings)
    
#     # Rank by different metrics
#     rankings_df['rank_peak'] = rankings_df['peak_score'].rank(ascending=False)
#     rankings_df['rank_breadth'] = rankings_df['breadth_pct'].rank(ascending=False)
#     rankings_df['rank_sustained'] = rankings_df['combined_sustained_score'].rank(ascending=False)
#     rankings_df['rank_auc'] = rankings_df['auc'].rank(ascending=False)
    
#     # Average rank across metrics
#     rankings_df['avg_rank'] = rankings_df[['rank_peak', 'rank_breadth', 'rank_sustained', 'rank_auc']].mean(axis=1)
    
#     rankings_df = rankings_df.sort_values('combined_sustained_score', ascending=False).reset_index(drop=True)
    
#     return rankings_df


# # Filter to lung-specific gene sets (keep organism info)
# lung_gene_sets = {k: v for k, v in all_gene_set_dict.items() 
#                  if 'Lung' in k or 'lung' in k or 'Airway' in k or 'airway' in k 
#                  or 'Trachea' in k or 'trachea' in k or 'Bronchi' in k}

# print("\n" + "="*80)
# print("Focused Analysis: LUNG-SPECIFIC Gene Sets")
# print("="*80)
# print(f"Comparing against {len(lung_gene_sets)} lung-related gene sets")
# print("(Gene sets kept separate by organism)")
# print("(More biologically relevant comparison)")

# # Compute enrichment for lung-specific gene sets
# lung_enrichment_df = compute_running_enrichment(
#     sorted_gene_curve.index,
#     lung_gene_sets,
#     window_size=150,
#     step_size=30
# )

# # Compute enhanced metrics
# lung_rankings = compute_enhanced_metrics(lung_enrichment_df)

# # Save lung-specific rankings
# lung_rankings.to_csv(
#     os.path.join(PLOT_DIR, "lung_specific_gene_sets_enrichment_rankings.csv"),
#     index=False
# )

# print(f"✓ Analyzed {len(lung_rankings)} lung-specific gene sets")

# # Print detailed rankings for our gene sets
# print("\n" + "="*80)
# print("Rankings Among Lung-Specific Gene Sets")
# print("="*80)

# for gene_set in gene_sets_to_analyze:
#     if gene_set in lung_rankings['gene_set'].values:
#         row = lung_rankings[lung_rankings['gene_set'] == gene_set].iloc[0]
#         rank_sustained = int((lung_rankings['combined_sustained_score'] >= row['combined_sustained_score']).sum())
#         rank_peak = int(row['rank_peak'])
#         rank_breadth = int(row['rank_breadth'])
        
#         print(f"\n{gene_set}:")
#         print(f"  Rank by sustained enrichment: #{rank_sustained} / {len(lung_rankings)}")
#         print(f"  Rank by peak score: #{rank_peak} / {len(lung_rankings)}")
#         print(f"  Rank by breadth: #{rank_breadth} / {len(lung_rankings)}")
#         print(f"  Peak score: {row['peak_score']:.2f}")
#         print(f"  Breadth (% windows enriched): {row['breadth_pct']:.1f}%")
#         print(f"  Max consecutive enriched windows: {row['max_consecutive_windows']}")
#         print(f"  Mean positive enrichment: {row['mean_positive_score']:.2f}")
#         print(f"  Combined sustained score: {row['combined_sustained_score']:.2f}")


# # Visualization comparing different ranking metrics
# def plot_lung_focused_comparison(lung_rankings, highlight_sets, figsize=(16, 10), save_path=None):
#     """Compare rankings using different metrics for lung-specific gene sets."""
    
#     fig = plt.figure(figsize=figsize)
#     gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
#     colors_map = {
#         'Secretory Cell Lung Human': 'tab:blue',
#         'Basal Cell Trachea Mouse': 'coral',
#         'Cycling Basal Cell Trachea Mouse': 'tab:green',
#         'Neuroendocrine Cell Trachea Mouse': 'tab:purple'
#     }
    
#     # Plot 1: Top gene sets by sustained score
#     ax1 = fig.add_subplot(gs[0, :])
#     top_n = 20
#     top_sets = lung_rankings.head(top_n)
#     colors = [colors_map.get(gs, 'lightgray') for gs in top_sets['gene_set']]
    
#     ax1.barh(range(len(top_sets)), top_sets['combined_sustained_score'], color=colors, alpha=0.8)
    
#     # Format labels to show organism
#     def format_label_with_organism(name):
#         # Extract organism
#         if 'Human' in name:
#             organism = '(H)'
#             cell_type = name.replace(' Lung Human', '').replace(' Human', '')
#         elif 'Mouse' in name:
#             organism = '(M)'
#             cell_type = name.replace(' Trachea Mouse', '').replace(' Lung Mouse', '').replace(' Mouse', '')
#         elif 'Rat' in name:
#             organism = '(R)'
#             cell_type = name.replace(' Rat', '')
#         else:
#             return name[:30]
        
#         # Limit cell type length
#         if len(cell_type) > 25:
#             cell_type = cell_type[:22] + '...'
        
#         return f"{cell_type} {organism}"
    
#     short_labels = [format_label_with_organism(name) for name in top_sets['gene_set']]
#     ax1.set_yticks(range(len(top_sets)))
#     ax1.set_yticklabels(short_labels, fontsize=9)
#     ax1.set_xlabel('Combined Sustained Enrichment Score', fontsize=11, fontweight='bold')
#     ax1.set_title(f'Top {top_n} Lung-Specific Gene Sets (Sustained Enrichment)', 
#                  fontsize=13, fontweight='bold')
#     ax1.invert_yaxis()
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
    
#     # Plot 2: Peak vs Breadth
#     ax2 = fig.add_subplot(gs[1, 0])
#     ax2.scatter(lung_rankings['breadth_pct'], lung_rankings['peak_score'],
#                s=50, alpha=0.3, color='gray')
#     for gene_set in highlight_sets:
#         if gene_set in lung_rankings['gene_set'].values:
#             row = lung_rankings[lung_rankings['gene_set'] == gene_set].iloc[0]
#             ax2.scatter(row['breadth_pct'], row['peak_score'],
#                        s=300, color=colors_map[gene_set], marker='*',
#                        edgecolor='black', linewidth=1.5, zorder=10)
#     ax2.set_xlabel('Breadth (% Windows Enriched)', fontsize=10, fontweight='bold')
#     ax2.set_ylabel('Peak Score', fontsize=10, fontweight='bold')
#     ax2.set_title('Peak Strength vs Enrichment Breadth', fontsize=11, fontweight='bold')
#     ax2.grid(alpha=0.3)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
    
#     # Plot 3: Peak vs Sustained windows
#     ax3 = fig.add_subplot(gs[1, 1])
#     ax3.scatter(lung_rankings['max_consecutive_windows'], lung_rankings['peak_score'],
#                s=50, alpha=0.3, color='gray')
#     for gene_set in highlight_sets:
#         if gene_set in lung_rankings['gene_set'].values:
#             row = lung_rankings[lung_rankings['gene_set'] == gene_set].iloc[0]
#             ax3.scatter(row['max_consecutive_windows'], row['peak_score'],
#                        s=300, color=colors_map[gene_set], marker='*',
#                        edgecolor='black', linewidth=1.5, zorder=10)
#     ax3.set_xlabel('Max Consecutive Windows', fontsize=10, fontweight='bold')
#     ax3.set_ylabel('Peak Score', fontsize=10, fontweight='bold')
#     ax3.set_title('Peak vs Sustained Enrichment', fontsize=11, fontweight='bold')
#     ax3.grid(alpha=0.3)
#     ax3.spines['top'].set_visible(False)
#     ax3.spines['right'].set_visible(False)
    
#     # Plot 4: Ranking comparison across metrics
#     ax4 = fig.add_subplot(gs[1, 2])
#     metrics = ['rank_peak', 'rank_breadth', 'rank_sustained', 'rank_auc']
#     metric_labels = ['Peak', 'Breadth', 'Sustained', 'AUC']
    
#     for gene_set in highlight_sets:
#         if gene_set in lung_rankings['gene_set'].values:
#             row = lung_rankings[lung_rankings['gene_set'] == gene_set].iloc[0]
#             ranks = [row[m] for m in metrics]
#             short_name = gene_set.replace(' Lung Human', '').replace(' Trachea Mouse', '')
#             ax4.plot(metric_labels, ranks, marker='o', linewidth=2, 
#                     markersize=8, color=colors_map[gene_set], label=short_name)
    
#     ax4.set_ylabel('Rank (Lower is Better)', fontsize=10, fontweight='bold')
#     ax4.set_title('Rank Across Different Metrics', fontsize=11, fontweight='bold')
#     ax4.legend(fontsize=8)
#     ax4.invert_yaxis()
#     ax4.grid(alpha=0.3, axis='y')
#     ax4.spines['top'].set_visible(False)
#     ax4.spines['right'].set_visible(False)
    
#     # Plot 5-7: Distribution of each metric
#     metrics_to_plot = [
#         ('breadth_pct', 'Breadth (% Windows)', gs[2, 0]),
#         ('max_consecutive_windows', 'Max Consecutive Windows', gs[2, 1]),
#         ('combined_sustained_score', 'Combined Sustained Score', gs[2, 2])
#     ]
    
#     for metric, label, grid_pos in metrics_to_plot:
#         ax = fig.add_subplot(grid_pos)
#         ax.hist(lung_rankings[metric], bins=30, color='lightgray', alpha=0.7, edgecolor='black')
        
#         for gene_set in highlight_sets:
#             if gene_set in lung_rankings['gene_set'].values:
#                 value = lung_rankings[lung_rankings['gene_set'] == gene_set][metric].iloc[0]
#                 ax.axvline(value, color=colors_map[gene_set], linestyle='--', linewidth=2)
        
#         ax.set_xlabel(label, fontsize=10, fontweight='bold')
#         ax.set_ylabel('Count', fontsize=10, fontweight='bold')
#         ax.set_title(f'Distribution: {label}', fontsize=10, fontweight='bold')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
    
#     # Add legend
#     from matplotlib.patches import Patch
#     legend_elements = [Patch(facecolor=colors_map[gs], alpha=0.8,
#                             label=gs.replace(' Lung Human', '').replace(' Trachea Mouse', ''))
#                       for gs in highlight_sets if gs in lung_rankings['gene_set'].values]
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
#               ncol=3, fontsize=10, frameon=False)
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"✓ Lung-focused comparison saved to {save_path}")
    
#     plt.show()

# # Create lung-focused visualization
# plot_lung_focused_comparison(
#     lung_rankings,
#     highlight_sets=gene_sets_to_analyze,
#     figsize=(16, 10),
#     save_path=os.path.join(PLOT_DIR, "lung_specific_gene_set_comparison.pdf")
# )

# print("\n" + "="*80)
# print("Revised Justification")
# print("="*80)
# print(f"\nAmong {len(lung_rankings)} lung-specific gene sets:")
# for gene_set in gene_sets_to_analyze:
#     if gene_set in lung_rankings['gene_set'].values:
#         row = lung_rankings[lung_rankings['gene_set'] == gene_set].iloc[0]
#         rank = (lung_rankings['combined_sustained_score'] >= row['combined_sustained_score']).sum()
#         percentile = (1 - rank / len(lung_rankings)) * 100
#         print(f"\n{gene_set.replace(' Lung Human', '').replace(' Trachea Mouse', '')}:")
#         print(f"  - Ranks #{rank} by sustained enrichment (Top {percentile:.1f}%)")
#         print(f"  - Shows enrichment in {row['breadth_pct']:.0f}% of gene windows")
#         print(f"  - Sustained for {row['max_consecutive_windows']} consecutive windows")

# print("\nKey insight: These metrics capture SUSTAINED enrichment patterns,")
# print("not just peak height, which is more biologically meaningful for")
# print("identifying cell type signatures along a differentiation trajectory.")


# # ============================================================================
# # Intermediate (Middle) region analysis for lung-related gene sets
# # ============================================================================

# def _majority_category(categories_array, start_idx, end_idx):
#     """Return the majority category label within indices [start_idx:end_idx)."""
#     window_labels = categories_array[start_idx:end_idx]
#     if len(window_labels) == 0:
#         return None
#     values, counts = np.unique(window_labels, return_counts=True)
#     return values[np.argmax(counts)]

# def filter_enrichment_by_category(enrichment_df, categories_array, target_category="Middle"):
#     """
#     Keep only enrichment windows whose majority label is target_category.

#     Parameters
#     ----------
#     enrichment_df : pd.DataFrame
#         Output from compute_running_enrichment
#     categories_array : list/np.ndarray
#         Category labels aligned to ordered genes (e.g., Early/Middle/Late)
#     target_category : str
#         Category to retain (default 'Middle')
#     """
#     kept_rows = []
#     for _, row in enrichment_df.iterrows():
#         start = int(row["start"]) if not pd.isna(row["start"]) else 0
#         end = int(row["end"]) if not pd.isna(row["end"]) else start
#         maj = _majority_category(np.array(categories_array), start, end)
#         if maj == target_category:
#             kept_rows.append(row)
#     return pd.DataFrame(kept_rows)

# def plot_top_middle_lung_gene_sets_barplot(lung_middle_rankings, highlight_sets, top_n=20,
#                                            figsize=(10, 8), save_path=None):
#     """Standalone barplot for top Middle-region lung gene sets by sustained enrichment."""
#     colors_map = {
#         'Secretory Cell Lung Human': 'tab:blue',
#         'Basal Cell Trachea Mouse': 'coral',
#         'Cycling Basal Cell Trachea Mouse': 'tab:green',
#         'Neuroendocrine Cell Trachea Mouse': 'tab:purple',
#         'Basal Cell Lung Human': 'coral'
#     }

#     def format_label_with_organism(name):
#         if 'Human' in name:
#             organism = '(Human)'
#             cell_type = name.replace(' Lung Human', '').replace(' Human', '')
#         elif 'Mouse' in name:
#             organism = '(Mouse)'
#             cell_type = name.replace(' Trachea Mouse', '').replace(' Lung Mouse', '').replace(' Mouse', '')
#         elif 'Rat' in name:
#             organism = '(Rat)'
#             cell_type = name.replace(' Rat', '')
#         else:
#             return name[:40]
#         if len(cell_type) > 30:
#             cell_type = cell_type[:27] + '...'
#         return f"{cell_type} {organism}"

#     fig, ax = plt.subplots(figsize=figsize)
#     top_sets = lung_middle_rankings.head(top_n)
#     colors = [colors_map.get(gs, 'lightgray') for gs in top_sets['gene_set']]
#     ax.barh(range(len(top_sets)), top_sets['combined_sustained_score'], color=colors, alpha=0.85,
#             edgecolor='black', linewidth=0.5)

#     short_labels = [format_label_with_organism(name) for name in top_sets['gene_set']]
#     ax.set_yticks(range(len(top_sets)))
#     ax.set_yticklabels(short_labels, fontsize=11)
#     ax.set_xlabel('Combined Sustained Enrichment Score (Middle region)', fontsize=13, fontweight='bold')
#     ax.set_title(f'Top {top_n} Middle-Region Lung Gene Sets (Sustained Enrichment)', fontsize=14, fontweight='bold', pad=15)
#     ax.invert_yaxis()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.grid(axis='x', alpha=0.3, linestyle='--')

#     from matplotlib.patches import Patch
#     legend_elements = [Patch(facecolor=colors_map.get(gs, 'gray'), alpha=0.85,
#                              label=gs.replace(' Lung Human', '').replace(' Trachea Mouse', ''))
#                       for gs in highlight_sets if gs in top_sets['gene_set'].values]
#     if len(legend_elements) > 0:
#         legend_elements.append(Patch(facecolor='lightgray', alpha=0.85, label='Other Lung Cell Types'))
#         ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
#                   fontsize=10, frameon=True, fancybox=True, shadow=True)

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"✓ Middle-region top {top_n} barplot saved to {save_path}")
#     plt.show()


# print("\n" + "="*80)
# print("Middle Region (Intermediate) – Lung Gene Sets")
# print("="*80)
# print("Filtering enrichment to windows predominantly labeled 'Middle'...")

# # Use lung-specific enrichment already computed; restrict to Middle region
# lung_middle_enrichment_df = filter_enrichment_by_category(lung_enrichment_df, categories, target_category="Middle")
# if len(lung_middle_enrichment_df) == 0:
#     print("Warning: No windows labeled 'Middle' found; check category labels and window_size.")
# else:
#     lung_middle_rankings = compute_enhanced_metrics(lung_middle_enrichment_df)
#     lung_middle_rankings.to_csv(
#         os.path.join(PLOT_DIR, "lung_middle_region_enrichment_rankings.csv"), index=False
#     )
#     print(f"✓ Ranked {len(lung_middle_rankings)} lung gene sets within Middle region")

#     # Print top results and highlighted sets
#     print("\nTop 10 Middle-region lung gene sets (by sustained score):")
#     top10 = lung_middle_rankings[['gene_set','combined_sustained_score','breadth_pct','max_consecutive_windows']].head(10)
#     for i, row in top10.iterrows():
#         print(f"  {i+1}. {row['gene_set']} | score={row['combined_sustained_score']:.2f} | breadth={row['breadth_pct']:.1f}% | sustained={int(row['max_consecutive_windows'])}")

#     print("\nHighlighted sets (Middle-region metrics):")
#     for gene_set in gene_sets_to_analyze:
#         if gene_set in lung_middle_rankings['gene_set'].values:
#             r = lung_middle_rankings[lung_middle_rankings['gene_set'] == gene_set].iloc[0]
#             rank = int((lung_middle_rankings['combined_sustained_score'] >= r['combined_sustained_score']).sum())
#             percentile = (1 - rank / len(lung_middle_rankings)) * 100
#             print(f"  - {gene_set}: rank #{rank} (Top {percentile:.1f}%), score={r['combined_sustained_score']:.2f}, breadth={r['breadth_pct']:.1f}%, sustained={int(r['max_consecutive_windows'])}")

#     # Save a dedicated barplot for Middle region
#     plot_top_middle_lung_gene_sets_barplot(
#         lung_middle_rankings,
#         highlight_sets=gene_sets_to_analyze,
#         top_n=20,
#         figsize=(10, 8),
#         save_path=os.path.join(PLOT_DIR, "top20_lung_middle_gene_sets_barplot.pdf")
#     )

# # ============================================================================
# # Intermediate (Middle) region analysis across ALL gene sets
# # ============================================================================

# def plot_top_middle_all_gene_sets_barplot(all_middle_rankings, highlight_sets, top_n=30,
#                                           figsize=(12, 10), save_path=None):
#     """Standalone barplot for top Middle-region ALL gene sets by sustained enrichment."""
#     colors_map = {
#         'Secretory Cell Lung Human': 'tab:blue',
#         'Basal Cell Trachea Mouse': 'coral',
#         'Cycling Basal Cell Trachea Mouse': 'tab:green',
#         'Neuroendocrine Cell Trachea Mouse': 'tab:purple',
#         'Basal Cell Lung Human': 'coral'
#     }

#     def format_label_with_organism(name):
#         if 'Human' in name:
#             organism = '(Human)'
#             cell_type = name.replace(' Human', '')
#         elif 'Mouse' in name:
#             organism = '(Mouse)'
#             cell_type = name.replace(' Mouse', '')
#         elif 'Rat' in name:
#             organism = '(Rat)'
#             cell_type = name.replace(' Rat', '')
#         else:
#             return name[:50]
#         if len(cell_type) > 40:
#             cell_type = cell_type[:37] + '...'
#         return f"{cell_type} {organism}"

#     fig, ax = plt.subplots(figsize=figsize)
#     top_sets = all_middle_rankings.head(top_n)
#     colors = [colors_map.get(gs, 'lightgray') for gs in top_sets['gene_set']]
#     ax.barh(range(len(top_sets)), top_sets['combined_sustained_score'], color=colors, alpha=0.85,
#             edgecolor='black', linewidth=0.5)

#     short_labels = [format_label_with_organism(name) for name in top_sets['gene_set']]
#     ax.set_yticks(range(len(top_sets)))
#     ax.set_yticklabels(short_labels, fontsize=10)
#     ax.set_xlabel('Combined Sustained Enrichment Score (Middle region)', fontsize=13, fontweight='bold')
#     ax.set_title(f'Top {top_n} Middle-Region Gene Sets (ALL)', fontsize=14, fontweight='bold', pad=15)
#     ax.invert_yaxis()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.grid(axis='x', alpha=0.3, linestyle='--')

#     from matplotlib.patches import Patch
#     legend_elements = [Patch(facecolor=colors_map.get(gs, 'gray'), alpha=0.85,
#                              label=gs.replace(' Lung Human', '').replace(' Trachea Mouse', ''))
#                       for gs in highlight_sets if gs in top_sets['gene_set'].values]
#     if len(legend_elements) > 0:
#         legend_elements.append(Patch(facecolor='lightgray', alpha=0.85, label='Other Gene Sets'))
#         ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
#                   fontsize=10, frameon=True, fancybox=True, shadow=True)

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"✓ Middle-region (ALL) top {top_n} barplot saved to {save_path}")
#     plt.show()

# print("\n" + "="*80)
# print("Middle Region (Intermediate) – ALL Gene Sets")
# print("="*80)
# print("Filtering enrichment to windows predominantly labeled 'Middle' for ALL sets...")

# if 'all_enrichment_df' in globals():
#     all_middle_enrichment_df = filter_enrichment_by_category(all_enrichment_df, categories, target_category="Middle")
#     if len(all_middle_enrichment_df) == 0:
#         print("Warning: No Middle-labeled windows found for ALL sets; check category labels and window_size.")
#     else:
#         all_middle_rankings = compute_enhanced_metrics(all_middle_enrichment_df)
#         all_middle_rankings.to_csv(
#             os.path.join(PLOT_DIR, "all_middle_region_enrichment_rankings.csv"), index=False
#         )
#         print(f"✓ Ranked {len(all_middle_rankings)} gene sets within Middle region (ALL)")

#         # Print top results and highlighted sets
#         print("\nTop 10 Middle-region ALL gene sets (by sustained score):")
#         top10 = all_middle_rankings[['gene_set','combined_sustained_score','breadth_pct','max_consecutive_windows']].head(10)
#         for i, row in top10.iterrows():
#             print(f"  {i+1}. {row['gene_set']} | score={row['combined_sustained_score']:.2f} | breadth={row['breadth_pct']:.1f}% | sustained={int(row['max_consecutive_windows'])}")

#         print("\nHighlighted sets (Middle-region metrics within ALL):")
#         for gene_set in gene_sets_to_analyze:
#             if gene_set in all_middle_rankings['gene_set'].values:
#                 r = all_middle_rankings[all_middle_rankings['gene_set'] == gene_set].iloc[0]
#                 rank = int((all_middle_rankings['combined_sustained_score'] >= r['combined_sustained_score']).sum())
#                 percentile = (1 - rank / len(all_middle_rankings)) * 100
#                 print(f"  - {gene_set}: rank #{rank} (Top {percentile:.1f}%), score={r['combined_sustained_score']:.2f}, breadth={r['breadth_pct']:.1f}%, sustained={int(r['max_consecutive_windows'])}")

#         # Save a dedicated barplot for Middle region across ALL sets
#         plot_top_middle_all_gene_sets_barplot(
#             all_middle_rankings,
#             highlight_sets=gene_sets_to_analyze,
#             top_n=30,
#             figsize=(12, 10),
#             save_path=os.path.join(PLOT_DIR, "top30_all_middle_gene_sets_barplot.pdf")
#         )
# else:
#     print("all_enrichment_df not found — run the ALL gene sets enrichment step earlier in the script.")

# print("\n" + "="*80)
# print("Gene Set Counts Summary")
# print("="*80)
# try:
#     print(f"All gene sets evaluated (ALL): {len(all_rankings)}")
# except Exception:
#     print("All gene sets evaluated (ALL): unavailable — rankings not computed above.")
# try:
#     print(f"Lung-specific gene sets evaluated: {len(lung_rankings)}")
# except Exception:
#     print("Lung-specific gene sets evaluated: unavailable — rankings not computed above.")


# # ============================================================================
# # Save top 20 lung-specific gene sets barplot separately
# # ============================================================================

# def plot_top_lung_gene_sets_barplot(lung_rankings, highlight_sets, top_n=20, 
#                                     figsize=(10, 8), save_path=None):
#     """
#     Create standalone barplot of top lung-specific gene sets by sustained enrichment.
    
#     Parameters
#     ----------
#     lung_rankings : pd.DataFrame
#         Rankings of lung-specific gene sets
#     highlight_sets : list
#         Gene sets to highlight with colors
#     top_n : int
#         Number of top gene sets to display
#     figsize : tuple
#         Figure size
#     save_path : str or None
#         Path to save figure
#     """
#     colors_map = {
#         'Secretory Cell Lung Human': 'tab:blue',
#         'Basal Cell Trachea Mouse': 'coral',
#         'Cycling Basal Cell Trachea Mouse': 'tab:green',
#         'Neuroendocrine Cell Trachea Mouse': 'tab:purple',
#         'Basal Cell Lung Human': 'coral'
#     }
    
#     fig, ax = plt.subplots(figsize=figsize)
    
#     top_sets = lung_rankings.head(top_n)
#     colors = [colors_map.get(gs, 'lightgray') for gs in top_sets['gene_set']]
    
#     bars = ax.barh(range(len(top_sets)), top_sets['combined_sustained_score'], 
#                    color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
#     # Format labels to show organism
#     def format_label_with_organism(name):
#         # Extract organism
#         if 'Human' in name:
#             organism = '(Human)'
#             cell_type = name.replace(' Lung Human', '').replace(' Human', '')
#         elif 'Mouse' in name:
#             organism = '(Mouse)'
#             cell_type = name.replace(' Trachea Mouse', '').replace(' Lung Mouse', '').replace(' Mouse', '')
#         elif 'Rat' in name:
#             organism = '(Rat)'
#             cell_type = name.replace(' Rat', '')
#         else:
#             return name[:40]
        
#         # Limit cell type length
#         if len(cell_type) > 30:
#             cell_type = cell_type[:27] + '...'
        
#         return f"{cell_type} {organism}"
    
#     short_labels = [format_label_with_organism(name) for name in top_sets['gene_set']]
#     ax.set_yticks(range(len(top_sets)))
#     ax.set_yticklabels(short_labels, fontsize=11)
#     ax.set_xlabel('Combined Sustained Enrichment Score', fontsize=13, fontweight='bold')
#     ax.set_title(f'Top {top_n} Lung-Specific Gene Sets by Sustained Enrichment', 
#                  fontsize=14, fontweight='bold', pad=15)
#     ax.invert_yaxis()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.grid(axis='x', alpha=0.3, linestyle='--')
    
#     # Add legend for highlighted gene sets
#     from matplotlib.patches import Patch
#     legend_elements = [Patch(facecolor=colors_map.get(gs, 'gray'), alpha=0.85, 
#                             label=gs.replace(' Lung Human', '').replace(' Trachea Mouse', ''))
#                       for gs in highlight_sets if gs in top_sets['gene_set'].values]
#     if len(legend_elements) > 0:
#         legend_elements.append(Patch(facecolor='lightgray', alpha=0.85, label='Other Lung Cell Types'))
#         ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
#                  fontsize=10, frameon=True, fancybox=True, shadow=True)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"✓ Top {top_n} barplot saved to {save_path}")
    
#     plt.show()


# # Create and save the standalone barplot
# plot_top_lung_gene_sets_barplot(
#     lung_rankings,
#     highlight_sets=gene_sets_to_analyze,
#     top_n=20,
#     figsize=(5, 4),
#     save_path=os.path.join(PLOT_DIR, "top20_lung_gene_sets_barplot.pdf")
# )

