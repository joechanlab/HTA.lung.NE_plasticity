#!/usr/bin/env python

import argparse
import os
import warnings

import h5py  # required by mudata / scanpy in some environments
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, rgb2hex, LinearSegmentedColormap
import mudata
import numpy as np
import pandas as pd
import pyBigWig
import pyranges as pr
import scanpy as sc
import scenicplus
from scenicplus.plotting import coverageplot
from scenicplus.utils import get_interaction_pr
from scenicplus.scenicplus_class import mudata_to_scenicplus
warnings.filterwarnings("ignore")

def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=256):
    """
    Create a truncated version of a matplotlib colormap.
    """
    base_cmap = plt.get_cmap(cmap_name)
    new_colors = base_cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})", new_colors
    )


def get_gene_body(gene, genome_annotation_df):
    """
    Select a single gene body region (chrom:start-end) for the given gene.
    If multiple entries exist, uses the median-length gene.
    """
    gene_annotation = genome_annotation_df[genome_annotation_df["gene_name"] == gene]
    bodies = []
    lengths = []
    for _, row in gene_annotation.iterrows():
        chrom, start, end = row["Chromosome"], row["Start"], row["End"]
        body = f"{chrom}:{start}-{end}"
        if body not in bodies:
            lengths.append(end - start)
            bodies.append(body)
    if len(bodies) > 1:
        order = np.argsort(lengths)
        bodies = [bodies[i] for i in order]
    if not bodies:
        raise ValueError(f"No gene body found for gene {gene}")
    return bodies[len(bodies) // 2]


def extend(region, left, right):
    chrom, start_end = region.split(":")
    start, end = map(int, start_end.split("-"))
    return f"{chrom}:{start - left}-{end + right}"


def get_chrom_start_end(region):
    chrom, start_end = region.split(":")
    start, end = map(int, start_end.split("-"))
    return chrom, start, end


def min_max_normalize(group):
    """
    Signed min–max normalization used for regulon importance.
    """
    if len(group) == 1:
        return group
    sign = np.sign(group)
    abs_vals = np.abs(group)
    return sign * (abs_vals - abs_vals.min()) / (abs_vals.max() - abs_vals.min())

def compute_da_peaks_scplus(
    scplus_mudata,
    celltype_key="cell_type",
    method="wilcoxon",
    min_logfc=0.25,
    max_qval=0.05,
    min_pct=0.05,
    groups_of_interest=None,
    layer=None,
):
    """
    Genome-wide differential accessibility (DA) for scATAC peaks
    using scanpy.tl.rank_genes_groups on the scATAC AnnData in a SCENIC+ MuData.

    Parameters
    ----------
    scplus_mudata : MuData
        SCENIC+ MuData object with 'scATAC_counts' modality.
    celltype_key : str
        Column in scplus_mudata['scATAC_counts'].obs with cell-type labels.
    method : {'wilcoxon','logreg','t-test','t-test_overestim_var'}
        Method passed to scanpy.tl.rank_genes_groups.
    min_logfc : float
        Minimum log2 fold-change for calling a peak as DA (only applied when building the result table).
    max_qval : float
        Maximum adjusted p value (Benjamini–Hochberg) for calling a peak as DA.
    min_pct : float
        Minimum fraction of cells with peak accessible in the group (pts) to keep.
    groups_of_interest : list or None
        Subset of cell types to test. If None, uses all categories in celltype_key.
    layer : str or None
        If scATAC counts are stored in a layer; otherwise uses .X.

    Returns
    -------
    da_df : pandas.DataFrame
        Long-format table with columns:
        ['peak', 'cell_type', 'logFC', 'pval', 'p_adj', 'pts', 'pts_rest'].
        One row per (peak, cell_type) DA result.
    """

    # 1. Extract scATAC AnnData
    atac = scplus_mudata["scATAC_counts"].copy()

    # If counts are in a layer (e.g. 'counts'), move them to X for scanpy
    if layer is not None and layer in atac.layers:
        atac.X = atac.layers[layer]
    atac.raw = None

    # (Optional) binarize accessibility; often done for DA on peaks
    # comment this out if you want raw counts
    if hasattr(atac.X, "toarray"):
        atac.X = (atac.X > 0).astype(np.float32)
    else:
        atac.X = (atac.X > 0).astype(np.float32)

    # 2. Ensure categorical cell-type column
    if celltype_key not in atac.obs:
        raise KeyError(f"'{celltype_key}' not found in atac.obs")
    atac.obs[celltype_key] = atac.obs[celltype_key].astype("category")

    all_groups = atac.obs[celltype_key].cat.categories.tolist()
    if groups_of_interest is None:
        groups_of_interest = all_groups

    results = []

    # 3. Run DA for each group vs rest, genome-wide
    for group in groups_of_interest:
        print(f"Computing DA peaks for {group} vs rest...")
        sc.tl.rank_genes_groups(
            atac,
            groupby=celltype_key,
            groups=[group],
            reference="rest",
            method=method,
            pts=True,
            use_raw=False,
        )
        df = sc.get.rank_genes_groups_df(atac, group=group)
        df = df.rename(
            columns={
                "names": "peak",
                "logfoldchanges": "logFC",
                "pvals": "pval",
                "pvals_adj": "p_adj",
                "scores": "score",
                "pct_nz_group": "pts",
                "pct_nz_reference": "pts_rest",
            }
        )

        df["cell_type"] = group

        df = df[
            (df["logFC"] >= min_logfc)
            & (df["p_adj"] <= max_qval)
            & (df["pts"] >= min_pct)
        ]

        results.append(df[["peak", "cell_type", "logFC", "pval", "p_adj", "pts", "pts_rest"]])

    da_df = pd.concat(results, ignore_index=True)
    return da_df

def summarize_da_peaks_by_gene(
    da_df,
    region_to_gene_df,
    celltype_of_interest,
):
    """
    Summarize DA peaks at the gene level using region_to_gene links.

    Parameters
    ----------
    da_df : DataFrame
        Output of compute_da_peaks_scplus, with columns
        ['peak', 'cell_type', 'logFC', 'pval', 'p_adj', 'pts', 'pts_rest'].
    region_to_gene_df : DataFrame
        region_to_gene_adj.tsv loaded as a DataFrame; must contain
        columns ['region', 'target', 'importance_R2G'].
    celltype_of_interest : str
        Cell type whose DA peaks you want to summarize.

    Returns
    -------
    gene_summary : DataFrame
        Indexed by gene (target), with columns:
        ['n_da_peaks', 'max_logFC', 'mean_logFC', 'max_importance_R2G'].
        Sorted by n_da_peaks desc, then max_logFC desc.
    """
    # subset DA results to the cell type of interest
    da_sub = da_df[da_df["cell_type"] == celltype_of_interest].copy()
    base_cols = ["region", "target"]
    if da_sub.empty:
        raise ValueError(
            f"No DA peaks found in da_df for cell_type == {celltype_of_interest}."
        )
    if "importance" in region_to_gene_df.columns:
        importance_col = "importance"
    elif "importance_R2G" in region_to_gene_df.columns:
        importance_col = "importance_R2G"
    else:
        importance_col = None

    if importance_col is not None:
        rtg = region_to_gene_df[base_cols + [importance_col]].copy()
        # standardize the name so downstream code is simpler
        rtg = rtg.rename(columns={importance_col: "importance_R2G"})
    else:
        rtg = region_to_gene_df[base_cols].copy()
        rtg["importance_R2G"] = np.nan
        print(
            "[WARN] No 'importance' or 'importance_R2G' column found in "
            "region_to_gene_df; filling importance with NaN."
        )
    # annotate DA peaks with linked genes
    annot = da_sub.merge(
        rtg,
        left_on="peak",
        right_on="region",
        how="inner",
    )

    if annot.empty:
        raise ValueError(
            f"No DA peaks for {celltype_of_interest} overlap region_to_gene links."
        )
        return pd.DataFrame()

    # summarize per gene
    gene_summary = (
        annot.groupby("target")
        .agg(
            n_da_peaks=("peak", "nunique"),
            max_logFC=("logFC", "max"),
            mean_logFC=("logFC", "mean"),
            max_importance_R2G=("importance_R2G", "max"),
        )
        .sort_values(["n_da_peaks", "max_logFC"], ascending=[False, False])
    )

    return gene_summary

# --------- global objects filled by setup_context() --------- #
scplus_mudata = None
ensgene_df = None
regulon_data = None
consensus_bed_pr = None
scplus_obj = None
interaction_df = None
all_regulon_regions = None
bw_dict = None


def setup_context(
    scplus_res_dir,
    gtf_path,
    consensus_bed_path,
    bw_dir,
    meta_data_key="cell_type",
    rename_nonne_to_adc=False,
    celltype_order=None,
    celltype_of_interest="ADC",
    min_logfc=0.25,
    max_qval=0.05,
    min_pct=0.05,
):
    """
    Load all necessary objects (MuData, annotations, bigWigs, interactions, regulons)
    to be used by plot_coverage().
    """
    global scplus_mudata, ensgene_df, regulon_data, consensus_bed_pr
    global scplus_obj, interaction_df, all_regulon_regions, bw_dict

    # ---- SCENIC+ MuData ---- #
    scplus_mudata = mudata.read(os.path.join(scplus_res_dir, "scplusmdata.h5mu"))

    # ---- region-to-gene mapping ---- #
    region_to_gene_df = pd.read_csv(
        os.path.join(scplus_res_dir, "region_to_gene_adj.tsv"), sep="\t"
    )

    # ---- gene annotation from GTF (used for gene bodies + plotting) ---- #
    ensgene_pr = pr.read_gtf(gtf_path)
    ensgene_df = ensgene_pr.df
    ensgene_df = ensgene_df[ensgene_df["Feature"] == "gene"]
    ensgene_df["transcript_type"] = "protein_coding"

    # ---- consensus peaks ---- #
    consensus_bed_pr = pr.read_bed(consensus_bed_path)
    # Rename NonNE -> ADC in peak names (optional; matches your notebook)
    if "Name" in consensus_bed_pr.columns:
        consensus_df = consensus_bed_pr.df
        consensus_df["Name"] = consensus_df["Name"].str.replace("NonNE", "ADC")
        consensus_bed_pr = pr.PyRanges(consensus_df)

    # ---- bigWig dictionary ---- #
    cell_types = scplus_mudata["scRNA_counts"].obs[meta_data_key].unique().tolist()
    bw_dict = {}
    for ct in cell_types:
        bw_path = os.path.join(bw_dir, f"{ct}.bw")
        if os.path.exists(bw_path):
            bw_dict[ct] = bw_path

    if rename_nonne_to_adc:
        # Rename in scRNA_counts (you already had this logic)
        if meta_data_key in scplus_mudata["scRNA_counts"].obs:
            scplus_mudata["scRNA_counts"].obs[meta_data_key] = (
                scplus_mudata["scRNA_counts"].obs[meta_data_key]
                .replace({"NonNE": "ADC"})
            )

        # NEW: rename in scATAC_counts as well
        if "scATAC_counts" in scplus_mudata.mod:
            if meta_data_key in scplus_mudata["scATAC_counts"].obs:
                scplus_mudata["scATAC_counts"].obs[meta_data_key] = (
                    scplus_mudata["scATAC_counts"].obs[meta_data_key]
                    .replace({"NonNE": "ADC"})
                )

    # ---- apply user-specified order, if any ---- #
    if celltype_order is not None:
        ordered = [ct for ct in celltype_order if ct in bw_dict]
        # optional: sanity check / warning if some requested types not found
        if len(ordered) == 0:
            raise ValueError(
                f"None of the requested cell types {celltype_order} found in bw_dict keys {list(bw_dict.keys())}"
            )
        bw_dict = {ct: bw_dict[ct] for ct in ordered}
    
    # ---- regulon information ---- #
    regulon_data = scplus_mudata.uns["direct_e_regulon_metadata"]

    # Add direct_e_regulons list expected by SCENIC+
    groupby_regulon = regulon_data.groupby("eRegulon_name")
    direct_e_regulons = []
    for group_name, group_df in groupby_regulon:
        e_regulon_df = pd.DataFrame(group_df["Region"])
        e_regulon_df.columns = ["target_regions"]
        e_regulon_df["eRegulon_name"] = group_name
        direct_e_regulons.append(e_regulon_df)
    scplus_mudata.uns["direct_e_regulons"] = direct_e_regulons

    # Add convenience columns for interactions
    regulon_data["region"] = regulon_data["Region"]
    regulon_data["rho"] = regulon_data["rho_R2G"]
    regulon_data["importance"] = regulon_data.groupby("Gene")["importance_R2G"].transform(
        min_max_normalize
    )

    # region_to_gene_df index for sign retrieval
    region_to_gene_df.index = region_to_gene_df.target + "_" + region_to_gene_df.region

    # ---- interactions ---- #
    scplus_mudata.region_names = scplus_mudata["scATAC_counts"].var_names
    scplus_mudata.uns["region_to_gene"] = region_to_gene_df

    blues = truncate_colormap("Blues")
    reds = truncate_colormap("Reds")

    interaction_pr = get_interaction_pr(
        SCENICPLUS_obj=scplus_mudata,
        species="hsapiens",
        assembly="hg38",
        region_to_gene_key="direct_e_regulon_metadata",
        eRegulons_key="direct_e_regulons",
        inplace=False,
        cmap_pos=blues,
        cmap_neg=reds,
        scale_by_gene=False,
    )
    interaction_df_local = interaction_pr.df
    if "value" not in interaction_df_local.columns:
        print("[WARN] 'value' column missing in interaction_df; setting all values to 1.0")
        interaction_df_local["value"] = 1.0

    def get_interaction_sign(row):
        region = row["Chromosome"] + ":" + str(row["Start"]) + "-" + str(row["End"])
        gene = row["name"].split("_")[0]
        return np.sign(region_to_gene_df.loc[f"{gene}_{region}"]["rho"])

    interaction_df_local["sign"] = interaction_df_local.apply(get_interaction_sign, axis=1)
    interaction_df_local["distance"] = (
        interaction_df_local["sourceStart"].astype(int)
        - interaction_df_local["targetStart"].astype(int)
    )

    interaction_df = interaction_df_local
    all_regulon_regions = regulon_data["Region"].unique()

    # ---- SCENIC+ object ---- #
    scplus_obj = mudata_to_scenicplus(scplus_mudata)
    if isinstance(scplus_obj.X_EXP, pd.DataFrame):
        scplus_obj.X_EXP = scplus_obj.X_EXP.values

    # ------ DA analysis -------#
    da_df = compute_da_peaks_scplus(
        scplus_mudata=scplus_mudata,
        celltype_key=meta_data_key,
        method="wilcoxon",
        min_logfc=min_logfc,
        max_qval=max_qval,
        min_pct=min_pct,
        groups_of_interest=[celltype_of_interest],
    )

    # store globals
    globals().update(
        dict(
            scplus_mudata=scplus_mudata,
            ensgene_df=ensgene_df,
            regulon_data=regulon_data,
            consensus_bed_pr=consensus_bed_pr,
            scplus_obj=scplus_obj,
            interaction_df=interaction_df,
            all_regulon_regions=all_regulon_regions,
            bw_dict=bw_dict,
            da_df=da_df,
        )
    )


def plot_coverage(
    gene,
    region=None,
    regulon=None,
    prune_consensus_bed=True,
    subset_gtf_to_gene=True,
    subset_interact_to_gene=True,
    color_regulon_peaks=True,
    figsize=(14, 8),
    fontsize_dict=None,
    bw_ymax=None,
    region_bed_height=1.0,
    arc_rad=0.5,
    activating_only=False,
    height_ratios_dict=None,
    arc_lw=1,
    exclude_promoter=False,
    genome_annotation=None,
    return_fig=False,
    genes_violin_plot=None,
    meta_data_key="cell_type",
    wspace=0.1,
    hspace=0.15,
    color_dict=None,
    tf_vspace=0.15,
    min_acc=200,
    min_fc=1.5,
    prox_thresh=0.05,
    tf_y1=0.66,
    tf_y2=0.4,
    tf_y3=0.77,
    celltype_of_interest="ADC",
    da_df=None,
    da_min_logfc=0.25,
    da_max_qval=0.10,
    max_tfs_per_peak=1,
):
    """
    Wrapper around scenicplus.plotting.coverageplot.coverage_plot with additional logic
    for differential peaks and TF annotation, closely matching your notebook code.
    Uses globals set by setup_context().
    """
    global regulon_data, consensus_bed_pr, interaction_df, all_regulon_regions
    global scplus_obj, bw_dict, ensgene_df

    if fontsize_dict is None:
        fontsize_dict = {
            "bigwig_label": 9,
            "gene_label": 9,
            "violinplots_xlabel": 9,
            "title": 12,
            "bigwig_tick_label": 5,
            "violinplots_ylabel": 9,
            "tf_label": 10,
        }

    if height_ratios_dict is None:
        height_ratios_dict = {"bigwig_violin": 1, "genes": 0.5, "arcs": 5}

    if genome_annotation is None:
        genome_annotation = ensgene_df

    # ---- determine region if not provided ---- #
    if region is None:
        linked_regions = regulon_data[regulon_data["Gene"] == gene]["Region"].unique()
        starts = [int(rg.split(":")[1].split("-")[0]) for rg in linked_regions]
        ends = [int(rg.split(":")[1].split("-")[1]) for rg in linked_regions]
        gene_body = get_gene_body(gene, genome_annotation)
        chrom, gene_start, gene_end = get_chrom_start_end(gene_body)
        points = np.concatenate((starts, ends, [gene_start, gene_end]))
        start = points.min()
        end = points.max()
        overhang = max(int((end - start) * 0.025), 1000)
        region = f"{chrom}:{start - overhang}-{end + overhang}"

    # ---- subset interactions ---- #
    interact_df = interaction_df.copy()
    if activating_only:
        interact_df = interact_df[interact_df["sign"] > 0]
    if exclude_promoter:
        interact_df = interact_df[interact_df["distance"].abs() > 1000]
    if subset_interact_to_gene:
        pr_interact = pr.PyRanges(interact_df[interact_df["name"].str.startswith(gene)])
    else:
        pr_interact = pr.PyRanges(interact_df)

    if len(pr_interact) == 0 or "value" not in pr_interact.df.columns:
        print("[WARN] pr_interact has no 'value' column or is empty; disabling arcs.")
        pr_interact = None

    # ---- color regulon peaks if requested ---- #
    if color_regulon_peaks and regulon is not None:
        norm = Normalize(0, 1)
        greens = truncate_colormap("Greens")
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=greens)
        interact_df = pr_interact.df.copy()
        interact_df["Region"] = (
            interact_df["Chromosome"].astype(str)
            + ":"
            + interact_df["Start"].astype(str)
            + "-"
            + interact_df["End"].astype(str)
        )
        regulon_regions = set(regulon["Region"].values)
        for i in interact_df.index:
            if interact_df.loc[i, "Region"] in regulon_regions:
                interact_df.loc[i, "color"] = rgb2hex(
                    mapper.to_rgba(interact_df.loc[i, "value"])
                )
        pr_interact = pr.PyRanges(interact_df)

    # ---- GTF subset ---- #
    if subset_gtf_to_gene:
        pr_gtf = pr.PyRanges(genome_annotation[genome_annotation["gene_name"] == gene])
    else:
        pr_gtf = pr.PyRanges(genome_annotation)

    # ---- consensus peaks subset ---- #
    if prune_consensus_bed:
        df = consensus_bed_pr.df.copy()
        df["region"] = (
            df["Chromosome"].astype(str)
            + ":"
            + df["Start"].astype(str)
            + "-"
            + df["End"].astype(str)
        )
        pr_consensus_peaks = pr.PyRanges(df[df["region"].isin(all_regulon_regions)])
    else:
        pr_consensus_peaks = consensus_bed_pr

    chrom, start, end = get_chrom_start_end(region)
    region_pr = pd.DataFrame([[chrom, start, end]], columns=["Chromosome", "Start", "End"])
    region_pr = pr.PyRanges(region_pr)
    pr_consensus_peaks = pr_consensus_peaks.intersect(region_pr)

    if len(pr_interact) == 0:
        pr_interact = None

    # ---- main coverage plot ---- #
    ret = coverageplot.coverage_plot(
        scplus_obj,
        bw_dict,
        region,
        pr_gtf=pr_gtf,
        pr_interact=pr_interact,
        pr_consensus_bed=pr_consensus_peaks,
        figsize=figsize,
        fontsize_dict=fontsize_dict,
        region_bed_height=region_bed_height,
        bw_ymax=bw_ymax,
        height_ratios_dict=height_ratios_dict,
        arc_rad=arc_rad,
        arc_lw=arc_lw,
        genes_violin_plot=genes_violin_plot,
        meta_data_key=meta_data_key,
        color_dict=color_dict,
    )

    axes = ret.axes
    if genes_violin_plot is not None and len(genes_violin_plot) > 0:
        # hide violin backgrounds above bigwig tracks
        axes = axes[::2]
        axes_vln = ret.axes[1::2]
        for i in range(len(bw_dict), len(axes_vln)):
            axes_vln[i].patch.set_alpha(0)

    # ---- differential peaks vs other cell types ---- #
    # Optionally restrict consensus peaks by overlap with interactions
    if pr_interact is not None:
        pr_consensus_peaks = pr_consensus_peaks.overlap(pr_interact)

    # Prepare list of (start, end) pairs for peaks in this region
    cons_df = pr_consensus_peaks.df.copy()
    cons_df["region_str"] = (
        cons_df["Chromosome"].astype(str) + ":" +
        cons_df["Start"].astype(str) + "-" +
        cons_df["End"].astype(str)
    )

    differential_peaks = []

    if da_df is not None:
        # --- Use genome-wide DA results --- #
        da_sub = da_df[
            (da_df["cell_type"] == celltype_of_interest) &
            (da_df["logFC"] >= da_min_logfc) &
            (da_df["p_adj"] <= da_max_qval)
        ].copy()

        da_regions = set(da_sub["peak"].values)

        # intersect DA regions with consensus peaks in this window
        mask = cons_df["region_str"].isin(da_regions)
        df_sig = cons_df[mask]

        for _, row in df_sig.iterrows():
            differential_peaks.append((int(row["Start"]), int(row["End"])))

    else:
        # --- Fallback: original bigWig-based differential logic --- #
        openbw_dict = {k: pyBigWig.open(v) for k, v in bw_dict.items()}
        other_celltypes = [k for k in openbw_dict if k != celltype_of_interest]

        for _, peak in cons_df.iterrows():
            chrom = peak["Chromosome"]
            start = int(peak["Start"])
            end = int(peak["End"])
            peakacc_dict = {
                k: np.nansum(v.values(chrom, start, end))
                for k, v in openbw_dict.items()
            }
            if (
                peakacc_dict[celltype_of_interest]
                > max([min_fc * peakacc_dict[c] for c in other_celltypes])
                and peakacc_dict[celltype_of_interest] > min_acc
            ):
                differential_peaks.append((start, end))

        for v in openbw_dict.values():
            v.close()

    chrom, start, end = get_chrom_start_end(region)
    i_of_interest = list(bw_dict.keys()).index(celltype_of_interest)
    ymid_alphas = [tf_y1] * len(differential_peaks)
    xrange = end - start
    differential_peaks = sorted(differential_peaks)

    # spread TF label vertical positions when peaks are close
    i = 0
    while i < len(differential_peaks) - 1:
        start1 = differential_peaks[i][0]
        start2 = differential_peaks[i + 1][0]
        if start2 - start1 < prox_thresh * xrange:
            if ymid_alphas[i] == tf_y1:
                ymid_alphas[i] = tf_y2
                ymid_alphas[i + 1] = tf_y3
            elif ymid_alphas[i] == tf_y2:
                ymid_alphas[i + 1] = tf_y3
            elif ymid_alphas[i] == tf_y3:
                ymid_alphas[i + 1] = tf_y2
        i += 1

    # highlight differential peaks and annotate TFs
    for k, (start_peak, end_peak) in enumerate(differential_peaks):
        for i_ax in range(len(bw_dict)):
            ax = axes[i_ax]
            ax.axvspan(start_peak, end_peak, color="yellow", alpha=0.35)
            if i_ax == i_of_interest:
                region_str = f"{chrom}:{start_peak}-{end_peak}"
                region_regulon = regulon_data[regulon_data["Gene"] == gene]
                region_regulon = region_regulon[region_regulon["Region"] == region_str]
                region_regulon = region_regulon[
                    region_regulon["eRegulon_name"].str.contains(r"\+/\+")
                ]
                tfs = region_regulon.sort_values("triplet_rank")["TF"].values[:max_tfs_per_peak]
                ymin, ymax = ax.get_ylim()
                ymid = ymid_alphas[k] * ymax + (1 - ymid_alphas[k]) * ymin
                yrange = ymax - ymin
                for j, tf in enumerate(tfs):
                    ax.text(
                        start_peak - 10,
                        ymid + tf_vspace * yrange * (j - len(tfs) / 2),
                        tf,
                        ha="right",
                        va="center",
                        fontsize=fontsize_dict["tf_label"],
                    )

    # ---- extra arrows over gene bodies (as in notebook) ---- #
    gtf_region_intersect = pr_gtf.intersect(region_pr)
    gtf_region_intersect = gtf_region_intersect[
        np.logical_and(
            np.logical_or(
                gtf_region_intersect.Feature == "gene",
                gtf_region_intersect.Feature == "exon",
            ),
            gtf_region_intersect.gene_type == "protein_coding",
        )
    ]
    ax_genes = axes[len(bw_dict)]
    genes_in_window = set(gtf_region_intersect.gene_name)
    n_genes_in_window = len(genes_in_window)
    gene_height = 1.0

    for idx, _gene in enumerate(genes_in_window):
        _gene_height = gene_height / n_genes_in_window
        _gene_bottom = -gene_height / 2 + _gene_height * idx
        gene_parts = gtf_region_intersect.df[
            gtf_region_intersect.df["gene_name"] == _gene
        ]

        # only protein-coding transcripts
        if not all(
            gene_parts["transcript_type"].dropna() == "protein_coding"
        ):
            continue

        for _, part in gene_parts.iterrows():
            if part["Feature"] == "gene":
                gene_start = part["Start"]
                gene_end = part["End"]
                strand = part["Strand"]
                x_middle = (gene_start + gene_end) / 2
                y_middle = _gene_bottom + _gene_height / 2
                arrow_half_height = _gene_height
                arrow_width = 500
                if strand == "-":
                    arrow_width = -arrow_width
                for i_off in range(3):
                    x_middle_ = x_middle + 350 * (i_off - 1)
                    ax_genes.plot(
                        (x_middle_, x_middle_ - arrow_width),
                        (y_middle, y_middle + arrow_half_height),
                        color="magenta",
                        linewidth=1,
                    )
                    ax_genes.plot(
                        (x_middle_, x_middle_ - arrow_width),
                        (y_middle, y_middle - arrow_half_height),
                        color="magenta",
                        linewidth=1,
                    )

    # adjust layout
    gs = axes[0].get_subplotspec().get_gridspec()
    gs.update(wspace=wspace, hspace=hspace)

    if return_fig:
        return ret


def main():
    parser = argparse.ArgumentParser(
        description="Generate SCENIC+ coverage plots for a given gene."
    )
    parser.add_argument(
        "--scplus-res-dir",
        required=True,
        help="SCENIC+ results directory (the one containing scplusmdata.h5mu).",
    )
    parser.add_argument(
        "--gtf",
        required=True,
        help="GTF file used for gene annotation (e.g. MANE.GRCh38.v1.4.ensembl_genomic.gtf).",
    )
    parser.add_argument(
        "--consensus-bed",
        required=True,
        help="Consensus peak BED file (e.g. peak_calling/consensus_regions.bed).",
    )
    parser.add_argument(
        "--bw-dir",
        required=True,
        help="Directory containing pseudobulk bigWig files named '<cell_type>.bw'.",
    )
    parser.add_argument(
        "--gene",
        default="PHOX2B",
        help="Gene symbol to plot (e.g. PHOX2B).",
    )
    parser.add_argument(
        "--celltype-of-interest",
        default="ADC",
        help="Cell type whose differential peaks are highlighted (default: ADC).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="coverage_plot.png",
        help="Output image file (e.g. PHOX2B_coverage.png).",
    )
    parser.add_argument(
        "--extend-left",
        type=int,
        default=50000,
        help="Bases to extend upstream of gene body (default: 50000).",
    )
    parser.add_argument(
        "--extend-right",
        type=int,
        default=50000,
        help="Bases to extend downstream of gene body (default: 50000).",
    )
    parser.add_argument(
        "--bw-ymax",
        type=float,
        default=None,
        help="Fixed y-axis max for bigWig tracks (optional).",
    )
    parser.add_argument(
        "--min-fc",
        type=float,
        default=1.75,
        help="Minimum fold-change for differential peak calling (default: 1.75).",
    )
    parser.add_argument(
        "--max-qval",
        type=float,
        default=0.05,
        help="Maximum adjusted p-value for differential peaks (default: 0.05).",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=0.05,
        help="Minimum percent of cells with accessibility for differential peaks (default: 0.05)",
    )
    parser.add_argument(
        "--min-acc",
        type=float,
        default=300,
        help="Minimum accessibility for differential peaks (default: 300).",
    )
    parser.add_argument(
        "--prox-thresh",
        type=float,
        default=0.01,
        help="Proximity threshold (fraction of window) to separate TF labels (default: 0.01).",
    )
    parser.add_argument(
        "--celltype-order",
        nargs="+",
        default=None,
        help="Space-separated desired order of cell types in the coverage plot. "
            "Names must match the cell_type labels after any NonNE→ADC renaming, "
            "e.g. --celltype-order ADC 'NE-high' 'NE-intermediate'."
    )
    parser.add_argument(
        "--top-da-genes",
        type=int,
        default=0,
        help="If >0, automatically select the top N genes with the most DA peaks "
            "linked via region_to_gene and plot coverage for each. "
            "If 0, only the gene given by --gene is plotted."
    )

    args = parser.parse_args()
    # set up all global objects
    setup_context(
        scplus_res_dir=args.scplus_res_dir,
        gtf_path=args.gtf,
        consensus_bed_path=args.consensus_bed,
        bw_dir=args.bw_dir,
        meta_data_key="cell_type",
        rename_nonne_to_adc=True,
        celltype_order=args.celltype_order,
        celltype_of_interest=args.celltype_of_interest,
        min_logfc=args.min_fc,
        max_qval=args.max_qval,
        min_pct=args.min_pct,
    )
    # ---- DA peaks already computed in setup_context ---- #
    da_df = globals()["da_df"]
    print("[DEBUG] da_df shape:", da_df.shape)
    print("[DEBUG] da_df shape:", da_df.shape)
    print("[DEBUG] DA peak counts per cell type:")
    print("[DEBUG] DA peak counts per cell type:")

    da_df.to_csv(
        os.path.join(args.scplus_res_dir, "DA_peaks.tsv"),
        sep="\t",
        index=False,
    )

    # ---- decide which genes to plot ---- #
    # if args.top_da_genes > 0:
    #     region_to_gene_df = pd.read_csv(
    #         os.path.join(args.scplus_res_dir, "region_to_gene_adj.tsv"),
    #         sep="\t",
    #     )
    #     gene_summary = summarize_da_peaks_by_gene(
    #         da_df=da_df,
    #         region_to_gene_df=region_to_gene_df,
    #         celltype_of_interest=args.celltype_of_interest,
    #     )
    #     if gene_summary.empty:
    #         print(
    #             f"[WARN] No DA-linked genes for {args.celltype_of_interest}. "
    #             f"Falling back to plotting only --gene ({args.gene})."
    #         )
    #         genes_to_plot = [args.gene]
    #     else:
    #         genes_to_plot = gene_summary.head(args.top_da_genes).index.tolist()
    #     print(
    #         f"Top {args.top_da_genes} genes by DA peaks for "
    #         f"{args.celltype_of_interest}: {genes_to_plot}"
    #     )
    # else:
    #     genes_to_plot = [args.gene]

    # # ---- plotting parameters ---- #
    # fontsize_dict = {
    #     "bigwig_label": 14,
    #     "gene_label": 14,
    #     "violinplots_xlabel": 14,
    #     "violinplots_ylabel": 14,
    #     "title": 16,
    #     "bigwig_tick_label": 7,
    #     "tf_label": 12,
    # }
    # height_ratios_dict = {"bigwig_violin": 1, "genes": 0.5, "arcs": 3}
    # # ---- loop over genes to plot ---- #
    # for gene in genes_to_plot:
    #     print(f"Plotting coverage for {gene}...")

    #     # window: gene body ± extension
    #     gene_body = get_gene_body(gene, ensgene_df)
    #     region = extend(gene_body, args.extend_left, args.extend_right)

    #     fig = plot_coverage(
    #         gene=gene,
    #         region=region,
    #         regulon=None,
    #         prune_consensus_bed=True,
    #         subset_gtf_to_gene=True,
    #         subset_interact_to_gene=False,
    #         color_regulon_peaks=False,
    #         figsize=(9, 3),
    #         exclude_promoter=True,
    #         genome_annotation=ensgene_df,
    #         height_ratios_dict=height_ratios_dict,
    #         bw_ymax=args.bw_ymax,
    #         min_fc=args.min_fc,
    #         min_acc=args.min_acc,
    #         prox_thresh=args.prox_thresh,
    #         meta_data_key="cell_type",
    #         celltype_of_interest=args.celltype_of_interest,
    #         da_df=da_df,        
    #         da_min_logfc=0.25,
    #         da_max_qval=0.05,
    #         max_tfs_per_peak=1,
    #         # color_dict=palette,
    #         return_fig=True,
    #     )

    #     fig.subplots_adjust(top=0.95)

    #     # choose output filename
    #     if args.top_da_genes > 0:
    #         out_path = f"{gene}_{args.celltype_of_interest}_DA_coverage.png"
    #     else:
    #         out_path = args.output

    #     fig.savefig(out_path, dpi=300, bbox_inches="tight")
    #     print(f"Saved coverage plot for {gene} to {out_path}")


if __name__ == "__main__":
    main()