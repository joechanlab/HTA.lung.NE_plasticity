import scanpy as sc
import os
import pickle
import pandas as pd
import numpy as np
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy.external as sce
sc.settings.figdir = "../../results/figures/figure7/"
sc.set_figure_params(
    dpi=120,
    dpi_save=600,
    figsize=(3, 2),
    fontsize=9,             # base font size for labels and ticks
    color_map='Spectral_r',
    frameon=False,
    vector_friendly=True,
)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.grid": False,
    "axes.titlesize": 10,       # titles like “PHOX2B expression”
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,         # embeds text as editable in Illustrator
    "ps.fonttype": 42,
})

def load_chromvar_matrix(path: str) -> pd.DataFrame:
    """Load ChromVAR motif deviation scores."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ChromVAR file not found: {path}")
    mat = pd.read_csv(path, sep="\t", index_col=0)
    return mat


def _to_numpy_1d(x):
    """Convert dense/sparse vectors to a 1D numpy array."""
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def _short_sample_label(sample_name: str) -> str:
    """Compress full sample IDs (e.g., RU151_PDX123_PHOX2B) to RU151."""
    m = re.search(r"(RU\d+)", str(sample_name))
    return m.group(1) if m else str(sample_name)


def _dodge_ondata_labels(ax, min_frac: float = 0.06, step_frac: float = 0.02, max_iter: int = 200):
    """Simple pairwise repulsion to reduce overlaps in Scanpy on-data text labels."""
    texts = [t for t in ax.texts if t.get_text().strip()]
    if len(texts) < 2:
        return

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = max(abs(x1 - x0), 1e-9)
    yspan = max(abs(y1 - y0), 1e-9)
    min_dx = min_frac * xspan
    min_dy = min_frac * yspan
    step_x = step_frac * xspan
    step_y = step_frac * yspan

    for _ in range(max_iter):
        moved = False
        for i in range(len(texts)):
            xi, yi = texts[i].get_position()
            for j in range(i):
                xj, yj = texts[j].get_position()
                dx = xi - xj
                dy = yi - yj
                if abs(dx) < min_dx and abs(dy) < min_dy:
                    sx = step_x if dx >= 0 else -step_x
                    sy = step_y if dy >= 0 else -step_y
                    xi = np.clip(xi + sx, min(x0, x1), max(x0, x1))
                    yi = np.clip(yi + sy, min(y0, y1), max(y0, y1))
                    xj = np.clip(xj - sx, min(x0, x1), max(x0, x1))
                    yj = np.clip(yj - sy, min(y0, y1), max(y0, y1))
                    texts[i].set_position((xi, yi))
                    texts[j].set_position((xj, yj))
                    moved = True
        if not moved:
            break


# Sex-linked genes that commonly confound batch integration when samples differ by sex.
# These are excluded from HVG selection, PCA, and DC correlation gene selection.
SEX_GENES = [
    "XIST", "TSIX",                                 # X-inactivation / female markers
    "RPS4Y1", "EIF1AY", "DDX3Y", "KDM5D",          # Y chromosome markers
    "USP9Y", "PRKY", "TMSB4Y", "ZFY",
    "NLGN4Y", "TBL1Y", "TTTY14", "TTTY15",
    "CYorf15A", "CYorf15B", "RPS4Y2",
]


def select_dc_genes(adata_norm, dc_coords, n_per_axis: int = 5, exclude_genes=None):
    """Select genes most correlated with DC1 and DC2 from HVGs."""
    if "highly_variable" in adata_norm.var:
        gene_pool = adata_norm.var_names[adata_norm.var["highly_variable"]].tolist()
    else:
        gene_pool = adata_norm.var_names.tolist()
    if exclude_genes:
        gene_pool = [g for g in gene_pool if g not in set(exclude_genes)]

    x = adata_norm[:, gene_pool].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=float)

    selected = []
    for ax_idx in range(min(dc_coords.shape[1], 2)):
        y = dc_coords[:, ax_idx]
        y_c = y - y.mean()
        y_std = y_c.std()
        if y_std == 0:
            continue
        x_c = x - x.mean(axis=0)
        denom = x_c.std(axis=0) * y_std
        denom[denom == 0] = np.nan
        corrs = (x_c * y_c[:, None]).mean(axis=0) / denom
        corr_s = pd.Series(corrs, index=gene_pool).dropna()
        for g in corr_s.nlargest(n_per_axis).index:
            if g not in selected:
                selected.append(g)
        for g in corr_s.nsmallest(n_per_axis).index:
            if g not in selected:
                selected.append(g)
    return selected


def select_hvgs_by_axis_covariation(
    adata_expr,
    hvg_mask,
    axis_values,
    n_keep: int,
    exclude_genes=None,
):
    """Subset preselected HVGs by absolute correlation with a target axis."""
    exclude = set(exclude_genes or [])
    hvg_genes = adata_expr.var_names[np.asarray(hvg_mask)].tolist()
    hvg_genes = [g for g in hvg_genes if g not in exclude]
    if len(hvg_genes) == 0:
        return []

    y = np.asarray(axis_values).ravel()
    y_c = y - y.mean()
    y_std = y_c.std()
    if y_std == 0:
        return hvg_genes[:n_keep]

    x = adata_expr[:, hvg_genes].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=float)
    x_c = x - x.mean(axis=0)
    x_std = x_c.std(axis=0)
    denom = x_std * y_std
    denom[denom == 0] = np.nan
    corrs = (x_c * y_c[:, None]).mean(axis=0) / denom
    corr_s = pd.Series(np.abs(corrs), index=hvg_genes).dropna()
    if corr_s.empty:
        return hvg_genes[:n_keep]

    n_keep = max(1, min(int(n_keep), corr_s.shape[0]))
    selected = corr_s.nlargest(n_keep).index.tolist()
    return selected


def select_hvgs_by_axis_covariation_per_batch(
    adata_expr,
    hvg_mask,
    axis_values,
    batch_key,
    n_keep: int,
    exclude_genes=None,
):
    """
    Subset preselected HVGs by |correlation| with a target axis, computed
    separately within each batch and then unioned (topped up from the pooled
    ranking if needed to reach n_keep).

    A single pooled correlation across batches of very different size is
    dominated by the larger batch's cells/variance, so a smaller batch's own
    covarying genes can be entirely excluded even though they would drive a
    real within-batch axis of variation. Ranking within each batch first
    guarantees every batch gets a fair share of genes before topping up.
    """
    exclude = set(exclude_genes or [])
    hvg_genes = adata_expr.var_names[np.asarray(hvg_mask)].tolist()
    hvg_genes = [g for g in hvg_genes if g not in exclude]
    if len(hvg_genes) == 0:
        return []

    y_full = np.asarray(axis_values).ravel()
    batches = adata_expr.obs[batch_key].astype(str).values
    unique_batches = pd.unique(batches)
    n_per_batch = max(1, int(n_keep) // max(1, len(unique_batches)))

    selected = []
    for b in unique_batches:
        mask = batches == b
        y = y_full[mask]
        y_c = y - y.mean()
        y_std = y_c.std()
        if y_std == 0:
            continue
        x = adata_expr[mask, :][:, hvg_genes].X
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=float)
        x_c = x - x.mean(axis=0)
        denom = x_c.std(axis=0) * y_std
        denom[denom == 0] = np.nan
        corrs = (x_c * y_c[:, None]).mean(axis=0) / denom
        corr_s = pd.Series(np.abs(corrs), index=hvg_genes).dropna()
        if corr_s.empty:
            continue
        for g in corr_s.nlargest(min(n_per_batch, corr_s.shape[0])).index:
            if g not in selected:
                selected.append(g)

    if len(selected) < n_keep:
        pooled = select_hvgs_by_axis_covariation(
            adata_expr, hvg_mask, axis_values, n_keep=n_keep, exclude_genes=exclude_genes
        )
        for g in pooled:
            if g not in selected:
                selected.append(g)
            if len(selected) >= n_keep:
                break

    return selected


def pick_phox2b_dc(dc_matrix, phox2b_signal, confound_signals=None):
    """
    Rank diffusion components by |r(PHOX2B)| - sum(|r(confound)|).
    Returns column indices into dc_matrix sorted best-first.
    """
    scores = []
    for i in range(dc_matrix.shape[1]):
        col = dc_matrix[:, i]
        if col.std() == 0:
            scores.append(-np.inf)
            continue
        score = abs(np.corrcoef(col, phox2b_signal)[0, 1])
        for sig in (confound_signals or []):
            score -= abs(np.corrcoef(col, sig)[0, 1])
        scores.append(score)
    return np.argsort(scores)[::-1]


def select_phox2b_genes(
    adata_magic,
    axis_values: np.ndarray,
    base_genes,
    n_pos: int = 4,
    n_neg: int = 4,
    exclude_genes=None,
):
    """Select genes most positively/negatively correlated with a PHOX2B-related axis."""
    _exclude = set(exclude_genes or [])
    present_base = [g for g in base_genes if g in adata_magic.var_names and g not in _exclude]
    hvg_genes = (
        adata_magic.var_names[adata_magic.var["highly_variable"]].tolist()
        if "highly_variable" in adata_magic.var
        else adata_magic.var_names.tolist()
    )
    hvg_genes = [g for g in hvg_genes if g not in _exclude]
    candidate_genes = list(dict.fromkeys(present_base + hvg_genes))
    if len(candidate_genes) == 0:
        return present_base

    x = adata_magic[:, candidate_genes].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x)
    y = np.asarray(axis_values).ravel()

    y_center = y - y.mean()
    y_std = y_center.std()
    if y_std == 0:
        return present_base

    x_center = x - x.mean(axis=0)
    x_std = x_center.std(axis=0)
    denom = (x_std * y_std)
    denom[denom == 0] = np.nan
    corrs = (x_center * y_center[:, None]).mean(axis=0) / denom

    corr_df = pd.DataFrame({"gene": candidate_genes, "corr": corrs}).dropna()
    if corr_df.empty:
        return present_base

    pos_genes = corr_df.sort_values("corr", ascending=False).head(n_pos)["gene"].tolist()
    neg_genes = corr_df.sort_values("corr", ascending=True).head(n_neg)["gene"].tolist()

    selected = []
    for g in ["PHOX2B", "FOXA1"] + present_base + pos_genes + neg_genes:
        if g in adata_magic.var_names and g not in selected:
            selected.append(g)
    return selected

BASE_DIR = "/data1/chanj3/wangm10/HTA_data/"
SAMPLES = ["RU151_PDX123_PHOX2B", "RU581_LN_SCLC"]
SAMPLE_TO_ADATA = {
    "RU581_LN_SCLC": "/data1/chanj3/wangm10/HTA_data/nf-scrnaseq_multiome/rna_postprocessing/RU581_LN_SCLC.h5ad",
    "RU581_LN": "/data1/chanj3/wangm10/HTA_data/nf-scrnaseq_multiome/rna_postprocessing/RU581_LN_filtered.h5ad",
    "RU151_PDX123_PHOX2B": "/data1/chanj3/HTA.multiome_plasticity.results/adatas/RU151_PDX123_cell_type_balanced.h5ad",
}
TFs_OF_INTEREST = ["PHOX2B", "FOXA1"]
TOP_N_PEAKS = 1000
BASE_GENES_OF_INTEREST = ["PHOX2B", "FOXA1", "ASCL1", "NEUROD1", "SOX2", "REST", "EGFR", "MYC", "CHGA"]
PLOT_BASES = ["dc", "umap"]
USE_HARMONY = True
HARMONY_KEY = "sample"
SAMPLE_SHORT_KEY = "sample_short"
USE_MAGIC_FOR_RNA_PLOTS = True
MAGIC_KNN = 3
MAGIC_T = 1
GENE_PLOT_ALPHA = 0.85
PHOX2B_HVG_TOP_N = 800
# Diffusion component pairs to visualize from X_diffmap (DC index starts at 1; DC0 is skipped).
DC_PAIRS_TO_PLOT = [(1, 2), (1, 3), (2, 3), (1, 4)]

adatas = []
chromvar_mats = []

for SAMPLE in SAMPLES:
    CHROMVAR_FILE = os.path.join(
        BASE_DIR, "nf-multiome_atac", "atac_chromvar", f"{SAMPLE}_chromvar", "tf_score.ChromVAR.tsv"
    )
    ADATA_FILE = SAMPLE_TO_ADATA[SAMPLE]
    chromvar = load_chromvar_matrix(CHROMVAR_FILE)
    adata = sc.read_h5ad(ADATA_FILE)
    chromvar_cells = chromvar.columns.str.split("#").str[1]
    shared_cells = adata.obs_names.intersection(chromvar_cells)
    adata = adata[shared_cells, :].copy()

    prefixed_cells = pd.Index([f"{SAMPLE}#{bc}" for bc in shared_cells])
    adata.obs_names = prefixed_cells
    adata.obs[HARMONY_KEY] = SAMPLE
    adata.obs['cell_type'] = adata.obs['cell_type'].cat.remove_unused_categories()
    print(adata.shape[0], "cells loaded for sample", SAMPLE)

    for tf in TFs_OF_INTEREST:
        # Create a Series with cell barcodes as index and align with adata.obs
        chromvar_series = pd.Series(chromvar.loc[tf, :].values, index=chromvar_cells)
        aligned_series = chromvar_series.reindex(shared_cells)
        aligned_series.index = prefixed_cells
        adata.obs[f'ChromVAR_{tf}'] = aligned_series.values

    # Keep TF x cell ChromVAR matrix for joint downstream correlation.
    cell_to_column = pd.Series(chromvar.columns, index=chromvar_cells)
    selected_columns = cell_to_column.reindex(shared_cells).dropna().values
    chromvar_shared = chromvar.loc[:, selected_columns].copy()
    chromvar_shared.columns = prefixed_cells

    # Drop precomputed embeddings/metadata containers to avoid concat type conflicts
    # across source h5ad files (e.g., DataFrame in one sample vs ndarray in another).
    # Keep the raw 'counts' layer: adata.X in these h5ad files is already
    # normalized ("normcounts" per adata.uns["X_name"]), which silently breaks
    # seurat_v3 HVG selection (it expects raw counts) and double-normalizes
    # downstream if left in .X.
    _counts_layer = adata.layers.get("counts")
    adata.obsm.clear()
    adata.varm.clear()
    adata.obsp.clear()
    adata.varp.clear()
    adata.layers.clear()
    adata.uns = {}
    if _counts_layer is not None:
        adata.layers["counts"] = _counts_layer
    else:
        print(f"[{SAMPLE}] WARNING: no 'counts' layer found; adata.X may not be raw counts.")

    adatas.append(adata)
    chromvar_mats.append(chromvar_shared)

adata = sc.concat(adatas, join="inner", merge="same")
adata.obs['cell_type'] = adata.obs['cell_type'].cat.remove_unused_categories()
adata.obs[SAMPLE_SHORT_KEY] = adata.obs[HARMONY_KEY].astype(str).map(_short_sample_label)
adata.obs[SAMPLE_SHORT_KEY] = pd.Categorical(adata.obs[SAMPLE_SHORT_KEY])
chromvar_joint = pd.concat(chromvar_mats, axis=1)

print(f"Joint dataset has {adata.n_obs} cells across {adata.obs[HARMONY_KEY].nunique()} samples")

if "counts" in adata.layers:
    adata.X = adata.layers["counts"].copy()
    print("[joint] Restored raw 'counts' layer into .X for HVG selection and PCA input.")
else:
    print("[joint] WARNING: 'counts' layer missing after concat; assuming adata.X already holds raw counts.")

N_PCS = 10   # suitable for ~700 cells; fewer PCs = less noise going into integration
sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=1500, batch_key=HARMONY_KEY)
# Prevent sex-linked genes from driving principal components / diffusion axes.
_sex_in_var = [g for g in SEX_GENES if g in adata.var_names]
if _sex_in_var:
    print(f"[joint] Masking {len(_sex_in_var)} sex-linked genes from HVGs: {_sex_in_var}")
    adata.var.loc[_sex_in_var, "highly_variable"] = False

# Further subset HVGs to those most covarying with PHOX2B to stabilize downstream axes.
_hvg_mask_initial = adata.var["highly_variable"].values.copy()
_hvg_rank_expr = adata.copy()
sc.pp.normalize_total(_hvg_rank_expr, target_sum=1e4)
sc.pp.log1p(_hvg_rank_expr)
if "PHOX2B" in _hvg_rank_expr.var_names:
    _phox2b_hvg_axis = _to_numpy_1d(_hvg_rank_expr[:, "PHOX2B"].X)
elif "ChromVAR_PHOX2B" in adata.obs:
    _phox2b_hvg_axis = _to_numpy_1d(adata.obs["ChromVAR_PHOX2B"].values)
else:
    _phox2b_hvg_axis = None

if _phox2b_hvg_axis is not None:
    _selected_hvgs = select_hvgs_by_axis_covariation_per_batch(
        _hvg_rank_expr,
        hvg_mask=_hvg_mask_initial,
        axis_values=_phox2b_hvg_axis,
        batch_key=HARMONY_KEY,
        n_keep=min(PHOX2B_HVG_TOP_N, int(np.sum(_hvg_mask_initial))),
        exclude_genes=SEX_GENES,
    )
    if len(_selected_hvgs) > 0:
        adata.var["highly_variable"] = adata.var_names.isin(_selected_hvgs)
        print(
            f"[joint] PHOX2B-covarying HVGs retained: {len(_selected_hvgs)} "
            f"from {int(np.sum(_hvg_mask_initial))} initial HVGs (selected per-sample, then unioned)"
        )

# Normalize + log1p + scale before PCA (standard practice; the previous version ran
# PCA directly on raw counts). Scaling in particular prevents the larger sample
# (RU151, 544 cells) from dominating the PCs purely on variance magnitude, which
# was smothering RU581's (157 cells) internal PHOX2B-driven structure.
_pca_input = adata.copy()
sc.pp.normalize_total(_pca_input, target_sum=1e4)
sc.pp.log1p(_pca_input)
sc.pp.scale(_pca_input, max_value=10)
sc.tl.pca(_pca_input, svd_solver='arpack', n_comps=N_PCS, use_highly_variable=True)
adata.obsm["X_pca"] = _pca_input.obsm["X_pca"]
adata.varm["PCs"] = _pca_input.varm["PCs"]
adata.uns["pca"] = _pca_input.uns["pca"]
del _pca_input

# bbknn builds batch-balanced cross-sample edges directly in the neighbor graph,
# which forces the subsequent diffmap to see both samples as one connected manifold.
# This is more effective than Harmony here because Harmony corrects the PCA
# representation but the diffmap can still re-separate samples along biology.
if USE_HARMONY:
    try:
        import bbknn
        bbknn.bbknn(adata, batch_key=HARMONY_KEY, n_pcs=N_PCS, neighbors_within_batch=5)
        print("Using bbknn batch-balanced graph for neighbors.")
    except Exception as e:
        print(f"bbknn failed; falling back to standard neighbors. Error: {e}")
        sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_pca")
else:
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_pca")
sc.tl.diffmap(adata)
if adata.obsm["X_diffmap"].shape[1] >= 3:
    adata.obsm["X_dc"] = adata.obsm["X_diffmap"][:, [1, 2]].copy()
else:
    adata.obsm["X_dc"] = adata.obsm["X_diffmap"][:, :2].copy()

if 'paga' in adata.uns:
    del adata.uns['paga']
if adata.obs.cell_type.nunique() > 1:
    sc.tl.paga(adata, groups='cell_type')
    sc.pl.paga(adata, show=False)
    sc.tl.umap(adata, init_pos='paga', min_dist=0.6, spread=1.5)
else:
    sc.tl.umap(adata, min_dist=0.6, spread=1.5)

# Library-size normalize + log1p for cross-sample comparable gene expression plots.
# Assumes .X still holds raw counts (seurat_v3 HVG selection requires this).
adata_norm = adata.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)

# Re-order diffusion components so DC0 = most PHOX2B-related, least sex/sample-confounded.
_phox2b_sig = (
    _to_numpy_1d(adata.obs["ChromVAR_PHOX2B"].values)
    if "ChromVAR_PHOX2B" in adata.obs
    else _to_numpy_1d(adata_norm[:, "PHOX2B"].X)
    if "PHOX2B" in adata_norm.var_names
    else None
)
if _phox2b_sig is not None:
    _confounds = [(adata.obs[HARMONY_KEY] == SAMPLES[0]).astype(float).values]
    if "XIST" in adata_norm.var_names:
        _confounds.append(_to_numpy_1d(adata_norm[:, "XIST"].X))
    _all_dcs = adata.obsm["X_diffmap"][:, 1:]   # skip trivial DC0
    _order = pick_phox2b_dc(_all_dcs, _phox2b_sig, confound_signals=_confounds)
    _best = _order[:2]
    print(f"[joint] Selected DC indices (from diffmap[:,1:]): {_best}  "
          f"(scores: {[round(abs(float(np.corrcoef(_all_dcs[:,i], _phox2b_sig)[0,1])), 3) for i in _best]})")
    adata.obsm["X_dc"] = _all_dcs[:, _best].copy()
    adata_norm.obsm["X_dc"] = adata.obsm["X_dc"].copy()

joint_label = "joint_" + "_".join(SAMPLES)

for basis in PLOT_BASES:
    fig = sc.pl.embedding(
        adata,
        basis=basis,
        color=[f'ChromVAR_{tf}' for tf in TFs_OF_INTEREST],
        title=[f'ChromVAR {tf}' for tf in TFs_OF_INTEREST],
        vmin=-0.8,
        vmax=0.8,
        s=45,
        add_outline=True,
        return_fig=True,
        sort_order=True,
    )
    for ax in fig.axes:
        ax.title.set_fontsize(14)
    fig.figure.savefig(
        os.path.join(sc.settings.figdir, f"{basis}_{joint_label}_fig7_ChromVAR_TF.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    fig = sc.pl.embedding(
        adata,
        basis=basis,
        color=[SAMPLE_SHORT_KEY, "cell_type"],
        title=["Sample", f"Cell type"],
        s=45,
        legend_loc="right margin",
        legend_fontoutline=2,
        add_outline=True,
        return_fig=True,
    )
    for ax in fig.axes:
        ax.title.set_fontsize(14)
    fig.figure.savefig(
        os.path.join(sc.settings.figdir, f"{basis}_{joint_label}_fig7_sample_cell_type.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

if "ChromVAR_PHOX2B" in adata.obs:
    phox2b_axis = _to_numpy_1d(adata.obs["ChromVAR_PHOX2B"].values)
elif "PHOX2B" in adata_norm.var_names:
    phox2b_axis = _to_numpy_1d(adata_norm[:, "PHOX2B"].X)
else:
    phox2b_axis = _to_numpy_1d(adata.obsm["X_dc"][:, 0])

genes_of_interest = select_phox2b_genes(
    adata_norm,
    axis_values=phox2b_axis,
    base_genes=BASE_GENES_OF_INTEREST,
    n_pos=0,
    n_neg=0,
    exclude_genes=SEX_GENES,
)
print(f"[joint] PHOX2B-focused genes for plotting: {', '.join(genes_of_interest)}")

dc_genes = select_dc_genes(adata_norm, adata.obsm["X_dc"], n_per_axis=5, exclude_genes=SEX_GENES)
print(f"[joint] DC-correlated genes for plotting: {', '.join(dc_genes)}")
basis_genes = {"umap": genes_of_interest, "dc": genes_of_interest}

adata_expr_plot = adata_norm
adata_magic_plot = None
if globals().get("USE_MAGIC_FOR_RNA_PLOTS", True):
    magic_genes = list(dict.fromkeys(genes_of_interest))
    try:
        adata_magic_plot = sce.pp.magic(
            adata_norm,
            knn=MAGIC_KNN,
            t=MAGIC_T,
            name_list=magic_genes,
            copy=True,
        )
        # Preserve embedding coordinates for plotting from the source normalized object.
        for emb_key in ["X_umap", "X_dc", "X_diffmap"]:
            if emb_key in adata_norm.obsm:
                adata_magic_plot.obsm[emb_key] = adata_norm.obsm[emb_key].copy()
        adata_expr_plot = adata_magic_plot
        print(
            f"[joint] Using MAGIC-imputed RNA expression for plotting genes: "
            f"{', '.join(magic_genes)} (knn={MAGIC_KNN}, t={MAGIC_T})"
        )
    except Exception as e:
        print(f"[joint] MAGIC imputation failed; using log-normalized RNA expression. Error: {e}")

# Plot configs: (adata_object, file suffix label)
_plot_layers = [("raw", adata_norm, "lognorm")]
if adata_magic_plot is not None:
    _plot_layers.append(("magic", adata_magic_plot, "magic"))

for basis in PLOT_BASES:
    color = basis_genes.get(basis, genes_of_interest)
    for _layer_name, _adata_plot, _layer_suffix in _plot_layers:
        fig = sc.pl.embedding(
            _adata_plot,
            basis=basis,
            color=color,
            frameon=False,
            legend_fontoutline=2,
            wspace=0,
            ncols=min(5, max(1, len(color))),
            use_raw=False,
            add_outline=True,
            sort_order=True,
            return_fig=True,
            s=28,
            alpha=GENE_PLOT_ALPHA,
        )
        for ax in fig.axes:
            ax.title.set_fontsize(16)
        suffix = ("dc_genes" if basis == "dc" else "genes") + f"_{_layer_suffix}"
        fig.figure.savefig(
            os.path.join(sc.settings.figdir, f"{basis}_{joint_label}_fig7_{suffix}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

# ---------------------------------------------------------
# Ranked Correlation Analysis (Hockey Stick Plot)
# ---------------------------------------------------------
TARGET_TF = "PHOX2B"
PARTNER_TF = "FOXA1"

if TARGET_TF in chromvar_joint.index:
    correlations = chromvar_joint.T.corrwith(chromvar_joint.loc[TARGET_TF, :], method='pearson')
    correlations = correlations.drop(TARGET_TF)
    correlations = correlations.sort_values(ascending=False)

    rank_df = pd.DataFrame({
        'TF': correlations.index,
        'Correlation': correlations.values,
        'Rank': range(1, len(correlations) + 1),
    })

    fig, ax = plt.subplots(figsize=(3, 1.5))
    ax.scatter(rank_df['Rank'], rank_df['Correlation'], c='lightgrey', s=5, alpha=0.6, label='Other TFs')

    partner_data = rank_df[rank_df['TF'] == PARTNER_TF]
    if not partner_data.empty:
        ax.scatter(
            partner_data['Rank'],
            partner_data['Correlation'],
            c='red',
            s=50,
            edgecolors='black',
            linewidth=0.5,
            label=PARTNER_TF,
        )
        ax.text(
            x=partner_data['Rank'].values[0] + 50,
            y=partner_data['Correlation'].values[0],
            s=f"{PARTNER_TF}\n(r={partner_data['Correlation'].values[0]:.2f})",
            color='red',
            fontsize=8,
            fontweight='bold',
            va='center',
        )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(sc.settings.figdir, f"rank_corr_{joint_label}_{TARGET_TF}_vs_Genome.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"[joint] Generated ranked correlation plot. "
        f"{PARTNER_TF} is Rank #{partner_data['Rank'].values[0] if not partner_data.empty else 'NaN'} "
        f"out of {len(rank_df)} TFs."
    )

# --- Additional plots: multiple raw DC pair embeddings (no PHOX2B axis reordering) ---
_n_diffmap_dims = adata.obsm["X_diffmap"].shape[1]
for dc1, dc2 in DC_PAIRS_TO_PLOT:
    if dc1 <= 0 or dc2 <= 0:
        continue
    if dc1 >= _n_diffmap_dims or dc2 >= _n_diffmap_dims:
        print(f"[joint] Skipping DC pair ({dc1}, {dc2}); only {_n_diffmap_dims - 1} non-trivial DCs available.")
        continue

    pair_coords = adata.obsm["X_diffmap"][:, [dc1, dc2]]
    pair_basis = f"dc_raw_{dc1}_{dc2}"
    pair_label = f"rawDC{dc1}_{dc2}"

    adata.obsm[f"X_{pair_basis}"] = pair_coords
    adata_norm.obsm[f"X_{pair_basis}"] = pair_coords.copy()
    if adata_magic_plot is not None:
        adata_magic_plot.obsm[f"X_{pair_basis}"] = pair_coords.copy()

    fig = sc.pl.embedding(
        adata,
        basis=pair_basis,
        color=[f'ChromVAR_{tf}' for tf in TFs_OF_INTEREST],
        title=[f'ChromVAR {tf} (DC{dc1}/DC{dc2})' for tf in TFs_OF_INTEREST],
        vmin=-0.8,
        vmax=0.8,
        s=45,
        add_outline=True,
        return_fig=True,
        sort_order=True,
    )
    for ax in fig.axes:
        ax.title.set_fontsize(14)
    fig.figure.savefig(
        os.path.join(sc.settings.figdir, f"dc_{pair_label}_{joint_label}_fig7_ChromVAR_TF.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = sc.pl.embedding(
        adata,
        basis=pair_basis,
        color=[SAMPLE_SHORT_KEY, "cell_type"],
        title=[f"Sample (DC{dc1}/DC{dc2})", f"Cell type (DC{dc1}/DC{dc2})"],
        s=45,
        legend_loc="right margin",
        legend_fontoutline=2,
        add_outline=True,
        return_fig=True,
    )
    for ax in fig.axes:
        ax.title.set_fontsize(14)
    fig.figure.savefig(
        os.path.join(sc.settings.figdir, f"dc_{pair_label}_{joint_label}_fig7_sample_cell_type.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    dc_genes_raw = select_dc_genes(adata_norm, pair_coords, n_per_axis=5, exclude_genes=SEX_GENES)
    print(f"[joint] DC-correlated genes for plotting (DC{dc1}/DC{dc2}): {', '.join(dc_genes_raw)}")
    for _layer_name, _adata_plot, _layer_suffix in _plot_layers:
        _adata_plot.obsm[f"X_{pair_basis}"] = pair_coords.copy()
        fig = sc.pl.embedding(
            _adata_plot,
            basis=pair_basis,
            color=genes_of_interest,
            frameon=False,
            legend_fontoutline=2,
            wspace=0,
            ncols=min(5, max(1, len(genes_of_interest))),
            use_raw=False,
            add_outline=True,
            sort_order=True,
            return_fig=True,
            s=28,
            alpha=GENE_PLOT_ALPHA,
        )
        for ax in fig.axes:
            ax.title.set_fontsize(16)
        fig.figure.savefig(
            os.path.join(sc.settings.figdir, f"dc_{pair_label}_{joint_label}_fig7_dc_genes_{_layer_suffix}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)