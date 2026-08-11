import scanpy as sc
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_predict
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp

sc.settings.figdir = "../../results/figures/figure7/"
sc.set_figure_params(
    dpi=120,
    dpi_save=600,
    figsize=(3, 2),
    fontsize=9,
    color_map='Spectral_r',
    frameon=False,
    vector_friendly=True,
)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.grid": False,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_chromvar_matrix(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ChromVAR file not found: {path}")
    return pd.read_csv(path, sep="\t", index_col=0)


def _to_numpy_1d(x):
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def select_embedding_genes(adata_norm, embed_coords, n_per_axis: int = 5):
    """Genes most correlated with either axis of a 2D embedding (UMAP/FDL/etc)."""
    if "highly_variable" in adata_norm.var:
        gene_pool = adata_norm.var_names[adata_norm.var["highly_variable"]].tolist()
    else:
        gene_pool = adata_norm.var_names.tolist()

    x = adata_norm[:, gene_pool].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=float)

    selected = []
    for ax_idx in range(min(embed_coords.shape[1], 2)):
        y = embed_coords[:, ax_idx]
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


def _safe_abs_corr(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return abs(float(np.corrcoef(x, y)[0, 1]))


def _knn_overlap_score(high_dim, low_dim, k=30):
    k = int(min(k, high_dim.shape[0] - 1, low_dim.shape[0] - 1))
    if k < 2:
        return 0.0

    nn_hi = NearestNeighbors(n_neighbors=k + 1).fit(high_dim)
    nn_lo = NearestNeighbors(n_neighbors=k + 1).fit(low_dim)
    idx_hi = nn_hi.kneighbors(return_distance=False)[:, 1:]
    idx_lo = nn_lo.kneighbors(return_distance=False)[:, 1:]

    overlaps = [
        len(set(idx_hi[i]).intersection(set(idx_lo[i]))) / float(k)
        for i in range(high_dim.shape[0])
    ]
    return float(np.mean(overlaps))


def _put_phox2b_on_x(coords, phox2b_signal):
    """Orient a 2D embedding so the x-axis is the dimension more correlated with PHOX2B,
    and its sign points toward higher PHOX2B. Unlike diffusion components, UMAP/FDL output
    dimensions aren't individually meaningful axes to "pick" -- this is just a post-hoc
    rotation/reflection for plot consistency, it doesn't touch the embedding's geometry."""
    if phox2b_signal is None or coords.shape[1] < 2:
        return coords
    coords = coords.copy()
    for ax_i in range(coords.shape[1]):
        if coords[:, ax_i].std() > 0 and np.corrcoef(coords[:, ax_i], phox2b_signal)[0, 1] < 0:
            coords[:, ax_i] *= -1
    x_corr = _safe_abs_corr(coords[:, 0], phox2b_signal)
    y_corr = _safe_abs_corr(coords[:, 1], phox2b_signal)
    if y_corr > x_corr:
        coords = coords[:, [1, 0]]
    return coords


def _score_embedding_by_phox2b(coords, phox2b_signal, k=10, n_splits=5):
    """
    5-fold CV KNN-regression correlation between 2D embedding coordinates and the PHOX2B
    signal. Used as the grid-search objective for UMAP/FDL settings: unlike diffusion
    components (individually orderable axes you can rank and pick from), a UMAP/FDL layout
    is a single non-separable 2D result, so scoring has to look at the whole embedding's
    local neighborhood structure rather than one axis's linear correlation.
    """
    y = np.asarray(phox2b_signal).ravel()
    if y.std() == 0 or coords.shape[0] < n_splits * 2:
        return 0.0
    k = int(min(k, coords.shape[0] - 1))
    if k < 1:
        return 0.0
    knn = KNeighborsRegressor(n_neighbors=k)
    pred = cross_val_predict(knn, coords, y, cv=KFold(n_splits, shuffle=True, random_state=0))
    if pred.std() == 0:
        return 0.0
    return float(np.corrcoef(pred, y)[0, 1])


def optimize_expression_embedding(
    adata_expr,
    phox2b_signal,
    n_pcs_grid=(15, 25, 35),
    n_neighbors_grid=(15, 25, 35),
    prefer_gauss=False,
    method="umap",
):
    """
    Grid-search expression-based UMAP/FDL settings using PHOX2B alignment (via
    _score_embedding_by_phox2b) as the sole optimization objective.
    """
    best = None
    score_rows = []

    for n_pcs in n_pcs_grid:
        for n_neighbors in n_neighbors_grid:
            ad = adata_expr.copy()
            max_pcs = min(int(n_pcs), ad.n_obs - 1, ad.n_vars - 1)
            if max_pcs < 3:
                continue

            sc.tl.pca(ad, n_comps=max_pcs, use_highly_variable=True, svd_solver="arpack")
            if prefer_gauss:
                # Gaussian kernel tends to produce smoother neighborhoods in small cohorts.
                sc.pp.neighbors(
                    ad,
                    n_neighbors=int(n_neighbors),
                    n_pcs=max_pcs,
                    metric="cosine",
                    knn=False,
                    method="gauss",
                )
            else:
                sc.pp.neighbors(ad, n_neighbors=int(n_neighbors), n_pcs=max_pcs, metric="cosine")

            # sc.tl.umap / sc.tl.draw_graph require sparse connectivities; the gaussian
            # kernel above (knn=False) returns a dense array.
            if not sp.issparse(ad.obsp["connectivities"]):
                ad.obsp["connectivities"] = sp.csr_matrix(ad.obsp["connectivities"])
            if not sp.issparse(ad.obsp["distances"]):
                ad.obsp["distances"] = sp.csr_matrix(ad.obsp["distances"])

            if method == "fdl":
                sc.tl.draw_graph(ad, layout="fa")
                coords = np.asarray(ad.obsm["X_draw_graph_fa"][:, :2], dtype=float)
            else:
                sc.tl.umap(ad, min_dist=0.5, spread=1.0)
                coords = np.asarray(ad.obsm["X_umap"][:, :2], dtype=float)

            coords = _put_phox2b_on_x(coords, phox2b_signal)

            hi = ad.obsm["X_pca"][:, :max_pcs]
            trust = float(trustworthiness(hi, coords, n_neighbors=min(30, int(n_neighbors))))
            overlap = _knn_overlap_score(hi, coords, k=min(30, int(n_neighbors)))
            phox = _score_embedding_by_phox2b(coords, phox2b_signal)
            n_cc, _ = connected_components(ad.obsp["connectivities"], directed=False)

            row = {
                "n_pcs": int(max_pcs),
                "n_neighbors": int(n_neighbors),
                "trustworthiness": trust,
                "knn_overlap": overlap,
                "phox2b_cv_corr": phox,
                "n_connected_components": int(n_cc),
                "score": phox,
                "coords": coords,
            }
            score_rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = row

    if best is None:
        raise RuntimeError(f"No valid {method} configuration found for expression-based embedding.")

    return best, pd.DataFrame(score_rows).drop(columns=["coords"])


def select_phox2b_genes(adata_magic, axis_values, base_genes, n_pos=4, n_neg=4):
    present_base = [g for g in base_genes if g in adata_magic.var_names]
    hvg_genes = (
        adata_magic.var_names[adata_magic.var["highly_variable"]].tolist()
        if "highly_variable" in adata_magic.var
        else adata_magic.var_names.tolist()
    )
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
    denom = x_center.std(axis=0) * y_std
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


def select_embedding_features_by_phox2b(
    adata_norm,
    phox2b_signal,
    n_hvg=1200,
    n_pos=200,
    n_neg=120,
    anchor_genes=("PHOX2B", "FOXA1", "ASCL1", "NEUROD1", "CHGA"),
):
    """Select expression features enriched for PHOX2B-correlated biology while preserving global variation."""
    if "highly_variable" in adata_norm.var:
        hvg = adata_norm.var_names[adata_norm.var["highly_variable"]].tolist()
    else:
        hvg = adata_norm.var_names.tolist()

    if len(hvg) > n_hvg:
        hvg = hvg[:n_hvg]

    x = adata_norm[:, hvg].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=float)
    y = np.asarray(phox2b_signal).ravel()

    if y.std() == 0 or x.shape[1] == 0:
        base = [g for g in anchor_genes if g in adata_norm.var_names]
        return list(dict.fromkeys(base + hvg))

    y_c = y - y.mean()
    x_c = x - x.mean(axis=0)
    denom = x_c.std(axis=0) * y_c.std()
    denom[denom == 0] = np.nan
    corrs = (x_c * y_c[:, None]).mean(axis=0) / denom
    corr_df = pd.DataFrame({"gene": hvg, "corr": corrs}).dropna()

    pos = corr_df.sort_values("corr", ascending=False).head(n_pos)["gene"].tolist()
    neg = corr_df.sort_values("corr", ascending=True).head(n_neg)["gene"].tolist()
    anchors = [g for g in anchor_genes if g in adata_norm.var_names]

    return list(dict.fromkeys(anchors + pos + neg + hvg))


def report_expr_chromvar_consistency(adata_norm, chromvar_cells_df, tf_list, sample):
    """Print per-sample expression vs ChromVAR TF consistency on shared aligned cells."""
    for tf in tf_list:
        cv_col = f"ChromVAR_{tf}"
        if cv_col not in adata_norm.obs:
            print(f"[{sample}] {cv_col} missing in obs")
            continue
        if tf not in adata_norm.var_names:
            print(f"[{sample}] {tf} missing in expression matrix")
            continue

        expr = _to_numpy_1d(adata_norm[:, tf].X)
        cv = _to_numpy_1d(adata_norm.obs[cv_col].values)
        ok = np.isfinite(expr) & np.isfinite(cv)
        if ok.sum() < 3:
            print(f"[{sample}] {tf}: insufficient finite values for expr-vs-ChromVAR correlation")
            continue
        corr = float(np.corrcoef(expr[ok], cv[ok])[0, 1]) if expr[ok].std() > 0 and cv[ok].std() > 0 else np.nan
        print(
            f"[{sample}] {tf} expr-vs-ChromVAR: r={corr:.3f}, "
            f"expr_nonzero_frac={(expr[ok] > 0).mean():.3f}, "
            f"ChromVAR_range=({np.nanmin(cv):.3f}, {np.nanmax(cv):.3f})"
        )


BASE_DIR = "/data1/chanj3/wangm10/HTA_data/"
SAMPLES = ["RU151_PDX123_PHOX2B", "RU581_LN_SCLC"]
SAMPLE_TO_ADATA = {
    "RU581_LN_SCLC": "/data1/chanj3/wangm10/HTA_data/nf-scrnaseq_multiome/rna_postprocessing/RU581_LN_SCLC.h5ad",
    "RU581_LN": "/data1/chanj3/wangm10/HTA_data/nf-scrnaseq_multiome/rna_postprocessing/RU581_LN_filtered.h5ad",
    "RU151_PDX123_PHOX2B": "/data1/chanj3/HTA.multiome_plasticity.results/adatas/RU151_PDX123_cell_type_balanced.h5ad",
}
TFs_OF_INTEREST = ["PHOX2B", "FOXA1"]
BASE_GENES_OF_INTEREST = ["PHOX2B", "FOXA1", "ASCL1", "NEUROD1", "SOX2", "REST", "EGFR", "MYC", "CHGA"]
TARGET_TF = "PHOX2B"
PARTNER_TF = "FOXA1"
N_PCS = 20
EMBEDDING_METHOD = "umap"   # "umap" or "fdl" (force-directed / ForceAtlas2 via sc.tl.draw_graph)
EMBEDDING_BASIS = EMBEDDING_METHOD  # sc.pl.embedding basis string; obsm key is "X_<this>"

# ---------------------------------------------------------
# Per-sample embedding + ranked correlation
# ---------------------------------------------------------
for SAMPLE in SAMPLES:
    CHROMVAR_FILE = os.path.join(
        BASE_DIR, "nf-multiome_atac", "atac_chromvar", f"{SAMPLE}_chromvar", "tf_score.ChromVAR.tsv"
    )
    chromvar_s = load_chromvar_matrix(CHROMVAR_FILE)
    adata_s = sc.read_h5ad(SAMPLE_TO_ADATA[SAMPLE])

    chromvar_cells_s = chromvar_s.columns.str.split("#").str[1]
    shared_cells_s = adata_s.obs_names.intersection(chromvar_cells_s)
    if len(shared_cells_s) == 0:
        raise RuntimeError(f"[{SAMPLE}] No shared cells between RNA and ChromVAR matrices.")
    adata_s = adata_s[shared_cells_s, :].copy()
    print(f"[{SAMPLE}] {adata_s.n_obs} cells")

    for tf in TFs_OF_INTEREST:
        chromvar_series_s = pd.Series(chromvar_s.loc[tf, :].values, index=chromvar_cells_s)
        adata_s.obs[f'ChromVAR_{tf}'] = chromvar_series_s.reindex(shared_cells_s).values

    # Alignment diagnostics: detect potential barcode/key mismatches early.
    _na_stats = {
        tf: int(pd.isna(adata_s.obs[f"ChromVAR_{tf}"]).sum())
        for tf in TFs_OF_INTEREST
        if f"ChromVAR_{tf}" in adata_s.obs
    }
    print(f"[{SAMPLE}] ChromVAR NA counts after alignment: {_na_stats}")

    # Reassign cell_type (NonNE / NE-intermediate / NE-high) from ChromVAR PHOX2B tertiles,
    # so "NE-high" directly means "highest PHOX2B" and the label tracks the same ChromVAR
    # signal being plotted/optimized against below. The pre-existing annotation was skewed
    # almost entirely into one category (e.g. 481/544 "NE-high" for RU151) and didn't track
    # ChromVAR PHOX2B (r=0.2-0.3 with RNA PHOX2B, let alone the ChromVAR score itself).
    # RU581 keeps its original uniform "SCLC-A" label as-is (not reassigned).
    if SAMPLE == "RU151_PDX123_PHOX2B" and "ChromVAR_PHOX2B" in adata_s.obs:
        _phox2b_tertiles_s = pd.qcut(
            adata_s.obs["ChromVAR_PHOX2B"], q=3, labels=["NonNE", "NE-intermediate", "NE-high"],
            duplicates="drop",
        )
        adata_s.obs["cell_type"] = pd.Categorical(
            _phox2b_tertiles_s, categories=["NonNE", "NE-intermediate", "NE-high"], ordered=True
        )
        print(
            f"[{SAMPLE}] Reassigned cell_type from ChromVAR PHOX2B tertiles: "
            f"{adata_s.obs['cell_type'].value_counts().to_dict()}"
        )

    # Subset chromvar to shared cells for rank correlation
    cell_to_col_s = pd.Series(chromvar_s.columns.tolist(), index=chromvar_cells_s)
    selected_cols_s = cell_to_col_s.reindex(shared_cells_s).dropna().values
    chromvar_s_cells = chromvar_s.loc[:, selected_cols_s]

    # Keep the raw 'counts' layer: adata_s.X in these h5ad files is already normalized
    # ("normcounts" per adata_s.uns["X_name"]), which silently breaks seurat_v3 HVG
    # selection (it expects raw counts) and double-normalizes if left in .X as-is.
    _counts_layer_s = adata_s.layers.get("counts")
    adata_s.obsm.clear()
    adata_s.varm.clear()
    adata_s.obsp.clear()
    adata_s.varp.clear()
    adata_s.layers.clear()
    adata_s.uns = {}
    if _counts_layer_s is not None:
        adata_s.X = _counts_layer_s.copy()
    else:
        print(f"[{SAMPLE}] WARNING: no 'counts' layer found; adata_s.X may not be raw counts.")

    # Normalize RNA for gene expression plots (seurat_v3 HVG needs raw counts first)
    adata_s_norm = adata_s.copy()
    sc.pp.highly_variable_genes(adata_s_norm, flavor='seurat_v3', n_top_genes=2000)
    sc.pp.normalize_total(adata_s_norm, target_sum=1e4)
    sc.pp.log1p(adata_s_norm)
    report_expr_chromvar_consistency(adata_s_norm, chromvar_s_cells, TFs_OF_INTEREST, SAMPLE)

    # Prefer the ChromVAR PHOX2B score over RNA expression to guide feature selection and
    # embedding scoring: it's the actual signal being plotted/concentrated, and RNA PHOX2B
    # is sparse and only weakly consistent with it (r=0.2-0.3, see report_expr_chromvar_consistency
    # above) -- optimizing against RNA was a weaker proxy for what we actually want concentrated.
    _phox2b_s = (
        _to_numpy_1d(adata_s.obs["ChromVAR_PHOX2B"].values)
        if "ChromVAR_PHOX2B" in adata_s.obs
        else _to_numpy_1d(adata_s_norm[:, "PHOX2B"].X)
        if "PHOX2B" in adata_s_norm.var_names
        else None
    )

    # PHOX2B-guided feature selection (expression only) to encourage one DC to follow PHOX2B.
    _phox2b_for_fs = _phox2b_s if _phox2b_s is not None else np.zeros(adata_s_norm.n_obs)
    _embed_features = select_embedding_features_by_phox2b(
        adata_s_norm,
        phox2b_signal=_phox2b_for_fs,
        n_hvg=2000,
        n_pos=200,
        n_neg=120,
    )
    print(f"[{SAMPLE}] embedding features: {len(_embed_features)} genes")

    # Build diffmap from selected expression features only.
    adata_s_embed = adata_s_norm[:, _embed_features].copy()
    adata_s_embed.var["highly_variable"] = True
    sc.pp.scale(adata_s_embed, max_value=10)

    # For small samples (e.g., RU581), use broader neighborhoods and gaussian graph for continuity.
    if adata_s_embed.n_obs < 350:
        _n_obs = int(adata_s_embed.n_obs)
        _nn_grid = sorted(
            set([
                min(max(8, int(round(_n_obs * 0.06))), _n_obs - 1),
                min(max(12, int(round(_n_obs * 0.10))), _n_obs - 1),
                min(max(16, int(round(_n_obs * 0.14))), _n_obs - 1),
                min(24, _n_obs - 1),
            ])
        )
        _pcs_grid = sorted(
            set([
                min(12, adata_s_embed.n_obs - 1, adata_s_embed.n_vars - 1),
                min(20, adata_s_embed.n_obs - 1, adata_s_embed.n_vars - 1),
                min(30, adata_s_embed.n_obs - 1, adata_s_embed.n_vars - 1),
            ])
        )
        _prefer_gauss = True
    else:
        _nn_grid = (15, 25, 35)
        _pcs_grid = (15, 25, min(35, adata_s_embed.n_obs - 1, adata_s_embed.n_vars - 1))
        _prefer_gauss = False

    best_cfg_s, metrics_df_s = optimize_expression_embedding(
        adata_s_embed,
        phox2b_signal=_phox2b_s if _phox2b_s is not None else np.zeros(adata_s_embed.n_obs),
        n_pcs_grid=_pcs_grid,
        n_neighbors_grid=_nn_grid,
        prefer_gauss=_prefer_gauss,
        method=EMBEDDING_METHOD,
    )
    embed_coords_s = best_cfg_s["coords"]
    print(
        f"[{SAMPLE}] best {EMBEDDING_METHOD} embedding: n_pcs={best_cfg_s['n_pcs']}, "
        f"n_neighbors={best_cfg_s['n_neighbors']}, "
        f"trust={best_cfg_s['trustworthiness']:.3f}, "
        f"overlap={best_cfg_s['knn_overlap']:.3f}, "
        f"phox2b_cv_corr={best_cfg_s['phox2b_cv_corr']:.3f}"
    )
    print(f"[{SAMPLE}] top settings:\n{metrics_df_s.sort_values('score', ascending=False).head(5).to_string(index=False)}")

    adata_s.obsm[f"X_{EMBEDDING_BASIS}"] = embed_coords_s
    adata_s_norm.obsm[f"X_{EMBEDDING_BASIS}"] = embed_coords_s.copy()

    # ChromVAR TF scores
    fig = sc.pl.embedding(
        adata_s, basis=EMBEDDING_BASIS,
        color=[f'ChromVAR_{tf}' for tf in TFs_OF_INTEREST],
        title=[f'ChromVAR {tf}' for tf in TFs_OF_INTEREST],
        vmin=-0.8, vmax=0.8,
        s=45, add_outline=True, return_fig=True, sort_order=True,
    )
    for ax in fig.axes:
        ax.title.set_fontsize(14)
    fig.figure.savefig(
        os.path.join(sc.settings.figdir, f"{EMBEDDING_BASIS}_{SAMPLE}_ChromVAR_TF.pdf"),
        dpi=300, bbox_inches="tight",
    )
    plt.close("all")

    # Cell type (plotted even when uniform, e.g. RU581's single "SCLC-A" label)
    _cell_type_palette = {"SCLC-A": "tab:red"} if SAMPLE == "RU581_LN_SCLC" else None
    fig = sc.pl.embedding(
        adata_s, basis=EMBEDDING_BASIS, color=["cell_type"],
        title=[f"{SAMPLE} cell type"],
        s=45, legend_loc="on data", legend_fontoutline=2,
        add_outline=True, return_fig=True,
        palette=_cell_type_palette,
    )
    for ax in fig.axes:
        ax.title.set_fontsize(14)
    fig.figure.savefig(
        os.path.join(sc.settings.figdir, f"{EMBEDDING_BASIS}_{SAMPLE}_cell_type.pdf"),
        dpi=300, bbox_inches="tight",
    )
    plt.close("all")

    # Gene expression
    _axis_s = _phox2b_s if _phox2b_s is not None else embed_coords_s[:, 0]
    genes_s = select_phox2b_genes(
        adata_s_norm, axis_values=_axis_s,
        base_genes=BASE_GENES_OF_INTEREST, n_pos=0, n_neg=0,
    )
    embed_genes_s = select_embedding_genes(adata_s_norm, embed_coords_s, n_per_axis=5)
    color_genes_s = list(dict.fromkeys(genes_s + [g for g in embed_genes_s if g not in genes_s]))
    print(f"[{SAMPLE}] {EMBEDDING_METHOD}-correlated genes: {', '.join(color_genes_s)}")

    fig = sc.pl.embedding(
        adata_s_norm, basis=EMBEDDING_BASIS, color=color_genes_s,
        frameon=False, wspace=0,
        ncols=min(5, max(1, len(color_genes_s))),
        use_raw=False, add_outline=True, return_fig=True, s=28, alpha=1,
    )
    for ax in fig.axes:
        ax.title.set_fontsize(16)
    fig.figure.savefig(
        os.path.join(sc.settings.figdir, f"{EMBEDDING_BASIS}_{SAMPLE}_embedding_genes.pdf"),
        dpi=300, bbox_inches="tight",
    )
    plt.close("all")

    # Ranked correlation (hockey stick)
    if TARGET_TF in chromvar_s_cells.index:
        correlations_s = chromvar_s_cells.T.corrwith(
            chromvar_s_cells.loc[TARGET_TF, :], method='pearson'
        ).drop(TARGET_TF).sort_values(ascending=False)

        rank_df_s = pd.DataFrame({
            'TF': correlations_s.index,
            'Correlation': correlations_s.values,
            'Rank': range(1, len(correlations_s) + 1),
        })

        fig, ax = plt.subplots(figsize=(3, 1.5))
        ax.scatter(rank_df_s['Rank'], rank_df_s['Correlation'], c='lightgrey', s=5, alpha=0.6)

        partner_s = rank_df_s[rank_df_s['TF'] == PARTNER_TF]
        if not partner_s.empty:
            ax.scatter(
                partner_s['Rank'], partner_s['Correlation'],
                c='red', s=50, edgecolors='black', linewidth=0.5,
            )
            ax.text(
                x=partner_s['Rank'].values[0] + 50,
                y=partner_s['Correlation'].values[0],
                s=f"{PARTNER_TF}\n(r={partner_s['Correlation'].values[0]:.2f})",
                color='red', fontsize=8, fontweight='bold', va='center',
            )

        ax.set_xlabel("Rank")
        ax.set_ylabel(f"r with {TARGET_TF}")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(sc.settings.figdir, f"rank_corr_{SAMPLE}_{TARGET_TF}_vs_Genome.pdf"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(
            f"[{SAMPLE}] {PARTNER_TF} is Rank "
            f"#{partner_s['Rank'].values[0] if not partner_s.empty else 'NaN'} "
            f"out of {len(rank_df_s)} TFs."
        )
