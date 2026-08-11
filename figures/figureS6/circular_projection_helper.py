"""Render a single patient's CellRank circular_projection (polytope) plot to PNG.

Runs under the `cellrank` conda env (/data1/chanj3/wangm10/conda_envs/cellrank),
which is not the notebook's own kernel (scdebussy-py311, which lacks cellrank).
Invoked via subprocess from s6.ipynb Panel D, one call per patient.
"""
import argparse
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scanpy as sc
import cellrank as cr
from cellrank._utils._lineage import Lineage

# same palette as Panel A/C in s6.ipynb, kept in sync manually since this script
# runs in a separate process/env and can't import the notebook's namespace
COL_DICT = {
    "LUAD": "gold",
    "LUSC": "orange",
    "SCLC-A": "tab:red",
    "SCLC-N": "tab:cyan",
    "SCLC-N (FOXI1+)": "tab:green",
    "NonNE SCLC": "tab:purple",
    "EMT": "#836953",
}
CELLTYPE_RENAME = {
    "NSCLC": "LUAD",
    "UBA52+ SCLC": "NonNE SCLC",
    "FOXI1+ NonNE SCLC-N": "SCLC-N (FOXI1+)",
}
# only for the polytope vertex text (not the ct_key/COL_DICT lookup, which
# must stay on the single-line canonical name) -- "SCLC-N (FOXI1+)" is long
# enough that it visibly widens the rotated-label bounding box
LABEL_WRAP = {"SCLC-N (FOXI1+)": "SCLC-N\n(FOXI1+)"}


def split_macrostate(cat):
    m = re.match(r"^(.*)_(\d+)$", cat)
    prefix, suffix = (m.group(1), m.group(2)) if m else (cat, None)
    return CELLTYPE_RENAME.get(prefix, prefix), suffix


def rename_macrostate(cat):
    prefix, suffix = split_macrostate(cat)
    display_prefix = LABEL_WRAP.get(prefix, prefix)
    return f"{display_prefix}_{suffix}" if suffix else display_prefix


def autocrop(path, pad=2):
    # cr.pl.circular_projection's own save doesn't use bbox_inches="tight", so
    # the saved PNG has a sizeable blank margin. Crop to content so the grid
    # cell can be filled edge-to-edge instead (combined with a smaller
    # label_distance below and aspect="auto" on imshow in the notebook). Some
    # empty corners are unavoidable -- a triangle/polygon doesn't tile a
    # rectangle as densely as a scatter plot's naturally rectangular extent.
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    nonwhite = np.where(np.any(arr < 250, axis=2))
    if nonwhite[0].size == 0:
        return
    y0, y1 = max(nonwhite[0].min() - pad, 0), min(nonwhite[0].max() + pad, arr.shape[0])
    x0, x1 = max(nonwhite[1].min() - pad, 0), min(nonwhite[1].max() + pad, arr.shape[1])
    img.crop((x0, y0, x1, y1)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adata = sc.read_h5ad(args.path)

    # rename macrostate prefixes (e.g. "NSCLC_1" -> "LUAD_1") and color every
    # lineage vertex/edge/label by its cell-type prefix, matching Panel A/C
    term_cats = adata.obs["term_states_fwd"].cat.categories.tolist()
    renamed_cats = [rename_macrostate(c) for c in term_cats]
    lineage_colors = [COL_DICT.get(split_macrostate(c)[0], "dimgray") for c in term_cats]
    adata.obsm["lineages_fwd"] = Lineage(
        adata.obsm["lineages_fwd"], names=renamed_cats, colors=lineage_colors
    )

    ct_key = "new_cell_type" if "new_cell_type" in adata.obs.columns else "cell_type_final2"
    adata.obs[ct_key] = (
        adata.obs[ct_key].astype(str).replace(CELLTYPE_RENAME).astype("category")
    )
    adata.uns[f"{ct_key}_colors"] = [
        COL_DICT.get(c, "dimgray") for c in adata.obs[ct_key].cat.categories
    ]

    cr.pl.circular_projection(
        adata, keys=[ct_key], lineages=renamed_cats,
        title="", size=10, legend_loc="none",
        dpi=200, figsize=(5, 5),
        # "optimal" solves a TSP over pairwise fate-probability distances so
        # that vertex adjacency reflects actual similarity between terminal
        # states; "default" just uses term_states_fwd's (alphabetical)
        # category order, which can place unrelated/anti-correlated fates
        # next to each other by coincidence (e.g. RU1518's EMT vs SCLC-N)
        lineage_order="optimal",
        label_distance=1.5,
        save=None,  # cellrank's own save_fig() hardcodes bbox_inches="tight" with
                    # matplotlib's default pad_inches=0.1 and doesn't expose it as a
                    # parameter -- that fixed margin is independent of label_distance,
                    # so save the current figure ourselves with a much smaller pad
    )
    plt.gcf().savefig(args.out, bbox_inches="tight", pad_inches=0.02, transparent=True)
    autocrop(args.out)


if __name__ == "__main__":
    main()
