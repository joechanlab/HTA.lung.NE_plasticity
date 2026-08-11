"""
Table S8 - scDeBussy trajectory-associated genes and pathways.

Evidence table for the main-text statement that the basal (NSCLC) state
coincides with early enrichment of inflammatory signaling (NF-kB and
JAK/STAT programs), followed by stress-related pathways (AKT/mTOR
signaling, reactive oxygen species, and hypoxia response), along the
scDeBussy-aligned NSCLC -> SCLC-A -> SCLC-N transdifferentiation trajectory.
Also includes, as a second line of evidence, the CellMarker_2024 cell-type
identity programs (Secretory / Basal / Neuroendocrine / Fibroblast / Neural
Progenitor) run the same way in basal_ridge_plot.py (CellMarker_ridge_final.pdf).

Genes were GAM-smoothed along scDeBussy-aligned pseudotime (7 patients with
all three histologies: RU1083, RU151, RU1444, RU263, RU581, RU831, RU942;
34,227 cells; 2,060 trajectory-informative genes after HVG + GAM-quality
filtering) and ordered by their peak expression timing. Reference gene
programs (Spectra NE/NonNE library, and separately CellMarker_2024 lung/
trachea/brain marker sets) were tested for over-representation in a sliding
window along that ordering using a one-sided hypergeometric test (window
size 200 for Spectra, 300 for CellMarker -- matching basal_ridge_plot.py's
two separate calls); the reported statistics are for each program's single
best (peak-score) window. DNA repair is included as a late-trajectory
comparator, not a program discussed in the main text.

Reproduces, from the cached intermediate files written by
regenerate_gam_and_pathway_enrichment.py (SLURM job 2198966), the same
process_gene_data / clean_gene_sets / compute_running_enrichment logic as
basal_ridge_plot.py -- this script only re-runs the fast (~seconds-minutes)
downstream steps, not the GAM fitting or scDeBussy alignment themselves.
"""
import os
import pickle
import sys

import gseapy as gp
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Basal"))
from regenerate_gam_and_pathway_enrichment import (
    ENRICHMENT_STEP_SIZE,
    ENRICHMENT_WINDOW_SIZE,
    GAM_RESULTS_PATH,
    GENE_SET_PATH,
    GENE_SETS_OF_INTEREST,
    clean_gene_sets,
    compute_running_enrichment,
    process_gene_data,
)

OUT_DIR = "../../results/tables/figure4"
os.makedirs(OUT_DIR, exist_ok=True)

CELLMARKER_WINDOW_SIZE = 300  # matches basal_ridge_plot.py's "Final run" for CellMarker_2024

SPECTRA_PATHWAY_INFO = {
    "all_TNF-via-NFkB_signaling": "TNF signaling via NF-κB",
    "all_JAK-STAT_signaling": "JAK/STAT signaling",
    "all_IL6-JAK-STAT3_signaling": "IL6-JAK-STAT3 signaling",
    "all_hypoxia-response": "Hypoxia response",
    "all_mTORC1_signaling": "mTORC1 signaling",
    "all_PI3K-AKT-mTOR_signaling": "PI3K/AKT/mTOR signaling",
    "all_ros_response": "Reactive oxygen species response",
    "all_ROS-detoxification": "ROS detoxification",
    "all_DNA-repair": "DNA repair",
}

# Exactly the 6 CellMarker_2024 sets plotted in basal_ridge_plot.py's
# CellMarker_ridge_final.pdf (gene_sets_to_plot / short_names there).
CELLMARKER_SETS_OF_INTEREST = [
    "Secretory Cell Lung Human",
    "Krt4/13+ Cell Trachea Mouse",
    "Basal Cell Trachea Mouse",
    "Neuroendocrine Cell Trachea Mouse",
    "Fibroblast Lung Mouse",
    "Neural Progenitor Cell Embryonic Prefrontal Cortex Human",
]


def get_sorted_gene_curve_and_phase():
    with open(GAM_RESULTS_PATH, "rb") as fh:
        gam_results = pickle.load(fh)
    scores_df = gam_results["scores_df"]
    gene_curve = gam_results["gene_curve"]

    color_map = {"NSCLC": "gold", "SCLC-A": "tab:red", "SCLC-N": "tab:cyan"}
    colors = [color_map[x] for x in ["NSCLC", "SCLC-A"]]
    sorted_gene_curve, _row_colors, _col_colors, categories, _gene_order = process_gene_data(
        scores_df, gene_curve, colors, [1, 2],
        n_clusters=3, label_names=["Early", "Middle", "Late"],
        hierarchical=True, enforce_contiguity=False, band_width=10,
    )
    gene_to_phase = pd.Series(categories, index=sorted_gene_curve.index)
    return sorted_gene_curve, gene_to_phase


def peak_stats_to_records(enrichment_df, sets_of_interest, info_map, source, gene_to_phase, window_size):
    # Peak_trajectory_position_pct holds the raw gene-position index here; the
    # caller divides by total_genes and rounds once it knows that count.
    records = []
    for gs_key in sets_of_interest:
        sub = enrichment_df[enrichment_df["gene_set"] == gs_key]
        if len(sub) == 0:
            continue
        peak = sub.loc[sub["score"].idxmax()]
        overlap_genes = sorted(set(peak["window_genes"]) & set(peak["gene_set_present"]))
        phase_counts = gene_to_phase.reindex(overlap_genes).value_counts()
        peak_phase = phase_counts.index[0] if len(phase_counts) else "NA"
        display_name = info_map[gs_key]
        records.append({
            "Pathway": display_name,
            "Source": source,
            "Gene_set_ID": gs_key,
            "N_genes_in_program_detected": len(peak["gene_set_present"]),
            "Peak_trajectory_position_pct": peak["position"],  # filled in properly by caller
            "Peak_temporal_phase": peak_phase,
            "Enrichment_score": round(peak["score"], 3),
            "Peak_p_value": peak["pval"],
            "Significant_p_lt_0.05": bool(peak["pval"] < 0.05),
            "Enrichment_ratio_obs_over_expected": round(peak["enrichment_ratio"], 2),
            "N_overlapping_genes_at_peak": peak["overlap"],
            "Window_size": window_size,
            "Leading_genes_at_peak": ";".join(overlap_genes),
        })
    return records


def run_spectra_analysis(sorted_gene_curve, gene_to_phase):
    with open(GENE_SET_PATH, "rb") as fh:
        gene_set_dict = clean_gene_sets(pickle.load(fh)["global"])

    ordered_genes = list(sorted_gene_curve.index)
    total_genes = len(ordered_genes)
    enrichment_df = compute_running_enrichment(
        ordered_genes, gene_set_dict,
        window_size=ENRICHMENT_WINDOW_SIZE, step_size=ENRICHMENT_STEP_SIZE,
    )

    records = peak_stats_to_records(
        enrichment_df, GENE_SETS_OF_INTEREST, SPECTRA_PATHWAY_INFO,
        source="Spectra NE/NonNE", gene_to_phase=gene_to_phase, window_size=ENRICHMENT_WINDOW_SIZE,
    )
    for r in records:
        r["Peak_trajectory_position_pct"] = round(100.0 * r["Peak_trajectory_position_pct"] / total_genes, 1)
    return records


def run_cellmarker_analysis(sorted_gene_curve, gene_to_phase):
    print("Fetching CellMarker_2024 library from Enrichr...")
    full_library = gp.get_library(name="CellMarker_2024")
    lung_trachea_brain_terms = [
        name for name in full_library.keys()
        if any(term in name.lower() for term in ["lung", "trachea", "brain", "prefrontal cortex"])
    ]
    gene_set_dict = {name: full_library[name] for name in lung_trachea_brain_terms}
    gene_set_dict = clean_gene_sets(gene_set_dict)
    print(f"{len(gene_set_dict)} CellMarker_2024 lung/trachea/brain gene sets after cell-cycle-gene cleanup.")

    ordered_genes = list(sorted_gene_curve.index)
    total_genes = len(ordered_genes)
    enrichment_df = compute_running_enrichment(
        ordered_genes, gene_set_dict,
        window_size=CELLMARKER_WINDOW_SIZE, step_size=ENRICHMENT_STEP_SIZE,
    )

    info_map = {name: name for name in CELLMARKER_SETS_OF_INTEREST}
    records = peak_stats_to_records(
        enrichment_df, CELLMARKER_SETS_OF_INTEREST, info_map,
        source="CellMarker_2024", gene_to_phase=gene_to_phase, window_size=CELLMARKER_WINDOW_SIZE,
    )
    for r in records:
        r["Peak_trajectory_position_pct"] = round(100.0 * r["Peak_trajectory_position_pct"] / total_genes, 1)
    return records


def main():
    sorted_gene_curve, gene_to_phase = get_sorted_gene_curve_and_phase()

    spectra_records = run_spectra_analysis(sorted_gene_curve, gene_to_phase)
    cellmarker_records = run_cellmarker_analysis(sorted_gene_curve, gene_to_phase)

    table = pd.DataFrame(spectra_records + cellmarker_records)
    table = table.sort_values(["Source", "Enrichment_score"], ascending=[True, False]).reset_index(drop=True)

    csv_path = os.path.join(OUT_DIR, "Table_S8_scDeBussy_trajectory_pathways.csv")
    table.to_csv(csv_path, index=False)

    print(f"Wrote {len(table)} rows ({len(spectra_records)} pathway + {len(cellmarker_records)} cell-marker)")
    print(table[["Pathway", "Source", "Peak_trajectory_position_pct", "Peak_temporal_phase",
                  "Peak_p_value", "Significant_p_lt_0.05"]].to_string(index=False))
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
