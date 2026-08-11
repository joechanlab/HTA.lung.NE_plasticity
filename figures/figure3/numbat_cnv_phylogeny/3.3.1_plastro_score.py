import pandas as pd
import numpy as np
import plastro
import os
import argparse
import re

STR2BINDICT = {
    65: 0b100, 67: 0b110, 71: 0b101, 84: 0b011, 78: 0b111,
    97: 0b100, 99: 0b110, 103: 0b101, 116: 0b011, 110: 0b111,
    'A': 0b100, 'C': 0b110, 'G': 0b101, 'T': 0b011, 'N': 0b111,
    'a': 0b100, 'c': 0b110, 'g': 0b101, 't': 0b011, 'n': 0b111
}


def encode(sequence) -> int:
    res = 0
    for nucleotide in sequence:
        res <<= 3
        res += STR2BINDICT[nucleotide]
    return res


def translate_lineage_barcode(barcode: str) -> str:
    patterns = [
        r"^(.*)_\[?([ACGTNacgtn]+)\]?-1$",
        r"^(.*)-\[?([ACGTNacgtn]+)\]?-1$",
    ]
    for pattern in patterns:
        match = re.match(pattern, barcode)
        if match:
            prefix, sequence = match.groups()
            return f"{prefix}_{encode(sequence)}"
    return barcode


def translate_lineage_matrix_ids(lineage_dist: pd.DataFrame) -> pd.DataFrame:
    translated_columns = [translate_lineage_barcode(str(col)) for col in lineage_dist.columns]
    translated_index = [translate_lineage_barcode(str(idx)) for idx in lineage_dist.index]
    lineage_dist = lineage_dist.copy()
    lineage_dist.columns = translated_columns
    lineage_dist.index = translated_index
    common = lineage_dist.index.intersection(lineage_dist.columns)
    lineage_dist = lineage_dist.loc[common, common]
    return lineage_dist


def load_lineage_distance_matrix(lineage_path: str) -> pd.DataFrame:
    with_index = pd.read_csv(lineage_path, index_col=0)
    if with_index.shape[0] == with_index.shape[1]:
        return with_index

    without_index = pd.read_csv(lineage_path, index_col=None)
    if without_index.shape[0] == without_index.shape[1]:
        without_index.index = without_index.columns
        return without_index

    raise ValueError(
        f"Unsupported lineage matrix shape for {lineage_path}: "
        f"index_col=0 -> {with_index.shape}, index_col=None -> {without_index.shape}"
    )

def run_plastro_for_sample(sample_id, base_dir, radii=np.linspace(0, 1, 100)):
    print(f"--- Processing {sample_id} ---")
    
    # 1. Define Paths
    lineage_path = os.path.join(base_dir, sample_id, 'numbat_output_FACETS_HISENS', f"{sample_id}_pairwise_distances.csv")
    pheno_path = os.path.join(base_dir, sample_id, 'GEX', f"{sample_id}_phenotype_distances.csv") 
    print(f"lineage_path: {lineage_path}")
    print(f"pheno_path: {pheno_path}")
    try:
        lineage_dist = load_lineage_distance_matrix(lineage_path)
        lineage_dist = translate_lineage_matrix_ids(lineage_dist)
        pheno_dist = pd.read_csv(pheno_path, index_col=0)

        common_cells = lineage_dist.index.intersection(pheno_dist.index)
        print(f"Matched {len(common_cells)} common cells after barcode translation.")
        L = lineage_dist.loc[common_cells, common_cells]
        P = pheno_dist.loc[common_cells, common_cells]

        overlaps = plastro.overlap._compute_overlaps_precomputed_optimized(
            L, P, radii
        )
        decoupling_scores = 1 - overlaps.mean(axis=1)

        results = pd.DataFrame({
            'cell': common_cells,
            'mean_overlap': overlaps.mean(axis=1),
            'plastro_score': decoupling_scores
        }).set_index('cell')

        return results

    except Exception as e:
        print(f"Error processing {sample_id}: {str(e)}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Plastro scores for one or more patients/samples."
    )
    parser.add_argument(
        "patients",
        nargs="+",
        help="One or more patient/sample IDs (e.g., RU151 RU1518 RU1444)."
    )
    parser.add_argument(
        "--base-dir",
        default="/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/",
        help="Base directory containing per-patient input files."
    )
    parser.add_argument(
        "--radii-steps",
        type=int,
        default=100,
        help="Number of radii points (default: 100)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    interval = 1
    radii = np.arange(1, args.radii_steps + 1, interval)
    all_results = {}

    for sid in args.patients:
        sid = sid.split("_")[0] 
        print(sid)
        res = run_plastro_for_sample(sid, args.base_dir, radii=radii)
        if res is not None:
            all_results[sid] = res
            output_path = os.path.join(args.base_dir, sid, 'numbat_output_FACETS_HISENS', f"{sid}_plastro_results.csv")
            print(f"Saving results to {output_path}")
            res.to_csv(output_path)

    print("Pipeline Complete.")


if __name__ == "__main__":
    main()
