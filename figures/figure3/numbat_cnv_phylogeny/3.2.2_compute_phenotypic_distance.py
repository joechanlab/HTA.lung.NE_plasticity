import os
import scanpy as sc
import pandas as pd
import plastro
import argparse

def compute_sample_phenotype(sample_id, h5ad_path, output_file, embedding_key):
    print(f"--- Processing Phenotype: {sample_id} ---")
    try:
        adata = sc.read_h5ad(h5ad_path)
        print(f"Calculating distance matrix for {adata.n_obs} cells...")
        pheno_dist = plastro.overlap.compute_phenotype_distances(adata, embedding_key)
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pheno_dist.to_csv(output_file)
        print(f"Successfully saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error processing {sample_id}: {str(e)}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute phenotype distance matrix for one patient/sample."
    )
    parser.add_argument("patient_id", help="Patient/sample ID (e.g., RU151)")
    parser.add_argument(
        "--base-data-path",
        default="/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/",
        help="Directory containing patient .h5ad files."
    )
    parser.add_argument(
        "--numbat-base",
        default="/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/",
        help="Base directory for Numbat per-patient folders."
    )
    parser.add_argument(
        "--embedding-key",
        default="X_pca_slalom",
        help="AnnData obsm key used to compute phenotype distances."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sid = args.patient_id

    h5_path = os.path.join(args.base_data_path, f"{sid}.no_cc.hvg_2000.090124.h5ad")
    output_file = os.path.join(args.numbat_base, sid, "GEX", f"{sid}_phenotype_distances.csv")

    compute_sample_phenotype(
        sid,
        h5_path,
        output_file,
        args.embedding_key,
    )


if __name__ == "__main__":
    main()