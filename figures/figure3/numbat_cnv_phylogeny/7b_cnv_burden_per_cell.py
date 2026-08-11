#!/usr/bin/env python3
"""
7b_cnv_burden_per_cell.py

Compute per-cell CNV burden from Numbat joint_post_filtered output.

Three metrics are computed for each cell:
  cnv_burden_bp   : Σ(seg_len × p_cnv) / autosomal_genome_size
                    Probabilistic fraction of genome altered; directly
                    comparable to CNV-clock FGA (uses soft p_cnv weights
                    instead of hard calls).

  cnv_burden_fga  : Σ(seg_len for segments where p_cnv ≥ threshold) /
                    autosomal_genome_size
                    Hard-call FGA; matches the binary definition used by
                    CNV clock and ABSOLUTE.

  cnv_burden_segs : mean(p_cnv) across all non-neutral segments
                    Segment-count-weighted burden; analogous to the
                    inferCNV per-cell CNV score (mean posterior deviation).

If --gene-bed is supplied (BED3 file: chrom, start, end, one row per gene),
a fourth metric is added:
  cnv_burden_gene : Σ(n_genes_in_seg × p_cnv) / total_autosomal_genes
                    Closest analog to the inferCNV CNV score computed over
                    individual genes rather than segments.

Output: cnv_burden_per_cell.tsv          (metacell, default)
        cnv_burden_per_cell_sc.tsv       (--single-cell)
  patient_id | cell | cell_type | sample_id | n_segs |
  cnv_burden_bp | cnv_burden_fga | cnv_burden_segs [| cnv_burden_gene]

Usage:
    python 7b_cnv_burden_per_cell.py [--patients RU151 ...]
                                     [--fga-threshold 0.5]
                                     [--gene-bed /path/to/genes.bed]
                                     [--single-cell]
"""

import argparse
import gzip
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

# GRCh38 autosomal chromosome lengths (chr1–22), bp
CHROM_LENGTHS_HG38 = {
    1: 248_956_422, 2: 242_193_529, 3: 198_295_559, 4: 190_214_555,
    5: 181_538_259, 6: 170_805_979, 7: 159_345_973, 8: 145_138_636,
    9: 138_394_717, 10: 133_797_422, 11: 135_086_622, 12: 133_275_309,
    13: 114_364_328, 14: 107_043_718, 15: 101_991_189, 16:  90_338_345,
    17:  83_257_441, 18:  80_373_285, 19:  58_617_616, 20:  64_444_167,
    21:  46_709_983, 22:  50_818_468,
}
AUTOSOMAL_GENOME_BP = sum(CHROM_LENGTHS_HG38.values())  # 2,875,001,522

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
BASE             = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
OUTDIR_METACELL  = "numbat_output_FACETS_HISENS_metacell"
OUTDIR_SC        = "numbat_output_FACETS_HISENS"

# CellRanger GRCh38 reference GTF — same gene set used during scRNA-seq processing
GENE_GTF_DEFAULT = ("/admin/software/cellranger/"
                    "refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz")


# ── Gene annotation helpers ───────────────────────────────────────────────────

def parse_gene_gtf(path: str) -> pd.DataFrame:
    """
    Parse a GTF (plain or gzipped) and return autosomal gene loci.
    Keeps only feature == 'gene'. Strips 'chr' prefix; drops non-numeric chroms.
    Returns DataFrame with columns: chrom (int), start (int), end (int).
    Coordinates are converted to 0-based half-open [start, end).
    """
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom = parts[0].lstrip("chr")
            if not chrom.isdigit():
                continue
            chrom_int = int(chrom)
            if chrom_int not in CHROM_LENGTHS_HG38:
                continue
            rows.append((chrom_int, int(parts[3]) - 1, int(parts[4])))

    genes = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    print(f"  Gene annotation: {len(genes):,} autosomal genes from {os.path.basename(path)}")
    return genes


def load_gene_bed(path: str) -> pd.DataFrame:
    """
    Load a BED3 gene file (chrom, start, end). Chromosome column may be
    'chr1' or '1'; both are normalised to int. Returns chrom (int), start, end.
    """
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chrom", "start", "end"])
    df["chrom"] = df["chrom"].astype(str).str.lstrip("chr")
    df = df[df["chrom"].str.match(r"^\d+$")]
    df["chrom"] = df["chrom"].astype(int)
    df = df[df["chrom"].isin(CHROM_LENGTHS_HG38)]
    return df.reset_index(drop=True)


def count_genes_per_segment(segs: pd.DataFrame,
                             genes_by_chrom: dict) -> list:
    """
    For each row in segs (CHROM, seg_start, seg_end), count overlapping genes.
    genes_by_chrom: dict chrom_int → (starts_array, ends_array) pre-grouped.
    Uses vectorised numpy comparison per chromosome — fast for ~60 segments.
    """
    counts = []
    for _, row in segs.iterrows():
        c = int(row["CHROM"])
        s = int(row["seg_start"])
        e = int(row["seg_end"])
        if c not in genes_by_chrom:
            counts.append(0)
            continue
        g_starts, g_ends = genes_by_chrom[c]
        counts.append(int(((g_starts < e) & (g_ends > s)).sum()))
    return counts


def build_genes_by_chrom(genes_df: pd.DataFrame) -> dict:
    """Pre-group genes by chromosome into numpy arrays for fast overlap queries."""
    result = {}
    for chrom, grp in genes_df.groupby("chrom"):
        result[chrom] = (grp["start"].values, grp["end"].values)
    return result


# ── Per-patient computation ───────────────────────────────────────────────────

def compute_burden(patient, df_raw, fga_threshold, genes_by_chrom, total_genes):
    """
    df_raw        : joint_post_filtered DataFrame for this patient.
    genes_by_chrom: dict chrom → (starts, ends) from build_genes_by_chrom(),
                    or None to skip gene-weighted burden.
    total_genes   : total autosomal gene count (denominator for cnv_burden_gene).
    Returns a per-cell DataFrame.
    """
    df = df_raw.copy()
    df["seg_len"] = (df["seg_end"] - df["seg_start"] + 1).clip(lower=1)
    df["patient_id"] = patient

    # sample_id: strip the trailing barcode/SEACell component
    # SC cells  : RU151_PDX_TTTGGTTCAGGTCGTC-1  → RU151_PDX
    # Metacells : RU151_PDX_SEACell-0            → RU151_PDX
    df["sample_id"] = df["cell"].apply(
        lambda x: x.rsplit("_SEACell", 1)[0] if "SEACell" in x
                  else x.rsplit("_", 1)[0]
    )

    # Gene counts per segment (computed once per patient, broadcast to all cells)
    if genes_by_chrom is not None:
        seg_meta = (
            df[["seg", "CHROM", "seg_start", "seg_end"]]
            .drop_duplicates("seg")
            .reset_index(drop=True)
        )
        seg_meta["n_genes"] = count_genes_per_segment(seg_meta, genes_by_chrom)
        df = df.merge(seg_meta[["seg", "n_genes"]], on="seg", how="left")

    records = []
    for cell_id, grp in df.groupby("cell", sort=False):
        n_segs = len(grp)
        burden_bp   = (grp["seg_len"] * grp["p_cnv"]).sum() / AUTOSOMAL_GENOME_BP
        called      = grp[grp["p_cnv"] >= fga_threshold]
        burden_fga  = called["seg_len"].sum() / AUTOSOMAL_GENOME_BP
        burden_segs = grp["p_cnv"].mean()

        rec = {
            "patient_id":      patient,
            "cell":            cell_id,
            "cell_type":       grp["cell_type"].iloc[0] if "cell_type" in grp.columns else None,
            "sample_id":       grp["sample_id"].iloc[0],
            "n_segs":          n_segs,
            "cnv_burden_bp":   round(burden_bp,   6),
            "cnv_burden_fga":  round(burden_fga,  6),
            "cnv_burden_segs": round(burden_segs, 6),
        }

        if genes_by_chrom is not None:
            burden_gene = (grp["n_genes"] * grp["p_cnv"]).sum() / total_genes
            rec["cnv_burden_gene"] = round(burden_gene, 6)
            # Binary (hard-threshold) gene-weighted FGA
            burden_gene_bin = grp.loc[grp["p_cnv"] >= fga_threshold, "n_genes"].sum() / total_genes
            rec["cnv_burden_gene_bin"] = round(burden_gene_bin, 6)

        records.append(rec)

    return pd.DataFrame(records)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-cell CNV burden from Numbat joint_post_filtered")
    parser.add_argument("--patients", nargs="*",
                        help="Restrict to these patient IDs")
    parser.add_argument("--fga-threshold", type=float, default=0.5,
                        help="p_cnv threshold for hard-call FGA (default: 0.5)")
    parser.add_argument("--gene-bed", default=None,
                        help="BED3 gene annotation for gene-weighted burden. "
                             "Defaults to the CellRanger GRCh38 GTF if available.")
    parser.add_argument("--no-gene-counts", action="store_true",
                        help="Skip gene-weighted burden (cnv_burden_gene).")
    parser.add_argument("--single-cell", action="store_true",
                        help="Use single-cell Numbat output (numbat_output_FACETS_HISENS). "
                             "Auto-discovers patients from directory; outputs "
                             "cnv_burden_per_cell_sc.tsv.")
    args = parser.parse_args()

    sc_mode   = args.single_cell
    outdir    = OUTDIR_SC if sc_mode else OUTDIR_METACELL
    out_tsv   = os.path.join(SCRIPT_DIR,
                              "cnv_burden_per_cell_sc.tsv" if sc_mode
                              else "cnv_burden_per_cell.tsv")

    # ── Load gene annotation ──────────────────────────────────────────────────
    genes_by_chrom = None
    total_genes    = 1
    if not args.no_gene_counts:
        if args.gene_bed:
            print(f"Loading gene BED: {args.gene_bed}")
            genes_df = load_gene_bed(args.gene_bed)
        elif os.path.exists(GENE_GTF_DEFAULT):
            print(f"Loading gene annotation: {GENE_GTF_DEFAULT}")
            genes_df = parse_gene_gtf(GENE_GTF_DEFAULT)
        else:
            print("No gene annotation found; skipping cnv_burden_gene. "
                  "Provide --gene-bed or ensure CellRanger GTF is available.")
            genes_df = None

        if genes_df is not None and not genes_df.empty:
            genes_by_chrom = build_genes_by_chrom(genes_df)
            total_genes    = len(genes_df)
            if total_genes == 0:
                total_genes = 1

    if sc_mode:
        # Discover all patients that have a joint_post_filtered in the SC directory
        pid_entries = sorted([
            d for d in os.listdir(BASE)
            if os.path.isfile(os.path.join(BASE, d, outdir,
                                           f"{d}_joint_post_filtered.csv"))
        ])
        print(f"Single-cell mode: found {len(pid_entries)} patients in {outdir}")
    else:
        patient_list_file = os.path.join(SCRIPT_DIR, "patient_list.txt")
        with open(patient_list_file) as f:
            pid_entries = [l.strip() for l in f if l.strip()]

    subset = set(args.patients) if args.patients else None
    all_records = []

    for pid_entry in pid_entries:
        patient = pid_entry.split("_")[0]
        if subset and patient not in subset and pid_entry not in subset:
            continue

        jp_path = os.path.join(BASE, patient, outdir,
                               f"{patient}_joint_post_filtered.csv")
        if not os.path.exists(jp_path):
            # Fallback: sample-level file (e.g. RU1518_T inside RU1518 dir)
            fallback = os.path.join(BASE, patient, outdir, pid_entry,
                                    f"{pid_entry}_joint_post_filtered.csv")
            if os.path.exists(fallback):
                jp_path = fallback
                print(f"[{patient}] using sample-level fallback: {fallback}")
            else:
                print(f"[{patient}] joint_post_filtered not found — skipping")
                continue

        print(f"[{patient}] {jp_path}")
        usecols = ["cell", "CHROM", "seg", "cnv_state",
                   "seg_start", "seg_end", "p_cnv", "cell_type"]
        try:
            df_raw = pd.read_csv(jp_path, usecols=usecols)
        except ValueError:
            df_raw = pd.read_csv(jp_path)

        n_cells = df_raw["cell"].nunique()
        n_segs  = df_raw["seg"].nunique()
        print(f"  {n_cells} cells × {n_segs} non-neutral segments")

        burden_df = compute_burden(patient, df_raw, args.fga_threshold,
                                   genes_by_chrom, total_genes)
        all_records.append(burden_df)

        # Quick per-cell-type summary
        if "cell_type" in burden_df.columns:
            summary = (
                burden_df.groupby("cell_type")["cnv_burden_bp"]
                .agg(["mean", "median", "count"])
                .round(4)
            )
            print(summary.to_string())

    if not all_records:
        print("No data collected.")
        sys.exit(1)

    out_df = pd.concat(all_records, ignore_index=True)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nOutput: {out_tsv}  ({len(out_df):,} cells, "
          f"{out_df['patient_id'].nunique()} patients)")
    print(f"\nColumn descriptions:")
    print(f"  cnv_burden_bp   — probabilistic FGA (p_cnv-weighted bp / {AUTOSOMAL_GENOME_BP:,} bp)")
    print(f"  cnv_burden_fga  — binary FGA (p_cnv ≥ {args.fga_threshold} / {AUTOSOMAL_GENOME_BP:,} bp)")
    print(f"  cnv_burden_segs — mean p_cnv across non-neutral segments")
    if genes_by_chrom is not None:
        print(f"  cnv_burden_gene — gene-weighted FGA "
              f"(p_cnv-weighted gene count / {total_genes:,} autosomal genes)")


if __name__ == "__main__":
    main()
