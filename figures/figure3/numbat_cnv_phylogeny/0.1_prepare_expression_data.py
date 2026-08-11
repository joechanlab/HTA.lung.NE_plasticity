# RU151
import argparse
import os
from turtle import pd
import pandas
import numpy as np
import scanpy as sc
from sys import maxsize
import scipy.io
import scipy.sparse as sp
import multiprocessing as mp
from functools import partial

parser = argparse.ArgumentParser(description='Prepare expression data for Numbat')
parser.add_argument('--patient_id', type=str, required=True, help='ID of the patient to process')
parser.add_argument('--mode', type=str, default="All", choices=["All", "Tumor"], help='Whether to process all cells or only tumor cells')
args = parser.parse_args()
patient_id = args.patient_id.split("_")[0]
MODE = args.mode
if MODE == "Tumor":
    cell_type_col = 'cell_type_final2'
else:
    cell_type_col = 'cell_type_final'
NCORES = 7
CHEMISTRY_MAP = pandas.read_csv("bam_mapping.txt", delimiter='\t', header=None)
CHEMISTRY_MAP.columns = ["patient", "sample", "path", "chemistry"]
CHEMISTRY_MAP = dict(zip(CHEMISTRY_MAP['sample'], CHEMISTRY_MAP['chemistry']))

# from https://github.com/dpeerlab/seqc/blob/master/src/seqc/sequence/encodings.py

class DNA3Bit(object):
    """
    Compact 3-bit encoding scheme for sequence data.
    """
    
    @staticmethod
    def bits_per_base():
        return 3

    str2bindict = {65: 0b100, 67: 0b110, 71: 0b101, 84: 0b011, 78: 0b111,
                   97: 0b100, 99: 0b110, 103: 0b101, 116: 0b011, 110: 0b111,
                   'A': 0b100, 'C': 0b110, 'G': 0b101, 'T': 0b011, 'N': 0b111,
                   'a': 0b100, 'c': 0b110, 'g': 0b101, 't': 0b011, 'n': 0b111}
    bin2strdict = {0b100: b'A', 0b110: b'C', 0b101: b'G', 0b011: b'T', 0b111: b'N'}
    
    @staticmethod
    def encode(b) -> int:
        """
        Convert string nucleotide sequence into binary, note: string is stored so
        that the first nucleotide is in the MSB position

        :param bytes|str b: sequence containing nucleotides to be encoded
        """
        res = 0
        for c in b:
            res <<= 3
            res += DNA3Bit.str2bindict[c]
        return res
        
    @staticmethod
    def decode(i: int) -> bytes:
        """
        Convert binary nucleotide sequence into string

        :param i: int, encoded sequence to be converted back to nucleotides
        """
        if i < 0:
            message = 'i must be an unsigned (positive) integer, not {0!s}'.format(i)
            raise ValueError(message)
        r = b''
        while i > 0:
            r = DNA3Bit.bin2strdict[i & 0b111] + r
            i >>= 3
        return r
        
    # TODO: another ooption is to use i.bit_length and take into account preceding 0's
    @staticmethod
    def seq_len(i: int) -> int:
        """
        Return the length of an encoded sequence based on its binary representation

        :param i: int, encoded sequence
        """
        l = 0
        while i > 0:
            l += 1
            i >>= 3
        return l
        
    @staticmethod
    def contains(s: int, char: int) -> bool:
        """
        return true if the char (bin representation) is contained in seq (binary
        representation)

        :param char: int, encoded character (one must be only one nucleotide)
        :param s: int, sequence of encoded nucleotides
        """
        while s > 0:
            if char == (s & 0b111):
                return True
            s >>= 3
        return False
    
    @staticmethod
    def ints2int(ints):
        """
        convert an iterable of sequences [i1, i2, i3] into a concatenated single integer
        0bi1i2i3. In cases where the sequence is longer than 64 bits, python will
        transition seamlessly to a long int representation, however the user must be
        aware that downsteam interaction with numpy or other fixed-size representations
        may not function

        :param ints: iterable of encoded sequences to concatenate
        """

        res = 0
        for num in ints:
            tmp = num
            # Get length of next number to concatenate (with enough room for leading 0's)
            while tmp > 0:
                res <<= 3
                tmp >>= 3
            res += num
        return res
    
    @staticmethod
    def count(seq, char_bin):
        """
        count how many times char is in seq.
        char needs to be an encoded value of one of the bases.
        """
        if char_bin not in DNA3Bit.bin2strdict.keys():
            raise ValueError("DNA3Bit.count was called with an invalid char code - "
                             "{}".format(char_bin))
        res = 0
        while seq > 0:
            if seq & 0b111 == char_bin:
                res += 1
            seq >>= 3
        return res



# todo document me
def generate_hamming_dist_1(seq):
    """ Return a list of all sequences that are up to 1 hamming distance from seq
    :param seq:
    """
    res = []
    l = DNA3Bit.seq_len(seq)
    #=barcode
    
    # generate all sequences that are dist 1
    for i in range(l):
        mask = 0b111 << (i * 3)
        cur_chr = (seq & mask) >> (i * 3)
        res += [seq & (~mask) | (new_chr << (i * 3))
                for new_chr in DNA3Bit.bin2strdict.keys() if new_chr != cur_chr]

    return res


def find_correct_barcode(code, barcodes_list, exact_match=False):
    """
    For a given barcode find the closest correct barcode to it from the list (limited to
    one ED), a string representing the error and the edit distance
    NOTE: for now this function looks for a barcode with ED==1 and does not bother
    looking for the minimum

    :param exact_match:
    :param barcodes_list:
    :param code:
    :returns:
    """

    # Return the barcode if it exists
    if code in barcodes_list:
        return code, 0

    # If perfect match is required, return an error since the barcode does not appear
    # in the correct barcode list
    if exact_match:
        return 0, maxsize

    min_ed = maxsize
    cor_code = 0
    for bc in barcodes_list:
        hamm_d = hamming_dist_bin(code, bc)
        if hamm_d == 1:
            min_ed = 1
            cor_code = bc
            break
        if hamm_d < min_ed:
            min_ed = hamm_d
            cor_code = bc

    return cor_code, min_ed
        
        
def hamming_dist_bin(c1, c2):
    """Return the hamming distance between two numbers representing a sequence (3 bits
    per base)

    :param c1:
    :param c2:
    :return:
    """
    if DNA3Bit.seq_len(c1) != DNA3Bit.seq_len(c2):
        return maxsize
    d = 0
    while c1 > 0:
        if c1 & 0b111 != c2 & 0b111:
            d += 1
        c1 >>= 3
        c2 >>= 3
    return d


def list_errors(s1, s2):
    """
    Return the list of nucleotide transformations that turn s1 to s2.
    An error is a six bit int representing a two chr string of type "AG","CT", etc.

    :param s2:
    :param s1:

    :returns:
    """

    # return the actual error
    err_list = []
    while s1 > 0:
        if s1 & 0b111 != s2 & 0b111:
            err_list.append((s1 & 0b111, s2 & 0b111))
        s1 >>= 3
        s2 >>= 3
    return err_list

def process_prefix_group(prefix_data, full_adata, OUT_DIR):
    prefix, group_indices = prefix_data
    print(f"Processing {prefix}...")
    sub_adata = full_adata[group_indices].copy()
    barcodes = sub_adata.obs_names.str.split("_").str[-1].astype(int)
    encoder = DNA3Bit()
    if CHEMISTRY_MAP[prefix] =='ten_x_v2':
        print('processing chemistry v2')
        tenx_bcs = pandas.read_csv("/data1/chanj3/shared/barcodes.10X/v2/737K-august-2016.txt",
                        header=None).iloc[:,0].values
    elif (CHEMISTRY_MAP[prefix] =='ten_x_v3'):
        print('processing chemistry v3')
        tenx_bcs = pandas.read_csv("/data1/chanj3/shared/barcodes.10X/v3/3M-february-2018.txt",
                        header=None).iloc[:,0].values
    tenx_bcs_bit = [encoder.encode(bc) for bc in tenx_bcs]
    corrected = [find_correct_barcode(bc, tenx_bcs_bit) for bc in barcodes]
    decoded = [DNA3Bit.decode(bc[0]).decode('utf-8') + "-1" for bc in corrected]
    sub_adata.obs_names = [prefix + "_" + d for d in decoded]
    PREFIX_OUT_DIR = os.path.join(OUT_DIR, prefix)
    os.makedirs(PREFIX_OUT_DIR, exist_ok=True)
    counts_matrix = sp.csc_matrix(sub_adata.raw.X.T)
    scipy.io.mmwrite(os.path.join(PREFIX_OUT_DIR, f"{prefix}.expression.mtx"), counts_matrix)
    print(f"Saved expression matrix for {prefix}")
    pandas.DataFrame(sub_adata.obs_names).to_csv(
            os.path.join(PREFIX_OUT_DIR, "barcodes.tsv"), 
            sep='\t', index=False, header=False
        )
    print(f"Saved barcodes for {prefix}")
    sub_adata.obs.loc[:, [cell_type_col]].to_csv(
        os.path.join(PREFIX_OUT_DIR, f"{prefix}.cell_types.tsv"), 
        sep='\t', index=True, header=True
    )
    print(f"Saved cell types for {prefix}")
    pandas.DataFrame(sub_adata.var_names).to_csv(
        os.path.join(PREFIX_OUT_DIR, "genes.tsv"), 
        sep='\t', index=False, header=False
    )
    print(f"Saved genes for {prefix}")
    return 


if patient_id == "Healthy_Lung":
    adata = sc.read_h5ad("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/scvi_latent150/adata.scvi.all_genes.final_subset.062124.h5ad")
    healthy_lung = adata[adata.obs.cell_type.isin(['Tumor']) == False, :]
    OUT_DIR=f"/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_{MODE}/Healthy_Lung/GEX/"
    scipy.io.mmwrite(os.path.join(OUT_DIR, "healthy_lung.expression.mtx"), sp.csc_matrix(healthy_lung.raw.X.T))
    pandas.DataFrame(healthy_lung.var_names).to_csv(os.path.join(OUT_DIR, "genes.tsv"), sep='\t', index=False, header=False)
    pandas.DataFrame(healthy_lung.obs_names).to_csv(os.path.join(OUT_DIR, "barcodes.tsv"), sep='\t', index=False, header=False)
    pandas.DataFrame(healthy_lung.obs[['patient', 'sample', cell_type_col]]).to_csv(os.path.join(OUT_DIR, "healthy_lung.cell_annotations.tsv"), sep='\t', index=True, header=True)
    print("Finished processing Healthy Lung sample")
else:
    if MODE == "Tumor":
        if patient_id == 'RU151':
            adata = sc.read_h5ad(f"/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/{patient_id}.no_cc.hvg_2000.h5ad")
        else:
            adata = sc.read_h5ad(f"/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/{patient_id}.no_cc.hvg_2000.090124.h5ad")
        print(f"Finished loading tumor data for {patient_id}, shape: {adata.shape}")
    elif MODE == "All":
        adata = sc.read_h5ad("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/scvi_latent150/adata.scvi.all_genes.final_subset.062124.h5ad")
        adata = adata[adata.obs.patient == patient_id, :].copy()
        print(f"Finished loading data for {patient_id}, shape: {adata.shape}")
    OUT_DIR=f"/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_{MODE}/{patient_id}/GEX/"
    adata.obs['barcode_prefix'] = adata.obs_names.str.split("_").str[:-1].str.join("_")
    groups = [(prefix, group.index) for prefix, group in adata.obs.groupby('barcode_prefix')]
    with mp.Pool(processes=NCORES) as pool:
        worker = partial(process_prefix_group, full_adata=adata, OUT_DIR=OUT_DIR)
        results = pool.map(worker, groups)
