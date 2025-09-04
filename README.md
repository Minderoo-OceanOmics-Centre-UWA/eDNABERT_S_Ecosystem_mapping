# Calculate per-site embeddings using 12S/16S finetuned DNABERT-S


This script takes a FAIRe-formatted Excel sheet with ASV results of 12S (MiFishU, Miya et al) or 16S sequencing (Berry et al) ASVs, downloads the corresponding finetuned OceanOmics DNABERT-S models, and averages all embeddings for each site by weighing the embeddings using per-site read counts of all ASVs.

It then stores the per-site embeddings in a paraquet file, and optionally runs tSNE or UMAP on those embeddings to get per-site representations.

# Usage

    python calculate.py --cache-dir './cache' --asv-seqs asv_seqs.12S.parquet --reads asv_reads.12S.parquet --run-tsne   --outdir './results' --run-umap > stdout 2> stderr

# Installation

There's a conda environment with the DNABERT-S needed dependencies in DNABERT_S.yml

    conda env create -f DNABERT_S.yml

# Example data

This repository comes with a FAIRe-formatted excel sheet from an OceanOmics transect from Perth to the Cocos Keeling Islands, only 12S (MiFish-U, Miya et al.)

OceanOmics - Computational Biology\Bioinformatics\Results\V10_CKIP1\results_MiFishU_curateddb\07-faire
