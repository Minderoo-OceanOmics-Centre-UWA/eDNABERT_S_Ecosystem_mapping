# Calculate per-site embeddings using 12S/16S finetuned DNABERT-S

Using eDNA-data while ignoring taxonomy to learn more about ecosystems!

This script takes a FAIRe-formatted Excel sheet with ASV results of 12S (MiFishU, Miya et al) or 16S sequencing (Berry et al) ASVs, downloads the corresponding finetuned OceanOmics DNABERT-S models, and averages all embeddings for each site by weighing the embeddings using per-site read counts of all ASVs. We get per-site embeddings!

It then stores the per-site embeddings in a paraquet file, and optionally runs tSNE or UMAP on those embeddings to get per-site representations.

This is what the tSNE clustering of those per-site embeddings looks like along a latitude gradient, split up by sampling device:

<img width="1024" height="768" alt="image" src="https://github.com/user-attachments/assets/62fa01e0-a8f8-4b61-b62f-e41ee22265fc" />

Each dot is one site, colored by their latitude/longitude.

# Usage

    python calculate.py --cache-dir './cache' --12s-files run1.xlsx --16s-files run2.xlsx
        --run-tsne   --outdir './results' --run-umap

<pre>
usage: calculatePerSite.py [-h] [--12s-files [12S_FILES ...]] [--16s-files [16S_FILES ...]] --outdir OUTDIR [--cache-dir CACHE_DIR]
                           [--min-asv-length MIN_ASV_LENGTH] [--max-asv-length MAX_ASV_LENGTH] [--force] [--model-12s MODEL_12S] [--model-16s MODEL_16S]
                           [--base-config BASE_CONFIG] [--pooling-token {mean,cls}] [--batch-size BATCH_SIZE] [--use-amp] [--max-length MAX_LENGTH]
                           [--weight-mode {hellinger,log,relative,softmax_tau3}] [--site-pooling {l2_weighted_mean,weighted_mean,gem_p2,gem_p3}]
                           [--run-tsne] [--run-umap] [--perplexity PERPLEXITY] [--n-neighbors N_NEIGHBORS] [--metric {cosine,euclidean}] [--seed SEED]
                           [--fuse {none,concat}]

eDNA DNABERT-S embedding pipeline (Excel -> ASVs -> sites -> t-SNE/UMAP)

optional arguments:
  -h, --help            show this help message and exit
  --12s-files [12S_FILES ...]
                        Path(s) to Excel file(s) containing 12S data
  --16s-files [16S_FILES ...]
                        Path(s) to Excel file(s) containing 16S data
  --outdir OUTDIR       Output directory
  --cache-dir CACHE_DIR
                        HuggingFace cache dir (optional)
  --min-asv-length MIN_ASV_LENGTH
                        Minimum ASV sequence length (optional)
  --max-asv-length MAX_ASV_LENGTH
                        Maximum ASV sequence length (optional)
  --force               Force recalculation of all steps, ignoring existing intermediate files
  --model-12s MODEL_12S
  --model-16s MODEL_16S
  --base-config BASE_CONFIG
  --pooling-token {mean,cls}
  --batch-size BATCH_SIZE
  --use-amp             Enable mixed precision on CUDA
  --max-length MAX_LENGTH
                        Longest tokenized length for the tokenizer
  --weight-mode {hellinger,log,relative,softmax_tau3}
  --site-pooling {l2_weighted_mean,weighted_mean,gem_p2,gem_p3}
  --run-tsne
  --run-umap
  --perplexity PERPLEXITY
                        Perplexity setting for tSNE
  --n-neighbors N_NEIGHBORS
                        Number of neighbours for UMAP
  --metric {cosine,euclidean}
  --seed SEED
  --fuse {none,concat}  How to fuse 12S+16S site vectors (concat or none)

</pre>

# Results

The results folder will contain at least two files: the per-ASV embeddings in parquet and the per-site embeddings in parquet.
If you've turned on `run-tsne` and/or `run-umap`, there will be CSV files with TSNE1/TSNE2 and UMAP1/UMAP2 values for all sites.

## Runtime

It usually runs only for a few minutes, but so far I've only tested it on systems without GPUs.

# Installation

There's a conda environment with the DNABERT-S needed dependencies in DNABERT_S.yml

    conda env create -f DNABERT_S.yml

# AI statement

Most of this code is written by GPT5 after a long discussion! Every mode has heaps options, human-me would've never done that. 

# Regression

I am working on predicting latitude/longitude from the site embeddings alone. That is happening in `regress.py`. I am unclear how to handle replicates best as there's a lot of variability in between - by not accounting for replicates I get a test R2 of about 0.7, with grouped KFold stratification by replicate it goes down to 0.5 (better). 

# Example data

This repository comes with a FAIRe-formatted excel sheet from an OceanOmics transect from Perth to the Cocos Keeling Islands, only 12S (MiFish-U, Miya et al.)
