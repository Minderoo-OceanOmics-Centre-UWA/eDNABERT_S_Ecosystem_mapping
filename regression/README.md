# Regression experiments


baseline_regress.py - regresses on decimalLatitude using encoded ASV species names and ASV counts

regress.py - regresses on decimalLatitude using embeddings alone

download_sst_from_oisst.py - downloads SST for our samples


## COMMANDS

python baseline_regress.py --asv-data asv_lca_with_fishbase_output.tsv --sample-metadata samplemetadata.tsv --output regress --model xgb --optimize-hyperparams > regress/regress_log.txt

python regress.py --site-embeddings ../analysis_for_paper/ckip10_results_perplexity5/site_embeddings_12S.parquet --coordinates samplemetadata.tsv --model xgb --optimize-hyperparams --output embedding_regress_output > embedding_regress_output/embedding_regress_log.txt


#### ON TEMPERATURES

python download_sst_from_oisst.py --input samplemetadata.tsv --output samplemetadata_sst.tsv --dataset final --vars sst anom err --interp nearest --insecure

python regress_sst.py --site-embeddings ../analysis_for_paper/ckip10_results_perplexity5/site_embeddings_12S.parquet --metadata samplemetadata.tsv --sst-table samplemetadata_sst.tsv --model xgb --optimize-hyperparams --output sst_embedding_regress_output > sst_embedding_regress_output/sst_embedding_regress_log.txt

python baseline_regress_sst.py --asv-data asv_lca_with_fishbase_output.tsv --sst-table samplemetadata_sst.tsv --sample-metadata samplemetadata.tsv --output baseline_regress_sst --model xgb --optimize-hyperparams > baseline_regress_sst/baseline_regress_sst_log.txt

python baseline_regress_sst.py --asv-data asv_lca_with_fishbase_output.tsv --sst-table samplemetadata_sst.tsv --sample-metadata samplemetadata.tsv --output rf_baseline_regress_sst --model rf --optimize-hyperparams > rf_baseline_regress_sst/rf_baseline_regress_sst_log.txt
