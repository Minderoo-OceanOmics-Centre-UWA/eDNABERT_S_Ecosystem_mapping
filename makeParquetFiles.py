  #parser.add_argument("--asv-seqs", required=True, type=Path,
  #                      help="Path to asv_sequences.[csv|parquet] (columns: asv_id, assay, sequence)")
  #  parser.add_argument("--reads", required=True, type=Path,
  #                      help="Path to reads_long.[csv|parquet] (columns: site_id, assay, asv_id, reads)")

import pandas as pd
asvs = pd.read_excel('asv_final_faire_metadata.xlsx', sheet_name= 'taxaRaw', skiprows=2)
x = asvs.loc[:, ['seq_id', 'dna_sequence']]
x = x.drop_duplicates()
x = x.rename(columns = {'seq_id': 'asv_id', 'dna_sequence':'sequence'})
x['assay'] = '12S'
x[ ['asv_id', 'assay', 'sequence'] ].to_parquet('asv_seqs.12S.parquet')

reads = pd.read_excel('asv_final_faire_metadata.xlsx', sheet_name= 'otuRaw')
reads = reads.drop('Unnamed: 0', axis = 1)
reads = reads.rename(columns = {'ASV':'asv_id'})
reads_long = reads.drop('ASV_sequence', axis = 1).melt(id_vars = ['asv_id'], var_name = 'site_id', value_name = 'reads')
reads_long['assay'] = '12S'

reads_long[ ['site_id', 'assay', 'asv_id', 'reads']].to_parquet('asv_reads.12S.parquet')
