# site id, latitude, longitude

import pandas as pd

df = pd.read_excel('../asv_final_faire_metadata.xlsx', sheet_name = 'sampleMetadata', skiprows = 2)
df = df[ ['samp_name', 'decimalLatitude', 'decimalLongitude']]
df = df.rename(columns={'samp_name':'site_id', 'decimalLatitude':'latitude', 'decimalLongitude':'longitude'})
df = df[df['latitude'].notnull()]

df.to_csv('coordinates.csv', index =False)
