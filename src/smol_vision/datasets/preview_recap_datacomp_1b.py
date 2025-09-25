import dask.dataframe as dd

splits = {'preview': 'data/preview_data/preview-00000-of-00001.parquet', 'train': 'data/train_data/train-*.parquet'}
df = dd.read_parquet("hf://datasets/UCSC-VLAA/Recap-DataComp-1B/" + splits["preview"])
print(df.columns)