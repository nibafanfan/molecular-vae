from types import SimpleNamespace
import pandas as pd

import biobricks as bb
ns = bb.assets('tox21') # get the paths for the 'tox21' brick

# Access the attributes
tox21_parquet_path = ns.tox21_parquet
tox21lib_parquet_path = ns.tox21lib_parquet
tox21_aggregated_parquet_path = ns.tox21_aggregated_parquet

# Now you have the paths in variables
print(tox21_parquet_path)
print(tox21lib_parquet_path)
print(tox21_aggregated_parquet_path)

# Reading the Parquet files using pandas
tox21_df = pd.read_parquet(ns.tox21_parquet)
tox21lib_df = pd.read_parquet(ns.tox21lib_parquet)
tox21_aggregated_df = pd.read_parquet(ns.tox21_aggregated_parquet)

# Print the size of each DataFrame
print(f"tox21_df size: {tox21_df.shape}")
print(f"tox21lib_df size: {tox21lib_df.shape}")
print(f"tox21_aggregated_df size: {tox21_aggregated_df.shape}")

# Print the first element of each DataFrame vertically
print("First element of tox21_df:")
print(tox21_df.iloc[0].to_frame())
print("\nFirst element of tox21lib_df:")
print(tox21lib_df.iloc[0].to_frame())
print("\nFirst element of tox21_aggregated_df:")
print(tox21_aggregated_df.iloc[0].to_frame())
