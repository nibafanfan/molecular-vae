import h5py
import sys
import numpy as np

# Open the file
with h5py.File(sys.argv[1], 'r') as file:
    # Iterate over each dataset provided in the list
    for dataset_name in file.keys():
        # Check if the dataset exists in the file
        if dataset_name in file:
            # Get the dataset
            dataset = file[dataset_name]
            # Print a representative value
            # Here, we print the first value. Adjust the indexing if necessary.
            print(len(dataset))
            print(f"{dataset_name}: {dataset[0]}")
            print(f"Data shape {np.shape(dataset[0])}" )
        else:
            print(f"Dataset {dataset_name} not found in the file.")
