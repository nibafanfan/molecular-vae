import torch
# from types import SimpleNamespace
import pandas as pd
import Molecular_VAE
import biobricks as bb
import numpy as np

ns = bb.assets('tox21') # get the paths for the 'tox21' brick

# build charset 
def get_charset_from_smiles(smiles):
    charset = set()
    for smile in smiles:
        # print(type(smile),smile)
        if smile is None:
            continue
        else:
            for char in smile:
                charset.add(char)
    return charset

def one_hot_encode(smiles, charset,desired_length):
    # Initialize a matrix of zeros
    matrix = np.zeros((desired_length, len(charset)))

    # Create a dictionary to map characters to column indices
    char_to_int = dict((c, i) for i, c in enumerate(charset))

    # Fill the matrix with 1s where appropriate
    for i, char in enumerate(smiles):
        if char == ' ':
            continue
        else:
            matrix[i, char_to_int[char]] = 1

    return matrix
desired_length = 120
tox21_df = pd.read_parquet(ns.tox21_parquet)
tox21_smiles=tox21_df['SMILES'].dropna().str.slice(0, desired_length).str.pad(width=desired_length, side='right', fillchar=' ')

charset = get_charset_from_smiles(tox21_smiles)
print(len(charset))
print(charset)

tox21_smiles_small=tox21_smiles.iloc[0:100]
print(len(tox21_smiles_small[5]))


# Apply one-hot encoding to each SMILES string
one_hot_encoded = tox21_smiles_small.apply(lambda x: one_hot_encode(x, charset, desired_length))
print(one_hot_encoded)
print(one_hot_encoded[5])
print(len(one_hot_encoded[7]))
print(one_hot_encoded[0][0])
print(sum(one_hot_encoded[0][8]))
print(len(one_hot_encoded[0][3]))
# Now `one_hot_encoded` is a Series of matrices



    

