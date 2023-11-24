import torch
# from types import SimpleNamespace
import pandas as pd
import Molecular_VAE
import biobricks as bb
import numpy as np
from Molecular_VAE import one_hot_array,one_hot_index,decode_smiles_from_indexes, load_dataset, char_to_index, one_hot_encode
ns = bb.assets('tox21') # get the paths for the 'tox21' brick
# h5_file = load_dataset("./data/processed.h5")
# h5_char_set = h5_file[2]

# index_dict = {}
# for idx, element in enumerate(h5_char_set):
#     if element in index_dict:
#         index_dict[element].append(idx)
#     else:
#         index_dict[element] = [idx]

# print(index_dict)

#cti = char_to_index(h5_char_set)

# Now you have the paths in variables
print(ns.tox21_parquet)
print(ns.tox21lib_parquet)
print(ns.tox21_aggregated_parquet)

# Reading the Parquet files using pandas
tox21_df = pd.read_parquet(ns.tox21_parquet)
tox21lib_df = pd.read_parquet(ns.tox21lib_parquet)
tox21_aggregated_df = pd.read_parquet(ns.tox21_aggregated_parquet)

tox21_df
# Print the size of each DataFrame
print(f"tox21_df size: {tox21_df.shape}")
# print(f"tox21lib_df size: {tox21lib_df.shape}")
# print(f"tox21_aggregated_df size: {tox21_aggregated_df.shape}")

# Print the first element of each DataFrame vertically
print("First element of tox21_df:")
print(tox21_df.iloc[0].to_frame())
# print("\nFirst element of tox21lib_df:")
# print(tox21lib_df.iloc[0].to_frame())
# print("\nFirst element of tox21_aggregated_df:")
# print(tox21_aggregated_df.iloc[0].to_frame())

tox21_smiles=tox21_df['SMILES'].iloc[:100]
tox21_smiles

char_set = set()

# Iterate over each string in the dictionary values
for string in tox21_smiles:
    # Add each character to the set
    if string:
        char_set.update(string)

byte_strings = np.array([np.bytes_(char) for char in char_set], dtype='S')
print("here")

for smile in tox21_smiles:
    if not smile:
        smile = "".rjust(120,'X')
    elif len(smile) < 120:
        smile = smile.rjust(120,'X')
    elif len(smile) > 120:
        length = len(smile)
        smile = smile[length - 120:]
    
    for char in smile:
        if char == 'X' or b'P':
            one_hot_encoding = one_hot_array(-1,len(byte_strings))
        else:
            char_byte = np.array([char], dtype='S').item()
            index = index_dict[char_byte][0]
            one_hot_encoding = one_hot_array(index,len(byte_strings))
            print(sum(list(one_hot_encoding)))
            # print(smile,char,list(one_hot_encoding))


'''
mvae = Molecular_VAE.MolecularVAE()
mvae
def train():
    data_train, data_test = s#split 90% train, 10% test 
    data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=250, shuffle=True)

    torch.manual_seed(42)

    epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MolecularVAE().to(device)
    optimizer = optim.Adam(model.parameters())

    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(device)
            optimizer.zero_grad()
            output, mean, logvar = model(data)
            
            if batch_idx==0:
                inp = data.cpu().numpy()
                outp = output.cpu().detach().numpy()
                lab = data.cpu().numpy()
                print("Input:")
                print(decode_smiles_from_indexes(map(from_one_hot_array, inp[0]), charset))
                print("Label:")
                print(decode_smiles_from_indexes(map(from_one_hot_array, lab[0]), charset))
                sampled = outp[0].reshape(1, 120, len(charset)).argmax(axis=2)[0]
                print("Output:")
                print(decode_smiles_from_indexes(sampled, charset))
            
            loss = vae_loss(output, data, mean, logvar)
            loss.backward()
            train_loss += loss
            optimizer.step()
    #         if batch_idx % 100 == 0:
    #             print(f'{epoch} / {batch_idx}\t{loss:.4f}')
        print('train', train_loss / len(train_loader.dataset))
        return train_loss / len(train_loader.dataset)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
'''
