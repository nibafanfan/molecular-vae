import torch
import pandas as pd
import numpy as np
import biobricks as bb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader 
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import model_selection
 
desired_length = 120
STARTCHAR = '^'
ENDCHAR = '$'
ns = bb.assets('tox21')  # get the paths for the 'tox21' brick

# Updated function to build charset
def get_charset_from_smiles(smiles):
    charset = set([STARTCHAR, ENDCHAR])  # Corrected initialization of charset
    for smile in tqdm(smiles, desc="Processing SMILES"):
        if smile is None:
            continue
        else:
            for char in smile:
                charset.add(char)
    return charset

# Updated one_hot_encode function for padding

# Load and preprocess SMILES data

tox21_df = pd.read_parquet(ns.tox21_parquet)

tox21_smiles=list(set(tox21_df['SMILES'].dropna().str.slice(0, desired_length)\
    .str.pad(width=desired_length, side='right', fillchar=' ')))

# add progress bar
charset = get_charset_from_smiles(tox21_smiles)

char_to_int = dict((c, i) for i, c in enumerate(charset))

def one_hot_encode(smiles, charset):
    matrix = np.zeros((desired_length, len(charset)))
    
    for i, char in enumerate(smiles):
        if i < desired_length:  # Adjusting for the length
            matrix[i, char_to_int[char]] = 1
    tensor =torch.tensor(matrix,dtype = torch.float32)
    return matrix


tox21_smiles_small=tox21_smiles[0:100]
print(len(tox21_smiles_small[5]))
smiles_encoding =  [one_hot_encode(x,charset) for x in tox21_smiles_small]
tox21_smiles_small_df = pd.DataFrame({'SMILES': list(tox21_smiles_small)})
# Apply one-hot encoding to each SMILES string
tqdm.pandas()
one_hot_encoded = tox21_smiles_small_df['SMILES'].progress_apply(lambda x: one_hot_encode(x, charset))
print(size(one_hot_encoded))
print('here1')


# class MolecularVAE(nn.Module):
#     def __init__(self):
#         super(MolecularVAE, self).__init__()

#         self.conv_1 = nn.Conv1d(120, 9, kernel_size=9)
#         self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
#         self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
#         self.linear_0 = nn.Linear(70, 435)
#         self.linear_1 = nn.Linear(435, 292)
#         self.linear_2 = nn.Linear(435, 292)

#         self.linear_3 = nn.Linear(292, 292)
#         self.gru = nn.GRU(292, 501, 3, batch_first=True)
#         self.linear_4 = nn.Linear(501, 33)
        
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax()

#     def encode(self, x):
#         x = self.relu(self.conv_1(x))
#         x = self.relu(self.conv_2(x))
#         x = self.relu(self.conv_3(x))
#         x = x.view(x.size(0), -1)
#         x = F.selu(self.linear_0(x))
#         return self.linear_1(x), self.linear_2(x)

#     def sampling(self, z_mean, z_logvar):
#         epsilon = 1e-2 * torch.randn_like(z_logvar)
#         return torch.exp(0.5 * z_logvar) * epsilon + z_mean

#     def decode(self, z):
#         z = F.selu(self.linear_3(z))
#         z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
#         output, hn = self.gru(z)
#         out_reshape = output.contiguous().view(-1, output.size(-1))
#         y0 = F.softmax(self.linear_4(out_reshape), dim=1)
#         y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
#         return y

#     def forward(self, x):
#         z_mean, z_logvar = self.encode(x)
#         z = self.sampling(z_mean, z_logvar)
#         return self.decode(z), z_mean, z_logvar
    


# def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
#     xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
#     kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
#     return xent_loss + kl_loss

# stacked_data = np.stack(one_hot_encoded.values)

# # Splitting the data into training and testing sets
# data_train_np = stacked_data[:80]
# data_test_np = stacked_data[81:]

# # Convert these NumPy arrays to PyTorch tensors
# data_train_tensor = torch.from_numpy(data_train_np).float()
# data_test_tensor = torch.from_numpy(data_test_np).float()
# # Create TensorDataset objects
# data_train_dataset = torch.utils.data.TensorDataset(data_train_tensor)
# data_test_dataset = torch.utils.data.TensorDataset(data_test_tensor)

# # Create a DataLoader for the training set
# train_loader = torch.utils.data.DataLoader(data_train_dataset, batch_size=250, shuffle=True)

# # data_train= one_hot_encoded[:80]
# # data_test= one_hot_encoded[81:]
# # data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
# # train_loader = torch.utils.data.DataLoader(data_train, batch_size=250, shuffle=True)

# torch.manual_seed(42)

# epochs = 30
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = MolecularVAE().to(device)
# optimizer = optim.Adam(model.parameters())

# def train(epoch):
#     model.train()
#     train_loss = 0
#     for batch_idx, data in enumerate(train_loader):
#         data = data[0].to(device)
#         optimizer.zero_grad()
#         output, mean, logvar = model(data)
        
#         if batch_idx==0:
#               inp = data.cpu().numpy()
#               outp = output.cpu().detach().numpy()
#               lab = data.cpu().numpy()
#               print("Input:")
#               print(decode_smiles_from_indexes(map(from_one_hot_array, inp[0]), charset))
#               print("Label:")
#               print(decode_smiles_from_indexes(map(from_one_hot_array, lab[0]), charset))
#               sampled = outp[0].reshape(1, 120, len(charset)).argmax(axis=2)[0]
#               print("Output:")
#               print(decode_smiles_from_indexes(sampled, charset))
        
#         loss = vae_loss(output, data, mean, logvar)
#         loss.backward()
#         train_loss += loss
#         optimizer.step()
# #         if batch_idx % 100 == 0:
# #             print(f'{epoch} / {batch_idx}\t{loss:.4f}')
#     print('train', train_loss / len(train_loader.dataset))
#     return train_loss / len(train_loader.dataset)

# for epoch in range(1, epochs + 1):
#     train_loss = train(epoch)