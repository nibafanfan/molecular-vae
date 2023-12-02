# This attempts to overfit by 1. remove variational aspect(the sampling part)
# 2. Simplify the model (also don't reduce the dimension)
# 3. Initialize weights for perfect reconstruction
# drop encoding layer & initialize weights
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
 
desired_length = 120
STARTCHAR = '^'
ENDCHAR = '$'
ns = bb.assets('tox21')  # get the paths for the 'tox21' brick

def get_charset_from_smiles(smiles):
    charset = set([STARTCHAR, ENDCHAR])  # Corrected initialization of charset
    for smile in tqdm(smiles, desc="Processing SMILES"):
        charset.update(smile)
    return charset

# Updated one_hot_encode function for padding

# Load and preprocess SMILES data

tox21_df = pd.read_parquet(ns.tox21_parquet)

tox21_smiles=list(set(tox21_df['SMILES'].dropna().str.slice(0, desired_length)\
    .str.pad(width=desired_length, side='right', fillchar=' ')))

# add progress bar
charset = get_charset_from_smiles(tox21_smiles)

char_to_int = dict((c, i) for i, c in enumerate(charset))
int_to_char = dict((i, c) for i, c in enumerate(charset))

def one_hot_encode(smiles, charset):
    matrix = np.zeros((len(charset), desired_length))
    for i, char in enumerate(smiles):
        if i < desired_length:  # Adjusting for the length
            matrix[char_to_int[char], i] = 1
    return matrix

def decode_one_hot_tensor(encoded_tensor, int_to_char):
    decoded_smiles = ''
    for i in range(encoded_tensor.shape[1]):
        char_index = torch.argmax(encoded_tensor[:, i]).item()
        decoded_smiles += int_to_char[char_index]
    return decoded_smiles.strip()


tox21_smiles_small=tox21_smiles[0:1000]
print(len(tox21_smiles_small[5]))
smiles_encoding =  [one_hot_encode(x,charset) for x in tox21_smiles_small]
tox21_smiles_small_df = pd.DataFrame({'SMILES': list(tox21_smiles_small)})
# Apply one-hot encoding to each SMILES string
tqdm.pandas()
one_hot_encoded = tox21_smiles_small_df['SMILES'].progress_apply(lambda x: one_hot_encode(x, charset))

print('here1')
# just a try. Need to modify  
newmatrix=[np.array(matrix, dtype='float32') for matrix in one_hot_encoded]
data_train= np.array(newmatrix)

print('here')

self.conv_1 = self.conv_1.to(device)
self.conv_3 = self.conv_3.to(device)

class MolecularAutoencoder(nn.Module):
    
    def __init__(self, char_to_int):
        super(MolecularAutoencoder, self).__init__()
        # Encoder layers
        self.conv_1 = nn.Conv1d(62, 128, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        # Decoder layers
        self.conv_3 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.ConvTranspose1d(128, 62, kernel_size=3, stride=1, padding=1)
    def encode(self, x):
        x1 = F.relu(self.conv_1(x))
        x2 = F.relu(self.conv_2(x1))
        return x
    def decode(self, z):
        z = F.relu(self.conv_3(z))
        z = torch.sigmoid(self.conv_4(z))
        return z
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
# # Assuming data_train is a numpy array of shape [100, 62, 120]?
data_train_tensor = torch.from_numpy(data_train).float()
data_train_tensor.shape

data_train_dataset = TensorDataset(data_train_tensor)
train_loader = DataLoader(data_train_dataset, batch_size=10, shuffle=True)  # Adjust batch size as needed


def autoencoder_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, reduction='sum')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and optimizer
model = MolecularAutoencoder(char_to_int).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
def autoencoder_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, reduction='sum')

# Training function
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = autoencoder_loss(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    return train_loss / len(train_loader.dataset)

# Train the model
epochs = 10
for epoch in range(1, epochs + 1):
    avg_loss = train(epoch)
    print(f'Epoch {epoch} average loss: {avg_loss}')

# Test the model
# Fetch a batch of data
batch_idx, (x,) = next(enumerate(train_loader))
x = x.to(device)

# Run the model
with torch.no_grad():
    recon_batch = model(x)

# Display some results
# Assuming you have a function to convert the output back to SMILES
for i in range(5):
    original_smiles = decode_one_hot_tensor(x[i], int_to_char)
    reconstructed_smiles = decode_one_hot_tensor(recon_batch[i], int_to_char)
    print(f'Original: {original_smiles}, Reconstructed: {reconstructed_smiles}')
print('done')

# TESTING THE RESULTS =======================================================================
# test the batches

