# Trying to overfit with transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

import biobricks as bb


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
# modified_char_list = [element.replace('\\\\', '\\') for element in charset]

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


tox21_smiles_small=tox21_smiles[0:100]
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

# data_train_tensor = torch.from_numpy(data_train).float()
# data_train_tensor.shape

# data_train_dataset = TensorDataset(data_train_tensor)
# train_loader = DataLoader(data_train_dataset, batch_size=10, shuffle=True)  # Adjust batch size as needed

# add validation

from sklearn.model_selection import train_test_split

# Assuming data_train is your full dataset
train_data, val_data = train_test_split(data_train, test_size=0.2, random_state=42)  # 20% data for validation

train_tensor = torch.from_numpy(train_data).float()
val_tensor = torch.from_numpy(val_data).float()

train_dataset = TensorDataset(train_tensor)
val_dataset = TensorDataset(val_tensor)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)



# Define the Custom Transformer Encoder Layer
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = src.permute(1, 0, 2)  # Switch to (seq_len, batch, feature)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src.permute(1, 0, 2)  # Switch back to (batch, seq_len, feature)

# Define the MolecularVAE class with the Custom Transformer Encoder
class MolecularVAE(nn.Module):
    def __init__(self, char_to_int, nhead, dim_feedforward, num_transformer_layers):
        super(MolecularVAE, self).__init__()
        # Encoder layers
        self.conv_1 = nn.Conv1d(62, 128, kernel_size=11, stride=1, padding=5)
        self.conv_2 = nn.Conv1d(128, 256, kernel_size=11, stride=1, padding=5)
        self.conv_3 = nn.Conv1d(256, 512, kernel_size=11, stride=1, padding=5)

        # Custom Transformer Encoder Layer
        transformer_layer = CustomTransformerEncoderLayer(
            d_model=512,  # Output feature size of the conv_3 layer
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # VAE layers
        self.fc1 = nn.Linear(512*120, 1024)
        self.fc21 = nn.Linear(1024, 512)
        self.fc22 = nn.Linear(1024, 512)

        # GRU layers
        self.gru = nn.GRU(input_size=512, hidden_size=1024, num_layers=4, batch_first=True)

        # Decoder layers
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 512*120)

        # Reconstruction layers
        self.conv_4 = nn.ConvTranspose1d(512, 256, kernel_size=11, stride=1, padding=5)
        self.conv_5 = nn.ConvTranspose1d(256, 128, kernel_size=11, stride=1, padding=5)
        self.conv_6 = nn.ConvTranspose1d(128, 62, kernel_size=11, stride=1, padding=5)

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x1 = self.relu(self.conv_1(x))
        x2 = self.relu(self.conv_2(x1))
        x3 = self.relu(self.conv_3(x2)) # 10 x 128 x 120
        x4 = x3.view(x3.size(0),-1) # 10 x 15360
        x5 = F.selu(self.fc1(x4)) # 10 X 256
        return self.fc21(x5), self.fc22(x5) # 10 X 128
    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar) # 10 X 128
        # return torch.exp(0.5 * z_logvar) * epsilon + z_mean # 10 X 128
        return z_mean
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) #10 X 128
        eps = torch.randn_like(std) #10 X 128
        return mu + eps*std
    def decode(self, z):
        z1 = self.relu(self.fc3(z)) # 10 x 256
        z2 = self.relu(self.fc4(z1)) # 10 x 15360
        z3 = z2.view(-1, 512, 120) #[10, 256, 120]
        z4 = self.relu(self.conv_4(z3)) # [10, 64, 120]
        z5 = self.relu(self.conv_5(z4)) # [10, 32, 120]
        z6 = self.sigmoid(self.conv_6(z5))  # [10, 120, 120], Sigmoid activation for output layer
        return z6
    def forward(self, x):
        mu, logvar = self.encode(x) #10 X 128?
        z = self.reparameterize(mu, logvar) #10 X 128
        return self.decode(z), mu, logvar

# Initialize the model with Transformer parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MolecularVAE(char_to_int, nhead=8, dim_feedforward=2048, num_transformer_layers=2).to(device)

# Training Setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # tensor(1.3217, device='cuda:0', grad_fn=<MulBackward0>)
    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)  # Ensure data is on the correct device
        optimizer.zero_grad()
        # Forward pass
        recon_batch, mu, logvar = model(data) #[10, 120, 15360], # 10 X 128, # 10 X 128
        # Calculate loss and backpropagate
        # print('batch',recon_batch.shape,'data',data.shape)
        # print('batch',recon_batch)
        # print('data',data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('decode_one_hot_tensor', decode_one_hot_tensor(recon_batch[5],int_to_char))
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    # print('train', train_loss / len(train_loader.dataset))
    return train_loss/len(train_loader.dataset)
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data[0].to(device)
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            val_loss += loss.item()
    return val_loss / len(val_loader.dataset)

epochs = 100

import matplotlib.pyplot as plt

# Initialize lists to store the losses
train_losses = []
val_losses = []

for epoch in tqdm(range(1, epochs + 1), desc='Training Progress'):
    train_loss = train(epoch)
    val_loss = validate(model, val_loader, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f'Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

    # Early stopping logic (if you choose to use it)
    # ...

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()





# for epoch in tqdm(range(1, epochs + 1), desc='Training Progress'):
#     train_loss = train(epoch)
#     val_loss = validate(model, val_loader, device)
#     print(f'Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}')



# # Train the model
# epochs = 100
# for epoch in tqdm(range(1, epochs + 1), desc='Training Progress'):
#     train_loss = train(epoch)  # Assuming 'train' is your training function
#     print(f'Epoch {epoch}, Loss: {train_loss}')
