# This attempts to overfit the simpyfied model
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

# self.conv_1 = self.conv_1.to(device)

class MolecularVAE(nn.Module):
    
    def __init__(self, char_to_int):
        super(MolecularVAE, self).__init__()
        # Encoder
        self.conv_1 = nn.Conv1d(62, 128, kernel_size=11, stride=1, padding=5)  # Increased filters and kernel size
        self.conv_2 = nn.Conv1d(128, 256, kernel_size=11, stride=1, padding=5)
        self.conv_3 = nn.Conv1d(256, 512, kernel_size=11, stride=1, padding=5)
        # Increase the number of neurons in Linear layers for the VAE latent space
        self.fc1 = nn.Linear(512*120, 1024)
        self.fc21 = nn.Linear(1024, 512)
        self.fc22 = nn.Linear(1024, 512)
        # Increase GRU complexity
        self.gru = nn.GRU(input_size=512, hidden_size=1024, num_layers=4, batch_first=True)  # Increased layers and hidden size
        # Decoder layers
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 512*120)
        # Reconstruction layer
        self.conv_4 = nn.ConvTranspose1d(512, 256, kernel_size=11, stride=1, padding=5)
        self.conv_5 = nn.ConvTranspose1d(256, 128, kernel_size=11, stride=1, padding=5)
        self.conv_6 = nn.ConvTranspose1d(128, 62, kernel_size=11, stride=1, padding=5)
        # Activation functions
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    #     self.conv_1 = nn.Conv1d(62, 64, kernel_size=9, stride=1, padding=4)
    #     self.conv_2 = nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4)
    #     self.conv_3 = nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=4)
    #     # Linear layers for the VAE latent space
    #     self.fc1 = nn.Linear(256*120, 512)  # Increase the number of neurons
    #     self.fc21 = nn.Linear(512, 256)  # Increase the size of the latent space
    #     self.fc22 = nn.Linear(512, 256)
    #     # Increase GRU complexity
    #     self.gru = nn.GRU(input_size=512, hidden_size=512, num_layers=3, batch_first=True)
    #     # Decoder layers
    #     self.fc3 = nn.Linear(256, 512)
    #     self.fc4 = nn.Linear(512, 256*120)
    #     # Reconstruction layer
    #     self.conv_4 = nn.ConvTranspose1d(256, 128, kernel_size=9, stride=1, padding=4)
    #     self.conv_5 = nn.ConvTranspose1d(128, 64, kernel_size=9, stride=1, padding=4)
    #     self.conv_6 = nn.ConvTranspose1d(64, 62, kernel_size=9, stride=1, padding=4)
    #     # Activation functions
    #     self.relu = nn.LeakyReLU()
    #     self.sigmoid = nn.Sigmoid()
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
# # Assuming data_train is a numpy array of shape [100, 62, 120]?
data_train_tensor = torch.from_numpy(data_train).float()
data_train_tensor.shape

data_train_dataset = TensorDataset(data_train_tensor)
train_loader = DataLoader(data_train_dataset, batch_size=10, shuffle=True)  # Adjust batch size as needed


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test the batches
# batch_idx, (x,) = next(enumerate(train_loader))
# x = x.to(device) #[10, 62, 120]
model = MolecularVAE(char_to_int).to(device)
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

epochs = 50

for epoch in tqdm(range(1, epochs + 1), desc='Training Progress'):
    train(epoch)

print('done')

# TESTING THE RESULTS =======================================================================
# test the batches
batch_idx, (x,) = next(enumerate(train_loader))
x1 = x[:1].to(device) #[10, 62, 120]
recon_batch, mu, logvar = model(x1)
recon_batch.shape
recon_batch[0]
tensor = recon_batch[0,:,0]
max_value, max_index = torch.max(tensor, 0)
tensor.shape
max_index
max_value    
char_to_int
decode_one_hot_tensor(recon_batch[0],int_to_char)
decode_one_hot_tensor(x1[0],int_to_char)


# 
sample_smiles = r'CC\\C(=C/c1ccc(cc1)O)/C'



# Perform the one-hot encoding
encoded_matrix = one_hot_encode(sample_smiles, charset)

# Convert the numpy matrix to a PyTorch tensor for decoding
encoded_tensor = torch.tensor(encoded_matrix, dtype=torch.float32)

# Decode the tensor back to SMILES
decoded_smiles = decode_one_hot_tensor(encoded_tensor, int_to_char)

sample_smiles, decoded_smiles