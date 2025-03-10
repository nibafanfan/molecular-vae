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

# def old_one_hot_encode(smiles, charset):
#     matrix = np.zeros((desired_length, len(charset)))
#     for i, char in enumerate(smiles):
#         if i < desired_length:  # Adjusting for the length
#             matrix[i, char_to_int[char]] = 1
#     return matrix

def one_hot_encode(smiles, charset):
    matrix = np.zeros((len(charset), desired_length))
    for i, char in enumerate(smiles):
        if i < desired_length:  # Adjusting for the length
            matrix[char_to_int[char], i] = 1
    return matrix


tox21_smiles_small=tox21_smiles[0:1000]
print(len(tox21_smiles_small[5]))
smiles_encoding =  [one_hot_encode(x,charset) for x in tox21_smiles_small]
tox21_smiles_small_df = pd.DataFrame({'SMILES': list(tox21_smiles_small)})
# Apply one-hot encoding to each SMILES string
tqdm.pandas()
one_hot_encoded = tox21_smiles_small_df['SMILES'].progress_apply(lambda x: one_hot_encode(x, charset))
# print(one_hot_encoded)
# print(one_hot_encoded[5])
# print(len(one_hot_encoded[7]))
# print(one_hot_encoded[0][0])
# print(sum(one_hot_encoded[0][8]))
# print(len(one_hot_encoded[0][3]))
# Now `one_hot_encoded` is a Series of matrices


print('here1')
# just a try. Need to modify  
newmatrix=[np.array(matrix, dtype='float32') for matrix in one_hot_encoded]
data_train= np.array(newmatrix)

print('here')

# self.conv_1 = self.conv_1.to(device)

class MolecularVAE(nn.Module):
    
    def __init__(self, char_to_int):
        super(MolecularVAE, self).__init__()
        # Convolution layers
        self.conv_1 = nn.Conv1d(62, 32, kernel_size=9, stride=1, padding=4)
        self.conv_2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.conv_3 = nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4)
        # Linear layers for the VAE latent space
        self.fc1= nn.Linear(128*120, 256)  # Flatten before passing
        self.fc21 = nn.Linear(256, 128)  # Mean of the latent space
        self.fc22 = nn.Linear(256, 128)  # Standard deviation of the latent space
        # Decoder layers
        self.fc3 = nn.Linear(128, 256)
         # Add GRU layer definition if needed
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        self.fc4 = nn.Linear(256, 128*120)
        # Reconstruction layer
        self.conv_4 = nn.ConvTranspose1d(128, 64, kernel_size=9, stride=1, padding=4)
        self.conv_5 = nn.ConvTranspose1d(64, 32, kernel_size=9, stride=1, padding=4)
        self.conv_6 = nn.ConvTranspose1d(32, 120, kernel_size=9, stride=1, padding=4)
        # Activation functions
        self.relu = nn.ReLU()
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
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean # 10 X 128
    
    def decode(self, z):
        # transformer layer
        z1 = F.selu(self.fc3(z)) # 10 X 256
        z2 = z1.view(z1.size(0), 1, z1.size(-1)).repeat(1, 120, 1) #[10, 120, 256]
        output, hn = self.gru(z2) #[10,120,256], [2,10,256]
        out_reshape = output.contiguous().view(-1, output.size(-1)) # 1200 X 256
        y0 = F.softmax(self.fc4(out_reshape), dim=1) # 1200 X 15360
        # y = y0.contiguous().view(output.size(0), -1, y0.size(-1)) #[10, 120, 15360]
        y2 = y0.contiguous().view(output.size(0), y0.size(-1), -1)  # [10, 15360, 120]
         # Apply Conv1d layer
        self.convtemp = nn.Conv1d(in_channels=15360, out_channels=240, kernel_size=1)
        self.convtemp = self.convtemp.to(device)
        # Pooling layer to reduce sequence length
        self.pooltemp = nn.MaxPool1d(kernel_size=2, stride=2)
        y3 = F.relu(self.convtemp(y2))  # [10, 240, 120]
        # Apply Pooling
        y4 = self.pooltemp(y3)  # [10, 240, 60]
        # Reshape for fully connected layer
        y5 = y4.transpose(1, 2).contiguous().view(-1, 240)  # [600, 240]
        # Modify the fully connected layer to match the final output size
        self.fc5temp = nn.Linear(240, 62)
        self.fc5temp = self.fc5temp.to(device)
        # Apply Fully connected layer
        y6 = self.fc5temp(y5)  # [600, 62]
        # Final Reshape to desired output
        y7 = y6.view(output.size(0), -1, 120)  # [10, 31, 120]?
        target_size= torch.Size([10, 62, 120]) 
        y8 = torch.nn.functional.interpolate(y7.unsqueeze(0), size=(target_size[1], target_size[2]), mode='nearest').squeeze(0) # [10, 62, 120]
        y9 = torch.sigmoid(y8) # normalize it between 0 to 1
        return y9
    def forward(self, x):
        z_mean, z_logvar = self.encode(x) # 10 X 128
        z = self.sampling(z_mean, z_logvar) # 10 X 128
        # print(z.size())
        return self.decode(z), z_mean, z_logvar  #[10, 120, 15360], # 10 X 128, # 10 X 128
    
    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps*std
    
    # def decode(self, z):
    #     z = self.relu(self.fc3(z))
    #     z = self.relu(self.fc4(z))
    #     z = z.view(-1, 128, 120)
    #     z = self.relu(self.conv_4(z))
    #     z = self.relu(self.conv_5(z))
    #     z = self.sigmoid(self.conv_6(z))  # Sigmoid activation for output layer
    #     return z
    
    # def forward(self, x):
    #     mu, logvar = self.encode(x)
    #     z = self.reparameterize(mu, logvar)
    #     return self.decode(z), mu, logvar
    
# # Assuming data_train is a numpy array of shape [100, 62, 120]?
data_train_tensor = torch.from_numpy(data_train).float()
data_train_tensor.shape

data_train_dataset = TensorDataset(data_train_tensor)
train_loader = DataLoader(data_train_dataset, batch_size=10, shuffle=True)  # Adjust batch size as needed

# test the batches
# batch_idx, (x,) = next(enumerate(train_loader))
# x = x.to(device)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MolecularVAE(char_to_int).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')


    print('train', train_loss / len(train_loader.dataset))
    return train_loss/len(train_loader.dataset)
epochs = 30

for epoch in tqdm(range(1, epochs + 1), desc='Training Progress'):
    train(epoch)

print('done')



    

