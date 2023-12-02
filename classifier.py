import pandas as pd
import numpy as np
import biobricks as bb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Data Preparation
desired_length = 120
STARTCHAR = '^'
ENDCHAR = '$'
ns = bb.assets('tox21')

tox21_df = pd.read_parquet(ns.tox21_parquet)
data_df_raw = tox21_df[['SMILES','ASSAY_OUTCOME','PROTOCOL_NAME']]

# Drop rows where 'ASSAY_OUTCOME' is missing
# data_df = data_df_raw.dropna()
# Filter for rows with PROTOCOL_NAME as 'tox21-ahr-p1'
data_df = data_df_raw[data_df_raw['PROTOCOL_NAME'] == 'tox21-ahr-p1'].dropna()


# set(data_df['ASSAY_OUTCOME']){'inactive', 'inconclusive agonist', 'active agonist', 'inconclusive antagonist', 'inconclusive', 'inconclusive antagonist (cytotoxic)', 'active antagonist', 'inconclusive agonist (cytotoxic)'}
# >>> set(data_df['PROTOCOL_NAME']) {'tox21-er-luc-bg1-4e2-antagonist-p2', 'tox21-ahr-p1', 'tox21-esre-bla-p1', ....

# Create a label encoder object
label_encoder = LabelEncoder()

# Fit the encoder to the ASSAY_OUTCOME column to learn the mapping
data_df['ASSAY_OUTCOME'] = label_encoder.fit_transform(data_df['ASSAY_OUTCOME'].astype(str))

# Preprocess SMILES strings
data_df['SMILES'] = data_df['SMILES'].str.slice(0, desired_length)\
    .str.pad(width=desired_length, side='right', fillchar=' ')

# Now ASSAY_OUTCOME is converted to integers
outcomes = data_df['ASSAY_OUTCOME'].values


def get_charset_from_smiles(smiles):
    charset = set([STARTCHAR, ENDCHAR])  # Corrected initialization of charset
    for smile in tqdm(smiles, desc="Processing SMILES"):
        charset.update(smile)
    return charset

# Preprocess SMILES
# My original method was turning it into a set. Now I'm using drop_duplicates
# Identify unique SMILES
unique_smiles = data_df['SMILES'].drop_duplicates()

charset = get_charset_from_smiles(unique_smiles)
char_to_int = {c: i for i, c in enumerate(charset)}
int_to_char = {i: c for i, c in enumerate(charset)}
# One-hot encoding function
def one_hot_encode(smiles, char_to_int, max_length):
    matrix = np.zeros((max_length, len(charset)), dtype=np.int8)
    for i, char in enumerate(smiles):
        if char in char_to_int and i < max_length:
            matrix[i, char_to_int[char]] = 1
    return matrix

# def one_hot_encode(smiles, charset):
#     matrix = np.zeros((len(charset), desired_length))
#     for i, char in enumerate(smiles):
#         if i < desired_length:  # Adjusting for the length
#             matrix[char_to_int[char], i] = 1
#         else:
#             raise ValueError(f"Unexpected character {char} in SMILES string")
#     return matrix

def decode_one_hot_tensor(encoded_tensor, int_to_char):
    decoded_smiles = ''
    for i in range(encoded_tensor.shape[1]):
        char_index = torch.argmax(encoded_tensor[:, i]).item()
        decoded_smiles += int_to_char[char_index]
    return decoded_smiles.strip()

# Encode unique SMILES
unique_smiles_encoded = {smile: one_hot_encode(smile, char_to_int, desired_length) for smile in unique_smiles}

# Apply encoding to all SMILES in the dataset using the mapping
all_smiles_encoded = np.array([unique_smiles_encoded[smile] for smile in data_df['SMILES']])

# Convert to PyTorch tensors
smiles_tensor0 = torch.tensor(all_smiles_encoded, dtype=torch.float32)
smiles_tensor = smiles_tensor0.transpose(1, 2)
outcomes_tensor = torch.tensor(data_df['ASSAY_OUTCOME'].values, dtype=torch.int64)

# Create dataset and DataLoader
dataset = TensorDataset(smiles_tensor, outcomes_tensor)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Print shapes and types for verification. With this unique protocol name, size= 69780
print(f"SMILES Encoding Shape: {smiles_tensor.shape}, Type: {smiles_tensor.dtype}")
print(f"Outcomes Shape: {outcomes_tensor.shape}, Type: {outcomes_tensor.dtype}")



# # Ensure all elements in smiles_encoding have the same shape
# smiles_encoding = np.stack([one_hot_encode(x, charset) for x in tox21_smiles])

# # Convert SMILES encoding to PyTorch tensor
# data_train_tensor = torch.tensor(smiles_encoding, dtype=torch.float32)

# # Check and convert outcomes to PyTorch tensor
# if outcomes.dtype != np.int64 and outcomes.dtype != np.int32:
#     outcomes = outcomes.astype(np.int32)  # Convert to int32 if not already
# outcomes_tensor = torch.tensor(outcomes)


# Molecular Classifier
class MolecularClassifier(nn.Module):
    def __init__(self, char_to_int, num_classes):
        super(MolecularClassifier, self).__init__()
        # Encoder layers
        self.conv_1 = nn.Conv1d(len(char_to_int), 128, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        # Classifier layer
        self.classifier = nn.Linear(256 * desired_length, num_classes)
    def encode(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        return x
    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
# Initialize the model
num_classes = len(np.unique(outcomes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MolecularClassifier(char_to_int, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    return total_loss / len(train_loader.dataset)

# Training the Model
epochs = 10
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    print(f'Epoch: {epoch}, Loss: {train_loss}')

# [Add testing phase code here if needed]

print('Training complete.')


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ... [Your existing code up to the creation of the dataset and DataLoader]

# Split the dataset into training and testing sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# ... [Your existing code for the MolecularClassifier and model initialization]

# Testing Loop
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)\n')
    return test_loss

# Record training and testing losses
train_losses = []
test_losses = []

# Training the Model with Testing
epochs = 20
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    test_loss = test()
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f'Epoch: {epoch}, Training Loss: {train_loss}, Testing Loss: {test_loss}')

print('Training complete.')

# Plotting training and testing losses
plt.figure(figsize=(10,6))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), test_losses, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Over Epochs')
plt.legend()
plt.show()
