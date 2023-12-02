# This version corrects the overfitting.
    # Use mapping for one-hot-encode faster
    # A neural network model with dropout layers.
    # A training loop with early stopping and a validation phase.
    # A ReduceLROnPlateau scheduler to reduce the learning rate if the validation loss plateaus.
    # Saving the best model based on validation loss.
    # Plotting the training and validation losses over epochs.
import pandas as pd
import numpy as np
import biobricks as bb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Data Preparation
desired_length = 120
STARTCHAR = '^'
ENDCHAR = '$'
ns = bb.assets('tox21')

tox21_df = pd.read_parquet(ns.tox21_aggregated_parquet)
data_df_raw = tox21_df[['SMILES','ASSAY_OUTCOME','PROTOCOL_NAME']]

# Filter for rows with PROTOCOL_NAME as 'tox21-ahr-p1' and drop rows where 'ASSAY_OUTCOME' is missing
data_df = data_df_raw[data_df_raw['PROTOCOL_NAME'] == 'tox21-ahr-p1'].dropna()
data_df['ASSAY_OUTCOME'].value_counts()
# Create a label encoder object and encode 'ASSAY_OUTCOME'
label_encoder = LabelEncoder()
data_df['ASSAY_OUTCOME'] = label_encoder.fit_transform(data_df['ASSAY_OUTCOME'].astype(str))

# Preprocess SMILES strings
data_df['SMILES'] = data_df['SMILES'].str.slice(0, desired_length)\
    .str.pad(width=desired_length, side='right', fillchar=' ')

# Identify unique SMILES
unique_smiles = data_df['SMILES'].drop_duplicates()

charset = set([STARTCHAR, ENDCHAR])
for smile in unique_smiles:
    charset.update(smile)
char_to_int = {c: i for i, c in enumerate(charset)}

# One-hot encoding function
def one_hot_encode(smiles, char_to_int, max_length):
    matrix = np.zeros((max_length, len(charset)), dtype=np.int8)
    for i, char in enumerate(smiles):
        if char in char_to_int and i < max_length:
            matrix[i, char_to_int[char]] = 1
    return matrix

# Encode unique SMILES
unique_smiles_encoded = {smile: one_hot_encode(smile, char_to_int, desired_length) for smile in unique_smiles}

# Apply encoding to all SMILES in the dataset using the mapping
all_smiles_encoded = np.array([unique_smiles_encoded[smile] for smile in data_df['SMILES']])

# Convert to PyTorch tensors
smiles_tensor = torch.tensor(all_smiles_encoded, dtype=torch.float32).transpose(1, 2)
outcomes_tensor = torch.tensor(data_df['ASSAY_OUTCOME'].values, dtype=torch.int64)

# Split the dataset into training and testing sets
train_tensor, test_tensor, train_outcome_tensor, test_outcome_tensor = train_test_split(
    smiles_tensor, outcomes_tensor, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(train_tensor, train_outcome_tensor)
test_dataset = TensorDataset(test_tensor, test_outcome_tensor)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Define the MolecularClassifier with Dropout layers
class MolecularClassifier(nn.Module):
    def __init__(self, char_to_int, num_classes):
        super(MolecularClassifier, self).__init__()
        self.conv_1 = nn.Conv1d(len(char_to_int), 128, kernel_size=3, stride=1, padding=1)
        self.dropout_1 = nn.Dropout(0.5)
        self.conv_2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.dropout_2 = nn.Dropout(0.5)
        self.classifier = nn.Linear(256 * desired_length, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.dropout_1(x)
        x = F.relu(self.conv_2(x))
        x = self.dropout_2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Initialize the model
num_classes = len(label_encoder.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MolecularClassifier(char_to_int, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

# Training and validation
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []  # Lists to track accuracies
epochs = 10
early_stopping_patience = 5
best_val_loss = np.inf

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    correct_train = 0  # Counter for correct predictions
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        correct_train += (predicted == target).sum().item()
    train_losses.append(total_train_loss / len(train_loader))
    train_accuracy = correct_train / len(train_loader.dataset)
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    total_val_loss = 0
    correct_test = 0  # Counter for correct predictions
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct_test += (predicted == target).sum().item()
    val_loss = total_val_loss / len(test_loader)
    test_losses.append(val_loss)
    test_accuracy = correct_test / len(test_loader.dataset)
    test_accuracies.append(test_accuracy)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {test_losses[-1]:.4f}, '
          f'Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {test_accuracies[-1]:.4f}')

    # Learning rate scheduler step
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Best model saved at epoch {epoch+1}")

    # Early stopping
    if val_loss > best_val_loss:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Stopping early due to no improvement in validation loss.")
            break
    else:
        early_stopping_counter = 0

# Plotting training and testing losses and accuracies
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ax1.plot(train_losses, label='Training Loss', color=color)
ax1.plot(test_losses, label='Testing Loss', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(train_accuracies, label='Training Accuracy', color=color)
ax2.plot(test_accuracies, label='Testing Accuracy', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Training and Testing Loss and Accuracy Over Epochs')
plt.show()
