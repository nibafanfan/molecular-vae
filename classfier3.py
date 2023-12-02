import pandas as pd
import numpy as np
import biobricks as bb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Data Preparation
desired_length = 120
STARTCHAR = '^'
ENDCHAR = '$'
ns = bb.assets('tox21')

tox21_df = pd.read_parquet(ns.tox21_parquet)
data_df_raw = tox21_df[['SMILES','ASSAY_OUTCOME','PROTOCOL_NAME']]

# Filter for rows with PROTOCOL_NAME as 'tox21-ahr-p1' and drop rows where 'ASSAY_OUTCOME' is missing
data_df = data_df_raw[data_df_raw['PROTOCOL_NAME'] == 'tox21-ahr-p1'].dropna()

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

# Training and validation with metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
precisions, recalls, f1_scores = [], [], []
early_stopping_patience = 5
early_stopping_counter = 0
best_val_loss = np.inf

epochs= 3

for epoch in range(epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    total_correct_train = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total_correct_train += (predicted == target).sum().item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_correct_train / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    total_val_loss = 0
    total_correct_val = 0
    all_val_targets = []
    all_val_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_correct_val += (predicted == target).sum().item()
            all_val_targets.extend(target.cpu().numpy())
            all_val_predictions.extend(predicted.cpu().numpy())
    avg_val_loss = total_val_loss / len(test_loader)
    val_accuracy = total_correct_val / len(test_loader.dataset)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    
    # Calculate precision, recall, and F1 score using weighted average for multiclass classification
    precision = precision_score(all_val_targets, all_val_predictions, average='weighted')
    recall = recall_score(all_val_targets, all_val_predictions, average='weighted')
    f1 = f1_score(all_val_targets, all_val_predictions, average='weighted')
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
          f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, '
          f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # Learning rate scheduler step
    scheduler.step(avg_val_loss)

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter == early_stopping_patience:
            print('Early stopping!')
            break

# Plotting training and testing losses, accuracies, and other metrics
# ... [Plotting code here, similar to the previous example]

# Assuming you have a one_hot_encoded_smiles_tensor
# Let's take the first one as an example
example_encoded_smiles = smiles_tensor[0].cpu().numpy()
example_decoded_smiles = decode_one_hot_array(example_encoded_smiles, int_to_char)
print(f'Decoded SMILES: {example_decoded_smiles}')
