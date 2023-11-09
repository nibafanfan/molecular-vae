from types import SimpleNamespace
import pandas as pd
import Molecular_VAE
import biobricks as bb
ns = bb.assets('tox21') # get the paths for the 'tox21' brick

# Access the attributes
tox21_parquet_path = ns.tox21_parquet
tox21lib_parquet_path = ns.tox21lib_parquet
tox21_aggregated_parquet_path = ns.tox21_aggregated_parquet

# Now you have the paths in variables
print(tox21_parquet_path)
print(tox21lib_parquet_path)
print(tox21_aggregated_parquet_path)

# Reading the Parquet files using pandas
tox21_df = pd.read_parquet(ns.tox21_parquet)
tox21lib_df = pd.read_parquet(ns.tox21lib_parquet)
tox21_aggregated_df = pd.read_parquet(ns.tox21_aggregated_parquet)

# Print the size of each DataFrame
print(f"tox21_df size: {tox21_df.shape}")
print(f"tox21lib_df size: {tox21lib_df.shape}")
print(f"tox21_aggregated_df size: {tox21_aggregated_df.shape}")

# Print the first element of each DataFrame vertically
print("First element of tox21_df:")
print(tox21_df.iloc[0].to_frame())
print("\nFirst element of tox21lib_df:")
print(tox21lib_df.iloc[0].to_frame())
print("\nFirst element of tox21_aggregated_df:")
print(tox21_aggregated_df.iloc[0].to_frame())

mvae = Molecular_VAE.MolecularVAE()
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
