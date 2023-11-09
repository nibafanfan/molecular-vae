
import os
import torch
import torch.nn as nn

def main():
    here = os.path.abspath(os.path.dirname(__file__))

    #!rm -R 'molecular-vae'
    #!git clone https://github.com/aksub99/molecular-vae.git
    import zipfile
    dataFolder =  os.path.join(here, 'data')
    zipfilename = os.path.join(dataFolder, 'processed.zip')
    h5filename = os.path.join(dataFolder, 'processed.h5')
    print(zipfilename)
    if not os.path.exists(h5filename):
        zip_ref = zipfile.ZipFile(zipfilename, 'r')
        zip_ref.extractall(dataFolder)
        zip_ref.close()

    def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
        xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return xent_loss + kl_loss

    data_train, data_test, charset = load_dataset(h5filename)
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

if __name__ == "__main__":
    main()