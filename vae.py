import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 768
h1_dim = 512
h2_dim = 256
h3_dim = 128
h4_dim = 64
z_dim = 32
num_epochs = 15
batch_size = 8
learning_rate = 1e-5

# raw_datasets = load_dataset('snips_built_in_intents')
checkpoint = r'./bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
cls_representation = AutoModel.from_pretrained(checkpoint)


class DatasetFramework(Dataset):
    def __init__(self, file_path, **kwargs):
        self.samples = self._load_dataset(file_path)

    def _load_dataset(self, file_path):
        with open(os.path.join(file_path, 'intents.txt'), 'r', encoding='utf-8') as f:
            intents = [_.strip() for _ in f.readlines()]
        with open(os.path.join(file_path, 'sentences.txt'), 'r', encoding='utf-8') as f:
            sentences = [_.strip() for _ in f.readlines()]
        embeddings_path = os.path.join(file_path, 'embeddings.npy')
        embeddings = np.load(embeddings_path)
        samples = []
        for i in range(len(intents)):
            sample = {}
            sample['sentences'] = sentences[i]
            sample['intents'] = intents[i]
            sample['embeddings'] = embeddings[i]
            samples.append(sample)
        return samples

    def __getitem__(self, index):
        sample = self.samples[index]
        dataset = {'sentences': sample['sentences'], 'intents': sample['intents'], 'embeddings': sample['embeddings']}
        # _input = tokenizer(sample['sentences'], truncation=True, return_tensors='pt')
        # dataset['embeddings'] = self.cls_representation(**_input).pooler_output[0]
        return dataset

    def __len__(self):
        return len(self.samples)

train_data_path = os.path.join(r'./data/snips', 'train')
dev_data_path = os.path.join(r'./data/snips', 'dev')
test_data_path = os.path.join(r'./data/snips', 'test')
ood_data_path = os.path.join(r'./data/snips', 'ood')

train_dataset = DatasetFramework(train_data_path)
dev_dataset = DatasetFramework(dev_data_path)
test_dataset = DatasetFramework(test_data_path)
ood_dataset = DatasetFramework(ood_data_path)

train_data_loader = DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               # pin_memory=True,
                               num_workers=0)
dev_data_loader = DataLoader(dataset=dev_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             # pin_memory=True,
                             num_workers=0)
test_data_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              # pin_memory=True,
                              num_workers=0)
ood_data_loader = DataLoader(dataset=ood_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             # pin_memory=True,
                             num_workers=0)


class VAE(nn.Module):
    def __init__(self, input_size=input_size, h1_dim=h1_dim, h2_dim=h2_dim, h3_dim=h3_dim, h4_dim=h4_dim,
                 z_dim=z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, h3_dim)
        self.fc4 = nn.Linear(h3_dim, h4_dim)

        self.fc5 = nn.Linear(h4_dim, z_dim)
        self.fc6 = nn.Linear(h4_dim, z_dim)

        self.fc7 = nn.Linear(z_dim, h4_dim)
        self.fc8 = nn.Linear(h4_dim, h3_dim)
        self.fc9 = nn.Linear(h3_dim, h2_dim)
        self.fc10 = nn.Linear(h2_dim, h1_dim)
        self.fc11 = nn.Linear(h1_dim, input_size)

    def encoder(self, x):
        h1 = F.tanh(self.fc1(x))
        h2 = F.tanh(self.fc2(h1))
        h3 = F.tanh(self.fc3(h2))
        h4 = F.tanh(self.fc4(h3))
        return self.fc5(h4), self.fc6(h4)  # Calculate the second fully connected layer for the mean and log(σ2)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)  # Generate a random number from the standard plus distribution of the same size as std
        return mu + eps * std

    def decoder(self, z):
        h = F.tanh(self.fc7(z))
        h = F.tanh(self.fc8(h))
        h = F.tanh(self.fc9(h))
        h = F.tanh(self.fc10(h))
        return F.tanh(self.fc11(h))

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decoder(z)
        return x_reconst, mu, log_var  # return the mean and variance to calculate the KL divergence


model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_training_steps = len(train_data_loader) * num_epochs
progress_bar = tqdm(range(num_training_steps))
loss_fn = nn.CrossEntropyLoss(reduction='sum')
# Start training
for epoch in range(num_epochs):
    tra_loss = 0
    for i, x in enumerate(train_data_loader):
        x = torch.tensor(x['embeddings']).to(device).to(torch.float32)
        x_reconst, mu, log_var = model(x)
        # print(x.min(), x.max())
        # print(x_reconst.min(), x_reconst.max())
        # assert 1==0
        reconst_loss = loss_fn(x_reconst, x)
        kl_div = - 0.5 * torch.sum(1 + log_var - log_var.exp() - mu.pow(2))
        loss = (reconst_loss + kl_div) / batch_size
        tra_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.update(1)
        if (i + 1) % 100 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, len(train_data_loader), reconst_loss.item(), kl_div.item()))
    threshold = tra_loss / len(train_data_loader)
    ood_num = 0
    with torch.no_grad():  # test
        for i, x in enumerate(tqdm(dev_data_loader)):
            x = torch.tensor(x['embeddings']).to(device).to(torch.float32)
            x_reconst, mu, log_var = model(x)
            reconst_loss = loss_fn(x_reconst, x)
            kl_div = - 0.5 * torch.sum(1 + log_var - log_var.exp() - mu.pow(2))
            loss = (reconst_loss + kl_div) / batch_size
            if loss > threshold:
                ood_num += 1
    print(f'Accuracy:{ood_num / len(dev_data_loader)}')
