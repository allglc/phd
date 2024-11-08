import click
from pathlib import Path
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

path_data = Path.cwd().parent / 'data'

class MoonsDataset(Dataset):
    def __init__(self, n_samples=20000, noise=None, random_state=1):
        x, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        self.x, self.y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        colors = ['C0' if y_ == 0 else 'C1' for y_ in y]
        # plt.figure()
        # plt.scatter(x[:,0], x[:,1], c=colors, alpha=0.2)
        # plt.savefig(path_data / f'{n_samples}samples_noise{noise}.png')
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MoonsDataModule(pl.LightningDataModule):
    
    def __init__(self, n_samples=20000, noise=None, random_state=1):
        super().__init__()
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        
    def setup(self, stage=None):
        data = MoonsDataset(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        self.data_train, self.data_val = torch.utils.data.random_split(data, [int(self.n_samples*0.8), int(self.n_samples*0.2)])
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data_train, batch_size=256, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_val, batch_size=256, shuffle=False, num_workers=4)