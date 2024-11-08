#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import click
import copy
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18, ResNet18_Weights, convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.resnet import ResNet
from torchvision.models.convnext import ConvNeXt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pathlib import Path

from classifiers.models import CNN_MNIST
from stylegan2_ada_pytorch.training.dataset import ImageFolderDataset
from utils import EasyDict, create_experiment_folder, Logger

torch.manual_seed(0)
np.random.seed(0)
rng = np.random.default_rng(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path_results = Path.cwd().parent / 'results'


def load_data(dataset):
    path_data = Path.cwd().parent / 'data/MNIST' / f'{dataset}.zip'
    ds_original = ImageFolderDataset(path_data, use_labels=True)

    # LOAD CLASSIFIER

    # predict digits
    classifier_digits = CNN_MNIST(output_dim=10).to(device)
    # classifier_digits.load_state_dict(torch.load(path_results / 'classifiers' / 'CNN_MNIST_weights_20220411_0826.pth', map_location=device)) # Confiance
    classifier_digits.load_state_dict(torch.load(path_results / 'classifiers' / 'CNN_MNIST_weights_20220210_1601.pth', map_location=device))
    # classifier_digits.load_state_dict(torch.load(path_results / 'classifiers' / 'CNN_mnist_stylegan2_blur_noise_maxSeverity3_proba50_20220510_1124.pth', map_location=device))
    classifier_digits.eval()

    # # predict noise
    # classifier_noise = CNN_MNIST(output_dim=6).to(device)
    # # classifier_noise.load_state_dict(torch.load(path_results / 'classifiers' / 'CNN_noise_MNIST_weights_20220411_0841.pth', map_location=device)) # Confiance
    # classifier_noise.load_state_dict(torch.load(path_results / 'classifiers' / 'CNN_MNIST_noise_weights_20220210_1728.pth', map_location=device))
    # classifier_noise.eval()

    return ds_original, classifier_digits


def create_datasets(similarity, ds_original, classifier_digits, batch_size=128, train_split=0.8):

    if similarity == 'classifResult':

        # DATASETS

        digit_pred = 99*torch.ones((len(ds_original)), dtype=int)
        for idx, (x, y) in enumerate(DataLoader(ds_original, batch_size=256)):
            x = (x / 255)[:, :, 2:30, 2:30].to(device)
            digit_pred[idx*256:(idx+1)*256] = classifier_digits(x).argmax(dim=1).cpu()

        idx_positive = np.nonzero(digit_pred.numpy() == ds_original._raw_labels)[0]
        idx_negative = np.nonzero(digit_pred.numpy() != ds_original._raw_labels)[0]


        # ds_positive = Subset(ds_original, idx_positive)
        # ds_negative = Subset(ds_original, idx_negative)

        # for x, y in DataLoader(ds_positive, batch_size=256):
        #     x = (x / 255)[:, :, 2:30, 2:30].to(device)
        #     y_pred = classifier_digits(x).argmax(dim=1).cpu()
        #     y = y.argmax(dim=1).cpu()
        #     assert all(y == y_pred), 'should be well classified'

        # for x, y in DataLoader(ds_negative, batch_size=256): # WEIRD: logits change slightly with different batch sizes (even with model.eval())
        #     x = (x / 255)[:, :, 2:30, 2:30].to(device)
        #     y_pred = classifier_digits(x).argmax(dim=1).cpu()
        #     y = y.argmax(dim=1).cpu()
        #     assert all(y != y_pred), 'should be mis classified'

        # print('checked that subsets are classified as they should be')

        ds_correctness = copy.deepcopy(ds_original)
        ds_correctness[0] # somehow need to do this for next lines to work
        ds_correctness._label_shape = [2] # 2 classes: correct or incorrect
        ds_correctness._raw_labels[idx_positive] = 1
        ds_correctness._raw_labels[idx_negative] = 0

        print('{:.2f}% of images are correctly classified'.format(100*len(idx_positive)/len(ds_original)))
        print('{:.2f}% of images are incorrectly classified'.format(100*len(idx_negative)/len(ds_original)))

    elif similarity == 'trueClassProba':
        
        tcp = 99*torch.ones((len(ds_original)))
        for idx, (x, y) in enumerate(DataLoader(ds_original, batch_size=256)):
            x = (x / 255)[:, :, 2:30, 2:30].to(device)
            with torch.no_grad():
                tcp[idx*256:(idx+1)*256] = nn.functional.softmax(classifier_digits(x), dim=1)[y.bool()].cpu()

        idx_positive = np.nonzero(tcp.numpy() > 0.9)[0] # indices with high score
        idx_negative = np.nonzero(tcp.numpy() <= 0.9)[0] # indices with low score

        ds_correctness = copy.deepcopy(ds_original)
        ds_correctness[0] # somehow need to do this for next lines to work
        ds_correctness._label_shape = [1]
        ds_correctness._raw_labels = tcp.numpy()

        print('{:.2f}% of images have high TCP'.format(100*len(idx_positive)/len(ds_original)))
        print('{:.2f}% of images have low TCP'.format(100*len(idx_negative)/len(ds_original)))

    # DATALOADERS

    np.random.shuffle(idx_negative)
    np.random.shuffle(idx_positive)
    idx_negative_train, idx_negative_test = idx_negative[:int(train_split*len(idx_negative))], idx_negative[int(train_split*len(idx_negative)):]
    idx_positive_train, idx_positive_test = idx_positive[:int(train_split*len(idx_positive))], idx_positive[int(train_split*len(idx_positive)):]
    idx_train = np.concatenate((idx_positive_train, idx_negative_train))
    idx_test = np.concatenate((idx_positive_test, idx_negative_test))
    ds_correctness_train = Subset(ds_correctness, idx_train)
    ds_correctness_test = Subset(ds_correctness, idx_test)

    train_dataloader = DataLoader(ds_correctness_train, batch_size=batch_size, shuffle=True) # shuffle NECESSARY!!! because positive and negative are concatenated
    test_dataloader = DataLoader(ds_correctness_test, batch_size=batch_size, shuffle=True)
    
    class_imbalance = len(idx_positive)/len(idx_negative)

    return train_dataloader, test_dataloader, class_imbalance, ds_correctness_train, ds_correctness_test

#%%

class tSNECallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        ds_train, ds_test = pl_module.datasets
        
        n_samples = 1000
        
        u_all = None
        y_all = None
        for x, y in DataLoader(ds_train, batch_size=256, shuffle=True):
            y = y.argmax(1)
            with torch.no_grad():
                u = pl_module(x.to(device))
            u_all = u.cpu() if u_all is None else torch.cat((u_all, u.cpu()))
            y_all = y.cpu() if y_all is None else torch.cat((y_all, y.cpu()))
            if u_all.shape[0] > n_samples:
                u_all = u_all[:n_samples]
                y_all = y_all[:n_samples]
                break
        # print(y_all.float().mean())
        
        u_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(u_all.numpy())

        # wellclassified = (class_predicted == digits)[:n_samples].cpu().numpy()
        plt.figure(figsize=(4, 4))
        plt.scatter(u_embedded[y_all==1, 0], u_embedded[y_all==1, 1], c='C0', alpha=0.5, label='well-classified')
        plt.scatter(u_embedded[y_all==0, 0], u_embedded[y_all==0, 1], c='C1', alpha=0.5, label='misclassified')
        # plt.scatter(u_embedded[:, 0], u_embedded[:, 1], c=y_all.float(), cmap='bwr', alpha=0.5)
        # plt.scatter(z_embedded[wellclassified, 0], z_embedded[wellclassified, 1], c='C0', label='well-classified', alpha=0.2)
        # plt.scatter(z_embedded[np.logical_not(wellclassified), 0], z_embedded[np.logical_not(wellclassified), 1], c='C1', label='misclassified', alpha=0.2)
        plt.legend(loc="lower left")
        plt.xticks([])
        plt.yticks([])
        plt.title('t-SNE on train data')
        plt.tight_layout()
        plt.savefig(pl_module.exp_dir / f'tsne_epoch{pl_module.current_epoch}')

class Model(pl.LightningModule):
    
    def __init__(self, model_name, loss_name, output_dim, classifier_digits, similarity, datasets, exp_dir):
        super().__init__()
        
        if loss_name == 'TripletMarginLoss':
            self.loss_function = torch.nn.TripletMarginLoss()
        
        if model_name == 'ResNet':
            weights = ResNet18_Weights.DEFAULT
            self.preprocess = weights.transforms()
            self.model = resnet18()
            self.model.fc = nn.Linear(512, output_dim, device=device) # replace last layer
        elif model_name == 'ResNet_pretrained':
            weights = ResNet18_Weights.DEFAULT
            self.preprocess = weights.transforms()
            self.model = resnet18(weights=weights)
            self.model.fc = nn.Linear(512, output_dim, device=device) # replace last layer
        elif model_name == 'ConvNeXt':
            weights = ConvNeXt_Tiny_Weights.DEFAULT
            self.preprocess = weights.transforms()
            self.model = convnext_tiny()
            self.model.classifier[2] = nn.Linear(768, output_dim) # replace last layer
        elif model_name == 'ConvNeXt_pretrained':
            weights = ConvNeXt_Tiny_Weights.DEFAULT
            self.preprocess = weights.transforms()
            self.model = convnext_tiny(weights=weights)
            self.model.classifier[2] = nn.Linear(768, output_dim) # replace last layer
        elif model_name == 'CNN':
            self.model = CNN_MNIST(1, output_dim)
        elif model_name == 'CNN_pretrained':
            self.model = copy.deepcopy(classifier_digits)
            self.model.net[9] = nn.Linear(128, output_dim, device=device) # replace last layer
            
        self.similarity = similarity
        self.datasets = datasets
        self.exp_dir = exp_dir
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(self, x):
        if isinstance(self.model, ResNet) or isinstance(self.model, ConvNeXt):
            x = self.preprocess(x.expand(-1, 3, -1, -1))
        elif isinstance(self.model, CNN_MNIST):
            x = (x / 255.)[:, :, 2:30, 2:30]
        return self.model(x)
    
    def _compute_loss(self, batch, mode):
        x, y = batch
        
        if (self.similarity == 'classifResult') and isinstance(self.loss_function, torch.nn.TripletMarginLoss):
            y = y.argmax(1)
            u = self(x)
            anchors = torch.zeros(u.shape, device=device)
            positives = torch.zeros(u.shape, device=device)
            negatives = torch.zeros(u.shape, device=device)
            for i, u_i in enumerate(u):
                anchors[i] = u_i
                # idx (anchors.to(device) != u_i.to(device)).all(1).cpu().numpy()
                idx_excluding_i = (torch.arange(u.shape[0], device=device) != i)
                idx_positives = (y == y[i]) & idx_excluding_i
                idx_negatives = (y != y[i]) & idx_excluding_i
                positives[i] = u[rng.choice(idx_positives.cpu().numpy())]
                negatives[i] = u[rng.choice(idx_negatives.cpu().numpy())]
                
            loss = self.loss_function(anchors, positives, negatives)
            
        self.log(f"{mode}_loss", loss)
        return loss
            
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, mode='train')
        return loss

    def validation_step(self, batch, batch_idx):
        self._compute_loss(batch, mode='val')
        
#%%

# MAIN
config = EasyDict()
config.dataset = 'mnist_stylegan2_blur_noise_maxSeverity3_proba50'
config.model = 'ResNet_pretrained'
config.loss = 'TripletMarginLoss'
config.output_dim = 128
config.similarity = 'classifResult'
config.batch_size = 128
config.train_test_split = 0.8
config.epochs = 30

# Create experiment folder, save config and logs
path_results_exp = create_experiment_folder(path_results / 'contrastive_learning', 'test')
config.exp_dir = str(path_results_exp)
with open(os.path.join(path_results_exp, 'training_options.json'), 'wt') as f:
    json.dump(config, f, indent=2)
# Logger(file_name=path_results_exp/'log.txt', file_mode='a', should_flush=True)


# Initialization
print(f'{config.model } - {config.loss}')

ds_original, classifier_digits = load_data(config.dataset)

train_dataloader, test_dataloader, class_imbalance, ds_train, ds_test = create_datasets(
    config.similarity, ds_original, classifier_digits, config.batch_size, config.train_test_split)

# Train
logger = TensorBoardLogger(save_dir=path_results_exp, name='', version='')
trainer = pl.Trainer(max_epochs=config.epochs, accelerator="gpu", devices=1, 
                        logger=logger, callbacks=[tSNECallback()])
model = Model(config.model, config.loss, config.output_dim, classifier_digits, config.similarity, (ds_train, ds_test), path_results_exp)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

# %%
