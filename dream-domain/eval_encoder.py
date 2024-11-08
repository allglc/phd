import os
import numpy as np
import pandas as pd
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from evaluation import test_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PREPARE DATA
traindir = os.path.expandvars('$WORK/DATA/IN100/train')
valdir = os.path.expandvars('$WORK/DATA/IN100/val')

train_data = torchvision.datasets.ImageFolder(
    traindir,
    torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ]))

test_data = torchvision.datasets.ImageFolder(
    valdir,
    torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ]))
num_classes = 100

# Set batch size
batch_size = 128

# Create dataloaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Create the classifier
classifier = resnet50(weights=ResNet50_Weights.DEFAULT)
classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
classifier = classifier.to(device)

# Load
state_dict = torch.load('../results/classifier/resnet50_in100_epoch_9.pth')
classifier.load_state_dict(state_dict)
classifier.eval()

# Load the anchor embeddings
anchor = torch.from_numpy(np.load('dream-ood-main/token_embed_in100.npy')).cuda()


# MAIN LOOP
for dataloader_str, dataloader in zip(['test', 'train'], [test_dataloader, train_dataloader]):
    for encoder_path in ['encoder_custom_2bis']:
        for epoch in range(20):
            print(f"Testing {encoder_path} on {dataloader_str} at epoch {epoch}...")

            # Create the encoder
            encoder = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            encoder.fc = nn.Linear(encoder.fc.in_features, 768)
            encoder = encoder.to(device)

            state_dict = torch.load(f"../results/{encoder_path}/resnet50_in100_epoch_{epoch}.pth")
            encoder.load_state_dict(state_dict)
            encoder.eval()

            metrics = test_encoder(encoder, dataloader, num_classes, device, anchor, classifier)

            df_res = pd.DataFrame([{'dataloader': dataloader_str, 'encoder': encoder_path, 'epoch': epoch, **metrics}])
            f_path_test = f'../results/{encoder_path}/evaluation.csv'
            if os.path.exists(f_path_test):
                df_0 = pd.read_csv(f_path_test)
                df_res = pd.concat([df_0, df_res], axis=0)
            df_res.to_csv(f_path_test, index=False)