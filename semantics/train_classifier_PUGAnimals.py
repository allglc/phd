from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
import torch
import pandas as pd
from PIL import Image
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


PUG_ANIMAL_PATH = os.path.expandvars('$SCRATCH/PUG_Animal/')
path_results = '../results/PUG/Animal/'

model_name = 'ViT-B16'
models_and_weights_imagenet = {
    'ResNet-50': (resnet50, ResNet50_Weights.DEFAULT),
    'ViT-B16': (vit_b_16, ViT_B_16_Weights.DEFAULT), 
}


class UnrealDatasetCustom(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)
        self.df.astype(str)
        self.images_folder = images_folder
        self.transform = transform
        self.dict_labels_to_idx = {l: i for i, l in enumerate(sorted(self.df['character_name'].unique()))}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        filename = self.df['filename'][index]
        label = self.df['character_name'][index]
        image = Image.open(os.path.join(self.images_folder, label, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label, self.df['world_name'][index], self.df['character_scale'][index], self.df['character_texture'][index], self.df['camera_yaw'][index]
    
    def labels_to_idx(self, labels):
        return [self.dict_labels_to_idx[l] for l in labels]

attributes_names = ['world_name', 'character_name', 'character_scale', 'character_texture', 'camera_yaw']

path_model = path_results + f'{model_name}_finetuned_weights.pth'


def main():
    # Build model 
    architecture, weights = models_and_weights_imagenet[model_name]
    classifier = architecture(weights=weights).eval().cuda()
    transforms = weights.transforms()
    dataset = UnrealDatasetCustom(csv_path=PUG_ANIMAL_PATH+"labels_pug_animal.csv", images_folder=PUG_ANIMAL_PATH, transform=transforms)

    # Build balanced datasets: train, calib, test of size [0.5, 0.25, 0.25]
    train_idx, valid_idx = train_test_split(np.arange(len(dataset)), train_size=0.5, random_state=0, shuffle=True, stratify=dataset.df['character_name'])
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    # split validation into calib and test
    calib_idx, test_idx = train_test_split(np.arange(len(valid_dataset)), test_size=0.5, random_state=0, shuffle=True, stratify=dataset.df.iloc[valid_idx]['character_name'])
    calib_dataset = Subset(valid_dataset, calib_idx)
    test_dataset = Subset(valid_dataset, test_idx)

    # Build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
    calib_loader = DataLoader(calib_dataset, batch_size=128, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4, shuffle=False)


    # retrain last layer only
    # for param in classifier.parameters():
    #     param.requires_grad = False

    if model_name == 'ViT-B16':
        classifier.heads.head = torch.nn.Linear(classifier.heads.head.in_features, dataset.df['character_name'].nunique()).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100):
        # train
        classifier.train()
        running_loss = 0.0
        for data in train_loader:
            images = data[0].cuda()
            labels = dataset.labels_to_idx(data[1])
            labels = torch.tensor(labels).cuda()
            
            optimizer.zero_grad()
            logits = classifier(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_epoch_loss = running_loss / len(train_loader)

        # eval
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in calib_loader:
                images = data[0].cuda()
                labels = dataset.labels_to_idx(data[1])
                labels = torch.tensor(labels).cuda()
                logits = classifier(images)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}, Train Loss: {avg_epoch_loss:.4f}, Validation accuracy: {100*correct/total:.2f}%')
        torch.save(classifier.state_dict(), path_model)

if __name__ == "__main__":
    main()