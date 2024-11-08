import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

import sys
sys.path.append(os.path.expandvars('$WORK/dream-domain/code/dream-ood-main'))
from scripts.resnet import ResNet_Model

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


# TRAIN ENCODER

# Load the anchor embeddings
anchor = torch.from_numpy(np.load('dream-ood-main/token_embed_in100.npy')).cuda()

# Create the encoder
encoder = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
encoder.fc = nn.Linear(encoder.fc.in_features, 768)
encoder = encoder.to(device)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(encoder.parameters(), lr=0.001, momentum=0.9)

# Define the number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    loss_avg = 0.0
    
    # Set the encoder to training mode
    encoder.train()
    
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        features = encoder(images)
        features = features.unsqueeze(1).repeat(1, num_classes, 1)
        
        # Cosine similarity
        outputs = torch.nn.functional.cosine_similarity(anchor.unsqueeze(0).repeat(features.shape[0], 1, 1), features, 2) / 0.1
        
        # Compute the loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update statistics
        loss_avg = loss_avg * 0.8 + loss.item() * 0.2
    
    # Print the statistics for each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_avg}")
    
    # Save the model after each epoch
    torch.save(encoder.state_dict(), f"../results/encoder/resnet50_in100_epoch_{epoch}.pth")
    
    

# save ID features.
number_dict = {}
for i in range(num_classes):
    number_dict[i] = 0
encoder.eval()
data_dict = torch.zeros(num_classes, 1000, 768).cuda()
with torch.no_grad():
    for _, in_set in enumerate(train_dataloader):

        data = in_set[0]
        target = in_set[1]

        data, target = data.cuda(), target.cuda()
        # forward
        feat = encoder(data)
        target_numpy = target.cpu().data.numpy()
        for index in range(len(target)):
            dict_key = target_numpy[index]
            if number_dict[dict_key] < 1000:
                data_dict[dict_key][number_dict[dict_key]] = feat[index].detach()
                number_dict[dict_key] += 1
        
        if all(value == 1000 for value in number_dict.values()):
            break

np.save('../results/encoder/id_feat_in100.npy', data_dict.cpu().numpy())


