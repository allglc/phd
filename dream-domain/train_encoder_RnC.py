import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from rnc_loss import RnCLoss


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

# LOAD CLASSIFIER

# Create the classifier
classifier = resnet50(weights=ResNet50_Weights.DEFAULT)
classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
classifier = classifier.to(device)

# Load
state_dict = torch.load('../results/classifier/resnet50_in100_epoch_9.pth')
classifier.load_state_dict(state_dict)
classifier.eval()

# TRAIN ENCODER

# Load the anchor embeddings
anchor = torch.from_numpy(np.load('dream-ood-main/token_embed_in100.npy')).cuda()

# Create the encoder
encoder = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
encoder.fc = nn.Linear(encoder.fc.in_features, 768)
encoder = encoder.to(device)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
rnc = RnCLoss()
optimizer = optim.SGD(encoder.parameters(), lr=0.001, momentum=0.9)

# Define the number of epochs
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    loss_avg = 0.0
    true_classe_cosim_correct_avg = 0.0
    true_classe_cosim_incorrect_avg = 0.0
    
    # Set the encoder to training mode
    encoder.train()
    
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the frozen classifier
        with torch.no_grad():
            classif_outputs = classifier(images)
            probas = torch.softmax(classif_outputs, 1)
            confidences, predicted = torch.max(probas, 1)
            incorrect = (predicted != labels)

        # Forward pass
        features = encoder(images)
        features = features.unsqueeze(1).repeat(1, num_classes, 1)

        # Cosine similarity
        outputs = torch.nn.functional.cosine_similarity(anchor.unsqueeze(0).repeat(features.shape[0], 1, 1), features, 2) / 0.1
        
        # Compute the loss
        loss = criterion(outputs, labels)
        rnc_loss = rnc(features.repeat(1, 2, 1), confidences)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update statistics
        loss_avg = loss_avg * 0.8 + loss.item() * 0.2
        true_classe_cosim = 0.1 * outputs[torch.arange(outputs.shape[0]), labels].detach()
        true_classe_cosim_correct_avg = true_classe_cosim_correct_avg * 0.8 + true_classe_cosim[incorrect.bitwise_not()].mean().item() * 0.2
        if incorrect.any():
            true_classe_cosim_incorrect_avg = true_classe_cosim_incorrect_avg * 0.8 + true_classe_cosim[incorrect].mean().item() * 0.2
    
    # Print the statistics for each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_avg:.3f}, True class cosine similarity (correct): {true_classe_cosim_correct_avg:.3f}, True class cosine similarity (incorrect): {true_classe_cosim_incorrect_avg:.3f}")

    
    # Save the model after each epoch
    save_path = '../results/encoder_RnC'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(encoder.state_dict(), f"{save_path}/resnet50_in100_epoch_{epoch}.pth")




