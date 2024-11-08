import os
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

classifier = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
classifier = classifier.to(device)

# Set batch size
batch_size = 128

# Create dataloaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

def test():
    # Set the model to evaluation mode
    classifier.eval()

    correct = 0
    total = 0

    # Use torch.no_grad to skip the gradient calculation as we are only testing the model
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the model on the test images: %d%%' % accuracy)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

# Define the number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Set the model to training mode
    classifier.train()
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = classifier(images)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    test()

    # Print the statistics for each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_data)}, Accuracy: {(correct/total)*100}%")

    # Save the model after each epoch
    torch.save(classifier.state_dict(), f"../results/classifier/resnet50_in100_epoch_{epoch}.pth")
