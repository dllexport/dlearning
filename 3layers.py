#!/usr/bin/env python
# coding: utf-8


import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

epochs = 1

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize([0.5], [0.5])
    ])

# Download and load the training data
trainset = datasets.MNIST('./MNIST_data/train', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('./MNIST_data/test', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Define the loss
criterion = nn.CrossEntropyLoss()

from torch import optim

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.03)

for e in range(epochs):
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        print("loss: " + str(loss.item()))
        optimizer.step()

# now the training is finished
import helper
images, labels = next(iter(testloader))

import matplotlib.pyplot as plt
for index in range(10):
    img = images[index].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    #print(ps)
    helper.view_classify(img.view(1, 28, 28), ps)
    plt.show()
