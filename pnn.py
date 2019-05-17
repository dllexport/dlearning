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
trainset = datasets.MNIST('./MNIST_data/train', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=False)

testset = datasets.MNIST('./MNIST_data/test', download=False, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=True)

clist = []
for i in range(11):
    clist.append(list())

images, labels = next(iter(trainloader))
test_images, test_labels = next(iter(testloader))

for i in range(1024):
    if len(clist[labels[i] + 1]) != 80:
        clist[labels[i] + 1].append(images[i])

six = 25
def gaussian_func(x, y):
    sub = (x - y)
    sub_square = torch.pow(sub, 2)
    sum = torch.sum(sub_square)
    return torch.exp(-sum / (2 * six * six))



def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

def PNN(input):
    sum_list = []

    for i in range(1, 11):
        listj = []

        for j in range(0, 80):
            loss = gaussian_func(input, clist[i][j])
            print("loss: " + str(loss))
            listj.append(loss)

        sum_list.append(sum(listj))
    
    sum_list_tensor = torch.tensor(sum_list)
    return softmax(sum_list_tensor)
print(test_images[0].shape)

import helper
import matplotlib.pyplot as plt

for index in range(10):
    #select the first image
    img = test_images[index]
    # Turn off gradients to speed up this part
    
    ps = PNN(img)

    helper.view_classify(img.view(1, 28, 28), ps)
    plt.show()


