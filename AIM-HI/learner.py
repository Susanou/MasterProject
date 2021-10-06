import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import tensorflow as tf
from tensorflow import keras

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

##############
# Global Vars
##############

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.CrossEntropyLoss()

learners = []

local_trainset = torch.utils.data.Subset(trainset, list(range(0, 5000)))
print("local data", len(local_trainset))
global_trainset = torch.utils.data.Subset(trainset, list(range(5000, 50000)))
print("global data", len(global_trainset))
global_loader = torch.utils.data.DataLoader(global_trainset, batch_size=100, shuffle=False, num_workers=2)

n_learners = 2 # Change that later
theta = 4
local_ds = len(local_trainset)//n_learners
print("Length of the local dataset", local_ds)

def train_local(dataset, epochs, net=None):
    if net == None:
        net = models.resnet18()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    net.train()
    train_loss = 0

    for e in range(epochs):
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if e%10==0:
            print('Epoch%d, Loss: %.3f' % (e, train_loss/(i+1)))

    return net

# Training loop
for i in range(n_learners):
    local_dataset = torch.utils.data.Subset(local_trainset, list(range(i*local_ds, (i+1)*local_ds)))
    net = train_local(local_dataset, 1000)
    learners.append(net)

# Global training

global_predictions = []
for learner in learners:
    global_predictions.append([])

    for inputs, labels in global_loader:
        _, pred = learner(inputs.to(device)).max(1)
        global_predictions[-1] += pred.data.cpu().numpy().tolist()

global_predictions = np.array(global_predictions)

# Voting part loop
certain_global = []
count = 0
for i in range(len(global_trainset)): 
    tmp = np.zeros(10) #10 classes
    for pred in global_predictions[:, i]:
        tmp[pred] += 1
    if tmp.max() >= theta:
        certain_global.append((global_trainset[i][0], np.argmax(tmp)))
        if np.argmax(tmp) == global_trainset[i][1]:
            count += 1

print("Certain predictions amount", len(certain_global), "with correct in them", count)