from resnet import ResNet18
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
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

def train(net, trainloader, optimizer, epoch):
    net.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if epoch%10==0:
        print('Epoch%d, Loss: %.3f' % (epoch, train_loss/(batch_idx+1)))

def train_learner(dataset, epochs, net=None):
    if net is None:
        net = ResNet18()
        net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    for epoch in range(epochs):
        train(net, trainloader, optimizer, epoch)
    return net

def tuning(learners, local_ds, theta, epochs):
    global_predictions = []
    for learner in learners:
        global_predictions.append([])
        for inp,lab in global_loader:
            _, preds = learner(inp.to(device)).max(1)
            global_predictions[-1] += preds.data.cpu().numpy().tolist()
    global_predictions = np.array(global_predictions)

    certain_global = []
    correct_count = 0
    for i in range(len(global_trainset)):
        tmp = np.zeros(10) #10 classes
        for pred in global_predictions[:, i]:
            tmp[pred] += 1
        if tmp.max() >= theta:
            certain_global.append((global_trainset[i][0], np.argmax(tmp)))
            if np.argmax(tmp) == global_trainset[i][1]:
                correct_count += 1
    print("Certain predictions amount", len(certain_global), "with correct in them", correct_count)

    acc = 0
    for i in range(n_learners):
        print("learner", i)
        local_dataset = torch.utils.data.Subset(local_trainset, list(range(i*local_ds, (i+1)*local_ds)))
        net = train_learner(local_dataset + certain_global, epochs=epochs, net = learners[i])
        acc += test(net)
        learners[i] = net
        
    return learners, acc/n_learners