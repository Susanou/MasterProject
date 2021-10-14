import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from keras.models import load_model


import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

##############
# Global Vars
##############
"""
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



local_trainset = torch.utils.data.Subset(trainset, list(range(0, 5000)))
print("local data", len(local_trainset))
global_trainset = torch.utils.data.Subset(trainset, list(range(5000, 50000)))
print("global data", len(global_trainset))
global_loader = torch.utils.data.DataLoader(global_trainset, batch_size=100, shuffle=False, num_workers=2)
"""

learners = []

builder = tfds.builder('cifar10')
builder.download_and_prepare(download_dir='data/')
print(builder.info.splits.keys())
print(builder.info.splits['train'].num_examples)
print(builder.info.splits['test'].num_examples)

local_trainset = builder.as_dataset()['train']
print(local_trainset)
assert isinstance(local_trainset, tf.data.Dataset)
global_trainset = builder.as_dataset()['test']

n_learners = 2 # Change that later
theta = 4
local_ds = len(local_trainset)//n_learners
print("Length of the local dataset", local_ds)

trainsets = list(local_trainset.batch(2).as_numpy_iterator())
print(trainsets)

# Training loop
for i in range(n_learners):
    # Building model

    model = keras.Sequential(
        [
        keras.layers.Flatten(input_shape=(32,32, 3)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(50),
        keras.layers.Dense(10)
        ]
    )
    model.compile(  optimizer='adam',
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    
    model.fit(trainsets[i], epochs=10, shuffle=True)

    # Maybe better way but needed to save into a file at one point
    model.save(f'models/model_{i}.tf')
    del model
    learners.append(load_model(f'models/model_{i}.tf'))

# Target training 

model = keras.Sequential(
    [
    keras.layers.Flatten(input_shape=(32,32, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(50),
    keras.layers.Dense(10)
    ]
)
model.compile(  optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

model.fit(local_trainset, epochs=10, shuffle=True)
model.save(f'models/target.tf')
del model
target = load_model('models/target.tf')

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