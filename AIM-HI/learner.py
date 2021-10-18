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

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

n_learners = 2 # Change that later
theta = 4
local_ds = len(x_train)//n_learners
print("Length of the local dataset", local_ds)

train_a_x = x_train[:local_ds]
train_b_x = x_train[local_ds:]
train_a_y = y_train[:local_ds]
train_b_y = y_train[local_ds:]

trainsets = [(train_a_x, train_a_y), (train_b_x, train_b_y)]
#print(trainsets)

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
    
    model.fit(trainsets[i][0], trainsets[i][1], epochs=10, shuffle=True)

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

model.fit(x_train, y_train, epochs=10, shuffle=True)
model.save(f'models/target.tf')
del model
target = load_model('models/target.tf')

# Global training

global_predictions = []
for learner in learners:
    global_predictions.append(learner.predict(x_test))

global_predictions = np.array(global_predictions)

print(np.argmax(global_predictions[0][0]), np.argmax(global_predictions[1][0]), y_test[0])

# Voting part loop
certain_global = []
count = 0
for i in range(len(y_test)): 

    tmp = np.maximum.reduce(global_predictions)
    certain_global.append(tmp)

    if np.argmax(tmp) == y_test[i]:
        count += 1

print("Certain predictions amount", len(certain_global), "with correct in them", count)

