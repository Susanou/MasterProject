import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.autograd import Variable

import onnx
from onnx_tf.backend import prepare

import numpy as np
from IPython.display import display
from PIL import Image

import tensorflow as tf

from pytorch2keras.converter import pytorch_to_keras

import os
import re

class Cifar10PaperNet(nn.Module):
    def __init__(self):
        super(Cifar10PaperNet, self).__init__()   
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(2304, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


regex = r"model_round(.*).model"

directory = 'torch_models'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    #print(filename)

    if os.path.isfile(f):

        epoch = int(re.search(regex, filename).group(1))
        epoch = (epoch+1)//20
        #print(epoch)

        trained_model = Cifar10PaperNet()
        trained_model.load_state_dict(torch.load(f))

        input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
        input_var = Variable(torch.FloatTensor(input_np))
        print(input_var.shape)

        tf_ref = pytorch_to_keras(trained_model, input_var, [(3, 32, 32,)], verbose=False, change_ordering=False)
        print(tf_ref.summary())
        tf_ref.save(f'keras_models\\model_{epoch}.tf')