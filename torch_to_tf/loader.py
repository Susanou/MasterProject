import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import onnx
from onnx_tf.backend import prepare

import numpy as np
from IPython.display import display
from PIL import Image

import tensorflow as tf

from pytorch2keras.converter import pytorch_to_keras

class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Load the trained model from file
trained_model = Net()
trained_model.load_state_dict(torch.load('mnist.pth'))

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 1, 28, 28)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(trained_model, dummy_input, "mnist.onnx")


# Load the ONNX file
#model = onnx.load('mnist.onnx')

tf_ref = pytorch_to_keras(trained_model, dummy_input, [(1, 28, 28)], verbose=True)
print(tf_ref.summary())
tf_ref.save('mnist_real.pb')

# Import the ONNX model to Tensorflow
#tf_rep = prepare(model)

# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')
print(tf_rep.tensor_dict)

print('Image 1:')
img = Image.open('two.png').resize((28, 28)).convert('L')
display(img)
output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
print('The digit is classified as ', np.argmax(output))

print('Image 2:')
img = Image.open('three.png').resize((28, 28)).convert('L')
display(img)
output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
print('The digit is classified as ', np.argmax(output))


tf_rep.export_graph('mnist.pb')

m = tf.keras.models.load_model('mnist.pb')
print(m.summary())