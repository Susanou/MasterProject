import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets, layers, models

import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


def create_model():

    
    # CIFAR10 model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
              #run_eagerly=True #debug only
              )

    return model

def check_learner_acc(learner, images, labels):
    tp = 0

    for i, l in enumerate(learner.predict(images)):
        if labels[i] == np.argmax(l):
            tp += 1
    
    print(f"tp = {tp} => acc = tp/len = {float(tp/len(labels))}")
    print("Results from built-in tf function ", learner.evaluate(images, labels, batch_size=128, verbose=0))

def train_local(train_x, train_y, learners, i):

    try:
        model = learners[i]

        model.fit(train_x, train_y, epochs=10, shuffle=True, verbose=0)

        # Maybe better way but needed to save into a file at one point
        model.save(f'models/cifar/model_{i}.tf')
    except Exception as e:
        if culling:
            model = create_culled_model(6) # we are training them on 5 classes currently. make it modular later
        else:
            model = create_model()
    
        model.fit(train_x, train_y, epochs=10, shuffle=True, verbose=0)

        # Maybe better way but needed to save into a file at one point
        model.save(f'models/attack/model_{i}.tf')

def dataset_formatting_label_culling(train_x, train_y, global_size, fixed, FP):

    global culling
    culling = True

    if fixed:
        a_class = [0,4,6,8,9]
        b_class = [1,2,3,5,7]
    else:
        a_class = np.random.randint(10, size=5)
        b_class = np.random.randint(10, size=5)
    
    local_train_x = train_x[global_size:]
    local_train_y = train_y[global_size:]
    global_train_x = train_x[:global_size]
    global_train_y = train_y[:global_size]

    train_a_x = []
    train_b_x = []
    train_a_y = []
    train_b_y = []

    for i, e in enumerate(local_train_y):
        if e in a_class:
            train_a_x.append(local_train_x[i])
            train_a_y.append(a_class.index(local_train_y[i]))
        elif random.random() < FP:
            train_a_x.append(local_train_x[i])
            train_a_y.append(a_class.index(local_train_y[i]))
        
        if e in b_class:
            train_b_x.append(local_train_x[i])
            train_b_y.append(b_class.index(local_train_y[i]))
        elif random.random() < FP:
            train_b_x.append(local_train_x[i])
            train_b_y.append(b_class.index(local_train_y[i]))

    trainsets = [[np.array(train_a_x), np.array(train_a_y)], [np.array(train_b_x), np.array(train_b_y)]]

    return trainsets, global_train_x, global_train_y

def test_acc(learners, target):
    certain_global, count = vote(learners, x_test, y_test)
    print("OVER TEST DATA = Certain predictions amount", len(certain_global), "with correct in them", count)

    target_predict = target.predict(x_test)
    count = 0
    for i in range(len(y_test)):

        if np.argmax(target_predict[i]) == y_test[i]:
            count += 1

    print("OVER TEST DATA = TARGET Certain predictions amount", len(target_predict), "with correct in them", count)



##############
# Global Vars
##############

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Use GPUs
tf.config.set_soft_device_placement(True)
#tf.config.run_functions_eagerly(True) # debug for model not compiling
#tf.debugging.set_log_device_placement(True) #uncomment if need to check that it is executing off of GPU
tf.get_logger().setLevel('ERROR')

filename = "outputs/plotdata_100_1times_cifar_10Klocal_acc.csv"

f = open(filename, "a")
f.write("Epoch,Learner,Loss,Accuracy\n")
f.close()

(x_train, y_train), (x_test, y_test)= keras.datasets.cifar10.load_data()
#(x_train, y_train), (x_test, y_test)= keras.datasets.mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

global_size = 40000

assert global_size < len(x_train)

#trainsets, global_x, global_y, local_ds  = dataset_formatting(x_train, y_train, global_size, 10, 5)
trainsets, global_x, global_y = dataset_formatting_label_culling(x_train, y_train, 20000, True, 0.0)

# Set number of itterations either via local_ds or number of epochs to train
epochs = 50
#epochs = len(global_x) // (local_ds) + 1
local_ds = len(global_x) // epochs
repetition = 1


culling = False

learners = []

e = 0 #variable for the epoch number

#(x_train, y_train), (x_test, y_test)= keras.datasets.mnist.load_data()
attack = create_model()
attack.fit(trainsets[0][0], trainsets[0][1], epochs=10, shuffle=True, verbose=0)
attack.save(f'models\\attack\\model_attack.tf')