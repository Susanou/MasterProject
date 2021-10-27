import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.models import load_model

import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


def create_model():
    model = keras.Sequential(
        [
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10)
        ]
    )
    model.compile(  optimizer='adam',
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model

def vote(voters, images, labels):

    # Global Predications of all learners

    global_predictions = []
    for v in voters:
        global_predictions.append(v.predict(images))
        #print(v.predict(images))

    global_predictions = np.array(global_predictions)

#    print(global_predictions)

    # Voting part loop
    certain_global = []
    count = 0

    for i in range(len(labels)): 

        tmp = np.maximum.reduce(global_predictions[:, i])

        certain_global.append(np.argmax(tmp))

        if np.argmax(tmp) == labels[i]:
            count += 1

    return certain_global, count

def train_local(train_x, train_y, learners, i):

    try:
        model = learners[i]

        model.fit(train_x, train_y, epochs=10, shuffle=True)

        # Maybe better way but needed to save into a file at one point
        model.save(f'models/model_{i}.tf')
    except Exception as e:
        model = create_model()
    
        model.fit(train_x, train_y, epochs=10, shuffle=True)

        # Maybe better way but needed to save into a file at one point
        model.save(f'models/model_{i}.tf')

def dataset_formatting(train_x, train_y, global_size, percent):
    local_train_x = train_x[global_size:]
    local_train_y = train_y[global_size:]
    global_train_x = train_x[:global_size]
    global_train_y = train_y[:global_size]

    n_learners = 2 # Change that later
    theta = 4
    local_ds = (len(local_train_x)*percent)//100
    print("Length of the local dataset", local_ds)

    train_a_x = local_train_x[:local_ds]
    train_b_x = local_train_x[local_ds:]
    train_a_y = local_train_y[:local_ds]
    train_b_y = local_train_y[local_ds:]

    trainsets = [(train_a_x, train_a_y), (train_b_x, train_b_y)]

    return trainsets, global_train_x, global_train_y

def dataset_formatting_label_culling(train_x, train_y, global_size, fixed, FP):
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
            train_a_y.append(local_train_y[i])
        elif random.random() < FP:
            train_a_x.append(local_train_x[i])
            train_a_y.append(local_train_y[i])
        
        if e in b_class:
            train_b_x.append(local_train_x[i])
            train_b_y.append(local_train_y[i])
        elif random.random() < FP:
            train_b_x.append(local_train_x[i])
            train_b_y.append(local_train_y[i])
    
    trainsets = [(np.array(train_a_x), np.array(train_a_y)), (np.array(train_b_x), np.array(train_b_y))]

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

learners = []

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

trainsets, global_x, global_y = dataset_formatting(x_train, y_train, 40000, 10)
#trainsets, global_x, global_y = dataset_formatting_label_culling(x_train, y_train, 30000, False, 0.0)

# Training loop
for i in range(len(trainsets)):
    print("Building model")

    train_local(trainsets[i][0], trainsets[i][1], learners, i)
    learners.append(load_model(f'models/model_{i}.tf'))

print("learners: ", len(learners))

# Target training 

model = create_model()

model.fit(x_train, y_train, epochs=10, shuffle=True)
model.save(f'models/target.tf')
del model
target = load_model('models/target.tf')


certain_global, count = vote(learners, global_x, global_y)

certain_global = np.array(certain_global)
print("Certain predictions amount", len(certain_global), "with correct in them", count)

target_predict = target.predict(global_x)
count = 0
for i in range(len(global_y)):

    if np.argmax(target_predict[i]) == global_y[i]:
        count += 1

print("TARGET Certain predictions amount", len(target_predict), "with correct in them", count)

test_acc(learners, target)

# fit model to the new labels
# Training loop
for i in range(len(learners)):
    
    train_local(global_x, certain_global, learners, i)
    learners = []
    learners.append(load_model(f'models/model_{i}.tf'))

# Last round of perdictions to check accuracy changes

test_acc(learners, target)