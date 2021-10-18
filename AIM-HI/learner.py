import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.models import load_model


import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


def create_model():
    model = keras.Sequential(
        [
        keras.layers.Conv2D(input_shape=(32,32,3), filters=10, kernel_size=3,
                            strides=2, activation='relu', name='Conv1'),
        keras.layers.Flatten(),
        keras.layers.Dense(50),
        keras.layers.Dense(10)
        ]
    )
    model.compile(  optimizer='adam',
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model


##############
# Global Vars
##############

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
    print("Building model")
    # Building model

    model = create_model()
    
    model.fit(trainsets[i][0], trainsets[i][1], epochs=100, shuffle=True)

    # Maybe better way but needed to save into a file at one point
    model.save(f'models/model_{i}.tf')
    del model
    learners.append(load_model(f'models/model_{i}.tf'))

# Target training 

model = create_model()

model.fit(x_train, y_train, epochs=100, shuffle=True)
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

    tmp = np.maximum.reduce(global_predictions[:, i])

    certain_global.append(np.argmax(tmp))

    if np.argmax(tmp) == y_test[i]:
        count += 1

certain_global = np.array(certain_global)
print("Certain predictions amount", len(certain_global), "with correct in them", count)

# fit model to the new labels
# Training loop
for i in range(n_learners):
    
    model = learners[i]

    model.fit(x_test, certain_global, epochs=100, shuffle=True)

    # Maybe better way but needed to save into a file at one point
    model.save(f'models/model_{i}.tf')
    del model
    learners[i] = load_model(f'models/model_{i}.tf')

# Last round of perdications to check accuracy changes

global_predictions = []
for learner in learners:
    global_predictions.append(learner.predict(x_test))

global_predictions = np.array(global_predictions)

print(np.argmax(global_predictions[0][0]), np.argmax(global_predictions[1][0]), y_test[0])

# Voting part loop
certain_global = []
count = 0


for i in range(len(y_test)): 

    tmp = np.maximum.reduce(global_predictions[:, i])

    certain_global.append(tmp)

    if np.argmax(tmp) == y_test[i]:
        count += 1

certain_global = np.array(certain_global)
print("Certain predictions amount", len(certain_global), "with correct in them", count)