import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import ml_privacy_meter
import tensorflow as tf

# Load saved target model to attack
cprefix = 'target.tf'
cmodelA = tf.keras.models.load_model(cprefix)
cprefix = 'model_0.tf'
cmodelB = tf.keras.models.load_model(cprefix)

cmodelA.summary()
cmodelB.summary()

def preprocess_cifar10_dataset():
    input_shape = (32, 32, 3)
    num_classes = 10

    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    return x_train, y_train, x_test, y_test, input_shape, num_classes


x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_cifar10_dataset()

# training data of the target model
num_datapoints = 10000
x_target_train, y_target_train = x_train[:num_datapoints], y_train[:num_datapoints]

# population data (training data is a subset of this)
x_population = np.concatenate((x_train, x_test))
y_population = np.concatenate((y_train, y_test))

datahandlerA = ml_privacy_meter.utils.attack_data.AttackData(x_population=x_population,
                                                             y_population=y_population,
                                                             x_target_train=x_target_train,
                                                             y_target_train=y_target_train,
                                                             batch_size=100, # test if it affects AIMHI
                                                             attack_percentage=10, input_shape=input_shape,
                                                             normalization=True)

attackobj = ml_privacy_meter.attack.meminf.initialize(
    target_train_model=cmodelA,
    target_attack_model=cmodelB,
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[7],
    exploit_loss=False,
    device=None, epochs=100, model_name='target_vitcim_10e_black')

print("starting attack training")
attackobj.train_attack()

print("Generating plots")
attackobj.test_attack()
