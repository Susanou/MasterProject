import numpy as np
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import ml_privacy_meter
import tensorflow as tf

input_shape = (32, 32, 3)

# Load saved target model to attack
cprefix = 'target.tf'
cmodelA = tf.keras.models.load_model(cprefix)
cprefix = 'models_fl\\model_1.tf'
cmodelB = tf.keras.models.load_model(cprefix)

cmodelA.summary()
cmodelB.summary()

saved_path = "datasets/cifar10_train.txt.npy"

# Similar to `saved_path` being used to form the memberset for attack model,
# `dataset_path` is used for forming the nonmemberset of the training data of
# attack model.
dataset_path = 'datasets/cifar10.txt'

datahandlerA = ml_privacy_meter.utils.attack_data.attack_data(dataset_path=dataset_path,
                                                              member_dataset_path=saved_path,
                                                              batch_size=1000, #update this fro AIMHI
                                                              attack_percentage=90, input_shape=input_shape, #90 assumes server is attacking
                                                              normalization=True)

attackobj = ml_privacy_meter.attack.meminf.initialize(
    target_train_model=cmodelB,
    target_attack_model=cmodelB,
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[7],
    exploit_loss=False,
    device=None, epochs=30, model_name='target_vitcim_100e_black')

print("starting attack training")
attackobj.train_attack()

print("Generating plots")
attackobj.test_attack()
