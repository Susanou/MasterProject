import numpy as np
import os
import shutil

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import ml_privacy_meter
import tensorflow as tf


input_shape = (32, 32, 3)

epoch = 25
max_epochs = 100
from_dir = "logs/plots"

while epoch < max_epochs:
    print(f"epoch num{epoch}")
    to_dir = f"epoch_logs_fl/logs_fl_90%_{epoch-1}e"
    if os.path.isdir("logs"):
        shutil.copytree(from_dir, to_dir)
        shutil.rmtree("logs", ignore_errors=True)

    # Load saved target model to attack
    cprefix = 'target.tf'
    cmodelA = tf.keras.models.load_model(cprefix)
    cprefix = f'model_0.tf'
    cmodelB = tf.keras.models.load_model(cprefix)

    #cmodelA.summary()
    #cmodelB.summary()

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
        layers_to_exploit=[6, 7],
        gradients_to_exploit=[4],
        device=None, epochs=epoch if epoch != 0 else 1, model_name=f'target_vitcim_{epoch}e_white') # change number of epochs for FL
    
    print("starting attack training")
    attackobj.train_attack()
    
    print("Generating plots")
    attackobj.test_attack()
    epoch += 1