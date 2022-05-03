'''
The Attack class.
'''
import datetime
import itertools
import json
import os
import time

import numpy as np

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from ml_privacy_meter.utils.attack_utils import attack_utils, sanity_check
from ml_privacy_meter.utils.logger import get_logger
from ml_privacy_meter.utils.losses import CrossEntropyLoss, mse
from ml_privacy_meter.utils.optimizers import optimizer_op
from ml_privacy_meter.visualization.visualize import compare_models
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import accuracy_score, auc, roc_curve

from .meminf_modules.encoder import create_encoder
from .meminf_modules.create_cnn import (cnn_for_cnn_gradients,
                                        cnn_for_cnn_layeroutputs,
                                        cnn_for_fcn_gradients)
from .meminf_modules.create_fcn import fcn_module

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Sets soft placement below for GPU memory issues
tf.config.set_soft_device_placement(True)

ioldinit = tf.compat.v1.Session.__init__


def myinit(session_object, target='', graph=None, config=None):
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)


tf.compat.v1.Session.__init__ = myinit


# To decide what attack component (FCN or CNN) to
# use on the basis of the layer name.
# CNN_COMPONENTS_LIST are the layers requiring each input in 3 dimensions.
# GRAD_COMPONENTS_LIST are the layers which have trainable components for computing gradients
CNN_COMPONENT_LIST = ['Conv', 'MaxPool']
GRAD_LAYERS_LIST = ['Conv', 'Dense']


class initialize(object):
    """
    This attack was originally proposed by Nasr et al. It exploits
    one-hot encoding of true labels, loss value, intermediate layer 
    activations and gradients of intermediate layers of the target model 
    on data points, for training the attack model to infer membership 
    in training data.

    Paper link: https://arxiv.org/abs/1812.00910

    Args:
    ------
    target_train_model: The target (classification) model that'll 
                        be used to train the attack model.

    target_attack_model: The target (classification) model that we are
                         interested in quantifying the privacy risk of. 
                         The trained attack model will be used 
                         for attacking this model to quantify its membership
                         privacy leakage. 

    train_datahandler: an instance of `ml_privacy_meter.data.attack_data.load`,
                       that is used to retrieve dataset for training the 
                       attack model. The member set of this training set is
                       a subset of the classification model's
                       training set. Check Main README on how to 
                       load dataset for the attack.

    attack_datahandler: an instance of `ml_privacy_meter.data.attack_data.load`,
                        used to retrieve dataset for evaluating the attack 
                        model. The member set of this test/evaluation set is
                        a subset of the target attack model's train set minus
                        the training members of the target_train_model.

    optimizer: The optimizer op for training the attack model.
               Default op is "adam".

    layers_to_exploit: a list of integers specifying the indices of layers,
                       whose activations will be exploited by the attack model.
                       If the list has only a single element and 
                       it is equal to the index of last layer,
                       the attack can be considered as a "blackbox" attack.

    gradients_to_exploit: a list of integers specifying the indices 
                          of layers whose gradients will be 
                          exploited by the attack model. 

    exploit_loss: boolean; whether to exploit loss value of target model or not.
   
    exploit_label: boolean; whether to exploit one-hot encoded labels or not.                 
   
    learning_rate: learning rate for training the attack model 
    
    epochs: Number of epochs to train the attack model 

    Examples:
    """

    def __init__(self,
                 target_train_model,
                 target_attack_model,
                 train_datahandler,
                 attack_datahandler,
                 device=None,
                 optimizer="adam",
                 model_name="sample_model",
                 layers_to_exploit=None,
                 gradients_to_exploit=None,
                 exploit_loss=True,
                 exploit_label=True,
                 learning_rate=0.001,
                 epochs=100):

        # Set self.loggers (directory according to todays date)
        time_stamp = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        self.attack_utils = attack_utils()
        self.logger = get_logger(self.attack_utils.root_dir, "attack",
                                 "meminf", "info", time_stamp)
        self.aprefix = os.path.join('logs',
                                    "attack", "tensorboard")
        self.summary_writer = tf.summary.create_file_writer(self.aprefix)
        self.target_train_model = target_train_model
        self.target_attack_model = target_attack_model
        self.train_datahandler = train_datahandler
        self.attack_datahandler = attack_datahandler
        self.optimizer = optimizer_op(optimizer, learning_rate)
        self.layers_to_exploit = layers_to_exploit
        self.gradients_to_exploit = gradients_to_exploit
        self.exploit_loss = exploit_loss
        self.device = device
        self.exploit_label = exploit_label
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_size = int(target_train_model.output.shape[1])
        self.ohencoding = self.attack_utils.createOHE(self.output_size)
        self.model_name = model_name

        # Create input containers for attack & encoder model.
        self.create_input_containers()
        layers = target_train_model.layers

        # basic sanity checks
        sanity_check(layers, layers_to_exploit)
        sanity_check(layers, gradients_to_exploit)

        # Create individual attack components
        self.create_attack_components(layers)

        # Initialize the attack model
        self.initialize_attack_model()

        # Log info
        self.log_info()

    def log_info(self):
        """
        Logs vital information pertaining to training the attack model.
        Log files will be stored in `/ml_privacy_meter/logs/attack_logs/`.
        """
        self.logger.info("`exploit_loss` set to: {}".format(self.exploit_loss))
        self.logger.info(
            "`exploit_label` set to: {}".format(self.exploit_label))
        self.logger.info("`layers_to_exploit` set to: {}".format(
            self.layers_to_exploit))
        self.logger.info("`gradients_to_exploit` set to: {}".format(
            self.gradients_to_exploit))
        self.logger.info("Number of Epochs: {}".format(self.epochs))
        self.logger.info("Learning Rate: {}".format(self.learning_rate))
        self.logger.info("Optimizer: {}".format(self.learning_rate))

    def create_input_containers(self):
        """
        Creates arrays for inputs to the attack and 
        encoder model. 
        (NOTE: Although the encoder is a part of the attack model, 
        two sets of containers are required for connecting 
        the TensorFlow graph).
        """
        self.attackinputs = []
        self.encoderinputs = []

    def create_layer_components(self, layers):
        """
        Creates CNN or FCN components for layers to exploit
        """
        for l in self.layers_to_exploit:
            # For each layer to exploit, module created and added to self.attackinputs and self.encoderinputs
            layer = layers[l-1]
            input_shape = layer.output_shape[1]
            requires_cnn = map(lambda i: i in layer.__class__.__name__,
                               CNN_COMPONENT_LIST)
            if any(requires_cnn):
                module = cnn_for_cnn_layeroutputs(layer.output_shape)
            else:
                module = fcn_module(input_shape, 100)
            self.attackinputs.append(module.input)
            self.encoderinputs.append(module.output)

    def create_label_component(self, output_size):
        """
        Creates component if OHE label is to be exploited
        """
        module = fcn_module(output_size)
        self.attackinputs.append(module.input)
        self.encoderinputs.append(module.output)

    def create_loss_component(self):
        """
        Creates component if loss value is to be exploited
        """
        module = fcn_module(1, 100)
        self.attackinputs.append(module.input)
        self.encoderinputs.append(module.output)

    def create_gradient_components(self, model, layers):
        """
        Creates CNN/FCN component for gradient values of layers of gradients to exploit
        """
        grad_layers = []
        for layer in layers:
            if any(map(lambda i: i in layer.__class__.__name__, GRAD_LAYERS_LIST)):
                grad_layers.append(layer)
        variables = model.variables
        for layerindex in self.gradients_to_exploit:
            # For each gradient to exploit, module created and added to self.attackinputs and self.encoderinputs
            layer = grad_layers[layerindex-1]
            shape = self.attack_utils.get_gradshape(variables, layerindex)
            requires_cnn = map(lambda i: i in layer.__class__.__name__,
                               CNN_COMPONENT_LIST)
            if any(requires_cnn):
                module = cnn_for_cnn_gradients(shape)
            else:
                module = cnn_for_fcn_gradients(shape)
            self.attackinputs.append(module.input)
            self.encoderinputs.append(module.output)

    def create_attack_components(self, layers):
        """
        Creates FCN and CNN modules constituting the attack model.  
        """
        model = self.target_train_model

        # for layer outputs
        if self.layers_to_exploit and len(self.layers_to_exploit):
            self.create_layer_components(layers)

        # for one hot encoded labels
        if self.exploit_label:
            self.create_label_component(self.output_size)

        # for loss
        if self.exploit_loss:
            self.create_loss_component()

        # for gradients
        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            self.create_gradient_components(model, layers)

        # encoder module
        self.encoder = create_encoder(self.encoderinputs)

    def initialize_attack_model(self):
        """
        Initializes a `tf.keras.Model` object for attack model.
        The output of the attack is the output of the encoder module.
        """
        output = self.encoder
        self.attackmodel = tf.compat.v1.keras.Model(inputs=self.attackinputs,
                                                    outputs=output)

    def get_layer_outputs(self, model, features):
        """
        Get the intermediate computations (activations) of 
        the hidden layers of the given target model.
        """
        layers = model.layers
        for l in self.layers_to_exploit:
            x = model.input
            y = layers[l-1].output
            # Model created to get output of specified layer
            new_model = tf.compat.v1.keras.Model(x, y)
            predicted = new_model(features)
            self.inputArray.append(predicted)

    def get_labels(self, labels):
        """
        Retrieves the one-hot encoding of the given labels.
        """
        ohe_labels = self.attack_utils.one_hot_encoding(
            labels, self.ohencoding)
        return ohe_labels

    def get_loss(self, model, features, labels):
        """
        Computes the loss for given model on given features and labels
        """
        logits = model(features)
        loss = CrossEntropyLoss(logits, labels)

        return loss

    def compute_gradients(self, model, features, labels):
        """
        Computes gradients given the features and labels using the loss
        """
        split_features = self.attack_utils.split(features)
        split_labels = self.attack_utils.split(labels)
        gradient_arr = []
        for (feature, label) in zip(split_features, split_labels):
            with tf.GradientTape() as tape:
                logits = model(feature)
                loss = CrossEntropyLoss(logits, label)
            targetvars = model.variables
            grads = tape.gradient(loss, targetvars)
            # Add gradient wrt crossentropy loss to gradient_arr
            gradient_arr.append(grads)

        return gradient_arr

    def get_gradients(self, model, features, labels):
        """
        Retrieves the gradients for each example.
        """
        gradient_arr = self.compute_gradients(model, features, labels)
        batch_gradients = []
        for grads in gradient_arr:
            # gradient_arr is a list of size of number of layers having trainable parameters
            gradients_per_example = []
            for g in self.gradients_to_exploit:
                g = (g-1)*2
                shape = grads[g].shape
                reshaped = (int(shape[0]), int(shape[1]), 1)
                toappend = tf.reshape(grads[g], reshaped)
                gradients_per_example.append(toappend.numpy())
            batch_gradients.append(gradients_per_example)

        # Adding the gradient matrices fo batches
        batch_gradients = np.asarray(batch_gradients)
        splitted = np.hsplit(batch_gradients, batch_gradients.shape[1])
        for s in splitted:
            array = []
            for i in range(len(s)):
                array.append(s[i][0])
            array = np.asarray(array)

            self.inputArray.append(array)

    def get_gradient_norms(self, model, features, labels):
        """
        Retrieves the gradients for each example
        """
        gradient_arr = self.compute_gradients(model, features, labels)
        batch_gradients = []
        for grads in gradient_arr:
            batch_gradients.append(np.linalg.norm(grads[-1]))
        return batch_gradients

    def forward_pass(self, model, features, labels):
        """
        Computes and collects necessary inputs for attack model
        """
        # container to extract and collect inputs from target model
        self.inputArray = []

        # Getting the intermediate layer computations
        if self.layers_to_exploit and len(self.layers_to_exploit):
            self.get_layer_outputs(model, features)

        # Getting the one-hot-encoded labels
        if self.exploit_label:
            ohelabels = self.get_labels(labels)
            self.inputArray.append(ohelabels)

        # Getting the loss value
        if self.exploit_loss:
            loss = self.get_loss(model, features, labels)
            loss = tf.reshape(loss, (len(loss.numpy()), 1))
            self.inputArray.append(loss)

        # Getting the gradients
        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            self.get_gradients(model, features, labels)

        attack_outputs = self.attackmodel(self.inputArray)
        return attack_outputs

    def attack_accuracy(self, members, nonmembers):
        """
        Computes attack accuracy of the attack model.
        """
        attack_acc = tf.keras.metrics.Accuracy(
            'attack_acc', dtype=tf.float32)
        model = self.target_train_model

        for (membatch, nonmembatch) in zip(members, nonmembers):
            mfeatures, mlabels = membatch
            nmfeatures, nmlabels = nonmembatch

            # Computing the membership probabilities
            mprobs = self.forward_pass(model, mfeatures, mlabels)
            nonmprobs = self.forward_pass(model, nmfeatures, nmlabels)
            probs = tf.concat((mprobs, nonmprobs), 0)

            target_ones = tf.ones(mprobs.shape, dtype=bool)
            target_zeros = tf.zeros(nonmprobs.shape, dtype=bool)
            target = tf.concat((target_ones, target_zeros), 0)

            attack_acc(probs > 0.5, target)

        result = attack_acc.result()
        return result

    def train_attack(self):
        """
        Trains the attack model
        """
        assert self.attackmodel, "Attack model not initialized"
        mtrainset, nmtrainset, nm_features, nm_labels = self.train_datahandler.load_train()
        model = self.target_train_model

        pred = model(nm_features)
        acc = accuracy_score(nm_labels, np.argmax(pred, axis=1))
        print('Target model test accuracy', acc)

        mtestset, nmtestset = self.attack_datahandler.load_test()
        attack_acc = tf.keras.metrics.Accuracy(
            'attack_acc', dtype=tf.float32)

        mtestset = self.attack_utils.intersection(
            mtrainset, mtestset, self.attack_datahandler.batch_size)
        nmtestset = self.attack_utils.intersection(
            nmtrainset, nmtestset, self.attack_datahandler.batch_size)
        # main training procedure begins

        with tf.device(self.device):
            best_accuracy = 0
            for e in range(self.epochs):
                zipped = zip(mtrainset, nmtrainset)
                for((mfeatures, mlabels), (nmfeatures, nmlabels)) in zipped:
                    with tf.GradientTape() as tape:
                        tape.reset()
                        # Getting outputs of forward pass of attack model
                        moutputs = self.forward_pass(model, mfeatures, mlabels)
                        nmoutputs = self.forward_pass(
                            model, nmfeatures, nmlabels)
                        # Computing the true values for loss function according
                        memtrue = tf.ones(moutputs.shape)
                        nonmemtrue = tf.zeros(nmoutputs.shape)
                        target = tf.concat((memtrue, nonmemtrue), 0)
                        probs = tf.concat((moutputs, nmoutputs), 0)
                        attackloss = mse(target, probs)
                    # Computing gradients

                    grads = tape.gradient(attackloss,
                                          self.attackmodel.variables)
                    self.optimizer.apply_gradients(zip(grads,
                                                       self.attackmodel.variables))

                # Calculating Attack accuracy
                attack_acc(probs > 0.5, target)

                attack_accuracy = self.attack_accuracy(mtestset, nmtestset)
                if attack_accuracy > best_accuracy:
                    best_accuracy = attack_accuracy

                with self.summary_writer.as_default(), tf.name_scope(self.model_name):
                    tf.summary.scalar('loss', np.average(attackloss), step=e+1)
                    tf.summary.scalar('accuracy', attack_accuracy, step=e+1)

                print("Epoch {} over :"
                      "Attack test accuracy: {}, Best accuracy : {}"
                      .format(e, attack_accuracy, best_accuracy))

                self.logger.info("Epoch {} over,"
                                 "Attack loss: {},"
                                 "Attack accuracy: {}"
                                 .format(e, attackloss, attack_accuracy))
        # main training procedure ends

        data = None
        if os.path.isfile('logs/attack/results') and os.stat("logs/attack/results").st_size > 0:
            with open('logs/attack/results', 'r+') as json_file:
                data = json.load(json_file)
                if data:
                    data = data['result']
                else:
                    data = []
        if not data:
            data = []
        data.append(
            {self.model_name: {'target_acc': float(acc), 'attack_acc': float(best_accuracy.numpy())}})
        with open('logs/attack/results', 'w+') as json_file:
            json.dump({'result': data}, json_file)

        # logging best attack accuracy
        self.logger.info("Best attack accuracy %.2f%%\n\n",
                         100 * best_accuracy)

    def test_attack(self):
        '''
        Test the attack model on dataset and save plots for visualization.
        '''
        mtrainset, nmtrainset, _, _ = self.attack_datahandler.load_vis(2048)
        model = self.target_attack_model
        mpreds = []
        mlab = []
        nmpreds = []
        nmlab = []
        mfeat = []
        nmfeat = []
        mtrue = []
        nmtrue = []

        mgradnorm, nmgradnorm = [], []
        path = 'logs/plots'
        if not os.path.exists(path):
            os.makedirs(path)
        with tf.device(self.device):
            for(mfeatures, mlabels) in mtrainset:
                # Getting outputs of forward pass of attack model
                moutputs = self.forward_pass(model, mfeatures, mlabels)
                mgradientnorm = self.get_gradient_norms(
                    model, mfeatures, mlabels)
                # Computing the true values for loss function according
                mpreds.extend(moutputs.numpy())
                mlab.extend(mlabels)
                mfeat.extend(mfeatures)
                mgradnorm.extend(mgradientnorm)
                memtrue = np.ones(moutputs.shape)
                mtrue.extend(memtrue)

            for(nmfeatures, nmlabels) in nmtrainset:
                # Getting outputs of forward pass of attack model
                nmoutputs = self.forward_pass(
                    model, nmfeatures, nmlabels)
                nmgradientnorm = self.get_gradient_norms(
                    model, nmfeatures, nmlabels)
                # Computing the true values for loss function according
                nmpreds.extend(nmoutputs.numpy())
                nmlab.extend(nmlabels)
                nmfeat.extend(nmfeatures)
                nmgradnorm.extend(nmgradientnorm)
                nonmemtrue = np.zeros(nmoutputs.shape)
                nmtrue.extend(nonmemtrue)

            target = tf.concat((mtrue, nmtrue), 0)
            probs = tf.concat((mpreds, nmpreds), 0)

        font = {
            'weight': 'bold',
            'size': 10}

        matplotlib.rc('font', **font)
        unique_mem_lab = sorted(np.unique(mlab))
        unique_nmem_lab = sorted(np.unique(nmlab))

        # Creates a histogram for Membership Probability
        fig = plt.figure(1)
        plt.hist(np.array(mpreds).flatten(), color='xkcd:blue', alpha=0.7, bins=20,
                 histtype='bar', range=(0, 1), weights=(np.ones_like(mpreds) / len(mpreds)), label='Training Data (Members)')
        plt.hist(np.array(nmpreds).flatten(), color='xkcd:light blue', alpha=0.7, bins=20,
                 histtype='bar', range=(0, 1), weights=(np.ones_like(nmpreds) / len(nmpreds)), label='Population Data (Non-members)')
        plt.xlabel('Membership Probability')
        plt.ylabel('Fraction')
        plt.title('Privacy Risk')
        plt.legend(loc='upper left')
        plt.savefig('logs/plots/privacy_risk.png')
        plt.close()

        # Creates ROC curve for membership inference attack
        fpr, tpr, _ = roc_curve(target, probs)
        roc_auc = auc(fpr, tpr)
        plt.title('ROC of Membership Inference Attack')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('logs/plots/roc.png')
        plt.close()

        # Creates plot for gradient norm per label
        xs = []
        ys = []
        for lab in unique_mem_lab:
            gradnorm = []
            for l, p in zip(mlab, mgradientnorm):
                if l == lab:
                    gradnorm.append(p)
            xs.append(lab)
            ys.append(np.mean(gradnorm))

        plt.plot(xs, ys, 'g.', label='Training Data (Members)')

        xs = []
        ys = []
        for lab in unique_nmem_lab:
            gradnorm = []
            for l, p in zip(nmlab, nmgradientnorm):
                if l == lab:
                    gradnorm.append(p)
            xs.append(lab)
            ys.append(np.mean(gradnorm))
        plt.plot(xs, ys, 'r.', label='Population Data (Non-Members)')
        plt.title('Average Gradient Norms per Label')
        plt.xlabel('Label')
        plt.ylabel('Average Gradient Norm')
        plt.legend(loc="upper left")
        plt.savefig('logs/plots/gradient_norm.png')
        plt.close()

        # Collect data and creates histogram for each label separately
        for lab in range(len(unique_mem_lab)):
            labs = []
            for l, p in zip(mlab, mpreds):
                if l == lab:
                    labs.append(p)

            plt.hist(np.array(labs).flatten(), color='xkcd:blue', alpha=0.7, bins=20, label='Training Data (Members)',
                     histtype='bar', range=(0, 1), weights=(np.ones_like(labs) / len(labs)))

            labs = []
            for l, p in zip(nmlab, nmpreds):
                if l == lab:
                    labs.append(p)

            plt.hist(np.array(labs).flatten(), color='xkcd:light blue', alpha=0.7, bins=20, label='Population Data (Non-members)',
                     histtype='bar', range=(0, 1), weights=(np.ones_like(labs) / len(labs)))

            plt.legend()
            plt.xlabel('Membership Probability')
            plt.ylabel('Fraction')

            plt.title('Privacy Risk - Class ' + str(lab))
            plt.savefig('logs/plots/privacy_risk_label' + str(lab) + '.png')
            plt.close()

        np.save('logs/member_probs.npy', np.array(mpreds))
        np.save('logs/nonmember_probs.npy', np.array(nmpreds))

        compare_models()
