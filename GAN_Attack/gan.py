from __future__ import print_function, division

from sklearn import datasets
from sklearn.datasets import fetch_openml
from mlxtend.data import loadlocal_mnist

from mlfromscratch.deep_learning.optimizers import Adam
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.deep_learning.layers import Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization
from mlfromscratch.deep_learning import NeuralNetwork

import math
import matplotlib.pyplot as plt
import numpy as np
import progressbar

class GAN():

    def __init__(self):
        self.rows = 28
        self.cols = 28
        self.dim = self.rows * self.cols
        self.latent_dim = 100

        optimizer = Adam(learning_rate = 0.0002, b1=0.5)
        loss_function = CrossEntropy

        # Dicriminator
        self.discriminator = self.build_discriminator(optimizer, loss_function)
        
        # Generator
        self.generator = self.build_generator(optimizer, loss_function)

        # Combined model
        self.combined = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        self.combined.layers.extend(self.generator.layers)
        self.combined.layers.extend(self.discriminator.layers)

        print ()
        self.generator.summary(name="Generator")
        self.discriminator.summary(name="Discriminator")

    def build_generator(self, optimizer, loss_function):
        model = NeuralNetwork(optimizer=optimizer, loss=loss_function)

        model.add(Dense(256, input_shape=(self.latent_dim,)))
        model.add(Activation('leaky_relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(Activation('leaky_relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(Activation('leaky_relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.dim))
        model.add(Activation('tanh'))

        return model

    def build_discriminator(self, optimizer, loss_function):
        model = NeuralNetwork(optimizer=optimizer, loss=loss_function)

        model.add(Dense(512, input_shape=(self.dim,)))
        model.add(Activation('leaky_relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('leaky_relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model

    def train(self, n_epochs, batch_size=128, save_interval=50):
        
        X, y = loadlocal_mnist(images_path='./MNIST/train-images.idx3-ubyte', labels_path='./MNIST/train-labels.idx1-ubyte')

        # Normalize array
        X = (X.astype(np.float32) - 127.5) / 127.5

        half_batch = batch_size // 2

        for epoch in range(n_epochs):

            # train the discriminator

            self.discriminator.set_trainable(True)

            # Select random images from half_batch
            idx = np.random.randint(0, X.shape[0], half_batch)
            imgs = X[idx]

            # Sample noise
            noise = np.random.randint(0, 1, (half_batch, self.latent_dim))

            # Generate half batch images
            gen_imgs = self.generator.predict(noise)

            # Get valid [1,0] and fakes [0,1]
            valid = np.concatenate((np.ones((half_batch, 1)), np.zeros((half_batch, 1))), axis=1)
            fake = np.concatenate((np.zeros((half_batch, 1)), np.ones((half_batch, 1))), axis=1)
            
            # Train the discriminator
            d_loss_real, d_acc_real =self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake, d_acc_fake =self.discriminator.train_on_batch(gen_imgs, fake)            
            d_loss = 0.5 * (d_loss_real + d_loss_fake)            
            d_acc = 0.5 * (d_acc_real + d_acc_fake)


            # Train Generator

            # discriminator shouldn't be trained anymore
            self.discriminator.set_trainable(False)

            # Sample noise
            noise = np.random.randint(0, 1, (batch_size, self.latent_dim))

            # 
            valid = np.concatenate((np.ones((batch_size, 1)),np.zeros((batch_size, 1))), axis=1)

            # Train the generator            
            g_loss, g_acc = self.combined.train_on_batch(noise, valid)

            # Show progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, acc: %.2f%%]"% (epoch, d_loss, 100*d_acc, g_loss, 100*g_acc))

            # Save image if epoch is on the save interval
            if epoch % save_interval == 0:
                self.save_image(epoch)
        

    def save_image(self, epoch):
        r, c = 5, 5 # row and column for grid size
        noise = np.random.normal(0,1, (r*c, self.latent_dim))

        # generate the images, resize and reshape to img size
        gen_imgs = self.generator.predict(noise).reshape((-1, self.rows, self.cols))

        # Rescale the images
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        plt.suptitle("GAN Attack")
        counter = 0
        
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[counter,:,:], cmap='gray')
                axs[i, j].axis('off')
                counter += 1
        
        fig.savefig("minst_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(n_epochs=200000, batch_size=64, save_interval=400)
    
