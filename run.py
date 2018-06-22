#coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from AutoEncoder import autoencoder
from tensorflow.examples.tutorials.mnist import input_data

def run():

    # setup mode
    with tf.Session() as sess:
        ae = autoencoder(sess, 'AE', nodes=[784, 128])
        
        # train model
        ae.train(x_train, epochs=training_epochs, batch_size=128)

        # predict
        x_predict = ae.predict(x_test)
        
    return x_predict

def image_show(x, y):

    plt.figure()
    plt.subplot(1, 2, 1), plt.imshow(x.reshape(28, 28))
    plt.subplot(1, 2, 2), plt.imshow(y.reshape(28, 28))
    plt.show()

if __name__ == '__main__':

    # prepare data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_train = mnist.train
    x_test, _ = mnist.test.next_batch(1) 

    # hyper parameters
    training_epochs = 20
    batch_size = 128

    # run model
    x_predict = run()

    # plot image
    image_show(x_test, x_predict)


