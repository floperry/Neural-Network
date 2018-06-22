#coding=utf-8

import tensorflow as tf

class autoencoder(object):

    def __init__(self, sess, name, nodes):
        self._sess = sess
        self._name = name
        self._nodes = nodes
        self._build_net()

    @property
    def sess(self):
        return self._sess

    @property
    def name(self):
        return self._name

    @property
    def nodes(self):
        return self._nodes

    def _build_net(self):

        # check whether number of encode layers is valid
        if len(self._nodes) < 2:
            raise ValueError('At least 2 encode layers should be specified.')

        with tf.variable_scope(self._name) as scope:

            # define palceholder
            self.X = tf.placeholder(tf.float32, [None, self._nodes[0]])
            self.learning_rate = tf.placeholder(tf.float32)
            self.keep_prob = tf.placeholder(tf.float32)

            layer = self.X

            # define encode layer
            for i, node in enumerate(self._nodes[:-2]):
                layer = self._dense_layer(shape=self._nodes[i, i+2], inputs=layer, 
                                          layer_name='encoder_' + str(i+1))
                layer = self._activation(layer)
                layer = tf.nn.dropout(layer, keep_prob=self.keep_prob)

            # define last encode layer
            layer = self._dense_layer(shape=self._nodes[len(self._nodes)-2:len(self._nodes)], inputs=layer, 
                                      layer_name='encoder_' + str(len(self._nodes)-1))
            
            # get hidden layer feature
            self.hidden_feature = layer

            layer = self._activation(layer)

            # define decode layer
            for i, node in enumerate(self._nodes[::-1][:-2]):
                layer = self._dense_layer(shape=self._nodes[::-1][i, i+2], inputs=layer,
                                          layer_name='decoder_' + str(i+1))
                layer = self._activation(layer)

            self.outputs = self._dense_layer(shape=self._nodes[::-1][len(self._nodes)-2:len(self._nodes)],
                                             inputs=layer, layer_name='decoder_' + str(len(self._nodes)-1))

        # define loss function and optimizer
        self.loss = self._loss_function(self.outputs, self.X)
        self.optimizer = self._optimizer(self.loss)
    
    # define dense layer
    def _dense_layer(self, shape, inputs, layer_name):
        with tf.variable_scope(layer_name) as scope:
            W = tf.get_variable(layer_name + '_W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([shape[-1]]))
        return tf.add(tf.matmul(inputs, W), b)

    # define activation function
    def _activation(self, inputs):
        return tf.nn.relu(inputs)

    # define loss function
    def _loss_function(self, predict, labels):
        return tf.reduce_mean(tf.square(predict - labels))

    # define optimizer
    def _optimizer(self, loss):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

    # train model
    def train(self, x_data, epochs=100, batch_size=128, learning_rate=0.001, keep_prob=0.7):
        losses = []
        for epoch in range(epochs):
            avg_loss = 0
            total_batch = int(x_data.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, _ = x_data.next_batch(batch_size)
                l, _ = self.sess.run([self.loss, self.optimizer], 
                        feed_dict={self.X: batch_xs, 
                                   self.learning_rate: learning_rate, 
                                   self.keep_prob: keep_prob})
                avg_loss += l / total_batch
            print("Epoch ", "%04d" %(epoch+1), "loss = ", "{:.9f}".format(avg_loss))

            losses.append(avg_loss)
        return losses

    # predict
    def predict(self, x_data, keep_prob=1.0):
        return self.sess.run(self.outputs, feed_dict={self.X: x_data, self.keep_prob: keep_prob})

    # get hidden feature
    def get_feature(self, x_data, keep_prob=1.0):
        return self.sess.run(self.hidden_feature, feed_dict={self.X: x_data, self.keep_prob: keep_prob})

    # save model
    def save(self, path):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, path)

    # load model
    def load(self, path):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, path)


