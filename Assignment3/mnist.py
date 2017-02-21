#!/bin/python

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  variable = tf.Variable(initial)
  tf.add_to_collection('model_variables', variable)
  tf.add_to_collection('l2', tf.reduce_sum(tf.pow(variable,2)))
  return variable

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  variable = tf.Variable(initial)
  tf.add_to_collection('model_variables', variable)
  return variable

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    
class Model():
    def __init__(self, sess, data, nEpochs, learning_rate, lambduh):
        self.sess = sess
        self.data = data
        self.nEpochs = nEpochs
        self.learning_rate = learning_rate
        self.lambduh = lambduh
        self.build_model()
        
    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        self.keep_prob = tf.placeholder(tf.float32)

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        self.x_image = tf.reshape(self.x, [-1,28,28,1])

        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, W_conv1) + b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, W_conv2) + b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, W_fc1) + b_fc1)

        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.y_conv = tf.matmul(self.h_fc1_drop, W_fc2)+b_fc2

        self.logloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv, labels=self.y))
        self.l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = self.logloss + self.lambduh*self.l2_penalty

    def train_init(self):
        model_variables = tf.get_collection('model_variables')            
        self.optim = (
             tf.train.AdamOptimizer(learning_rate = self.learning_rate)
                .minimize(self.loss, var_list=model_variables)
            )
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.sess.run(tf.global_variables_initializer())
    def train(self):
        for i in range(self.nEpochs):
            batch = self.data.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = self.accuracy.eval(feed_dict= {self.x:batch[0], self.y: batch[1], self.keep_prob: 1.0}, session= self.sess)
                print("step %d, training accuracy %g"%(i, train_accuracy))
            self.optim.run(feed_dict= {self.x:batch[0], self.y: batch[1], self.keep_prob: 0.5}, session= self.sess)

    def infer(self):
        wrong = 0
        #for i, l in zip(self.data.validation.images, self.data.validation.labels):
        for i, l in zip(self.data.test.images, self.data.test.labels):
            i = np.reshape(i, (1,784))
            l = np.reshape(l, (1,10))
            if (self.accuracy.eval(feed_dict={self.x: i, self.y: l, self.keep_prob: 1.0}, session= self.sess) == 0):
                wrong += 1
        #print("validation accuracy %g"%((5000-wrong)/5000)
        print("test accuracy %g"%((10000-wrong)/10000))


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()
model = Model(sess, mnist, nEpochs=2000, learning_rate=1e-4, lambduh=1e-3)
model.train_init()
model.train()
model.infer()