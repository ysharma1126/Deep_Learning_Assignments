#!/bin/python

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#font = {'family' : 'Adobe Caslon Pro',
#        'size'   : 10}

#matplotlib.rc('font', **font)

def model_variable(shape, name):
        variable = tf.get_variable(name=name,
                                   dtype=tf.float32,
                                   shape=shape,
                                   initializer=tf.random_uniform_initializer(minval = 0, maxval = 1)
        )
        tf.add_to_collection('model_variables', variable)
        return variable
    
class Model():
    def __init__(self, sess, data, nEpochs, learning_rate, num_functions):
        self.sess = sess
        self.data = data
        self.nEpochs = nEpochs
        self.learning_rate = learning_rate
        self.num_functions = num_functions
        self.build_model()
        
    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[])
        self.y = tf.placeholder(tf.float32, shape=[])

        w = model_variable([self.num_functions], 'w')
        mu = model_variable([self.num_functions], 'mu')
        sig = model_variable([self.num_functions], 'sig')
        b = model_variable([], 'b')
            
        self.yhat =  tf.reduce_sum(w*tf.exp(-(self.x-mu)**2/(sig**2))) + b
        self.loss = 0.5*tf.square(self.y - self.yhat)
        
    def train_init(self):
        model_variables = tf.get_collection('model_variables')            
        self.optim = (
            tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
            .minimize(self.loss, var_list=model_variables)
            )
        self.sess.run(tf.initialize_all_variables())

    def train_iter(self, x, y):
        loss, _ = self.sess.run([self.loss, self.optim],feed_dict={self.x : x, self.y : y})
        print('loss: {}'.format(loss))

    def train(self):
        for _ in range(self.nEpochs):
            for x, y in self.data():
                self.train_iter(x, y)

    def infer(self, x):
        return self.sess.run(self.yhat, feed_dict={self.x : x})

def data():
    num_samp = 50
    sigma = 0.1
    np.random.seed(31415)
    for _ in range(num_samp):
        x = np.random.uniform()
        y = np.sin(2*np.pi*x) + np.random.normal()*sigma
        yield x, y

sess = tf.Session()
model = Model(sess, data, nEpochs=100, learning_rate=1e-2, num_functions=5)
model.train_init()
model.train()

collection = sess.run(tf.get_collection('model_variables'))
print(collection)

w = collection[0]
mu = collection[1]  
sig = collection[2]
b = collection[3]


# generate manifold and plot

x= np.linspace(-0.1, 1.1, 1000)

y = []
for a in x:
    y.append(model.infer(a))
y = np.array(y)

examples, targets = zip(*list(data()))

plt.subplot(1,2,1)
plt.plot(x, y, '-', x, np.sin(2*np.pi*x), '-', np.array(examples), np.array(targets), 'o')
plt.xlim([-0.1, 1.1])
plt.ylim([-1.5, 1.5])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fit')

plt.subplot(1,2,2)
for mean,stdev in zip(mu,sig):
    plt.plot(x, np.exp(-(x-mean)**2/(stdev**2)), '-')
plt.xlim([-0.1, 1.1])
plt.ylim([0, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bases for Fit')

plt.tight_layout()
plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')
