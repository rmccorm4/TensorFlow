import tensorflow as tf
sess = tf.Session()

import numpy as np

from keras import backend as K
# Use Keras on top of tensorflow session
K.set_session(sess)

# Placeholder for input digits compressed from matrix into vector
img = tf.placeholder(tf.float32, shape=(None, 784)) # 784 = 28*28

from keras.layers import Dense
x = Dense(128, activation='relu')(img) # fully connected layer
x = Dense(128, activation='relu')(x)
# Predictions
predictions = Dense(10, activation='softmax')(x) # Output layer 

# Place holders for labels and loss function
labels = tf.placeholder(tf.float32, shape=(None, 10))

from keras.objectives import categorical_crossentropy
# Loss function
loss = tf.reduce_mean(categorical_crossentropy(labels, predictions))

from tensorflow.examples.tutorials.mnist import input_data
# One hot is a vector of all 0's except for a 1 in the predicted output spot
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# Learning rate = 0.5
# Loss function wants to be minimized to decrease loss and improve accuracy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Training loop
with sess.as_default():
	for i in range(100):
		batch = mnist_data.train.next_batch(50)
		train_step.run(feed_dict={img : batch[0], labels : batch[1]})

# Evaluate model
from keras.metrics import categorical_accuracy as accuracy
acc_value = accuracy(labels, predictions)
with sess.as_default():
	print(acc_value.eval(feed_dict={img : mnist_data.test.images,
									labels : mnist_data.test.labels}))
