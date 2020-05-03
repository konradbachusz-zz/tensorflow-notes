from __future__ import absolute_import, division, print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)


IMAGE_SIZE = 28
HIDDEN_SIZE = 500
NUM_LABELS = 10
RAND_SEED = 42

def nn_layer(input_tensor, input_dim, output_dim, activate=tf.nn.relu):
	"""Reusable code for making a simple neural net layer."""

	weights = tf.Variable(tf.truncated_normal(
		[input_dim, output_dim], stddev=0.1, seed=RAND_SEED))
	biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))

	linear = tf.matmul(input_tensor, weights) + biases

	activations = activate(linear)

	return activations

def main(_):

	# Store the MNIST data in mnist_data/
	mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

	X = tf.placeholder(tf.float32, shape=[None, 784])

	y = tf.placeholder(tf.float32, shape=[None, 10])

	hidden = nn_layer(X, IMAGE_SIZE**2, HIDDEN_SIZE)
	logits = nn_layer(hidden, HIDDEN_SIZE, NUM_LABELS, tf.identity)

	y_ = tf.nn.softmax(logits)

	# xentropy = -(y * tf.log(y_))

	xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
	                                                   labels=y)
	loss = tf.reduce_mean(xentropy)
	optimizer = tf.train.AdamOptimizer(0.01)
	training_op = optimizer.minimize(loss)

	correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	sess = tf.InteractiveSession()

	if FLAGS.debug:
	    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

	n_epochs = 10
	batch_size = 100

	sess.run(tf.global_variables_initializer())

	for epoch in range(n_epochs):

	    num_iterations = mnist.train.num_examples // batch_size

	    for iteration in range(num_iterations):

	        X_batch, y_batch = mnist.train.next_batch(batch_size)

	        # Calculate and print the accuracy for every training batch
	        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
	        # sess.run(training_op, feed_dict={X: X_batch})


	    acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
	    acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})

	    print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
	    print()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda v: v.lower() == "true")

	parser.add_argument(
	      "--debug",
	      type="bool",
	      nargs="?",
	      const=True,
	      default=False,
	      help="Use debugger to track down bad values during training. ")

	FLAGS, unparsed = parser.parse_known_args()

	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)








