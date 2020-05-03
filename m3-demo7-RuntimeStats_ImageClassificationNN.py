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

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)

    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)

    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))

    tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, activate=tf.nn.relu):
    """Reusable code for making a simple neural net layer."""

    with tf.name_scope(layer_name):

        with tf.name_scope("weights"):
            weights = tf.Variable(tf.truncated_normal(
                [input_dim, output_dim], stddev=0.1, seed=RAND_SEED), name="W")
            variable_summaries(weights)
        

        with tf.name_scope("biases"):
            biases = tf.Variable(tf.constant(0.1, shape=[output_dim]), name="b")
            variable_summaries(biases)

        linear = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('linear', linear)

        activations = activate(linear)
        tf.summary.histogram('activations', activations)
        
        return activations

# Store the MNIST data in mnist_data/
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784], name="X")

y = tf.placeholder(tf.float32, shape=[None, 10], name="y")

hidden = nn_layer(X, IMAGE_SIZE**2, HIDDEN_SIZE, "hidden")
logits = nn_layer(hidden, HIDDEN_SIZE, NUM_LABELS, "logits", tf.identity)

with tf.name_scope("cross_entropy"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=y)

    with tf.name_scope("total"):
        loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(0.01)
    training_op = optimizer.minimize(loss)

with tf.name_scope("accuracy"):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1), name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

sess = tf.InteractiveSession()

n_epochs = 5
batch_size = 100

sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./m3_demo7', sess.graph)

for epoch in range(n_epochs):

    num_iterations = mnist.train.num_examples // batch_size

    for iteration in range(num_iterations):

        X_batch, y_batch = mnist.train.next_batch(batch_size)

        if iteration % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, summary = sess.run([training_op, merged], 
                                  feed_dict={X: X_batch, y: y_batch},
                                  options=run_options,
                                  run_metadata=run_metadata)
            writer.add_summary(summary, iteration)
            writer.add_run_metadata(run_metadata, 'epoch%d-step-%d' % (epoch, iteration))
        else:
            _, summary = sess.run([training_op, merged],
                                  feed_dict={X: X_batch, y: y_batch})

    acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
    acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})

    print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    print()


writer.close()
sess.close()



