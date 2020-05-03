from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

print(tf.__version__)
print(np.__version__)


from tensorflow.examples.tutorials.mnist import input_data

# Store the MNIST data in mnist_data/
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

def multilayer_dnn(X):
    fc1 = tf.layers.dense(X, 256, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)
    out = tf.layers.dense(fc2, 10, activation=None)
    return out, fc1, fc2


X = tf.placeholder(tf.float32, shape=[None, 784])

y = tf.placeholder(tf.int32, shape=[None, 10])


# ### The final output layer with softmax activation
#
# Do not apply the softmax activation to this layer.
# The *tf.nn.sparse_softmax_cross_entropy_with_logits* will apply the
# softmax activation as well as calculate the cross-entropy as our cost function

logits, fc1, fc2 = multilayer_dnn(X)


xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                   labels=y)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(0.001)
training_op = optimizer.minimize(loss)


# ### Check correctness and accuracy of the prediction
#
# * Check whether the highest probability output in logits is equal to the y-label
# * Check the accuracy across all predictions (How many predictions did we get right?)

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()


n_epochs = 5
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):

        num_iterations = mnist.train.num_examples // batch_size

        for iteration in range(num_iterations):

            X_batch, y_batch = mnist.train.next_batch(batch_size)

            _, loss_eval, fc1_eval, fc2_eval, logits_eval = \
                sess.run([training_op, loss, fc1, fc2, logits],
                         feed_dict={X: X_batch, y: y_batch})
            if iteration == num_iterations - 1:
                print("Layer            :  Mean     Standard Deviation")
                print("Fully connected 1: ", fc1_eval.mean(), fc1_eval.std())
                print("Fully connected 2: ", fc2_eval.mean(), fc2_eval.std())


        acc_train = accuracy.eval(
            feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(
            feed_dict={X: mnist.test.images, y: mnist.test.labels})

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        print()
