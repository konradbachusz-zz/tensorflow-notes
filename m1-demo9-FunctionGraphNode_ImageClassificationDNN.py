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

    def _debug_print_func(fc1_val, fc2_val):
        print('FC1 : {}, FC2 : {}'.format(fc1_val.shape, fc2_val.shape))

        return False

    # Step 1
    # debug_print_op = tf.py_func(_debug_print_func, [fc1, fc2], [tf.bool])

    # with tf.control_dependencies(debug_print_op):
    #     out = tf.layers.dense(fc2, 10, activation=None)


    out = tf.layers.dense(fc2, 10, activation=None)

    return out


X = tf.placeholder(tf.float32, shape=[None, 784])

y = tf.placeholder(tf.int32, shape=[None, 10])


# ### The final output layer with softmax activation
#
# Do not apply the softmax activation to this layer. The *tf.nn.sparse_softmax_cross_entropy_with_logits* will apply the softmax activation as well as calculate the cross-entropy as our cost function

logits = multilayer_dnn(X)


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

# Step 2
def _debug_func(accuracy_val):
    if (accuracy_val >= 0.98).any():
       import ipdb; ipdb.set_trace()
    return False

debug_op = tf.py_func(_debug_func, [accuracy], [tf.bool])

with tf.control_dependencies(debug_op):
    accuracy = tf.identity(accuracy, name='accuracy')


init = tf.global_variables_initializer()


n_epochs = 5
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):

        num_iterations = mnist.train.num_examples // batch_size

        for iteration in range(num_iterations):

            X_batch, y_batch = mnist.train.next_batch(batch_size)

            # Calculate and print the accuracy for every training batch
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        print()
