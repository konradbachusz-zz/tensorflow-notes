from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import debug as tf_debug

print(tf.__version__)

# Model parameters
# W = tf.Variable([.3], dtype=tf.float32)
# b = tf.Variable([-.3], dtype=tf.float32)

W = tf.Variable([.3], dtype=tf.float32, name="W")
b = tf.Variable([-.3], dtype=tf.float32, name="b")

# Model input and output
# x = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, name="x")

linear_model = W * x + b

# y = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32, name = "y")

# loss
# loss = tf.reduce_sum(tf.square(linear_model - y))
loss = tf.reduce_sum(tf.square(linear_model - y), name="loss")

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()


sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session())

sess.run(init)

for i in range(1000):
	sess.run(train, {x: x_train, y: y_train})

	# evaluate training accuracy
	curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})

	print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
