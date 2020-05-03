from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import debug as tf_debug

print(tf.__version__)

# Model parameters
W = tf.Variable([-.3], dtype=tf.float32, name="W")
b = tf.Variable([.3], dtype=tf.float32, name="b")

# Model input and output
x = tf.placeholder(tf.float32, name="x")

linear_model = W * x + b

y = tf.placeholder(tf.float32, name = "y")

# loss
loss = tf.reduce_sum(tf.square(linear_model - y), name="loss")

# optimizer
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# optimizer = tf.train.GradientDescentOptimizer(0.001)
optimizer = tf.train.AdamOptimizer(0.01)


train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4, 6, 23, 44, 55, 66]
y_train = [0, -1, -2, -3, -8, -28, -50, -67, -78]

# training loop
init = tf.global_variables_initializer()

sess = tf.Session()

sess = tf_debug.LocalCLIDebugWrapperSession(sess)

sess.run(init)

def zero_filter(_, tensor):
  return tensor == 0.0

sess.add_tensor_filter('threshold_filter', zero_filter)


for i in range(2000):
	sess.run(train, {x: x_train, y: y_train})

	# evaluate training accuracy
	curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})

	print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
