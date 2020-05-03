from __future__ import absolute_import, division, print_function

import tensorflow as tf

print(tf.__version__)


W = tf.constant([10, 100])

x = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

Wx = tf.multiply(W, x)

# y = Wx + b
y = tf.add(Wx, b)

# y_ = x^2 + b
y_pred = x**2 + b

# loss = (y - y_pred)^2
loss = (y - y_pred)**2


with tf.Session() as sess:
	print("Final result: Wx + b = ",
		  sess.run(y, feed_dict={x: [7, 70], b: [1, 2]}))

	print("Intermediate result: Wx = ",
	      sess.run(Wx, feed_dict={x: [3, 33]}))

	print("Intermediate specified: Wx + b = ", \
	      sess.run(y, feed_dict={Wx: [100, 1000], b: [3, 4]}))

	print("Loss  = ", \
		  sess.run(loss, feed_dict={x: [7, 70], b: [1, 2]}))

	print("y_pred  = ", \
	      sess.run(y_pred, feed_dict={x: [7, 70], b: [1, 2]}))

	print("y  = ", \
	      sess.run(y, feed_dict={x: [7, 70], b: [1, 2]}))
