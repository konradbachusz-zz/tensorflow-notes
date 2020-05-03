from __future__ import absolute_import, division, print_function

import tensorflow as tf

x = tf.constant([100, 200, 300])
y = tf.constant([1, 2, 3])
z = tf.placeholder(tf.int32)

final_div = tf.div(x, y)

final_add = tf.add(final_div, z)

with tf.Session() as sess:

	print ("final_div: ", sess.run(final_div))
	print ("final_add: ", sess.run(final_add, {z: [10, 100, 1000]}))

	writer = tf.summary.FileWriter('./m3_demo1', sess.graph)

	writer.close()
