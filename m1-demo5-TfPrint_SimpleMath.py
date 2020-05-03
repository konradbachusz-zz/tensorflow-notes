from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

print(tf.__version__)
print(np.__version__)

a = tf.placeholder(tf.float32, shape=[])
b = tf.placeholder(tf.float32, shape=[])
c = tf.placeholder(tf.float32, shape=[])

r1 = tf.add(a, b)
r2 = tf.multiply(r1, c)

r2_5 = tf.Print(r2, [r2], "Intermediate Result 2: ")

final = tf.square(r2)
final_print = tf.square(r2_5)


with tf.Session() as sess:

	final_eval = sess.run(final, {a: 1, b: 2, c: 10})
	print("Final result without a print statement: ", final_eval)

	final_print_eval = sess.run(final_print, {a: 1, b: 2, c: 10})
	print("Final result WITH a print statement: ", final_print_eval)
