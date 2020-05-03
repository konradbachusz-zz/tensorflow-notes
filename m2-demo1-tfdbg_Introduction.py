from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import debug as tf_debug

print(tf.__version__)

a = tf.placeholder(tf.float32, shape=[])
b = tf.placeholder(tf.float32, shape=[])
c = tf.placeholder(tf.float32, shape=[])
d = tf.placeholder(tf.float32, shape=[])

r1 = tf.add(a, b)
r2 = tf.multiply(r1, c)

r2_5 = tf.Print(r2, [r2], "Result 2: ")

final = tf.square(r2_5)

final_print = tf.Print(final, [final], "Final: ")

sess = tf.Session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)

sess.run(final_print, {a: 1, b: 2, c: 3})

sess.close()
