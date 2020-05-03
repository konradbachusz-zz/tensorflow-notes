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
r3 = tf.square(r2)

with tf.Session() as sess:
    h = sess.partial_run_setup([r1, r2, r3], [a, b, c])

    r1_eval = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})

    r2_eval = sess.partial_run(h, r2, feed_dict={c: r1_eval})

    r3_eval = sess.partial_run(h, [r3])
    
    print("r1: %s, r2: %s, r3:%s" % (r1_eval, r2_eval, r3_eval))
