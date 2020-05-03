# Original code at: https://www.tensorflow.org/programmers_guide/tensorboard_histograms

from __future__ import absolute_import, division, print_function

import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)

# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("./m3_demo5", sess.graph)

summaries = tf.summary.merge_all()

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
	k_val = step/float(N)
	summ = sess.run(summaries, feed_dict={k: k_val})
	writer.add_summary(summ, global_step=step)

writer.close()
sess.close()
