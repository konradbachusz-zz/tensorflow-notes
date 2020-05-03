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
r3 = tf.multiply(r1, b)

# Step 1: This will not be executed as it is not part of our computational graph 
#tf.Assert(tf.reduce_all(r2 > 0), [r2], name="assert_r2_positive")

final = tf.square(r2)

# Step 2: Set up the assert to be included in the graph
assert_op = tf.Assert(tf.reduce_all(r2 > 0), [r2], name="assert_r2_positive")

# Step 3: Different kinds of asserts and boolean checks
# Step 4: Add asserts to a collection
assert_op = tf.assert_positive(r2, [r2])
tf.add_to_collection('Asserts', assert_op)

# assert_op = tf.assert_negative(r2, [r2])
# tf.add_to_collection('Asserts', assert_op)

# assert_op = tf.assert_equal(r2, r3)
# tf.add_to_collection('Asserts', assert_op)

assert_op = tf.assert_type(r2, tf.float32)
tf.add_to_collection('Asserts', assert_op)

# assert_op = tf.assert_type(r2, tf.int32)
# tf.add_to_collection('Asserts', assert_op)


with tf.control_dependencies([assert_op]):
	final = tf.identity(final)



with tf.Session() as sess:
	
	# This should pass
	# final_eval = sess.run(final, {a: 1, b: 2, c: 10})

	# print("Final result: ", final_eval)

	# # This should fail
	# final_eval = sess.run(final, {a: 1, b: -22, c: 10})

	# print("Final result: ", final_eval)

	assert_op = tf.group(tf.get_collection('Asserts'))
	
	sess.run([final, assert_op], {a: 1, b: 2, c: 10})



