# Original code at:
# https://www.tensorflow.org/programmers_guide/debugger#debugging_tf-learn_estimators_and_experiments

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import experiment
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python import debug as tf_debug


# URLs to download data sets from, if necessary.
IRIS_TRAINING_DATA_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/monitors/iris_training.csv"
IRIS_TEST_DATA_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/monitors/iris_test.csv"

TRAIN_FILE = 'iris_training.csv'
TEST_FILE = 'iris_test.csv'


def maybe_download_data():
  if not os.path.isfile(TRAIN_FILE):
    train_file = open(TRAIN_FILE, "wt")
    urllib.request.urlretrieve(IRIS_TRAINING_DATA_URL, train_file.name)
    train_file.close()

    print("Training data are downloaded to %s" % train_file.name)

  if not os.path.isfile(TEST_FILE):
    test_file = open(TEST_FILE, "wt")
    urllib.request.urlretrieve(IRIS_TEST_DATA_URL, test_file.name)
    test_file.close()

    print("Test data are downloaded to %s" % test_file.name)


_IRIS_INPUT_DIM = 4

def iris_input_fn():
  iris = base.load_iris()

  features = tf.reshape(tf.constant(iris.data), [-1, _IRIS_INPUT_DIM])
  labels = tf.reshape(tf.constant(iris.target), [-1])

  return features, labels


def main(_):
    maybe_download_data()
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=TRAIN_FILE,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=TEST_FILE,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3)

    hooks = None
    if FLAGS.debug:
        debug_hook = tf_debug.LocalCLIDebugHook()
        hooks = [debug_hook]

    if not FLAGS.use_experiment:
      # Fit model.
      classifier.fit(x=training_set.data,
                     y=training_set.target,
                     steps=20,
                     monitors=hooks)

      # Evaluate accuracy.
      accuracy_score = classifier.evaluate(x=test_set.data,
                                           y=test_set.target,
                                           hooks=hooks)["accuracy"]
    else:
      ex = experiment.Experiment(classifier,
                                 train_input_fn=iris_input_fn,
                                 eval_input_fn=iris_input_fn,
                                 train_steps=20,
                                 eval_delay_secs=0,
                                 eval_steps=1,
                                 train_monitors=hooks,
                                 eval_hooks=hooks)
      ex.train()
      accuracy_score = ex.evaluate()["accuracy"]

    print("Accuracy = %f" % (accuracy_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--use_experiment",
        type="bool",
        nargs="?",
        const=True,
        default=False,
        help="Use tf.contrib.learn Experiment to run training and evaluation")
    parser.add_argument(
        "--debug",
        type="bool",
        nargs="?",
        const=True,
        default=False,
        help="Use debugger to track down bad values during training. ")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
