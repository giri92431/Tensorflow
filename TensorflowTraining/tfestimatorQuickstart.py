from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import urllib

import numpy as np 
import tensorflow as tf


# DataSet
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def main():
	# if train test are not there download them locally 

	if not os.path.exists(IRIS_TRAINING):
		raw = urllib.urlopen(IRIS_TRAINING_URL).read()
		with open(IRIS_TRAINING, "w") as f:
			f.write(raw)

	if not os.path.exists(IRIS_TEST):
		raw = urllib.urlopen(IRIS_TEST_URL).read()
		with open(IRIS_TEST, "w") as f:
			f.write(raw)

	#load Dataset
	traning_set = tf.contrib.learn.datasets.base.load_csv_with_header(
		filename =IRIS_TRAINING,
		target_dtype = np.int,
		features_dtype = np.float32)

	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
		filename = IRIS_TEST,
		target_dtype = np.int,
		features_dtype = np.float32)

	#specify that all features hve real-value data
	feature_columns - [tf.feature_column.nmeric_colums(x,shape[4])]

	# build 3 layer DNN with 10 ,20 , 10 units respectively 
	classifier = tf.estimator.DnnClassifier()
	