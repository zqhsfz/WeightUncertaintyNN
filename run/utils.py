# Utilities for running scripts
import tensorflow as tf
import numpy as np


def load_mnist(batch_size):
    # load original data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # get validation set split out of training set
    index_shuffle_array = np.random.permutation(x_train.shape[0])

    validation_split = int(x_train.shape[0] * 0.85)
    index_train_array = index_shuffle_array[:validation_split]
    index_validation_array = index_shuffle_array[validation_split:]

    x_validation, y_validation = x_train[index_validation_array], y_train[index_validation_array]
    x_train, y_train = x_train[index_train_array], y_train[index_train_array]

    print "Training set size: {:d}".format(x_train.shape[0])
    print "Validation set size: {:d}".format(x_validation.shape[0])
    print "Testing set size: {:d}".format(x_test.shape[0])

    # prepare dataset
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.shuffle(10000).batch(batch_size)

    dataset_validation = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
    dataset_validation = dataset_validation.batch(batch_size)

    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset_test = dataset_test.batch(batch_size)

    # metadata
    metadata = {
        "train_size": x_train.shape[0],
        "validation_size": x_validation.shape[0],
        "test_size": x_test.shape[0],
    }

    return dataset_train, dataset_validation, dataset_test, metadata
