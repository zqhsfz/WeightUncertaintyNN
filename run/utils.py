# Utilities for running scripts
import tensorflow as tf
import numpy as np
import pandas as pd


def load_mnist(batch_size, train_indices=None, validation_indices=None):
    """
    Load MNIST data, split it into train/validation sets, and prepare tf dataset out of it

    :param batch_size: Batch size
    :param train_indices: List of index for training set. None by default, in which case random partition would be
                          generated. Non-None input is mainly for re-using previous partition.
    :param validation_indices: Same as train_indices, but for validation set
    :return: Tuple of (train dataset, validation dataset, test dataset, metadata)
                      metadata is a dictionary containing all auxiliary information
    """

    # input check
    assert (train_indices is None) == (validation_indices is None), \
        "If specified, both training / validation index array must be specified"

    # load original data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # get validation set split out of training set
    if train_indices is None:
        index_shuffle_array = np.random.permutation(x_train.shape[0])

        validation_split = int(x_train.shape[0] * 0.85)
        index_train_array = index_shuffle_array[:validation_split]
        index_validation_array = index_shuffle_array[validation_split:]
    else:
        index_train_array = train_indices
        index_validation_array = validation_indices

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
        "train_indices": index_train_array,
        "validation_indices": index_validation_array,
    }

    return dataset_train, dataset_validation, dataset_test, metadata


def load_mushroom(data_path, batch_size, train_indices=None, validation_indices=None, test_indices=None):
    """
    Load Mushroom data for simple classification task

    Here is how we split: 85% for training, 15% for testing. Among training, 15% for validation, and remaining for
    actual training

    :param data_path: Path to the data parquet file
    :param batch_size: Batch size
    :param train_indices: List of index for training set. None by default, in which case random partition would be
                          generated. Non-None input is mainly for re-using previous partition.
    :param validation_indices: Similar to train_indices
    :param test_indices: Similar to train_indices
    :return: train, validation, test and metadata
    """

    # input check
    assert (train_indices is None) == (validation_indices is None) == (test_indices is None)

    # load original data
    data = pd.read_parquet(data_path)

    # train/validation/test splitting
    if train_indices is None:
        index_array_shuffled = np.random.permutation(data.shape[0])

        test_split = int(data.shape[0] * 0.15)
        train_split = test_split + int(data.shape[0] * 0.85 * 0.85)

        index_array_test = index_array_shuffled[:test_split]
        index_array_train = index_array_shuffled[test_split:train_split]
        index_array_validation = index_array_shuffled[train_split:]
    else:
        index_array_train = train_indices
        index_array_validation = validation_indices
        index_array_test = test_indices

    data_train = data.iloc[index_array_train]
    data_validation = data.iloc[index_array_validation]
    data_test = data.iloc[index_array_test]

    # convert from pandas to numpy matrix
    def pandas_to_np(df):
        matrix = df.as_matrix().astype(np.float32)

        label = matrix[:, 0].astype(np.int32)
        features = matrix[:, 1:]

        return features, label

    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = \
        pandas_to_np(data_train), pandas_to_np(data_validation), pandas_to_np(data_test)

    print "Training set size: {:d}".format(x_train.shape[0])
    print "Validation set size: {:d}".format(x_validation.shape[0])
    print "Testing set size: {:d}".format(x_test.shape[0])

    # prepare dataset
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.shuffle(1000).batch(batch_size)

    dataset_validation = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
    dataset_validation = dataset_validation.batch(batch_size)

    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset_test = dataset_test.batch(batch_size)

    # metadata
    metadata = {
        "train_size": x_train.shape[0],
        "validation_size": x_validation.shape[0],
        "test_size": x_test.shape[0],
        "train_indices": index_array_train,
        "validation_indices": index_array_validation,
        "test_indices": index_array_test,
    }

    return dataset_train, dataset_validation, dataset_test, metadata
