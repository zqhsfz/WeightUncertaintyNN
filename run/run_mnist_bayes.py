# Running script for model_mnist_bayes
from model.model_mnist_bayes import ModelMnistBayes
from run.utils import load_mnist

import time
import os
import numpy as np
import shutil


if __name__ == "__main__":
    # setting
    batch_size = 100
    config_name_in = None
    config_name_out = "2layer_1200units_lr1e-3"
    n_epoch = 1000

    # construct model
    model = ModelMnistBayes(
        config={
            # IO
            "output_path": "result/{:s}/".format(config_name_out),  # path to dump all outputs (logs, models, etc.)
            "model_path": "result/{:s}/models/".format(config_name_in),  # path to load cached model
            "model_save_freq": 100,
            # network
            "n_layers": 2,
            "n_hidden_units": 1200,
            "lr": 1e-4,
            # prior
            "prior_ratio": 0.5,
            "prior_log_sigma1": -0.,
            "prior_log_sigma2": -6.,
            # sampling
            "n_sample": 1,
        },
    ).build().initialize()

    # No need to touch codes below for running purpose
    ####################################################################################################################
    # No need to touch codes below for running purpose

    # load partition indices, if needed
    if config_name_in is None:
        train_indices = None
        validation_indices = None
    else:
        load_partition_indices = np.load("result/{:s}/partition.npz".format(config_name_in))
        train_indices = load_partition_indices["train_indices"]
        validation_indices = load_partition_indices["validation_indices"]

    # prepare data
    dataset_train, dataset_validation, dataset_test, metadata = load_mnist(
        batch_size=batch_size,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )

    # output used partition indices
    np.savez(
        "{:s}/partition.npz".format(model.get_config("output_path")),
        train_indices=metadata["train_indices"],
        validation_indices=metadata["validation_indices"],
    )

    # copy this file to output as well
    shutil.copy("run_mnist_bayes.py", model.get_config("output_path"))

    # training
    model = model.train(
        (dataset_train, dataset_validation),
        n_epoch=n_epoch,
        n_batch_train=metadata["train_size"] / batch_size,
        n_batch_validation=metadata["validation_size"] / batch_size,
    )

    # # evaluation on testing data
    # print model.evaluate_standalone(dataset_test, n_batch_eval=metadata["test_size"] / batch_size)
