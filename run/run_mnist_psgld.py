# Running script for model_mnist
import tensorflow as tf
import numpy as np
import shutil
import os
import dill

from model.model_mnist_psgld import ModelMnistPSGLD
from run.utils import load_mnist


if __name__ == "__main__":
    batch_size = 100
    n_epoch = 100
    output_path = "pSGLD/1200/pSGLD/run2"

    # delete path
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # no validation set, since we are not (heavily) tuning hyper-parameters
    dataset_train, dataset_validation, dataset_test, metadata = load_mnist(
        batch_size=batch_size,
        validation_frac=0.0,
    )
    model = ModelMnistPSGLD(
        config={
            # IO
            "output_path": output_path,
            # NN
            "n_layers": 2,
            "n_hidden_units": 1200,
            # SGLD
            "prior_log_sigma": 0.,
            "lr": 5e-4,
            "lr_decay_block": 20,
            "burnin": 300,
            "noise_factor": 1.0,
            "prenoise": 30,
            "thinning": 100,
            "momentum_decay": 0.95,
            "momentum_bias": 1e-5
        },
    ).build().initialize()

    # monitor test data result directly
    model.train(
        (dataset_train, dataset_test),
        n_epoch=n_epoch,
        train_size=metadata["train_size"],
    )

