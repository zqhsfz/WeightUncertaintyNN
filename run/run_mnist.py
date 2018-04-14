# Running script for model_mnist
import tensorflow as tf
import numpy as np

from model.model_mnist import ModelMnist
from run.utils import load_mnist


if __name__ == "__main__":
    dataset_train, dataset_validation, dataset_test, metadata = load_mnist(batch_size=128)
    model = ModelMnist(
        config={
            "n_layers": 1,
            "n_hidden_units": 800,
            "output_path": "test",
            "lr": 0.01,
            # "adam_epsilon": 0.001,
            "dropout": 1.0,
        },
    ).build().initialize()

    model = model.train((dataset_train, dataset_validation), n_epoch=1000)
    # model = model.train((dataset_train, dataset_test), n_epoch=600)
