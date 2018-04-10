# Running script for model_mnist
import tensorflow as tf
import numpy as np

from model.model_mnist import ModelMnist
from run.utils import load_mnist


if __name__ == "__main__":
    dataset_train, dataset_validation, dataset_test = load_mnist(batch_size=128)
    model = ModelMnist(
        config={
            "n_layers": 1,
            "n_hidden_units": 800,
            "output_path": "test",
            "lr": 1e-5,
            # "adam_epsilon": 0.001,
            "dropout": 0.5,
        },
    ).build().initialize()

    model = model.train((dataset_train, dataset_validation), n_epoch=1000)
    # model = model.train((dataset_train, dataset_test), n_epoch=600)
