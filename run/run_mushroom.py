# Running script for model_mushroom
import tensorflow as tf
import numpy as np

from model.model_mushroom import ModelMushroom
from run.utils import load_mushroom


if __name__ == "__main__":
    dataset_train, dataset_validation, dataset_test, metadata = load_mushroom(
        data_path="../rl_env/data/mushroom/data.parq",
        batch_size=128
    )

    model = ModelMushroom(
        config={
            "n_layers": 2,
            "n_hidden_units": 100,
            "output_path": "test",
            "lr": 0.001,
            "dropout": 0.5,
            "model_save_freq": 50,
        },
    ).build().initialize()

    model = model.train((dataset_train, dataset_validation), n_epoch=500)
