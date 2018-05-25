# Running script for model_mnist
import tensorflow as tf
import numpy as np
import shutil
import os
import dill

from model.model_mnist import ModelMnist
from run.utils import load_mnist


if __name__ == "__main__":
    batch_size = 100
    n_epoch = 100
    output_path = "pSGLD/1200/RMSProp/run1"

    # delete path
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # no validation set, since we are not (heavily) tuning hyper-parameters
    dataset_train, dataset_validation, dataset_test, metadata = load_mnist(
        batch_size=batch_size,
        validation_frac=0.0,
    )
    model = ModelMnist(
        config={
            # IO
            "output_path": output_path,
            "model_save_freq": 100,
            # NN
            "n_layers": 2,
            "n_hidden_units": 1200,
            "optimizer": "rmsprop",
            "lr": 5e-4,
            "lr_decay_block": 20,
            "dropout": 1.0,
        },
    ).build().initialize()

    # monitor test data result directly
    model = model.train((dataset_train, dataset_test), n_epoch=n_epoch)

    print "===> TEST RESULT <==="
    test_result = model.evaluate_standalone(dataset_test)
    print test_result
    dill.dump(test_result, open(output_path+"/test_result.dill", "w"))
