# Running script for model_mnist
import tensorflow as tf
import numpy as np
import shutil
import os
import dill

from model.model_mnist_sgld import ModelMnistSGLD
from run.utils import load_mnist


if __name__ == "__main__":
    batch_size = 100
    n_epoch = 100
    output_path = "pSGLD/400/SGLD/run1"

    # delete path
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # no validation set, since we are not (heavily) tuning hyper-parameters
    dataset_train, dataset_validation, dataset_test, metadata = load_mnist(
        batch_size=batch_size,
        validation_frac=0.0,
    )
    model = ModelMnistSGLD(
        config={
            # IO
            "output_path": output_path,
            # NN
            "n_layers": 2,
            "n_hidden_units": 400,
            # SGLD
            "prior_log_sigma": 0.,
            "lr": 5e-1,
            "lr_decay_block": 20,
            "burnin": 300,
            "prenoise": 0,
            "thinning": 100,
            "n_sample": 10,
        },
    ).build().initialize()

    # monitor test data result directly
    model.train(
        (dataset_train, dataset_test),
        n_epoch=n_epoch,
        train_size=metadata["train_size"],
        validation_size=metadata["test_size"]
    )

    print "===> TEST RESULT <==="
    # from last sample
    test_result = model.evaluate_standalone(dataset_test, eval_size=metadata["test_size"])
    print "From last sample:", test_result
    dill.dump(test_result, open(output_path+"/test_result.dill", "w"))
