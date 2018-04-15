# Running script for model_mnist_bayes
from model.model_mnist_bayes import ModelMnistBayes
from run.utils import load_mnist
import time

if __name__ == "__main__":
    # setting
    batch_size = 100
    config_name_in = None
    config_name_out = "default"

    # prepare data
    dataset_train, dataset_validation, dataset_test, metadata = load_mnist(batch_size=batch_size)

    model = ModelMnistBayes(
        config={
            # IO
            "output_path": "result/{:s}/".format(config_name_out),  # path to dump all outputs (logs, models, etc.)
            "model_path": "result/{:s}/models/".format(config_name_in),  # path to load cached model
            "model_save_freq": 50,
            # network
            "n_layers": 2,
            "n_hidden_units": 400,
            "lr": 0.01,
            # prior
            "prior_ratio": 0.25,
            "prior_log_sigma1": 0.,
            "prior_log_sigma2": -6.,
            # sampling
            "n_sample": 1,
        },
    ).build().initialize()

    model = model.train(
        (dataset_train, dataset_validation),
        n_epoch=2000,
        n_batch_train=metadata["train_size"] / batch_size,
        n_batch_validation=metadata["validation_size"] / batch_size,
    )
