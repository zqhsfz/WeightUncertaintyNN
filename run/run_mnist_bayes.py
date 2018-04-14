# Running script for model_mnist_bayes
from model.model_mnist_bayes import ModelMnistBayes
from run.utils import load_mnist
import time

if __name__ == "__main__":
    batch_size = 100
    dataset_train, dataset_validation, dataset_test, metadata = load_mnist(batch_size=batch_size)
    model = ModelMnistBayes(
        config={
            # IO
            "output_path": "test/{:d}/".format(int(time.time())),
            # network
            "n_layers": 1,
            "n_hidden_units": 400,
            "lr": 0.01,
            # prior
            "prior_ratio": 0.5,
            "prior_log_sigma1": 0.,
            "prior_log_sigma2": -6.,
            # sampling
            "n_sample": 1,
        },
    ).build().initialize()

    model = model.train(
        (dataset_train, dataset_validation),
        n_epoch=1000,
        n_batch_train=metadata["train_size"] / batch_size,
        n_batch_validation=metadata["validation_size"] / batch_size,
    )
