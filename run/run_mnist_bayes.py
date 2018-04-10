# Running script for model_mnist_bayes
from model.model_mnist_bayes import ModelMnistBayes
from run.utils import load_mnist

if __name__ == "__main__":
    dataset_train, dataset_validation, dataset_test = load_mnist(batch_size=128)
    model = ModelMnistBayes(
        config={
            "n_layers": 1,
            "n_hidden_units": 800,
            "output_path": "test",
            "lr": 1e-5,
            # "adam_epsilon": 0.001,
            "dropout": 1.0,
        },
    ).build().initialize()

    # model = model.train((dataset_train, dataset_validation), n_epoch=1000)
