import tensorflow as tf

from model.model_mnist import ModelMnist


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.shuffle(10000).batch(100)

    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset_test = dataset_test.batch(100)

    return dataset_train, dataset_test


if __name__ == "__main__":
    dataset_train, dataset_test = load_mnist()
    model = ModelMnist(
        config={
            "n_layers": 2,
            "n_hidden_units": 400,
            "lr": 1e-3,
            "output_path": "test",
        },
    ).build().initialize()

    model = model.train((dataset_train, dataset_test), n_epoch=600)
