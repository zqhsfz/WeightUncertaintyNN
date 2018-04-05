import tensorflow as tf

from model.model_mnist import ModelMnist


class ModelMnistBayes(ModelMnist):

    def _add_classifier(self):
        with tf.variable_scope("classifier"):
            net = tf.layers.flatten(self._image_placeholder, name="layer_input") / 128.0

            for i_layer in range(self.get_config("n_layers")):
                with tf.variable_scope("layer_{:d}".format(i_layer)):
                    # determine input dimension
                    input_dim = self._image_size * self._image_size if i_layer == 0 else self.get_config("n_hidden_units")

                    with tf.variable_scope("w"):
                        mu = tf.get_variable(
                            name="mu",
                            shape=[input_dim, self.get_config("n_hidden_units")],
                            dtype=tf.float32,
                            trainable=True,
                        )
                        rho = tf.get_variable(
                            name="rho",
                            shape=[input_dim, self.get_config("n_hidden_units")],
                            dtype=tf.float32,
                            trainable=True
                        ),

                w = tf.get_variable(name="w_layer_{:d}".format(i_layer), shape=)