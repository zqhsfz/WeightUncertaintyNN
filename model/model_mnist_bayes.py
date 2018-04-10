# Bayes-by-propogation model
import tensorflow as tf

from model.model_mnist import ModelMnist


class ModelMnistBayes(ModelMnist):
    @staticmethod
    def _add_weight(scope, input_dim, output_dim, weight_name):
        if input_dim <= 0:
            tensor_shape = [output_dim]
        else:
            tensor_shape = [input_dim, output_dim]

        with tf.variable_scope(scope):
            mu = tf.get_variable(
                name="mu",
                shape=tensor_shape,
                dtype=tf.float32,
                trainable=True,   # TODO: not totally sure here. Same on rho
            )
            rho = tf.get_variable(
                name="rho",
                shape=tensor_shape,
                dtype=tf.float32,
                trainable=True,
            )

            epsilon = tf.random_normal(
                shape=tensor_shape,
                name="epsilon",
            )

            w = tf.add(mu, tf.log(1 + tf.exp(rho)) * epsilon, name=weight_name)

        return w

    def _add_classifier(self):
        with tf.variable_scope("classifier"):
            net = tf.layers.flatten(self._image_placeholder, name="layer_input") / 128.0

            for i_layer in range(self.get_config("n_layers")):
                with tf.variable_scope("layer_{:d}".format(i_layer)):
                    # determine input/output dimension
                    if i_layer == 0:
                        input_dim = self._image_size * self._image_size
                    else:
                        input_dim = self.get_config("n_hidden_units")
                    output_dim = self.get_config("n_hidden_units")

                    # add kernel
                    w = self._add_weight("kernel", input_dim, output_dim, "w")

                    # add bias
                    b = self._add_weight("bias", 0, output_dim, "b")

                    # form output
                    with tf.variable_scope("output"):
                        output = tf.add(tf.matmul(net, w), b, name="linear_output")
                        net = tf.nn.relu(output, name="activation")

            with tf.variable_scope("layer_output"):
                # determine input/output dimension
                input_dim = self.get_config("n_hidden_units")
                output_dim = self._n_class

                # add kernel
                w = self._add_weight("kernel", input_dim, output_dim, "w")

                # add bias
                b = self._add_weight("bias", 0, output_dim, "b")

                # form output
                with tf.variable_scope("output"):
                    net = tf.add(tf.matmul(net, w), b, name="linear_output")

            self._logits = net
