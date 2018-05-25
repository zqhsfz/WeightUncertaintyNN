# NN with pSGLD
import tensorflow as tf
from tqdm import tqdm
from collections import deque
import numpy as np

from model.model_mnist_sgld import ModelMnistSGLD


class ModelMnistPSGLD(ModelMnistSGLD):
    def _add_train_op(self):
        noise_factor = self.get_config("noise_factor")
        momentum_decay = self.get_config("momentum_decay")
        momentum_bias = self.get_config("momentum_bias")

        with tf.variable_scope("optimizer"):
            self._global_step = tf.get_variable(
                "train_step",
                shape=(),
                dtype=tf.int32,
                initializer=tf.initializers.zeros(dtype=tf.int32),
                trainable=False
            )

            # obtain gradients (separately for prior and likelihood)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr_placeholder)
            list_grad_var_likelihood = optimizer.compute_gradients(self._loss_likelihood)
            list_grad_var_prior = optimizer.compute_gradients(self._loss_prior)

            list_grad_likelihood, list_var = zip(*list_grad_var_likelihood)
            list_grad_prior, _ = zip(*list_grad_var_prior)

            assert len(list_grad_likelihood) == len(list_grad_prior) == len(list_var)

            # clipping
            list_grad_likelihood_clipped = map(lambda grad: tf.clip_by_value(grad, -10., 10.), list_grad_likelihood)
            list_grad_prior_clipped = map(lambda grad: tf.clip_by_value(grad, -10., 10.), list_grad_prior)
            list_grad_clipped = [
                grad_likelihood + grad_prior
                for grad_likelihood, grad_prior in zip(list_grad_likelihood_clipped, list_grad_prior_clipped)
            ]

            # initialize internal variables
            self._optimizer_internal_momentum = [
                tf.get_variable(
                    "optimizer_internal_momentum_{:d}".format(index),
                    shape=var.get_shape(),
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                )
                for index, var in enumerate(list_var)
            ]

            # update internal momentum
            mom_update = [
                tf.assign_add(mom, (1. - momentum_decay) * (tf.square(grad) - mom))
                for mom, grad in zip(self._optimizer_internal_momentum, list_grad_likelihood_clipped)
            ]

            precondition = [1.0 / (momentum_bias + tf.sqrt(mom)) for mom in mom_update]

            # inject noise if enabled, along with the precondition
            stddev = tf.where(self._prenoise_over_placeholder, tf.rsqrt(0.5 * self._lr_placeholder), tf.zeros([]))

            list_grad_injected = [
                g * grad + tf.sqrt(g) * stddev * tf.random_normal(grad.get_shape()) / tf.pow(self._data_size_placeholder, noise_factor)
                for grad, g in zip(list_grad_clipped, precondition)
            ]

            list_grad_var_injected = zip(list_grad_injected, list_var)

            # apply gradients
            self._train_op = optimizer.apply_gradients(list_grad_var_injected, global_step=self._global_step, name="train_op")

            # get gradient norm
            self._grad_global_norm = tf.global_norm(list_grad_clipped, name="grad_global_norm")

        return self
