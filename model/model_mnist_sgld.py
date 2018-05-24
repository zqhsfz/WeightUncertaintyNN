# Baseline using normal DNN with dropout regularization
import tensorflow as tf
from tqdm import tqdm
from collections import deque
import numpy as np

from model.model_mnist import ModelMnist


def extract_from_result_cache(data, key, agg_func):
    return agg_func(map(lambda x: x[key], data))


class ModelMnistSGLD(ModelMnist):
    def train(self, data_train, **options):
        """
        :param data_train: A tuple of training dataset and validation dataset
        :param options: Contains following item
                        1. n_epoch: Number of epochs for training
                        2. train_size: Data size of training data
                        3. validation_size: Data size of validation data
        :return: evaluation cache
        """

        # parse options
        n_epoch = options["n_epoch"]
        assert n_epoch >= 1, "At least one epoch is required for training!"
        train_size = options["train_size"]
        validation_size = options["validation_size"]

        # parse training data
        dataset_train, dataset_validation = data_train

        # prepare iterators
        training_iterator = dataset_train.make_initializable_iterator()
        validation_iterator = dataset_validation.make_initializable_iterator()

        # prepare string handle
        training_handle = self._sess.run(training_iterator.string_handle())
        validation_handle = self._sess.run(validation_iterator.string_handle())

        # training loop
        with tqdm(total=n_epoch) as pbar:
            # learning rate
            lr = self.get_config("lr")
            lr_decay_block = self.get_config("lr_decay_block")

            # burn-in
            # This flag shows whether one can start collecting samples for prediction
            burnin = self.get_config("burnin")
            burnin_over = False

            # add_noise
            # This flag shows whether we can start stochastic Langevin process instead of pure SGD
            prenoise = self.get_config("prenoise")
            prenoise_over = False

            # thinning
            thinning = self.get_config("thinning")

            # number of samples to compute average
            n_sample = self.get_config("n_sample")

            # counter for training loop
            i_train = 0

            # cache for evaluation result
            train_eval_result_cache = deque([], maxlen=n_sample)
            validation_eval_result_cache = deque([], maxlen=n_sample)

            for i_epoch in range(n_epoch):
                # block decay learning rate
                if (i_epoch > 0) and (i_epoch % lr_decay_block == 0):
                    lr /= 2.0

                # training for one epoch
                self._sess.run(training_iterator.initializer)
                while True:
                    try:
                        # check if burn-in is over
                        if (not burnin_over) and (i_train > burnin):
                            burnin_over = True

                        # check if prenoise is over
                        if (not prenoise_over) and (i_train > prenoise):
                            prenoise_over = True

                        # run a traning update
                        _ = self._sess.run(
                            self._train_op,
                            feed_dict={
                                self._handle: training_handle,
                                self._lr_placeholder: lr,
                                self._dropout_placeholder: 1.0,
                                self._prenoise_over_placeholder: prenoise_over,
                                self._data_size_placeholder: train_size,
                            }
                        )

                        # run evaluation with current NN
                        if burnin_over and (i_train % thinning == 0):
                            result_training = self.evaluate((training_iterator, training_handle), eval_size=train_size)
                            result_validation = self.evaluate((validation_iterator, validation_handle), eval_size=validation_size)

                            train_eval_result_cache.append(result_training)
                            validation_eval_result_cache.append(result_validation)

                        # update counter
                        i_train += 1

                    except tf.errors.OutOfRangeError:
                        break

                # update pbar (per epoch)
                pbar.update(1)
                pbar.set_description("Training loss: {:.4f}".format(extract_from_result_cache(train_eval_result_cache, "loss", np.mean)))  # average

                # update tensorboard at the end of each epoch
                summary_op = np.mean
                self._record_summary(
                    i_epoch,
                    {
                        self._tb_training_loss_placeholder: extract_from_result_cache(train_eval_result_cache, "loss", summary_op),
                        self._tb_training_accuracy_placeholder: extract_from_result_cache(train_eval_result_cache, "accuracy", summary_op),
                        self._tb_training_grad_global_norm_placeholder: extract_from_result_cache(train_eval_result_cache, "grad_global_norm", summary_op),

                        self._tb_validation_loss_placeholder: extract_from_result_cache(validation_eval_result_cache, "loss", summary_op),
                        self._tb_validation_accuracy_placeholder: extract_from_result_cache(validation_eval_result_cache, "accuracy", summary_op),
                        self._tb_validation_grad_global_norm_placeholder: extract_from_result_cache(validation_eval_result_cache, "grad_global_norm", summary_op),
                    }
                )

        return train_eval_result_cache, validation_eval_result_cache

    def evaluate(self, data_eval, **options):
        """
        Run evaluation ONCE with CURRENT NN parameters
        Notice that no sampling is done here to reduce complication.

        :param data_eval: Tuple of (iterator, string_handle)
        :param options: List of following items:
                        1. eval_size: Data size of evaluation data
        :return: A dictionary of
                 1. loss
                 2. accuracy
                 3. grad_global_norm
        """

        # option parsing
        eval_size = options["eval_size"]

        # argument parsing
        data_eval_iterator, data_eval_handle = data_eval

        # initialization
        self._sess.run([
            data_eval_iterator.initializer,       # data initialization
            self._metric_accuracy_initializer,    # metric of accuracy initialization
        ])

        # loop through batches
        count = 0                  # counting on data points
        count_batch = 0            # counting on updates (batches)
        loss_all = 0.              # mean of the loss across whole dataset
        grad_global_norm_all = 0.  # mean of the global gradients across all updates (in one epoch)
        while True:
            try:
                batch_size, loss_batch, grad_global_norm_batch, _, loss_likelihood, loss_prior = self._sess.run(
                    [
                        self._batch_size,
                        self._loss,
                        self._grad_global_norm,
                        self._metric_accuracy_update,
                        # DEBUG
                        self._loss_likelihood,
                        self._loss_prior
                    ],
                    feed_dict={
                        self._handle: data_eval_handle,
                        self._dropout_placeholder: 1.0,  # no dropout in evaluation
                        self._data_size_placeholder: eval_size,
                    }
                )

                count += batch_size
                count_batch += 1

                if loss_batch == float('inf'):
                    print "WARNING! Infinite loss encountered!"
                    print loss_likelihood, loss_prior

                loss_all += 1.0 * (loss_batch - loss_all) * batch_size / count
                grad_global_norm_all += 1.0 * (grad_global_norm_batch - grad_global_norm_all) / count_batch
            except tf.errors.OutOfRangeError:
                break

        # extract metrics
        accuracy_all = self._sess.run(self._metric_accuracy)

        return dict(
            loss=loss_all,
            accuracy=accuracy_all,
            grad_global_norm=grad_global_norm_all,
        )

    ####################################################################################################################

    def _add_placeholder(self):
        with tf.variable_scope("placeholders"):
            # Use feedable iterator so that one can feed data in a flexible way
            self._handle = tf.placeholder(tf.string, shape=[], name="handle")
            iterator = tf.data.Iterator.from_string_handle(
                self._handle,
                output_types=(tf.uint8, tf.uint8),
                output_shapes=(tf.TensorShape([None, self._image_size, self._image_size]), tf.TensorShape([None]))
            )
            input_image, input_label = iterator.get_next()

            self._image_placeholder = tf.cast(input_image, dtype=tf.float32, name="image_placeholder")
            self._label_placeholder = tf.cast(input_label, dtype=tf.int32, name="label_placeholder")

            # other placeholders
            self._lr_placeholder = tf.placeholder(tf.float32, shape=[], name="lr_placeholder")
            self._dropout_placeholder = tf.placeholder(tf.float32, shape=[], name="dropout_placeholder")
            self._prenoise_over_placeholder = tf.placeholder(tf.bool, shape=[], name="prenoise_over_placeholder")
            self._data_size_placeholder = tf.placeholder(tf.float32, shape=[], name="data_size_placeholder")

            # utilities
            self._batch_size = tf.shape(self._label_placeholder)[0]

        return self

    # def _add_classifier(self):
    #     with tf.variable_scope("classifier"):
    #         net = tf.layers.flatten(self._image_placeholder, name="layer_input") / 128.0
    #         for i_layer in range(self.get_config("n_layers")):
    #             net = tf.layers.dense(
    #                 net,
    #                 units=self.get_config("n_hidden_units"),
    #                 activation=tf.nn.relu,
    #                 kernel_initializer=tf.random_normal_initializer(),
    #                 bias_initializer=tf.random_normal_initializer(),
    #                 name="layer_{:d}".format(i_layer)
    #             )
    #             net = tf.nn.dropout(net, keep_prob=self._dropout_placeholder, name="layer_{:d}_dropout".format(i_layer))
    #         net = tf.layers.dense(
    #             net,
    #             units=self._n_class,
    #             activation=None,
    #             kernel_initializer=tf.random_normal_initializer(),
    #             bias_initializer=tf.random_normal_initializer(),
    #             name="layer_output"
    #         )
    #
    #         self._logits = net
    #
    #     return self

    def _add_loss(self):
        prior_log_sigma = self.get_config("prior_log_sigma")

        with tf.variable_scope("loss"):
            # log-likelihood part
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._label_placeholder,
                logits=self._logits,
                name="cross_entropy"
            )
            self._loss_likelihood = tf.reduce_mean(ce, name="loss_likelihood")

            # prior part

            loss_prior = tf.add_n([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()])
            loss_prior = tf.divide(loss_prior, tf.exp(2 * prior_log_sigma))
            self._loss_prior = tf.divide(loss_prior, self._data_size_placeholder, name="loss_prior")

            # adds up
            # self._loss = self._loss_likelihood + self._loss_prior
            self._loss = self._loss_likelihood  # no prior

        return self

    def _add_train_op(self):
        with tf.variable_scope("optimizer"):
            self._global_step = tf.get_variable(
                "train_step",
                shape=(),
                dtype=tf.int32,
                initializer=tf.initializers.zeros(dtype=tf.int32),
                trainable=False
            )

            # obtain gradients
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr_placeholder)
            list_grad_var = optimizer.compute_gradients(self._loss)

            # clipping
            list_grad_var_clipped = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in list_grad_var]

            # inject noise if enabled
            stddev = tf.where(self._prenoise_over_placeholder, tf.rsqrt(0.5 * self._lr_placeholder), tf.zeros([]))
            # list_grad_var_injected = \
            #     [(0.5 * self._data_size_placeholder * grad + stddev * tf.random_normal(var.get_shape()), var)
            #      for grad, var in list_grad_var_clipped]
            list_grad_var_injected = \
                [(grad + stddev * tf.random_normal(var.get_shape()) / self._data_size_placeholder, var)
                 for grad, var in list_grad_var_clipped]

            # apply gradients
            self._train_op = optimizer.apply_gradients(list_grad_var_injected, global_step=self._global_step, name="train_op")

            # get gradient norm
            self._grad_global_norm = tf.global_norm(zip(*list_grad_var_clipped)[0], name="grad_global_norm")

        return self
