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
        :return: Self
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

            # cache for weights
            weights_cache = deque([], maxlen=n_sample)
            weights_latest = None

            for i_epoch in range(n_epoch):
                # block decay learning rate
                if (i_epoch > 0) and (i_epoch % lr_decay_block == 0):
                    lr /= 2.0

                # training for one epoch
                self._sess.run(training_iterator.initializer)
                if weights_latest is not None:
                    self._sess.run(tf.group(*[tf.assign(var, weight) for var, weight in zip(tf.trainable_variables(), weights_latest)]))
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

                        # cache weight
                        if burnin_over and (i_train % thinning == 0):
                            weights_snapshot = self._sess.run(tf.trainable_variables())
                            weights_cache.append(weights_snapshot)

                        weights_latest = self._sess.run(tf.trainable_variables())

                        # update counter
                        i_train += 1

                    except tf.errors.OutOfRangeError:
                        break

                # evaluation at the end of epoch, using ensembles
                train_eval_result = self.evaluate((training_iterator, training_handle), weights_samples=weights_cache)
                validation_eval_result = self.evaluate((validation_iterator, validation_handle), weights_samples=weights_cache)

                # update pbar (per epoch)
                pbar.update(1)
                pbar.set_description("Training score: {:.4f}".format(train_eval_result["accuracy"]))

                # update tensorboard at the end of each epoch
                self._record_summary(
                    i_epoch,
                    {
                        self._tb_training_loss_placeholder: train_eval_result["loss"],
                        self._tb_training_accuracy_placeholder: train_eval_result["accuracy"],
                        self._tb_training_grad_global_norm_placeholder: train_eval_result["grad_global_norm"],

                        self._tb_validation_loss_placeholder: validation_eval_result["loss"],
                        self._tb_validation_accuracy_placeholder: validation_eval_result["accuracy"],
                        self._tb_validation_grad_global_norm_placeholder: validation_eval_result["grad_global_norm"],
                    }
                )

        return self

    def evaluate(self, data_eval, **options):
        """
        Run evaluation ONCE with CURRENT NN parameters
        Notice that no sampling is done here to reduce complication.

        :param data_eval: Tuple of (iterator, string_handle)
                          Provided data must be deterministic in terms of order (i.e. no shuffling)
        :param options: List of following items:
                        1. weights_samples: List of weights as sampled before
        :return: A dictionary of
                 1. loss
                 2. accuracy
                 3. grad_global_norm
        """

        # option parsing
        weights_samples = options["weights_samples"]

        if len(weights_samples) == 0:
            return dict(
                loss=-1,
                accuracy=-1,
                grad_global_norm=-1,
            )

        # argument parsing
        data_eval_iterator, data_eval_handle = data_eval

        # loop through samples
        pred_prob_ensemble = []
        for sample in weights_samples:
            # assign weights
            self._sess.run(tf.group(*[tf.assign(var, weight) for var, weight in zip(tf.trainable_variables(), sample)]))

            # initialize data
            self._sess.run(data_eval_iterator.initializer)

            # make predictions
            pred_prob_list = []
            while True:
                try:
                    pred_prob_batch = self._sess.run(
                        self._pred_prob,
                        feed_dict={
                            self._handle: data_eval_handle,
                            self._dropout_placeholder: 1.0,  # no dropout in evaluation
                        }
                    )
                    pred_prob_list.append(pred_prob_batch)
                except tf.errors.OutOfRangeError:
                    break

            # concatenate predictions
            pred_prob_ensemble.append(pred_prob_list)

        # get average prediction
        pred_prob_avg = np.mean(pred_prob_ensemble, axis=0)
        pred_class_avg = np.argmax(pred_prob_avg, axis=-1)

        # initialization (again)
        self._sess.run([
            data_eval_iterator.initializer,
            self._metric_accuracy_initializer,
        ])

        # loop through batches
        for pred_class_batch in pred_class_avg:
            self._sess.run(
                self._metric_accuracy_update,
                feed_dict={
                    self._handle: data_eval_handle,
                    self._pred_class: pred_class_batch,
                }
            )

        # extract metric
        accuracy = self._sess.run(self._metric_accuracy)

        # return
        return dict(
            loss=-1,
            accuracy=accuracy,
            grad_global_norm=-1,
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
