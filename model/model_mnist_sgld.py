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
        :return: Self
        """

        # parse options
        n_epoch = options["n_epoch"]
        assert n_epoch >= 1, "At least one epoch is required for training!"
        train_size = options["train_size"]

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

            # counter for training loop
            i_train = 0

            # moving average of prediction probability on validation data
            pred_prob_avg = None
            sample_decay = 0

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
                                self._lr_placeholder: lr,  # * ((1 + i_train)**-0.55),
                                self._dropout_placeholder: 1.0,
                                self._prenoise_over_placeholder: prenoise_over,
                                self._data_size_placeholder: train_size,
                            }
                        )

                        # keep track of average prediction probability over sampling ensemble
                        if burnin_over and (i_train % thinning == 0):
                            # initialize validation data
                            self._sess.run(validation_iterator.initializer)

                            # loop through validation data
                            pred_prob_list = []
                            while True:
                                try:
                                    pred_prob_batch = self._sess.run(
                                        self._pred_prob,
                                        feed_dict={
                                            self._handle: validation_handle,
                                            self._dropout_placeholder: 1.0
                                        }
                                    )
                                    pred_prob_list.append(pred_prob_batch)
                                except tf.errors.OutOfRangeError:
                                    break
                            pred_prob_sample = np.concatenate(pred_prob_list, axis=0)

                            # update the moving average of prediction probability
                            sample_decay += 1
                            if pred_prob_avg is None:
                                pred_prob_avg = pred_prob_sample
                            else:
                                pred_prob_avg += (pred_prob_sample - pred_prob_avg) / sample_decay

                        # update counter
                        i_train += 1

                    except tf.errors.OutOfRangeError:
                        break

                # evaluation on validation set at the end of epoch
                if pred_prob_avg is None:
                    result_validation = dict(
                        accuracy=-1
                    )
                else:
                    result_validation = self.evaluate(
                        (validation_iterator, validation_handle),
                        pred_prob=pred_prob_avg
                    )

                # update pbar (per epoch)
                pbar.update(1)
                pbar.set_description("Validation accuracy: {:.4f}".format(result_validation["accuracy"]))

                # update tensorboard at the end of each epoch
                summary_op = np.mean
                self._record_summary(
                    i_epoch,
                    {
                        self._tb_training_loss_placeholder: -1,
                        self._tb_training_accuracy_placeholder: -1,
                        self._tb_training_grad_global_norm_placeholder: -1,

                        self._tb_validation_loss_placeholder: -1,
                        self._tb_validation_accuracy_placeholder: result_validation["accuracy"],
                        self._tb_validation_grad_global_norm_placeholder: -1,
                    }
                )

        return self

    def evaluate(self, data_eval, **options):
        """
        Run evaluation ONCE with CURRENT NN parameters
        Notice that no sampling is done here to reduce complication.
        Also please make sure provided data is deterministic in terms of ordering

        :param data_eval: Tuple of (iterator, string_handle)
        :param options: List of following items:
                        1. pred_prob: nd array of prediction probability on evaluation sample
        :return: A dictionary of
                 1. accuracy
        """

        # option parsing
        pred_prob = options["pred_prob"]

        # argument parsing
        data_eval_iterator, data_eval_handle = data_eval

        # initialization
        self._sess.run([
            data_eval_iterator.initializer,       # data initialization
        ])

        # get labels
        label_list = []
        while True:
            try:
                label_batch = self._sess.run(
                    self._label_placeholder,
                    feed_dict={
                        self._handle: data_eval_handle
                    }
                )
                label_list.append(label_batch)
            except tf.errors.OutOfRangeError:
                break
        label = np.concatenate(label_list, axis=0)

        # match with prediction probability
        pred_label = np.argmax(pred_prob, axis=1)

        assert len(pred_label) == len(label), "Inconsistent prediction and labels!"

        accuracy = 1.0 * np.sum(pred_label == label) / len(pred_label)

        # return
        return dict(
            accuracy=accuracy
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
            self._loss = self._loss_likelihood + self._loss_prior
            # self._loss = self._loss_likelihood  # no prior

        return self

    def _add_train_op(self):
        noise_factor = self.get_config("noise_factor")

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
            # # This is the standard version. However notice that gradient will be super big here so a very small
            # # step size is needed
            # list_grad_var_injected = \
            #     [(0.5 * self._data_size_placeholder * grad + stddev * tf.random_normal(var.get_shape()), var)
            #      for grad, var in list_grad_var_clipped]
            # # Scale update by 1/N
            # list_grad_var_injected = \
            #     [(grad + stddev * tf.random_normal(var.get_shape()) / self._data_size_placeholder, var)
            #      for grad, var in list_grad_var_clipped]
            # # Scale mean by 1/N, but std by 1/sqrt(N)
            # list_grad_var_injected = \
            #     [(grad + stddev * tf.random_normal(var.get_shape()) / tf.sqrt(self._data_size_placeholder), var)
            #      for grad, var in list_grad_var_clipped]
            # scale mean by 1/N, by std by N^-alpha
            list_grad_var_injected = \
                [(grad + stddev * tf.random_normal(var.get_shape()) / tf.pow(self._data_size_placeholder, noise_factor), var)
                 for grad, var in list_grad_var_clipped]

            # apply gradients
            self._train_op = optimizer.apply_gradients(list_grad_var_injected, global_step=self._global_step, name="train_op")

            # get gradient norm
            self._grad_global_norm = tf.global_norm(zip(*list_grad_var_clipped)[0], name="grad_global_norm")

        return self
