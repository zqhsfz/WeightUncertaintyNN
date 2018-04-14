# Bayes-by-propogation model
import tensorflow as tf
from tqdm import tqdm

from model.model_mnist import ModelMnist


class ModelMnistBayes(ModelMnist):
    def train(self, data_train, **options):
        """
        :param data_train: A tuple of training dataset and validation dataset
        :param options: Contains following item
                        1. n_epoch: Number of epochs for training
                        2. n_batch_train: Number of batches for training set
                        3. n_batch_validation: Number of batches for validation set
        :return: Self
        """

        # parse options
        n_epoch = options["n_epoch"]
        assert n_epoch >= 1, "At least one epoch is required for training!"
        n_batch_train = options["n_batch_train"]
        n_batch_validation = options["n_batch_validation"]

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
            lr = self.get_config("lr")
            for i_epoch in range(n_epoch):
                # training for one epoch
                self._sess.run(training_iterator.initializer)
                batch_index = 1
                while True:
                    try:
                        _ = self._sess.run(
                            self._train_op,
                            feed_dict={
                                self._handle: training_handle,
                                self._lr_placeholder: lr,
                                self._kl_schedule: 1. / (2**batch_index - 2**(batch_index - n_batch_train)),
                            }
                        )
                        batch_index += 1
                    except tf.errors.OutOfRangeError:
                        break

                # evaluation on training set
                result_training = self.evaluate((training_iterator, training_handle), n_batch_eval=n_batch_train)

                # evaluation on validation set
                result_validation = self.evaluate((validation_iterator, validation_handle), n_batch_eval=n_batch_validation)

                # update pbar
                pbar.update(1)
                pbar.set_description("Training loss: {:.4f}".format(result_training["loss"]))

                # update tensorboard
                self._record_summary(
                    i_epoch,
                    {
                        self._tb_training_loss_placeholder: result_training["loss"],
                        self._tb_training_accuracy_placeholder: result_training["accuracy"],
                        self._tb_training_grad_global_norm_placeholder: result_training["grad_global_norm"],

                        self._tb_validation_loss_placeholder: result_validation["loss"],
                        self._tb_validation_accuracy_placeholder: result_validation["accuracy"],
                        self._tb_validation_grad_global_norm_placeholder: result_validation["grad_global_norm"],
                    }
                )

        return self

    def evaluate(self, data_eval, **options):
        """
        :param data_eval: Tuple of (iterator, string_handle)
        :param options: Contains following item
                        1. n_batch_eval: Number of batches for evaluation set
        :return: A dictionary of
                 1. loss
                 2. accuracy
                 3. grad_global_norm
        """

        # option parsing
        n_batch_eval = options["n_batch_eval"]

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
                batch_size, loss_batch, grad_global_norm_batch, _ = self._sess.run(
                    [
                        self._batch_size,
                        self._loss,
                        self._grad_global_norm,
                        self._metric_accuracy_update,
                    ],
                    feed_dict={
                        self._handle: data_eval_handle,
                        self._kl_schedule: 1. / n_batch_eval,
                    }
                )

                count += batch_size
                count_batch += 1
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
            sigma = tf.log(1 + tf.exp(rho), name="sigma")

            epsilon = tf.random_normal(
                shape=tensor_shape,
                name="epsilon",
            )

            w = tf.add(mu, sigma * epsilon, name=weight_name)
            log_p = tf.reduce_sum(-0.5 * tf.square(epsilon) - tf.log(sigma))

        return w, log_p

    def _add_placeholder(self):
        # load whatever placeholder specified in parent class
        super(ModelMnistBayes, self)._add_placeholder()

        # add new placeholder
        with tf.variable_scope("placeholders"):
            self._kl_schedule = tf.placeholder(tf.float32, shape=[], name="kl_schedule")

        return self

    def _add_classifier(self):
        with tf.variable_scope("classifier"):
            # collection of weights and log probability of getting such weight from posterior
            weight_collection = []

            # flatten input image first
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
                    w, log_p = self._add_weight("kernel", input_dim, output_dim, "w")
                    weight_collection.append((w, log_p))

                    # add bias
                    b, log_p = self._add_weight("bias", 0, output_dim, "b")
                    weight_collection.append((b, log_p))

                    # form output
                    with tf.variable_scope("output"):
                        output = tf.add(tf.matmul(net, w), b, name="linear_output")
                        net = tf.nn.relu(output, name="activation")

            with tf.variable_scope("layer_output"):
                # determine input/output dimension
                input_dim = self.get_config("n_hidden_units")
                output_dim = self._n_class

                # add kernel
                w, log_p = self._add_weight("kernel", input_dim, output_dim, "w")
                weight_collection.append((w, log_p))

                # add bias
                b, log_p = self._add_weight("bias", 0, output_dim, "b")
                weight_collection.append((b, log_p))

                # form output
                with tf.variable_scope("output"):
                    net = tf.add(tf.matmul(net, w), b, name="linear_output")

            self._weight_collection = weight_collection
            self._logits = net

    def _add_loss(self):
        with tf.variable_scope("loss"):
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._label_placeholder,
                logits=self._logits,
                name="cross_entropy"
            )

            kl_posterior = tf.add_n(map(lambda x: x[1], self._weight_collection))
            kl_prior = self._add_prior()
            kl = kl_posterior - kl_prior

            kl_schedule = self._kl_schedule / tf.cast(self._batch_size, tf.float32)
            self._loss = kl_schedule * kl + tf.reduce_mean(ce)

    def _add_prior(self):
        prior_ratio = self.get_config("prior_ratio")
        prior_log_sigma1 = self.get_config("prior_log_sigma1")
        prior_log_sigma2 = self.get_config("prior_log_sigma2")

        with tf.variable_scope("prior"):
            list_to_sum = []
            for index, (weight, _) in enumerate(self._weight_collection):
                sum_prob = tf.add(
                    prior_ratio / tf.exp(2 * prior_log_sigma1) * tf.exp(-tf.square(weight) / (2 * tf.exp(2 * prior_log_sigma1))),
                    (1 - prior_ratio) / tf.exp(2 * prior_log_sigma2) * tf.exp(-tf.square(weight) / (2 * tf.exp(2 * prior_log_sigma2))),
                    name="sum_prob_{:d}".format(index)
                )

                list_to_sum.append(tf.reduce_sum(tf.log(sum_prob)))

            return tf.add_n(list_to_sum)
