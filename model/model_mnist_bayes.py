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

                # save model
                if (i_epoch % self.get_config("model_save_freq") == 0) or (i_epoch == n_epoch - 1):
                    self.save(i_epoch)

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

    ####################################################################################################################

    # Collection of useful utility functions here

    @staticmethod
    def _get_parameters(scope, shape):
        """
        Get variational posterior parameters. Source of everything
        """

        with tf.variable_scope(scope):
            loc = tf.get_variable(
                name="loc",
                shape=shape,
                dtype=tf.float32,
                trainable=True,
                initializer=tf.random_normal_initializer(mean=0., stddev=1.0),
            )
            rho = tf.get_variable(
                name="rho",
                shape=shape,
                dtype=tf.float32,
                trainable=True,
                initializer=tf.constant_initializer(-3.),
            )
            scale = tf.nn.softplus(rho, name="scale")

            return loc, scale

    @staticmethod
    def _get_weight(scope, weight_name, loc, scale, n_sample):
        """
        Sample model latent parameters (weights) with given distribution parameters, along with log-probability
        """

        shape = loc.get_shape().as_list()

        with tf.variable_scope(scope):
            # generate weight by sampling weight distribution
            epsilon = tf.random_normal(
                shape=[n_sample] + shape,  # first dimension represents which sample being used
                name="epsilon",
            )

            scale_expand = tf.expand_dims(scale, axis=0)
            loc_expand = tf.expand_dims(loc, axis=0)

            weights = tf.add(loc_expand, scale_expand * epsilon, name=weight_name)  # shape: [n_sample] + dist_shape

            # obtain posterior log-probability

            _, log_p_reduced_posterior = ModelMnistBayes._get_weight_log_prob("log_p_posterior", weights, loc, scale)

            return weights, log_p_reduced_posterior

    @staticmethod
    def _get_weight_log_prob(scope, weights, loc, scale):
        """
        Utility function to compute the (log) probability of weights

        Expected shape of weights: [n_sample] + dist_shape
        """

        # get distribution shape
        dist_shape = weights.get_shape().as_list()[1:]

        # broadcast utility
        def broadcast(tensor):
            if len(tensor.get_shape().as_list()) == 0:
                # scalar: some global value is expected to be broadcasted to expected shape
                return tensor + tf.zeros(shape=dist_shape, dtype=tensor.dtype)
            else:
                # non-scalar: should match distribution shape exactly
                assert tensor.get_shape().as_list() == dist_shape
                return tensor

        with tf.variable_scope(scope):
            # obtain log-probability of the sampled weight
            dist = tf.distributions.Normal(loc=broadcast(loc), scale=broadcast(scale),
                                           validate_args=True, allow_nan_stats=False, name="NormalDist")

            log_p = dist.log_prob(weights, name="log_p")  # shape: [n_sample] + dist_shape

            # summation of all log-probability in one sampling
            log_p_reduced = tf.reduce_sum(log_p, axis=range(1, len(log_p.get_shape().as_list())), name="log_p_reduced")

            # output shape
            # log_p: [n_sample] + dist_shape
            # log_p_reduced: [n_sample]
            return log_p, log_p_reduced

    def _get_weight_piror(self, scope, weights):
        """
        Obtain the log probability of given weights, according to prior

        Then assuming weights have shape of [n_sample] + dist_shape
        """

        # collect prior hyper-parameters here. All of them must be scalar
        prior_ratio = tf.constant(self.get_config("prior_ratio"), dtype=tf.float32, shape=[])
        prior_log_sigma1 = tf.constant(self.get_config("prior_log_sigma1"), dtype=tf.float32, shape=[])
        prior_log_sigma2 = tf.constant(self.get_config("prior_log_sigma2"), dtype=tf.float32, shape=[])

        with tf.variable_scope(scope):
            # first component
            log_p_first_component, _ = self._get_weight_log_prob("first_component",
                                                                 weights,
                                                                 tf.constant(0., dtype=tf.float32, shape=[]),
                                                                 tf.exp(prior_log_sigma1))

            # second component
            log_p_second_component, _ = self._get_weight_log_prob("second_component",
                                                                  weights,
                                                                  tf.constant(0., dtype=tf.float32, shape=[]),
                                                                  tf.exp(prior_log_sigma2))

            # combine two components
            log_p_combined = tf.log(prior_ratio * tf.exp(log_p_first_component) +
                                    (1. - prior_ratio) * tf.exp(log_p_second_component), name="log_p_combined")

            # summation within each sampling
            log_p_combined_reduced = tf.reduce_sum(log_p_combined,
                                                   axis=range(1, len(log_p_combined.get_shape().as_list())),
                                                   name="log_p_combined_reduced")

            # output shape: [n_sample]
            return log_p_combined_reduced

    ####################################################################################################################

    # Build the Graph

    def _add_placeholder(self):
        # load whatever placeholder specified in parent class
        super(ModelMnistBayes, self)._add_placeholder()

        # add new placeholder
        with tf.variable_scope("placeholders"):
            self._kl_schedule = tf.placeholder(tf.float32, shape=[], name="kl_schedule")

        return self

    def _add_classifier(self):
        # collection of hyper-parameters
        n_sample = self.get_config("n_sample")
        n_layers = self.get_config("n_layers")
        n_hidden_units = self.get_config("n_hidden_units")

        # collection for weights posterior / prior log-probability
        weights_logprob_posterior = []
        weights_logprob_prior = []

        with tf.variable_scope("classifier"):
            # flatten the input: [batch_size, input_dim]
            net = tf.layers.flatten(self._image_placeholder) / 128.0

            # explicit broadcast to: [n_sample, batch_size, input_dim]
            net = tf.expand_dims(net, axis=0, name="layer_input")
            net = net + tf.zeros(shape=[n_sample, 1, 1], dtype=tf.float32)

            # build up layer-by-layer
            for i_layer in range(n_layers + 1):  # extra one is for final output layer
                with tf.variable_scope("layer_{:d}".format(i_layer)):
                    # determine input dimension
                    if i_layer == 0:
                        input_dim = self._image_size * self._image_size
                    else:
                        input_dim = n_hidden_units

                    # determine output dimension
                    if i_layer == n_layers:
                        output_dim = self._n_class
                    else:
                        output_dim = self.get_config("n_hidden_units")

                    # kernel
                    with tf.variable_scope("kernel"):
                        loc, scale = self._get_parameters("posterior_parameters", [input_dim, output_dim])
                        weights_kernel, log_p_kernel = self._get_weight("weights", "w", loc, scale, n_sample)
                        # weights_kernel shape: [n_sample, input_dim, output_dim]

                        # prior
                        log_p_kernel_prior = self._get_weight_piror("prior", weights_kernel)

                        weights_logprob_posterior.append(log_p_kernel)
                        weights_logprob_prior.append(log_p_kernel_prior)

                    # bias
                    with tf.variable_scope("bias"):
                        loc, scale = self._get_parameters("posterior_parameters", [output_dim])
                        weights_bias, log_p_bias = self._get_weight("bias", "b", loc, scale, n_sample)
                        # weights_bias shape: [n_sample, output_dim]

                        # prior
                        log_p_bias_prior = self._get_weight_piror("prior", weights_bias)

                        weights_logprob_posterior.append(log_p_bias)
                        weights_logprob_prior.append(log_p_bias_prior)

                    # layer output
                    with tf.variable_scope("output"):
                        # shape expected for input: [n_sample, batch_size, input_dim]
                        multiplication = tf.matmul(net, weights_kernel, name="multiplication")
                        # shape expected for output: [n_sample, batch_size, output_dim]

                        # expand weights_bias to [n_sample, batch_size, output_dim]
                        weights_bias_expanded = tf.expand_dims(weights_bias, axis=1, name="bias_expansion")

                        # add them up: [n_sample, batch_size, output_dim]
                        output = tf.add(multiplication, weights_bias_expanded, name="output_linear")

                        if i_layer == self.get_config("n_layers"):
                            # last layer is the final output layer so it must be linear as logits
                            net = output
                        else:
                            # otherwise, we use relu activation
                            net = tf.nn.relu(output, name="output_activation")
                            # shape expected for output: [n_sample, batch_size, output_dim]

            # cache last output as logits
            # shape: [n_sample, batch_size, n_class]
            self._logits = net

            # cache weights posterior / prior
            self._weights_logprob_posterior = weights_logprob_posterior
            self._weights_logprob_prior = weights_logprob_prior

    def _add_loss(self):
        # collect hyper-parameters here
        n_sample = self.get_config("n_sample")

        with tf.variable_scope("loss"):
            # P(D|w) / cross-entropy / likelihood
            with tf.variable_scope("cross_entropy"):
                # shape of label_placeholder: [batch_size]
                # Need to expand it to be consistent with logits: [n_sample, batch_size]
                label_expanded = tf.expand_dims(self._label_placeholder, axis=0, name="label_expansion")
                label_expanded = label_expanded + tf.zeros(shape=[n_sample, 1], dtype=tf.int32)

                # compute cross-entropy, which is negative of log-p
                # shape should be same as label: [n_sample, batch_size]
                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=label_expanded,
                    logits=self._logits,
                    name="cross_entropy"
                )

                # Take average across all dimensions (i.e. across both sampling and batches)
                ce_reduced = tf.reduce_mean(ce, name="cross_entropy_mean")

            # KL between posterior p(w|theta) and prior p(w)
            with tf.variable_scope("kl"):
                # list of [n_sample] -> [n_sample, list_size]
                logprob_posterior = tf.stack(self._weights_logprob_posterior, axis=1, name="logprob_posterior")
                logprob_prior = tf.stack(self._weights_logprob_prior, axis=1, name="logprob_prior")

                # summation for each sampling respectively
                sum_logprob_posterior = tf.reduce_sum(logprob_posterior, axis=1, name="sum_logprob_posterior")
                sum_logprob_prior = tf.reduce_sum(logprob_prior, axis=1, name="sum_logprob_prior")

                # average across sampling
                avg_logprob_posterior = tf.reduce_mean(sum_logprob_posterior, axis=0, name="avg_logprob_posterior")
                avg_logprob_prior = tf.reduce_mean(sum_logprob_prior, axis=0, name="avg_logprob_prior")

                # kl should be a scalar now
                kl = tf.subtract(avg_logprob_posterior, avg_logprob_prior, name="kl")

                # add kl scheduling
                kl_schedule = self._kl_schedule / tf.cast(self._batch_size, tf.float32)

            # assemble into the final loss
            self._loss = kl_schedule * kl + ce_reduced

    def _add_prediction(self):
        with tf.variable_scope("prediction"):
            # [n_sample, batch_size, n_class]
            pred_prob = tf.nn.softmax(self._logits, name="prediction_prob")

            # take argmax
            # -> [n_sample, batch_size]
            pred_class = tf.argmax(pred_prob, axis=2, name="prediction_class_raw")

            # # unstack to a list of tensor of shape [n_sample]
            # # each tensor represents the predicted classes across different sampling, for each data point to predict
            # pred_class_list = tf.unstack(pred_class, axis=1, name="prediction_class_list")
            #
            # # extract majority voting
            # majority_class_list = []
            # for votes in pred_class_list:
            #     # shape of votes: [n_sample]
            #     values, _, counts = tf.unique_with_counts(votes, name="count_voting")
            #     majority_class = values[tf.argmax(counts)]
            #     majority_class_list.append(majority_class)
            #
            # # stack them back into a tensor
            # self._pred_class = tf.stack(majority_class_list, name="prediction_class")

            #
            # at this moment, our predict will be only the first sampling
            #

            self._pred_class = pred_class[0, :]

        return self

    def _add_evaluation(self):
        with tf.variable_scope("evaluation"):
            metric_accuracy, metric_accuracy_update = \
                tf.metrics.accuracy(self._label_placeholder, self._pred_class, name="metric_accuracy")

            self._metric_accuracy = metric_accuracy
            self._metric_accuracy_update = metric_accuracy_update

            local_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="evaluation/metric_accuracy")
            self._metric_accuracy_initializer = \
                tf.variables_initializer(var_list=local_variables, name="metric_accuracy_initializer")

        return self
