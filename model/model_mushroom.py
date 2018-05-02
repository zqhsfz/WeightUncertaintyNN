# Basic classification model
import tensorflow as tf
from tqdm import tqdm

from model.model_mnist import ModelMnist


class ModelMushroom(ModelMnist):
    def __init__(self, config):
        super(ModelMnist, self).__init__(config)

        self._feature_size = 117
        self._n_class = 2

    def train(self, data_train, **options):
        """
        :param data_train: A tuple of training dataset and validation dataset
        :param options: Contains following item
                        1. n_epoch: Number of epochs for training
        :return: Self
        """

        # parse options
        n_epoch = options["n_epoch"]
        assert n_epoch >= 1, "At least one epoch is required for training!"

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
                while True:
                    try:
                        _ = self._sess.run(
                            self._train_op,
                            feed_dict={
                                self._handle: training_handle,
                                self._lr_placeholder: lr,
                                self._dropout_placeholder: self.get_config("dropout"),
                            }
                        )
                    except tf.errors.OutOfRangeError:
                        break

                # evaluation on training set
                result_training = self.evaluate((training_iterator, training_handle))

                # evaluation on validation set
                result_validation = self.evaluate((validation_iterator, validation_handle))

                # update pbar
                pbar.update(1)
                pbar.set_description("Training loss: {:.4f}".format(result_training["loss"]))

                # update tensorboard
                self._record_summary(
                    i_epoch,
                    {
                        self._tb_training_loss_placeholder: result_training["loss"],
                        self._tb_training_accuracy_placeholder: result_training["accuracy"],
                        self._tb_training_auc_placeholder: result_training["auc"],
                        self._tb_training_grad_global_norm_placeholder: result_training["grad_global_norm"],

                        self._tb_validation_loss_placeholder: result_validation["loss"],
                        self._tb_validation_accuracy_placeholder: result_validation["accuracy"],
                        self._tb_validation_auc_placeholder: result_validation["auc"],
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
        :param options: Nothing for now
        :return: A dictionary of
                 1. loss
                 2. accuracy
                 3. auc
                 4. grad_global_norm
        """

        # argument parsing
        data_eval_iterator, data_eval_handle = data_eval

        # initialization
        self._sess.run([
            data_eval_iterator.initializer,       # data initialization
            self._metric_accuracy_initializer,    # metric of accuracy initialization
            self._metric_auc_initializer,         # metric of auc initialization
        ])

        # loop through batches
        count = 0                  # counting on data points
        count_batch = 0            # counting on updates (batches)
        loss_all = 0.              # mean of the loss across whole dataset
        grad_global_norm_all = 0.  # mean of the global gradients across all updates (in one epoch)
        while True:
            try:
                batch_size, loss_batch, grad_global_norm_batch, _, _ = self._sess.run(
                    [
                        self._batch_size,
                        self._loss,
                        self._grad_global_norm,
                        self._metric_accuracy_update,
                        self._metric_auc_update,
                    ],
                    feed_dict={
                        self._handle: data_eval_handle,
                        self._dropout_placeholder: 1.0,  # no dropout in evaluation
                    }
                )

                count += batch_size
                count_batch += 1
                loss_all += 1.0 * (loss_batch - loss_all) * batch_size / count
                grad_global_norm_all += 1.0 * (grad_global_norm_batch - grad_global_norm_all) / count_batch
            except tf.errors.OutOfRangeError:
                break

        # extract metrics
        accuracy_all, auc_all = self._sess.run([self._metric_accuracy, self._metric_auc])

        return dict(
            loss=loss_all,
            accuracy=accuracy_all,
            auc=auc_all,
            grad_global_norm=grad_global_norm_all,
        )

    def _add_placeholder(self):
        with tf.variable_scope("placeholders"):
            # Use feedable iterator so that one can feed data in a flexible way
            self._handle = tf.placeholder(tf.string, shape=[], name="handle")
            iterator = tf.data.Iterator.from_string_handle(
                self._handle,
                output_types=(tf.float32, tf.int32),
                output_shapes=(tf.TensorShape([None, self._feature_size]), tf.TensorShape([None]))
            )
            input_feature, input_label = iterator.get_next()

            self._feature_placeholder = tf.cast(input_feature, dtype=tf.float32, name="feature_placeholder")
            self._label_placeholder = tf.cast(input_label, dtype=tf.int32, name="label_placeholder")

            # other placeholders
            self._lr_placeholder = tf.placeholder(tf.float32, shape=[], name="lr_placeholder")
            self._dropout_placeholder = tf.placeholder(tf.float32, shape=[], name="dropout_placeholder")

            # utilities
            self._batch_size = tf.shape(self._label_placeholder)[0]

        return self

    def _add_classifier(self):
        with tf.variable_scope("classifier"):
            # input layer
            net = self._feature_placeholder

            # hidden layer
            for i_layer in range(self.get_config("n_layers")):
                net = tf.layers.dense(
                    net,
                    units=self.get_config("n_hidden_units"),
                    activation=tf.nn.relu,
                    name="layer_{:d}".format(i_layer)
                )
                net = tf.nn.dropout(net, keep_prob=self._dropout_placeholder, name="layer_{:d}_dropout".format(i_layer))

            # output layer
            net = tf.layers.dense(
                net,
                units=self._n_class,
                activation=None,
                name="layer_output"
            )
            self._logits = net

        return self

    def _add_evaluation(self):
        with tf.variable_scope("evaluation"):
            #
            # accuracy
            #

            metric_accuracy, metric_accuracy_update = \
                tf.metrics.accuracy(self._label_placeholder, self._pred_class, name="metric_accuracy")

            self._metric_accuracy = metric_accuracy
            self._metric_accuracy_update = metric_accuracy_update

            local_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="evaluation/metric_accuracy")
            self._metric_accuracy_initializer = \
                tf.variables_initializer(var_list=local_variables, name="metric_accuracy_initializer")

            #
            # auc
            #

            metric_auc, metric_auc_update = \
                tf.metrics.auc(self._label_placeholder, self._pred_prob[:, 1], name="metric_auc")

            self._metric_auc = metric_auc
            self._metric_auc_update = metric_auc_update

            local_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="evaluation/metric_auc")
            self._metric_auc_initializer = \
                tf.variables_initializer(var_list=local_variables, name="metric_auc_initializer")

        return self

    def _add_tensorboard(self):
        # placeholder to log stuff from python
        self._tb_training_loss_placeholder = tf.placeholder(tf.float32, shape=[],
                                                            name="tb_training_loss_placeholder")
        self._tb_training_accuracy_placeholder = tf.placeholder(tf.float32, shape=[],
                                                                name="tb_training_accuracy_placeholder")
        self._tb_training_auc_placeholder = tf.placeholder(tf.float32, shape=[], name="tb_training_auc_placeholder")
        self._tb_training_grad_global_norm_placeholder = tf.placeholder(
            tf.float32, shape=[],
            name="tb_training_grad_global_norm_placeholder"
        )

        self._tb_validation_loss_placeholder = tf.placeholder(tf.float32, shape=[],
                                                              name="tb_validation_loss_placeholder")
        self._tb_validation_accuracy_placeholder = tf.placeholder(tf.float32, shape=[],
                                                                  name="tb_validation_accuracy_placeholder")
        self._tb_validation_auc_placeholder = tf.placeholder(tf.float32, shape=[], name="tb_validation_auc_placeholder")
        self._tb_validation_grad_global_norm_placeholder = tf.placeholder(
            tf.float32, shape=[],
            name="tb_validation_grad_global_norm_placeholder"
        )

        # add summary
        tf.summary.scalar("Training Loss", self._tb_training_loss_placeholder)
        tf.summary.scalar("Training Accuracy", self._tb_training_accuracy_placeholder)
        tf.summary.scalar("Training Error", 1.0 - self._tb_training_accuracy_placeholder)
        tf.summary.scalar("Training AUC", self._tb_training_auc_placeholder)
        tf.summary.scalar("Training Gradient", self._tb_training_grad_global_norm_placeholder)

        tf.summary.scalar("Validation Loss", self._tb_validation_loss_placeholder)
        tf.summary.scalar("Validation Accuracy", self._tb_validation_accuracy_placeholder)
        tf.summary.scalar("Validation Error", 1.0 - self._tb_validation_accuracy_placeholder)
        tf.summary.scalar("Validation AUC", self._tb_validation_auc_placeholder)
        tf.summary.scalar("Validation Gradient", self._tb_validation_grad_global_norm_placeholder)

        # logging
        self._tb_merged = tf.summary.merge_all()
        self._tb_file_writer = tf.summary.FileWriter(
            "{:s}/log".format(self.get_config("output_path")),
            self._sess.graph
        )

        return self
