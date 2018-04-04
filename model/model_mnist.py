import tensorflow as tf
from tqdm import tqdm

from model.model_base import ModelBase


class ModelMnist(ModelBase):
    def __init__(self, config):
        super(ModelMnist, self).__init__(config)

        self._image_size = 28
        self._n_class = 10

    def build(self):
        # forward propagation
        self._add_placeholder()
        self._add_classifier()

        # loss and back propagation
        self._add_loss()
        self._add_train_op()

        # prediction / evaluation
        self._add_prediction()
        self._add_evaluation()

        return self

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
            # lr = self.get_config("lr")
            for i_epoch in range(n_epoch):
                # training for one epoch
                grad_global_norm_mean = 0.
                count_train = 0

                self._sess.run(training_iterator.initializer)
                while True:
                    try:
                        _, grad_global_norm = self._sess.run(
                            [self._train_op, self._grad_global_norm],
                            feed_dict={
                                self._handle: training_handle,
                                # self._lr_placeholder: lr,
                                self._dropout_placeholder: self.get_config("dropout"),
                            }
                        )

                        count_train += 1
                        grad_global_norm_mean += (grad_global_norm - grad_global_norm_mean) / count_train
                    except tf.errors.OutOfRangeError:
                        break

                # evaluation on validation set
                loss_validation = 0.
                count_validation = 0

                self._sess.run([self._metric_accuracy_initializer, validation_iterator.initializer])
                while True:
                    try:
                        loss_current, _ = self._sess.run(
                            [self._loss, self._metric_accuracy_update],
                            feed_dict={
                                self._handle: validation_handle,
                                self._dropout_placeholder: 1.0,   # no dropout in validation / testing
                            }
                        )

                        count_validation += 1
                        loss_validation += (loss_current - loss_validation) / count_validation
                    except tf.errors.OutOfRangeError:
                        break

                # summary per epoch
                pbar.update(1)
                pbar.set_description("Validation loss: {:.4f}".format(loss_validation))

                metric_accuracy = self._sess.run(self._metric_accuracy)

                self._record_summary(
                    i_epoch,
                    {
                        self._tb_validation_loss_placeholder: loss_validation,
                        self._tb_validation_accuracy_placeholder: metric_accuracy,
                        self._tb_grad_global_norm_placeholder: grad_global_norm_mean,
                    }
                )

        return self

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

        return self

    def _add_classifier(self):
        with tf.variable_scope("classifier"):
            net = tf.layers.flatten(self._image_placeholder, name="layer_input") / 128.0
            for i_layer in range(self.get_config("n_layers")):
                # net = tf.contrib.layers.fully_connected(
                #     net,
                #     num_outputs=self.get_config("n_hidden_units"),
                #     activation_fn=tf.nn.relu,
                #     scope="layer_{:d}".format(i_layer)
                # )
                net = tf.layers.dense(
                    net,
                    units=self.get_config("n_hidden_units"),
                    activation=tf.nn.relu,
                    name="layer_{:d}".format(i_layer)
                )
                net = tf.nn.dropout(net, keep_prob=self._dropout_placeholder, name="layer_{:d}_dropout".format(i_layer))
            # net = tf.contrib.layers.fully_connected(
            #     net,
            #     num_outputs=self._n_class,
            #     activation_fn=None,
            #     scope="layer_output"
            # )
            net = tf.layers.dense(
                net,
                units=self._n_class,
                activation=None,
                name="layer_output"
            )

            self._logits = net

        return self

    def _add_loss(self):
        with tf.variable_scope("loss"):
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._label_placeholder,
                logits=self._logits,
                name="cross_entropy"
            )
            self._loss = tf.reduce_mean(ce)

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

            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.get_config("lr", 1e-3),
                epsilon=self.get_config("adam_epsilon", 1e-8)
            )
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr_placeholder)

            list_grad_var = optimizer.compute_gradients(self._loss)
            self._train_op = optimizer.apply_gradients(list_grad_var, global_step=self._global_step, name="train_op")

            list_grad = map(lambda x: x[0], list_grad_var)
            self._grad_global_norm = tf.global_norm(list_grad, name="grad_global_norm")

        return self

    def _add_prediction(self):
        with tf.variable_scope("prediction"):
            self._pred_prob = tf.nn.softmax(self._logits, name="prediction_prob")
            self._pred_class = tf.argmax(self._logits, axis=1, name="prediction_class")

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

    def _add_tensorboard(self):
        # placeholder to log stuff from python
        self._tb_validation_loss_placeholder = tf.placeholder(tf.float32, shape=[],
                                                              name="tb_validation_loss_placeholder")
        self._tb_validation_accuracy_placeholder = tf.placeholder(tf.float32, shape=[],
                                                                  name="tb_validation_accuracy_placeholder")
        self._tb_grad_global_norm_placeholder = tf.placeholder(tf.float32, shape=[],
                                                               name="tb_grad_global_norm_placeholder")

        # add summary
        tf.summary.scalar("Validation Loss", self._tb_validation_loss_placeholder)
        tf.summary.scalar("Validation Accuracy", self._tb_validation_accuracy_placeholder)
        tf.summary.scalar("Validation Error", 1.0 - self._tb_validation_accuracy_placeholder)
        tf.summary.scalar("Gradient Global Norm", self._tb_grad_global_norm_placeholder)

        # logging
        self._tb_merged = tf.summary.merge_all()
        self._tb_file_writer = tf.summary.FileWriter(
            "{:s}/log".format(self.get_config("output_path")),
            self._sess.graph
        )

        return self

    def _record_summary(self, t, feed_dict):
        """
        This is where we actually log summary on tensorboard

        :param t: step index. The step unit in tensorboard
        :param feed_dict: Input feed dictionary as specified in _add_tensorboard()
        :return: Self
        """

        summary = self._sess.run(self._tb_merged, feed_dict=feed_dict)
        self._tb_file_writer.add_summary(summary, t)

        return self
