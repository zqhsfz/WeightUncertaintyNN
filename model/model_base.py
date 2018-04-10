# Templates for all models
import tensorflow as tf


class ModelBase(object):
    def __init__(self, config):
        self._config = config

    def get_config(self, key, default=None):
        if default is None:
            return self._config[key]
        else:
            return self._config.get(key, default)

    def build(self):
        """
        Build the DAG

        :return: Self
        """

        raise NotImplementedError("This is where DAG should be constructed. Please implement it in your sub-class!")

    def initialize(self, restore_id=None):
        """
        Assuming the graph has been constructed
        Create tf session and take care of basic model initialization / restoration

        :param restore_id: Model identity number to restore
                           If None (default), no model will be restored and we initialize from scratch
                           Otherwise we just load back the cached model
        :return: Self
        """

        # initialize a session
        self._sess = tf.Session()

        # add tensorboard
        self._add_tensorboard()

        # variable initialization / restoration
        if restore_id is None:
            self._sess.run(tf.global_variables_initializer())
            # self._sess.run(tf.local_variables_initializer())
            # Instead of a simple local variable initialization here, we might better do so explicitly for where it is
            # actually needed to avoid unintended behavior
            # http://ronny.rest/blog/post_2017_09_11_tf_metrics/
        else:
            self.restore(restore_id)

        return self

    def restore(self, restore_id):
        """
        Restore a model based on id

        :param restore_id: Model id
        :return: Self
        """

        raise NotImplementedError("This is where model gets restored. Please implement it in your sub-class!")

    def train(self, data_train, **options):
        """
        Define training procedure here

        :param data_train: Provided training data
        :param options: Other options that one would like to specify per training instead of per class instance
        :return: Self
        """

        raise NotImplementedError("This is where training procedure is defined. Please implement it in your sub-class!")

    def predict(self, data_pred, **options):
        """
        Define prediction procedure here

        :param data_pred: Provided data for prediction
        :param options: Other options that one would like to specify as external inputs
        :return: Prediction result
        """

        raise NotImplementedError("This is where prediction procedure is defined. Please implement it in your sub-class!")

    def evaluate(self, data_eval, **options):
        """
        Define evaluation procedure here

        :param data_eval: Provided data for evaluation
        :param options: Other options that one would like to specify as external inputs
        :return: Evaluation result
        """

        raise NotImplementedError("This is where evaluation procedure is defined. Please implement it in your sub-class!")

    def _add_tensorboard(self):
        """
        Add all stuffs that are related to tensorboard summary here.
        Do nothing by default

        :return: Self
        """

        return self
