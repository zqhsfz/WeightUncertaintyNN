import tensorflow as tf

from model.model_base import ModelBase


class ModelMnist(ModelBase):

    def build(self):
        x_placeholder = tf.placeholder