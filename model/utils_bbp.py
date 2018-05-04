# utilities for bayes-by-propagation
import tensorflow as tf


def get_parameters(scope, shape):
    """
    Get variational posterior parameters. Source of everything
    """

    with tf.variable_scope(scope):
        loc = tf.get_variable(
            name="loc",
            shape=shape,
            dtype=tf.float32,
            trainable=True,
            # initializer=tf.random_normal_initializer(mean=0., stddev=1.),
            initializer=tf.contrib.layers.xavier_initializer(),
        )
        rho = tf.get_variable(
            name="rho",
            shape=shape,
            dtype=tf.float32,
            trainable=True,
            # initializer=tf.constant_initializer(-3.),
            initializer=tf.random_normal_initializer(mean=-3., stddev=0.1)
        )
        scale = tf.nn.softplus(rho, name="scale")

        return loc, scale


def get_weight(scope, weight_name, loc, scale, n_sample):
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

        _, log_p_reduced_posterior = get_weight_log_prob("log_p_posterior", weights, loc, scale)

        return weights, log_p_reduced_posterior


def get_weight_log_prob(scope, weights, loc, scale):
    """
    Utility function to compute the (log) probability of weights

    Expected shape of weights: [n_sample] + dist_shape
    """

    # get distribution shape
    dist_shape = weights.get_shape().as_list()[1:]

    # broadcast function
    def broadcast(tensor):
        if len(tensor.get_shape().as_list()) == 0:
            # scalar: some global value is expected to be broadcasted to expected shape
            return tensor + tf.zeros(shape=dist_shape, dtype=tensor.dtype)
        else:
            # non-scalar: should match distribution shape exactly
            assert tensor.get_shape().as_list() == dist_shape
            return tensor

    with tf.variable_scope(scope):
        # broadcast loc / scale
        loc_broadcast = broadcast(loc)
        scale_broadcast = broadcast(scale)

        # obtain log-probability of the sampled weight
        dist = tf.distributions.Normal(loc=loc_broadcast, scale=scale_broadcast,
                                       validate_args=True, allow_nan_stats=False, name="NormalDist")
        # dist = tf.distributions.Normal(loc=loc_broadcast, scale=scale_broadcast, name="NormalDist")

        log_p = dist.log_prob(weights, name="log_p")  # shape: [n_sample] + dist_shape

        # summation of all log-probability in one sampling
        log_p_reduced = tf.reduce_sum(log_p, axis=range(1, len(log_p.get_shape().as_list())),
                                      name="log_p_reduced")

        # output shape
        # log_p: [n_sample] + dist_shape
        # log_p_reduced: [n_sample]
        return log_p, log_p_reduced
