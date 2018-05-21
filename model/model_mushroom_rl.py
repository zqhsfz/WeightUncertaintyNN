# Model for mushroom data RL task
import tensorflow as tf
import model.utils_bbp as utils
from model.model_base import ModelBase
import os
from collections import deque
import numpy as np
from tqdm import tqdm
import dill


class ModelMushroomRL(ModelBase):
    def __init__(self, config):
        super(ModelMushroomRL, self).__init__(config)

        self._feature_size = 117
        self._n_action = 2

    def build(self):
        # forward propagation
        self._add_placeholder()
        self._add_classifier()

        # loss and back propagation
        self._add_loss()
        self._add_train_op()

        # prediction
        self._add_prediction()

        # add saver
        self._saver = tf.train.Saver(max_to_keep=10)

        # create output directory
        os.makedirs(self.get_config("output_path"))

        return self

    def restore(self, restore_id):
        print "\n===> Restoring Model {:d} ... <===\n".format(restore_id)

        restore_path = "{:s}/model-{:d}".format(self.get_config("model_path"), restore_id)
        restore_path = restore_path.replace("//", "/")

        self._saver.restore(self._sess, save_path=restore_path)

        return self

    def save(self, identifier):
        print "\n===> Saving Model {:d} ... <===\n".format(identifier)

        save_path = "{:s}/models/model-{:d}".format(self.get_config("output_path"), identifier)

        self._saver.save(self._sess, save_path=save_path)

        return self

    def train(self, env, **options):
        """
        Training procedure while interacting with the environment

        :param env: Provided environment to interact with
        :param options: Contain following items:
                        1. buffer_size: Size of buffer
                        2. batch_size: The mini-batch size to be used in buffer
                        3. n_updates: Number of batches being used for each back-propagation. Must be at least 1.
                        4. train_steps: Total number of steps for training
                        5. sampled_steps: Number of steps in each sampling
        :return: Self
        """

        # option parsing
        buffer_size = options["buffer_size"]
        batch_size = options["batch_size"]
        n_updates = options["n_updates"]
        train_steps = options["train_steps"]
        sampled_steps = options["sampled_steps"]

        assert n_updates >= 1, "At least one batch per updates!"

        # reset environment
        obs, _, _, _ = env.reset()

        # prepare buffer and history
        buffer = deque([], maxlen=buffer_size)
        history = []
        cumulative_regret = 0.

        # loop through
        i_step = 0
        pbar = tqdm(total=train_steps)
        while i_step < train_steps:
            # move forward with current policy
            sampled_paths = self.predict(env, n_step=sampled_steps, reset=False)

            # add to buffer
            buffer.extend(map(lambda x: (x[0], x[1], x[2]), sampled_paths))

            # add to history
            history += sampled_paths
            cumulative_regret += sum(map(lambda x: x[3] - x[2], sampled_paths))

            # start updating NN only after collecting some amount of data
            if len(buffer) > 1 * batch_size:
                # prepare batches
                buffer_index_batches = [np.random.choice(len(buffer), batch_size, replace=False) for _ in range(n_updates)]

                # loop through batches for updates
                features, actions, rewards = zip(*buffer)
                features = np.array(features)
                actions = np.array(actions)
                rewards = np.array(rewards)

                n_batches = len(buffer_index_batches)
                for i_batch, index_batch in enumerate(buffer_index_batches):
                    features_batch = features[index_batch]
                    actions_batch = actions[index_batch]
                    rewards_batch = rewards[index_batch]

                    self._sess.run(self._train_op,
                                   feed_dict={
                                       self._feature_placeholder: features_batch,
                                       self._action_placeholder: actions_batch,
                                       self._reward_placeholder: rewards_batch,
                                       # self._kl_schedule: 1. / (2**(i_batch+1) - 2**((i_batch+1) - n_batches)),
                                       self._kl_schedule: 1. / n_batches,
                                   })

            # tensorboard
            self._record_summary(i_step,
                                 feed_dict={
                                     self._tb_cumulative_regret: cumulative_regret,
                                 })

            # next step
            i_step += 1
            pbar.update(1)
            pbar.set_description("Cumulative Regret: {:.4f}".format(cumulative_regret))

        # dump history to disk
        dill.dump(history, open("{:s}/history.dill".format(self.get_config("output_path")), "w"))

        return self

    def predict(self, env, **options):
        """
        Execution of current policy

        :param env: Provided environment to interact with
        :param options: Contain following items:
                        1. n_step: Number of steps to be executed
                        2. reset: Whether reset the environment
        :return: List of (context, chosen action, observed reward, optimal reward) in the same order of execution
        """

        # option parsing
        n_step = options["n_step"]
        reset = options["reset"]

        # reset environment
        if reset:
            obs, _, _, _ = env.reset()
        else:
            obs = env.get_observation()

        # execution
        i_step = 0
        history = []
        while i_step < n_step:
            # execute policy
            action = self._sess.run(self._predicted_action, feed_dict={self._feature_placeholder: obs[None]})[0]

            # cache obs
            obs_cache = obs

            # get optimal reward
            optimal_reward = env.step(env.oracle(), move_next=False)

            # next step
            obs, reward, _, _ = env.step(action)

            # update history
            history.append((obs_cache, action, reward, optimal_reward))
            i_step += 1

        return history

    def evaluate(self, data_eval, **options):
        pass

    ####################################################################################################################

    def _add_placeholder(self):
        with tf.variable_scope("placeholder"):
            # context from mushroom
            # assumed to be one-hot encoded.
            self._feature_placeholder = tf.placeholder(
                tf.int32,
                shape=[None, self._feature_size],
                name="feature_placeholder"
            )

            # action index
            # notice that this is index, NOT one-hot encoding
            self._action_placeholder = tf.placeholder(
                tf.int32,
                shape=[None],
                name="action_placeholder"
            )

            # reward
            self._reward_placeholder = tf.placeholder(
                tf.float32,
                shape=[None],
                name="reward_placeholder"
            )

            # kl scheduler
            self._kl_schedule = tf.placeholder(tf.float32, shape=[], name="kl_schedule")

            # batch size, for kl scheduling purpose
            self._batch_size = tf.shape(self._action_placeholder)[0]

        return self

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
            log_p_first_component, _ = utils.get_weight_log_prob("first_component",
                                                                 weights,
                                                                 tf.constant(0., dtype=tf.float32, shape=[]),
                                                                 tf.exp(prior_log_sigma1))

            # second component
            log_p_second_component, _ = utils.get_weight_log_prob("second_component",
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

    def _add_classifier(self):
        # collection of hyper-parameters
        n_sample = self.get_config("n_sample")
        n_layers = self.get_config("n_layers")
        n_hidden_units = self.get_config("n_hidden_units")

        # collection for weights posterior / prior log-probability
        weights_logprob_posterior = []
        weights_logprob_prior = []

        with tf.variable_scope("classifier"):
            # cast feature vector into float32
            features = tf.cast(self._feature_placeholder, dtype=tf.float32)

            # use contextual features as input layer
            net = features

            # explicit broadcast to: [n_sample, batch_size, input_dim]
            net = tf.expand_dims(net, axis=0, name="layer_input")
            net = net + tf.zeros(shape=[n_sample, 1, 1], dtype=tf.float32)

            # build up layer-by-layer
            for i_layer in range(n_layers + 1):  # extra one is for final output layer
                with tf.variable_scope("layer_{:d}".format(i_layer)):
                    # determine input dimension
                    if i_layer == 0:
                        input_dim = self._feature_size   # only context as input
                    else:
                        input_dim = n_hidden_units

                    # determine output dimension
                    if i_layer == n_layers:
                        output_dim = self._n_action  # one output for each action
                    else:
                        output_dim = n_hidden_units

                    # kernel
                    with tf.variable_scope("kernel"):
                        loc, scale = utils.get_parameters("posterior_parameters", [input_dim, output_dim])
                        weights_kernel, log_p_kernel = utils.get_weight("weights", "w", loc, scale, n_sample)
                        # weights_kernel shape: [n_sample, input_dim, output_dim]

                        # prior
                        log_p_kernel_prior = self._get_weight_piror("prior", weights_kernel)

                        weights_logprob_posterior.append(log_p_kernel)
                        weights_logprob_prior.append(log_p_kernel_prior)

                    # bias
                    with tf.variable_scope("bias"):
                        loc, scale = utils.get_parameters("posterior_parameters", [output_dim])
                        weights_bias, log_p_bias = utils.get_weight("bias", "b", loc, scale, n_sample)
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

                        if i_layer == n_layers:
                            # last layer is the final output layer so it must be linear
                            net = output
                        else:
                            # otherwise, we use relu activation
                            net = tf.nn.relu(output, name="output_activation")
                            # shape expected for output: [n_sample, batch_size, output_dim]

            # cache last output as reward
            # shape: [n_sample, batch_size, n_action]
            self._expected_reward = net

            # cache weights posterior / prior
            self._weights_logprob_posterior = weights_logprob_posterior
            self._weights_logprob_prior = weights_logprob_prior

    def _add_loss(self):
        # collect hyper-parameters here
        n_sample = self.get_config("n_sample")

        with tf.variable_scope("loss"):
            # P(D|w) / likelihood
            with tf.variable_scope("likelihood"):
                # pick expected rewards corresponding to actions #

                action_onehot = tf.one_hot(self._action_placeholder, self._n_action)  # [batch_size, n_action]
                action_onehot = tf.expand_dims(action_onehot, axis=0)  # [1, batch_size, n_action]
                action_onehot = action_onehot + tf.zeros(shape=[n_sample, 1, 1])  # [n_sample, batch_size, n_action]

                expected_reward_per_action = \
                    tf.reduce_sum(action_onehot * self._expected_reward, axis=2)  # [n_sample, batch_size]

                # reshape observed rewards #

                # shape of reward_placeholder: [batch_size]
                # Need to expand it to be consistent with expected reward: [n_sample, batch_size]
                reward_expanded = tf.expand_dims(self._reward_placeholder, axis=0, name="reward_expansion")
                reward_expanded = reward_expanded + tf.zeros(shape=[n_sample, 1])

                observed_reward_per_action = reward_expanded

                # compute loss, according to BBB #

                # compute mean-squared error, which is negative of log-p
                # shape should be same as label: [n_sample, batch_size]
                loss_likelihood = tf.square(expected_reward_per_action - observed_reward_per_action)

                # Take average across all dimensions (i.e. across both sampling and batches)
                loss_likelihood = tf.reduce_mean(loss_likelihood, name="loss_likelihood")

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
            self._loss = kl_schedule * kl + loss_likelihood

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
                learning_rate=self.get_config("lr"),
                epsilon=self.get_config("adam_epsilon", 1e-8)
            )

            list_grad_var = optimizer.compute_gradients(self._loss)
            self._train_op = optimizer.apply_gradients(list_grad_var, global_step=self._global_step, name="train_op")

            list_grad = map(lambda x: x[0], list_grad_var)
            self._grad_global_norm = tf.global_norm(list_grad, name="grad_global_norm")

        return self

    def _add_prediction(self):
        with tf.variable_scope("prediction"):
            # take average of sampling
            # get [batch_size, n_action]
            avg_reward = tf.reduce_mean(self._expected_reward, axis=0)

            # then get both the max reward and corresponding action
            reward_chosen = tf.reduce_max(avg_reward, axis=1)  # [batch_size]
            action_chosen = tf.argmax(avg_reward, axis=1)  # [batch_size]

            # write out
            self._predicted_reward = reward_chosen
            self._predicted_action = action_chosen

        return self

    def _add_tensorboard(self):
        # placeholder to log stuff from python
        self._tb_cumulative_regret = tf.placeholder(tf.float32, shape=[], name="tb_cumulative_regret")

        # add to summary
        tf.summary.scalar("Cumulative Regret", self._tb_cumulative_regret)

        # logging
        self._tb_merged = tf.summary.merge_all()
        self._tb_file_writer = tf.summary.FileWriter(
            "{:s}/log".format(self.get_config("output_path")),
            self._sess.graph
        )

        return self

    def _record_summary(self, t, feed_dict):
        """
        Plugin for passing python information to tensorboard

        :param t: step index. The step unit in tensorboard
        :param feed_dict: Input feed dictionary as specified in _add_tensorboard()
        :return: Self
        """

        summary = self._sess.run(self._tb_merged, feed_dict=feed_dict)
        self._tb_file_writer.add_summary(summary, t)

        return self
