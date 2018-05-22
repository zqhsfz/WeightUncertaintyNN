# Model for mushroom data RL task, using simple e-greedy
import tensorflow as tf
import model.utils_bbp as utils
from model.model_mushroom_rl import ModelMushroomRL
import os
from collections import deque
import numpy as np
from tqdm import tqdm
import dill


class ModelMushroomRLEGreedy(ModelMushroomRL):
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
            sampled_paths = self.predict(env, n_step=sampled_steps, reset=False, epsilon=self.get_config("epsilon"))

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

                for i_batch, index_batch in enumerate(buffer_index_batches):
                    features_batch = features[index_batch]
                    actions_batch = actions[index_batch]
                    rewards_batch = rewards[index_batch]

                    self._sess.run(self._train_op,
                                   feed_dict={
                                       self._feature_placeholder: features_batch,
                                       self._action_placeholder: actions_batch,
                                       self._reward_placeholder: rewards_batch,
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
                        3. epsilon: The exploration probability
        :return: List of (context, chosen action, observed reward, optimal reward) in the same order of execution
        """

        # option parsing
        n_step = options["n_step"]
        reset = options["reset"]
        epsilon = options["epsilon"]

        # reset environment
        if reset:
            obs, _, _, _ = env.reset()
        else:
            obs = env.get_observation()

        # execution
        i_step = 0
        history = []
        while i_step < n_step:
            # execute policy (e-greedy)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
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

        return self

    def _add_classifier(self):
        # collection of hyper-parameters
        n_layers = self.get_config("n_layers")
        n_hidden_units = self.get_config("n_hidden_units")

        with tf.variable_scope("classifier"):
            # cast feature vector into float32 as input layer
            # [batch_size, feature_size]
            net = tf.cast(self._feature_placeholder, dtype=tf.float32, name="layer_input")

            # build up layer-by-layer
            for i_layer in range(n_layers + 1):  # extra one is for final output layer
                if i_layer == n_layers:
                    output_dim = self._n_action
                    activation_func = None
                else:
                    output_dim = n_hidden_units
                    activation_func = tf.nn.relu

                net = tf.layers.dense(
                    net,
                    units=output_dim,
                    activation=activation_func,
                    name="layer_{:d}".format(i_layer)
                )

            # cache last output as reward
            # shape: [batch_size, n_action]
            self._expected_reward = net

    def _add_loss(self):
        with tf.variable_scope("loss"):
            # pick expected rewards corresponding to actions #

            action_onehot = tf.one_hot(self._action_placeholder, self._n_action)  # [batch_size, n_action]

            expected_reward_per_action = \
                tf.reduce_sum(action_onehot * self._expected_reward, axis=1)  # [batch_size]

            # observed rewards #

            observed_reward_per_action = self._reward_placeholder  # [batch_size]

            # compute loss #

            # compute mean-squared error, which is negative of log-p
            # shape should be same as label: [batch_size]
            loss_likelihood = tf.square(expected_reward_per_action - observed_reward_per_action)

            # Take average across all dimensions (i.e. across both sampling and batches)
            # scalar
            loss_likelihood = tf.reduce_mean(loss_likelihood, name="loss_likelihood")

            # assemble into the final loss
            self._loss = loss_likelihood

    def _add_prediction(self):
        with tf.variable_scope("prediction"):
            # take average of sampling
            # get [batch_size, n_action]
            avg_reward = self._expected_reward

            # then get both the max reward and corresponding action
            reward_chosen = tf.reduce_max(avg_reward, axis=1)  # [batch_size]
            action_chosen = tf.argmax(avg_reward, axis=1)  # [batch_size]

            # write out
            self._predicted_reward = reward_chosen
            self._predicted_action = action_chosen

        return self
