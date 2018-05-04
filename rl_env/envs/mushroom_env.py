# gym environment for UCI Mushroom data
# Original data is a supervised learning task. We adapt it into a contextual bandit problem according to
# https://arxiv.org/pdf/1505.05424.pdf

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandas as pd
import numpy as np


class MushroomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path=None):
        self._data = None
        self._n_sample = None
        self._n_obs = None
        self._state = None

        if data_path is not None:
            self.load_data(data_path)

    def load_data(self, data_path):
        # load data
        # At this moment, we still have strong assumption on input data that all feature columns are one-hot columns
        self._data = pd.read_parquet(data_path)
        self._n_sample = self._data.shape[0]

        # define observation / action space
        self._n_obs = self._data.shape[1] - 1  # exclude the label
        self.observation_space = spaces.MultiBinary(self._n_obs)
        self.action_space = spaces.Discrete(2)  # eat: 1, do nothing: 0

        # state is the current row of dataframe being sampled
        # as contextual bandit problem, it is actually stateless. We set this way so that env can interact with input
        # action to determine the reward. Evolution to next state is completely independent of current state / action.
        self._state = None

        return self

    def reset(self):
        self._state = self._sample()
        return self.get_observation(), 0., False, None

    def step(self, action, move_next=True):
        # get reward
        edible = self._state[0]
        if action == 0:
            # do nothing
            reward = 0.
        else:
            # eat it
            assert action == 1
            if edible:
                reward = 5.
            else:
                if np.random.rand() < 0.5:
                    reward = -35.
                else:
                    reward = 5.

        if move_next:
            # update to next state
            self._state = self._sample()

            # return
            return self.get_observation(), reward, False, None
        else:
            return reward

    def oracle(self):
        """
        Return optimal policy with truth information
        :return: action
        """

        if self._state[0]:
            action = 1
        else:
            action = 0

        return action

    def render(self, mode='human'):
        pass

    def get_observation(self):
        return np.array(self._state.tolist()[1:])

    def _sample(self):
        index = np.random.randint(low=0, high=self._n_sample)
        return self._data.iloc[index]
