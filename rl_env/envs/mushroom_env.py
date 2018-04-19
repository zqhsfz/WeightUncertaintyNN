# gym environment for UCI Mushroom data
# Original data is a supervised learning task. We adapt it into a contextual bandit problem according to
# https://arxiv.org/pdf/1505.05424.pdf

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandas as pd


class MushroomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path):
        self._data = pd.read_parquet(data_path)

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass