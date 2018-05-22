import tensorflow as tf
import numpy as np
from model.model_mushroom_rl_egreedy import ModelMushroomRLEGreedy
import gym
import rl_env.envs


if __name__ == "__main__":
    # setup environment
    env = gym.make("mushroom-v0").load_data("../rl_env/data/mushroom/data.parq")

    # setup model
    model = ModelMushroomRLEGreedy(
        config={
            # IO
            "output_path": "test-greedy",
            # network
            "n_layers": 2,
            "n_hidden_units": 100,
            "lr": 0.001,
            # epsilon
            "epsilon": 0.0,
        }
    ).build().initialize()

    # training
    model.train(
        env,
        buffer_size=4096,
        batch_size=64,
        n_updates=64*3,
        train_steps=50000,
        sampled_steps=1,
    )
