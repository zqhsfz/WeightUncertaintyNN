import tensorflow as tf
import numpy as np
from model.model_mushroom_rl import ModelMushroomRL
import gym
import rl_env.envs


if __name__ == "__main__":
    # setup environment
    env = gym.make("mushroom-v0").load_data("../rl_env/data/mushroom/data.parq")

    # setup model
    model = ModelMushroomRL(
        config={
            # IO
            "output_path": "test",
            # network
            "n_layers": 2,
            "n_hidden_units": 100,
            "lr": 0.001,
            # prior
            "prior_ratio": 0.5,
            "prior_log_sigma1": -0.,
            "prior_log_sigma2": -6.,
            # sampling
            "n_sample": 2,
        }
    ).build().initialize()

    # # run a test sampling
    # history = model.predict(env, n_step=100)
    # for obs, action, reward, optimal_reward in history:
    #     print action, reward, optimal_reward

    # training
    model.train(
        env,
        buffer_size=4096,
        batch_size=64,
        n_updates=64*3,
        train_steps=50000,
        sampled_steps=1,
    )
