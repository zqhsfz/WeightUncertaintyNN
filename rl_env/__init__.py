from gym.envs.registration import register

register(
    id='mushroom-v0',
    entry_point='rl_env.envs:MushroomEnv',
    # kwargs=,
)
