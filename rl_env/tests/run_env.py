import gym
import rl_env.envs


# setup data path
data_path = "../data/mushroom/data.parq"

# setup env
env = gym.make("mushroom-v0").load_data(data_path)

# test loop
n_total = 1000
n_step = 0

total_reward = 0

obs, _, _, _ = env.reset()
while n_step < n_total:
    # random action
    action = env.action_space.sample()

    # next step
    obs, reward, _, _ = env.step(action)
    total_reward += reward

    # print out
    print action, reward

    n_step += 1

print "Total reward: {:.4f}".format(total_reward)
