from rl_lunarlanding import agent, main,network
import gym
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DQN1_Net = network.DQN1()
DQNAgent = agent.DQNAgent('DQN1_test',DQN1_Net)

# DQN2_Net = network.DQN2()
# DQNAgent = agent.DQNAgent('DQN2_test',DQN2_Net)

""" TRAINING TEST """
env = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -7.0,
    enable_wind = False,
    )

main.auto_generation_from_random(env = env , agent_G = DQNAgent , nb_gen = 500)
