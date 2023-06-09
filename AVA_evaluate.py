from rl_lunarlanding import agent, main,network
import gym
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DQN1_Net = network.DQN1()
DQNAgent = agent.DQNAgent('DQN1_test',DQN1_Net)

# DQN2_Net = network.DQN2()
# DQNAgent = agent.DQNAgent('DQN2_test',DQN2_Net)

RandomAgent = agent.RandomAgent('random_test')

env = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -5.0,
    enable_wind = False,
    render_mode = "human"
    )

DQNAgent.net.load_state_dict(torch.load('saved_agents/DQN1_test_G273.pth'))
main.evaluate(env,DQNAgent,3)

# main.evaluate(env,RandomAgent,3)
