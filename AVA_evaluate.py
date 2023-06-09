from rl_lunarlanding import agent, main,network, environment
import gym
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DQN1_Net = network.DQN1()
DQNAgent = agent.DQNAgent('DQN_envtest',DQN1_Net)

# RandomAgent = agent.RandomAgent('random_test')

planet = environment.Planet('Neptune',human_render=True)

DQNAgent.net.load_state_dict(torch.load('local_saved_agents/DQN1_test/DQN1_test_G1700.pth'))
main.evaluate(planet.env,DQNAgent,3)

# main.evaluate(moon.env,RandomAgent,3)
