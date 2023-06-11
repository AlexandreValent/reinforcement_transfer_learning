from rl_lunarlanding import agent, main,network, environment
import gym
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DQN1_Net = network.DQN1()
DQNAgent = agent.DQNAgent('DQN_moon',DQN1_Net,DQN_target = False)
DQNAgent.net.load_state_dict(torch.load('local_saved_agents/DQN_tar_moon/DQN_tar_moon_G107.pth'))
planet = environment.Planet('Moon',human_render=True)

main.evaluate(planet.env,DQNAgent,3)

# RandomAgent = agent.RandomAgent('random_test')
# main.evaluate(planet.env,RandomAgent,3)
