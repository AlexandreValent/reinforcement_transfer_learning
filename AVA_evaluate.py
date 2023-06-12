from rl_lunarlanding import agent, main,network, environment
import gym
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DQN1_Net = network.DQN()
DQNAgent = agent.DQNAgent('DQN_test',DQN1_Net,DQN_target = True)
DQNAgent.net.load_state_dict(torch.load('local_saved_agents/FULL_TESTING/FULL_TESTING_G80.pth'))
planet = environment.Planet('Earth',human_render=True)

main.evaluate(planet.env,DQNAgent,3)

# RandomAgent = agent.RandomAgent('random_test')
# main.evaluate(planet.env,RandomAgent,3)
