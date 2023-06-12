from rl_lunarlanding import agent, main, network, environment, config
import warnings
import torch
warnings.filterwarnings("ignore", category=DeprecationWarning)

""" TRAINING with target"""

DQN_Net = network.DQN()
DQNAgent = agent.DQNAgent('baseline_earth',DQN_Net,DQN_target = True)
planet = environment.Planet('Earth',human_render=True)

main.training(planet = planet , agent_G = DQNAgent , nb_party= 100_000)

""" TRANSFERT LEARNING """

# DQN_Net = network.DQN()
# DQNAgent = agent.DQNAgent('moon_from_earth_DQN',DQN_Net,DQN_target = True)
# DQNAgent.net.load_state_dict(torch.load('local_saved_agents/Earth_DQN/Earth_DQN_G2000.pth'))
# planet = environment.Planet('Moon',human_render=True)

# main.training(planet = planet , agent_G = DQNAgent , nb_party= 2000)
