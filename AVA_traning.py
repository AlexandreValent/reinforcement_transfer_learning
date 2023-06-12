from rl_lunarlanding import agent, main, network, environment, config
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

""" TRAINING with target"""

DQN_Net = network.DQN()
DQNAgent = agent.DQNAgent('earth_testing',DQN_Net,DQN_target = True)
planet = environment.Planet('Earth',human_render=False)

main.training(planet = planet , agent_G = DQNAgent , nb_party= 2000)
