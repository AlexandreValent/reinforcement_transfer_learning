from rl_lunarlanding import agent, main, network, environment, config
import gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

""" TRAINING TEST """

# DQN1_Net = network.DQN1()
# DQNAgent = agent.DQNAgent('DQN_earth',DQN1_Net,DQN_target = False)
# planet = environment.Planet('Earth',human_render=False)

# main.auto_generation_from_random(env = planet.env , agent_G = DQNAgent , nb_gen = 2000)

# main.continue_auto_gen(env = planet.env , path= "local_saved_agents/DQN_earth/DQN_earth_G1112.pth" , nb_gen = 900)

# main.sampling_agent('DQN1_test',50)



""" TRAINING with target"""

DQN1_Net = network.DQN()
DQNAgent = agent.DQNAgent('FULL_TESTING',DQN1_Net,DQN_target = True)
planet = environment.Planet('Earth',human_render=False)

main.training(planet = planet , agent_G = DQNAgent , nb_party= 2000)
