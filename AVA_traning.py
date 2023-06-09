from rl_lunarlanding import agent, main, network, environment
import gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DQN1_Net = network.DQN1()
DQNAgent = agent.DQNAgent('DQN_earth',DQN1_Net)

""" TRAINING TEST """
planet = environment.Planet('Earth',human_render=False)

main.auto_generation_from_random(env = planet.env , agent_G = DQNAgent , nb_gen = 2000)

# main.continue_auto_gen(env = moon.env , path= "local_saved_agents/Patoche/Patoche_G4.pth" , nb_gen = 10)

# main.sampling_agent('DQN1_test',50)
