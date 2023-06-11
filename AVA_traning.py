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



DQN1_Net = network.DQN1()
DQN1_Net_target = network.DQN1()
DQNAgent = agent.DQNAgent('DQN_tar_moon',DQN1_Net,DQN_target = DQN1_Net_target)
planet = environment.Planet('Moon',human_render=False)

main.CFG.nb_obs_init = 10_000
main.CFG.nb_obs_run = 1_500

main.CFG.epsilon = 0.75
main.CFG.decrease_eps = 0.999

main.CFG.lr = 0.0001
main.CFG.decrease_lr = 0.9999

main.CFG.tau = 2

main.auto_generation_from_random(planet = planet , agent_G = DQNAgent , nb_gen = 2000)
