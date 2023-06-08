from rl_lunarlanding import agent, main,network
import gym
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# env = gym.make(
#     "LunarLander-v2",
#     continuous = False,
#     gravity = -1.0,
#     enable_wind = True,
#     wind_power = 15.0,
#     turbulence_power = 1.5,
#     render_mode = 'human'
#     )

# episodes = 10
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info_1, info_2 = env.step(action)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()


""" test pickle"""
# import pickle

# data1 = "embeddings"
# data2 = "names"
# with open('AVA_testing/embeddings.pickle', 'ab+') as fp:
#     pickle.dump(data1, fp)
#     pickle.dump(data2, fp)

# with open('AVA_testing/embeddings.pickle', 'rb') as fp:
#     print(pickle.load(fp))
#     print(pickle.load(fp))

# data = []
# with open('AVA_testing/embeddings.pickle', 'rb') as fr:
#     try:
#         while True:
#             data.append(pickle.load(fr))
#     except EOFError:
#         pass
# print(data)

# """ EVALUATE """


# env = gym.make(
#     "LunarLander-v2",
#     continuous = False,
#     gravity = -11.0,
#     enable_wind = False,
#     render_mode = 'human'
#     )

# DQN1_Net = network.DQN1()
# DQNAgent = agent.DQNAgent('evaluate',DQN1_Net)
# DQNAgent.net.load_state_dict(torch.load('saved_agents/test_G5.pth'))

# main.evaluate(env,DQNAgent,10)


DQN1_Net = network.DQN1()
DQNAgent = agent.DQNAgent('DQN1_test',DQN1_Net)


# DQN2_Net = network.DQN2()
# DQNAgent = agent.DQNAgent('DQN2_test',DQN2_Net)

""" TRAINING TEST """
env = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -5.0,
    enable_wind = False,
    )

main.auto_generation_from_random(env = env , agent_G0 = DQNAgent , nb_gen = 5)

""" EVALUATE """
# env = gym.make(
#     "LunarLander-v2",
#     continuous = False,
#     gravity = -5.0,
#     enable_wind = False,
#     render_mode = "human"
#     )

# DQNAgent.net.load_state_dict(torch.load('saved_agents/DQN1_test_G4.pth'))
# main.evaluate(env,DQNAgent,10)
