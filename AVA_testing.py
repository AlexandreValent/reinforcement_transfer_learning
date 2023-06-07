# import gym

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

""" train """
from rl_lunarlanding import agent, main
import gym
import torch
import time



env = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -5.0,
    enable_wind = False
    )

# start = time.time()

# random_agent = agent.RandomAgent('random_G0')
# DQNAgent_G0 = agent.DQNAgent('DQNAgent_G0')

# main.get_train_data(env,random_agent,100)
# main.lean_from_pickle("random_G0.pickle", DQNAgent_G0)

# DQNAgent_G1 = agent.DQNAgent('DQNAgent_G1')
# DQNAgent_G1.net.load_state_dict(torch.load('saved_agents/DQNAgent_G1.pth'))

# main.get_train_data(env,DQNAgent_G1,100)
# main.lean_from_pickle("DQNAgent_G1.pickle", DQNAgent_G1)

# DQNAgent_G2 = agent.DQNAgent('DQNAgent_G2')
# DQNAgent_G2.net.load_state_dict(torch.load('saved_agents/DQNAgent_G2.pth'))

# main.get_train_data(env,DQNAgent_G2,100)

# end = time.time()
# total_time = end - start
# print("\n"+ str(total_time))


DQNAgent_G1 = agent.DQNAgent('DQNAgent_G1')
DQNAgent_G1.net.load_state_dict(torch.load('saved_agents/DQNAgent_G1.pth'))
DQNAgent_G1.net.forward([ 0.306629  , -0.03466933,  0.38309968, -0.8690317 , -0.08171916,
        5.21138   ,  1.        ,  1.        ])
