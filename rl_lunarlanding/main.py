import os
import pickle
import numpy as np
import random
import torch
import re
from rl_lunarlanding import agent, network
from rl_lunarlanding.config import CFG

# Silent warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_train_data(env, agt, obs_number):
    """
    Get data for training with a specified agent.
    """
    #Initialisation du fichier pickle
    directory = 'training_data/'
    filename = 'temp.pickle'
    file_path = os.path.join(directory, filename)  # Combine the directory and filename to create the complete file path

    scores=[]
    obs = 0
    parties = 0

    print(f"👀 Start to get data with {agt.name}_G{agt.gen}.")
    while obs < obs_number:

        obs_old, _ = env.reset()
        done = False
        score=0
        frame = 0
        while not done and frame < 1500:

        # while not done and obs < obs_number :
            # We request an action from the agent.
            act = agt.choose(obs_old)

            # We apply the action on the environment.
            obs_new, rwd, done, _ , _ = env.step(act)
            score+=rwd

            # We store the data
            data = (obs_old, act, rwd, obs_new)

            # We add the needed data to our file :
            with open(file_path, 'ab+') as file:
                pickle.dump(data, file)

            # Update latest observation
            obs_old = obs_new

            # Adding one more obs and frame
            obs += 1
            frame +=1

            if obs % 10_000 == 0:
                print(f"Obs {obs}/{obs_number} done.")

        # print(f"{parties} party done.")
        scores.append(score)
        parties += 1

    avg_score = round(np.array(scores).mean(),2)
    print(f"✅ Get data done with average score of {avg_score} on {parties} parties.\n")

    return

def lean_from_pickle(agt):
    #Initialisation du fichier pickle
    directory = 'training_data/'
    filename = 'temp.pickle'
    file_path = os.path.join(directory, filename)  # Combine the directory and filename to create the complete file path

    # Extract data from pickle
    data = []
    with open(file_path, 'rb') as file:
        try:
            while True:
                data.append(pickle.load(file))
        except EOFError:
            pass

    # Shuffle pickle's data for rl purpose
    random.shuffle(data)

    # Training our agent
    print(f"📖 Start to train {agt.name}_G{agt.gen}.")
    for i, obs in enumerate(data):
        obs_old, act, rwd, obs_new = obs
        agt.learn(obs_old, act, rwd, obs_new)

        if i % 5000 == 0:
            print(f"Passing observation {i}/{len(data)}.")

    print(f"✅ {agt.name}_G{agt.gen} have learn from pickle and became {agt.name}_G{agt.gen+1}.")
    return

def auto_generation_from_random(env,agent_G,nb_gen):
    directory = 'training_data/'
    filename = 'temp.pickle'
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    print("🚀 Strating with generation 0")
    # Get random data
    random_agent = agent.RandomAgent('random')
    get_train_data(env,random_agent,CFG.nb_obs_init)
    # Train G0
    lean_from_pickle(agent_G)

    # Save G0
    os.remove("training_data/temp.pickle")
    torch.save(agent_G.net.state_dict(),f"saved_agents/{agent_G.name}_G{agent_G.gen}.pth")
    print(f"💾 {agent_G.name}_G{agent_G.gen} saved and pickel data deleted.")

    print("\n =================================== \n")

    for i in range (1,nb_gen):
        print(f"🧬 Doing generation {i}")

        get_train_data(env,agent_G,CFG.nb_obs_run)
        lean_from_pickle(agent_G)

        # Save new agent and delete pickel
        os.remove("training_data/temp.pickle")
        agent_G.gen += 1
        torch.save(agent_G.net.state_dict(),f"saved_agents/{agent_G.name}_G{agent_G.gen}.pth")
        print(f"💾 {agent_G.name}_G{agent_G.gen} saved and pickel data deleted.")

        print("\n =================================== \n")

        # decrease eps an lr
        CFG.lr = CFG.lr * CFG.decrease_lr
        CFG.epsilon = CFG.epsilon * CFG.decrease_lr

    print("🌚 Ready to land !")

    print("\n =================================== \n")
    return

def evaluate(env,agt,run_number):
    agt.train = False
    for party in range(1, run_number+1):
        frame = 0
        scores=[]
        score = 0
        obs_old, info = env.reset()
        done = False

        while not done :
            # We can visually render the learning environment. We disable it for performance.
            env.render()
            # We request an action from the agent.
            act = agt.choose(obs_old)
            # We apply the action on the environment.
            obs_new, rwd, done, _, _ = env.step(act)
            # Update latest observation
            obs_old = obs_new
            #We calculate the metrics needed
            frame += 1
            score += rwd
            scores.append(score)
        #print(f"Score = {score}")

    print(f"SCORE MEAN = {np.array(scores).mean()}")
    return
