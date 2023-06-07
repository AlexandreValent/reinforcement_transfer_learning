import os
import pickle
import numpy as np
import random
import torch
import re
from rl_lunarlanding import agent

# Silent warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_train_data(env, agt, run_number):
    """
    Get data for training with a specified agent.
    """
    #Initialisation du fichier pickle
    directory = 'training_data/'
    filename = f'{agt.name}.pickle' #PENSER A ADD
    file_path = os.path.join(directory, filename)  # Combine the directory and filename to create the complete file path

    scores=[]

    print(f"👀 Start to get data with {agt.name}.")
    for i in range(1, run_number+1):

        obs_old, _ = env.reset()
        done = False

        score=0

        while not done :

            # We request an action from the agent.
            act = agt.choose(obs_old, train = True)

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

        if i % 50 == 0:
            print(f"Party {i}/{run_number} done.")

        scores.append(score)

    avg_score = round(np.array(scores).mean(),2)
    print(f"✅ Get data done with average score of {avg_score} on {run_number} parties.")
    print("\n =================================== \n")

    return

def lean_from_pickle(file_name, agt):
    #Initialisation du fichier pickle
    directory = 'training_data/'
    filename = file_name
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
    print(f"👀 Start to train {agt.name}.")
    for i, obs in enumerate(data):
        obs_old, act, rwd, obs_new = obs
        agt.learn(obs_old, act, rwd, obs_new)

        if i % 5000 == 0:
            print(f"Passing observation {i}/{len(data)}.")

    print(f"✅ {agt.name} have learn from pickle.")

    old_gen = re.search(r"\d+$", agt.name).group()
    new_agent_name = agt.name.replace(old_gen, str(int(old_gen)+1))

    torch.save(agt.net.state_dict(),f"saved_agents/{new_agent_name}.pth")
    print(f"💾 {new_agent_name} saved.")

    print("\n =================================== \n")
    return

def auto_generation_from_random(env,nb_gen,name):

    random_agent = agent.RandomAgent(f'random_{name}')
    get_train_data(env,random_agent,100)

    DQNAgent_G0 = agent.DQNAgent(f'{name}_G0')
    lean_from_pickle(f'random_{name}.pickle', DQNAgent_G0)

    for i in range (1,nb_gen):
        print(f"Doing generation {i}")
        DQNAgent = agent.DQNAgent(f'{name}_G{i}')
        DQNAgent.net.load_state_dict(torch.load(f'saved_agents/{name}_G{i}.pth'))

        get_train_data(env,DQNAgent,500)
        lean_from_pickle(f"{name}_G{i}.pickle", DQNAgent)

    return
