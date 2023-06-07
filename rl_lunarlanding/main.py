import os
import pickle
import numpy as np
import random
import torch
import re


def get_train_data(env, agt, run_number):
    """
    Get data for training with a specified agent.
    """
    #Initialisation du fichier pickle
    directory = 'training_data/'
    filename = f'{agt.name}.pickle' #PENSER A ADD
    file_path = os.path.join(directory, filename)  # Combine the directory and filename to create the complete file path

    scores=[]

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

        scores.append(score)

    env.close()
    avg_score = round(np.array(scores).mean(),2)
    print(f"Training done with average score of {avg_score} on {run_number} parties.")

    return None


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
    for obs in data:
        obs_old, act, rwd, obs_new = obs
        agt.learn(obs_old, act, rwd, obs_new)
    print(f"{agt.name} have learn from pickle.")

    old_gen = re.search(r"\d+$", agt.name).group()
    new_agent_name = agt.name.replace(old_gen, str(int(old_gen)+1))

    torch.save(agt.net.state_dict(),f"saved_agents/{new_agent_name}.pth")
    print(f"{new_agent_name} saved.")
    return
