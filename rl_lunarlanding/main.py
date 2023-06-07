import os
import pickle
import numpy as np

def get_train_data(env, agt, run_number):
    """
    Get data for training with a specified agent.
    """

    #Initialisation du fichier pickle
    directory = 'training_data/'
    filename = f'{agt.name}.pickle' #Ajout l'env name

    # Combine the directory and filename to create the complete file path
    file_path = os.path.join(directory, filename)

    scores=[]

    for i in range(1, run_number+1):

        obs_old, _ = env.reset()
        done = False

        score=0

        while not done :

            # We request an action from the agent.
            act = agt.choose(obs_old, train = True)[0]

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
