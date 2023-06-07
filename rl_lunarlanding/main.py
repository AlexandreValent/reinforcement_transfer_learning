import os
import pickle


def get_train_data(env, agt, run_number):
    """
    Run a given environment with a given agent.
    """

    #Initialisation du fichier pickle
    # fichier_pickle = f"training_data/_{agt.name}.pickle"
    # file_path = "/trainin_data"
    # fichier_pickle = open("/training_data",'wb')
    directory = 'training_data/'
    filename = f'{agt.name}.pickle'

    # Combine the directory and filename to create the complete file path
    file_path = os.path.join(directory, filename)

    # Create the file
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('')
    data_list=[]

    for i in range(1, run_number+1):

        frame = 0
        score = 0
        obs_old, info = env.reset()
        done = False

        while not done :

            # We can visually render the learning environment. We disable it for performance.
            env.render()

            # We request an action from the agent.
            act = agt.choose(obs_old, train = True)[0]

            # We apply the action on the environment.
            obs_new, rwd, done, truncated, _ = env.step(act)

            # We store the data
            data = (obs_old, act, rwd, obs_new)

            # We add the needed data to our file :
            data_list.append(data)
            # Update latest observation
            obs_old = obs_new

    with open(file_path, 'wb') as file:
        pickle.dump(data_list, file)

    env.close()
    print("training done")
    return None
