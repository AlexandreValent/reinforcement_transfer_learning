


def get_train_data(env, agt, run_number):
    """
    Run a given environment with a given agent.
    """

    #Initialisation du fichier pickle
    fichier = f"training_data/{env.name}_{agt.name}.pickle"
    fichier_pickle = open(fichier,'wb')



    for _ in range(1, run_number+1):

        print(f"Run number: {run_number + 1}")
        frame = 0
        score = 0
        obs_old, info = env.reset()
        done = False

        while not done :

            # We can visually render the learning environment. We disable it for performance.
            env.render()

            # We request an action from the agent.
            act = agt.choose(obs_old, train = True)

            # We apply the action on the environment.
            obs_new, rwd, done, truncated, _ = env.step(act)

            # We store the data
            data = (obs_old, act, rwd, obs_new)

            # We add the needed data to our file :
            fichier_pickle.dump(data , fichier_pickle)

            # Update latest observation
            obs_old = obs_new



    env.close()
    print("training done")
    return fichier_pickle
