import gym
import pandas as pd


def get_env(render_mode = "human"):
    """
    Returns a gym environment. Replace by a custom environment if needed.
    """

    # We use the LunarLander env. Other environments are available.
    return gym.make("LunarLander-v2", render_mode)



def run_env(env, agt, run_number):
    """
    Run a given environment with a given agent.
    """

    #Initialisation du fichier pickle
    fichier = "data_saved.pickle"
    fichier_pickle = open(fichier,'wb')

    #obs_old, info = env.reset(seed=CFG.rnd_seed)
    obs_old, info = env.reset()

    # We get the action space.
    act_space = env.action_space

    frame = 0
    score = 0

    print(f"Run number: {run_number + 1}")

    for _ in range(1000):

        # We can visually render the learning environment. We disable it for performance.
        env.render()

        # We request an action from the agent.
        act = agt.get(obs_old, act_space)

        # We apply the action on the environment.
        obs_new, rwd, terminated, truncated, _ = env.step(act)

        # We perform a learning step.
        agt.set(obs_old, act, rwd, obs_new)

        # We store the data
        data = (obs_new , obs_old , rwd, terminated)

        # Update latest observation
        obs_old = obs_new

        # We add the needed data to our file :
        fichier_pickle.dump(data , fichier_pickle)

        #We update our information
        frame += 1
        score += rwd

        if terminated or truncated:
            obs_end, info = env.reset()

    # We define the macro data
    agent_type = agt
    game_over = env.game_over
    time = frame / 50 #Comment récupérer la variable
    score_reward = score

    # We add the macro data to a Df
    data_for_df = [agent_type,game_over,time,score_reward]
    df = pd.DataFrame(data_for_df,columns=['Agent','Game Over','Time','Score'])


    env.close()

    return df , fichier_pickle
