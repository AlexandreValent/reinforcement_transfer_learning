import os
import pickle
import numpy as np
import random
import torch
import re
from rl_lunarlanding import agent, network
from rl_lunarlanding.config import CFG
import shutil
import json

def get_train_data(env, agt, nb_party):
    """
    Get data for training with a specified agent.
    """
    #Initialisation du fichier pickle
    directory = 'local_training_data/'
    filename = 'temp.pickle'
    file_path = os.path.join(directory, filename)  # Combine the directory and filename to create the complete file path
    if os.path.exists(file_path): # delete pickle if exist
        os.remove(file_path)

    scores=[]
    party = 0

    agt.train = True

    print(f"👀 Start to get data with {agt.name}_G{agt.gen}.")
    while party < nb_party:
        obs_old, _ = env.reset()
        done = False
        score=0

        while not done:
            act = agt.choose(obs_old) # We request an action from the agent.
            obs_new, rwd, terminated, truncated , _ = env.step(act)  # We apply the action on the environment.
            done = terminated or truncated

            score+=rwd
            data = (obs_old, act, rwd, obs_new)

            with open(file_path, 'ab+') as file:
                pickle.dump(data, file)

            obs_old = obs_new

        scores.append(score)
        party += 1

    avg_score = round(np.array(scores).mean(),2)
    print(f"✅ Get data done with average score of {round(avg_score,3)} on {party} parties.\n")

    return

def lean_from_pickle(agt):
    #Initialisation du fichier pickle
    directory = 'local_training_data/'
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

    # Randomly chose data in pickle
    batch = random.sample(data,CFG.batch_size)

    # Training our agent
    print(f"📖 Start to train {agt.name}_G{agt.gen}.")
    agt.learn(batch)

    print(f"✅ {agt.name}_G{agt.gen} have learn from pickle and became {agt.name}_G{agt.gen+1}.")
    return

def auto_generation_from_random(planet,agent_G,nb_gen):
    # Get initial gen to deal with continue_auto_gen
    int_gen = agent_G.gen

    # Dealing with directory and path for saving agent
    saving_path = f"local_saved_agents/{agent_G.name}"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    # Saving rl info in a json file
    json_dict = {'epsilon_init' : CFG.epsilon,'decrease_eps' : CFG.decrease_eps, 'epsilon_min' : CFG.eps_min,
                 'lr_init' : CFG.lr,'decrease_lr' : CFG.decrease_lr,
                 'gamma' : CFG.gamma,
                 'nb_party_init' : CFG.nb_party_init, 'nb_party_run' : CFG.nb_party_run,
                 'training_env' : planet.name
                 }
    if not agent_G.net_target:
        json_dict['DQN_target'] = False
        json_dict['target_tau'] = False
    else :
        json_dict['DQN_target'] = True
        json_dict['target_tau'] = CFG.tau
    with open(f'{saving_path}/rl_info.json', 'w') as outfile:
        json.dump(json_dict, outfile)

    # Restart pickle
    os.remove("local_training_data/temp.pickle")

    env = planet.env
    if int_gen == 0:
        print("🚀 Strating with generation 0")
        # Get random data
        random_agent = agent.RandomAgent('random')
        get_train_data(env,random_agent,CFG.nb_party_init)
        # Train G0
        lean_from_pickle(agent_G)

        # Save G0
        torch.save(agent_G.net.state_dict(),f"{saving_path}/{agent_G.name}_G{agent_G.gen}.pth")
        print(f"💾 {agent_G.name}_G{agent_G.gen} saved and pickel data deleted.")

        print("\n =================================== \n")

    for i in range (int_gen, int_gen + nb_gen):
        print(f"🧬 Doing generation {i} with eps = {round(CFG.epsilon,3)} & lr ={CFG.lr}\n")

        get_train_data(env,agent_G,CFG.nb_party_run)
        lean_from_pickle(agent_G)

        # Save new agent and delete pickel
        agent_G.gen += 1
        torch.save(agent_G.net.state_dict(),f"{saving_path}/{agent_G.name}_G{agent_G.gen}.pth")
        print(f"💾 {agent_G.name}_G{agent_G.gen} saved and pickel data deleted.")

        print("\n =================================== \n")

        # decrease eps, lr and incremente nb_upd_target
        CFG.lr = CFG.lr * CFG.decrease_lr
        CFG.epsilon = CFG.epsilon * CFG.decrease_eps
        agent_G.nb_upd_target +=1
        agent_G.upg_target_done = False

    print("🌚 Ready to land !")

    print("\n =================================== \n")
    return

def continue_auto_gen(env,path,nb_gen,net_target = False):
    """ Used to continue trainging with a specific agent (only working with network.DQN1)
    """

    print("⌛ Starting to restore agent")
    # Get info from path
    name = re.search(r"([^/]+)(?=_G)", path[:-4]).group()
    gen = int(re.search(r"\d+$", path[:-4]).group())


    # Get last gen to calculate eps and lr
    CFG.lr = CFG.lr * CFG.decrease_lr**gen
    CFG.epsilon = CFG.epsilon * CFG.decrease_lr**gen

    # Instanciate and load agent
    DQN1_Net = network.DQN1()
    if net_target :
        DQN1_Net_target = network.DQN1()
        DQNAgent = agent.DQNAgent(name,DQN1_Net,DQN1_Net_target)
        DQNAgent.net.load_state_dict(torch.load(path))
        DQNAgent.net_target.load_state_dict(torch.load(path))
    else :
        DQNAgent = agent.DQNAgent(name,DQN1_Net)
        DQNAgent.net.load_state_dict(torch.load(path))

    DQNAgent.gen = gen

    print("✅ Agent fully restore")
    print("\n =================================== \n")
    return auto_generation_from_random(env,DQNAgent,nb_gen)

def evaluate(env,agt,run_number):
    agt.train = False
    for party in range(1, run_number+1):
        scores=[]
        score = 0
        obs_old, info = env.reset()
        done = False
        out_of_x = False

        while not done :
            # We can visually render the learning environment. We disable it for performance.
            env.render()
            # We request an action from the agent.
            act = agt.choose(obs_old)
            # We apply the action on the environment.
            obs_new, rwd, terminated , truncated , _ = env.step(act)
            if abs(obs_new[0]) > 1:
                out_of_x = True
            done = terminated or truncated
            # Update latest observation
            obs_old = obs_new
            #We calculate the metrics needed
            score += rwd
            scores.append(score)

        if env.game_over :
            print(f"❌ Run {party} : Lander crash with a score of {score}")
        elif out_of_x :
            print(f"❌ Run {party} : Lander go outside the sceen with a score of {score}")
        elif truncated:
            print(f"❌ Run {party} : Run has been truncate with a score of {score}")
        else :
            print(f"✅ Run {party} : Lander has landed with a score of {score}")
    return

def sampling_agent(agent_name,rate):
    print("⌛ Trying to sample agents for storage on GH.")

    saving_path = f"saved_agents/{agent_name}"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    try:
        i = 0
        while True :
            shutil.copy(f"local_saved_agents/{agent_name}/{agent_name}_G{i*rate}.pth", f"saved_agents/{agent_name}")
            i += 1
    except:
        if i == 0:
            print("❌ Unable to save a sampling of your agent.")
        else :
            print(f"✅ Saving a sampling of {i} agents.")
        return

def training(planet,agent_G,nb_party):
    memory = []
    party = 0
    obs = 0
    scores = []

    # Dealing with directory and path for saving agent
    saving_path = f"local_saved_agents/{agent_G.name}"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    while party < nb_party:
        # Reset params
        state, _ = planet.env.reset()
        done = False
        out_of_x = False
        score = 0

        # While party is not over
        while not done:
            # Chose an action and calculate new state
            action = agent_G.choose(state)
            new_state, reward, terminated, truncated , _ = planet.env.step(action)

            # Update party's variables and add data in memory
            score += reward
            if len(memory) >= CFG.memmory_len:
                memory.pop(0)
            memory.append((state, action, reward, new_state,terminated)) # Add data in memory
            obs += 1
            state = new_state
            if abs(new_state[0]) > 1: # Checking if lander go outside
                out_of_x = True
            done = terminated or truncated # Checking if party is done.

            # Learning from memory
            if obs > CFG.batch_size and obs % CFG.learn_every == 0: # TODO : pas de learn avant learn < 1000 ?
                batch = random.sample(memory,CFG.batch_size)
                agent_G.learn(batch)
            # TODO : Add clean memory if > X

        # Print result of finish party
        score = round(score,2)
        scores.append(score)
        party += 1
        if planet.env.game_over :
            print(f"💥 Party {party} (rwd = {score}, eps = {round(CFG.epsilon,2)}): Lander crash")
        elif truncated:
            print(f"⌛ Party {party} (rwd = {score}, eps = {round(CFG.epsilon,2)}): Run truncated ")
        elif out_of_x :
            print(f"❌ Party {party} (rwd = {score}, eps = {round(CFG.epsilon,2)}): Lander go outside")
        else :
            print(f"✅ Party {party} (rwd = {score}, eps = {round(CFG.epsilon,2)}): Lander has landed !")

        # Update epsilon
        CFG.epsilon = max(CFG.epsilon * CFG.decrease_eps , CFG.eps_min)

        # Print average reward and save agent
        if party % CFG.AVERAGE_EVERY == 0:
            average = round(sum(scores[-CFG.AVERAGE_EVERY:]) / CFG.AVERAGE_EVERY,2)
            agent_G.gen += 1
            torch.save(agent_G.net.state_dict(),f"{saving_path}/{agent_G.name}_G{agent_G.gen}.pth")

            print('\n' + '=' * 30 + '\n')
            print(f"🎯 Average score on last {CFG.AVERAGE_EVERY} parties : {average}")
            print(f"💾 {agent_G.name}_G{agent_G.gen} saved.")
            print('\n' + '=' * 30 + '\n')
