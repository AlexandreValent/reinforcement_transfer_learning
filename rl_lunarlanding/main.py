import os, csv
import random
import torch
from rl_lunarlanding.config import CFG

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

def training(planet,agent_G,nb_party):
    memory = []
    party = 0
    obs = 0
    scores = []
    dict_metrics = {"Party": None, "Score": None,"Frame":None,
            "Success":None, "Crash": None,"Outside": None,"Truncate": None,
            "max_from_center" : None, "max_angle" : None,"main_engine_activation" : None,"side_engines_activation" : None}

    # Dealing with directory and path for saving agent and csv
    saving_path = f"local_saved_agents/{agent_G.name}" # Creating directory

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    csv_path = os.path.join(saving_path, f"{agent_G.name}.csv")
    if os.path.isfile(csv_path): # Delete csv if existing
        os.remove(csv_path)
    with open(csv_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(dict_metrics.keys())


    while party < nb_party:
        # Reset params
        state, _ = planet.env.reset()
        done = False
        party += 1

        dict_metrics = {"Party": party, "Score": 0,"Frame":0,
        "Success":False, "Crash": False,"Outside": False,"Truncate": False,
        "max_from_center" : abs(state[0]), "max_angle" : abs(state[4]),"main_engine_activation" : 0,"side_engines_activation" : 0}

        # While party is not over
        while not done:
            # Chose an action and calculate new state
            action = agent_G.choose(state)
            new_state, reward, terminated, dict_metrics['Truncate'] , _ = planet.env.step(action)

            # Update party's variables and add data in memory
            dict_metrics['Score'] += reward
            if len(memory) >= CFG.memmory_len:
                memory.pop(0)
            memory.append((state, action, reward, new_state,terminated)) # Add data in memory
            obs += 1
            state = new_state
            if abs(state[0]) > 1: # Checking if lander go outside
                dict_metrics['Outside'] = True
            dict_metrics['Frame'] += 1
            if abs(state[0]) > dict_metrics['max_from_center'] :
                dict_metrics['max_from_center']  = abs(state[0])
            if abs(state[4]) > dict_metrics['max_angle']  :
                dict_metrics['max_angle']  = abs(state[4])
            if action == 1 or action == 3:
                dict_metrics['side_engines_activation'] += 1
            if action == 2:
                dict_metrics['main_engine_activation'] += 1

            done = terminated or dict_metrics['Truncate'] # Checking if party is done.

            # Learning from memory
            if obs > CFG.batch_size and obs % CFG.learn_every == 0:
                batch = random.sample(memory,CFG.batch_size)
                agent_G.learn(batch)
                CFG.epsilon = max(CFG.epsilon * CFG.decrease_eps , CFG.eps_min)

        # Print result of finish party
        dict_metrics['Score'] = round(dict_metrics['Score'],2)
        scores.append(dict_metrics['Score']) # Append to scores for log purpose

        dict_metrics['Success'] = not planet.env.lander.awake
        dict_metrics['Crash'] = planet.env.game_over

        # Saving in CSV
        with open(csv_path, "a") as file:
            writer = csv.DictWriter(file, fieldnames=dict_metrics.keys())
            writer.writerow(dict_metrics)

        if dict_metrics['Crash'] :
            print(f"💥 Party {party} : Lander crash (rwd = {dict_metrics['Score']}, eps = {round(CFG.epsilon,2)})")
        if dict_metrics['Truncate']:
            print(f"⌛ Party {party} : Run truncated (rwd = {dict_metrics['Score']}, eps = {round(CFG.epsilon,2)})")
        if dict_metrics['Outside'] :
            print(f"❌ Party {party} : Lander go outside (rwd = {dict_metrics['Score']}, eps = {round(CFG.epsilon,2)})")
        if dict_metrics['Success'] :
            print(f"✅ Party {party} : Lander has landed (rwd = {dict_metrics['Score']}, eps = {round(CFG.epsilon,2)})")


        # Print average reward and save agent
        if party % CFG.AVERAGE_EVERY == 0:
            average = round(sum(scores[-CFG.AVERAGE_EVERY:]) / CFG.AVERAGE_EVERY,2)
            agent_G.gen += 1
            torch.save(agent_G.net.state_dict(),f"{saving_path}/{planet.name}_DQN_P{party}.pth")

            print('\n' + '=' * 30 + '\n')
            print(f"🎯 Average score on last {CFG.AVERAGE_EVERY} parties from party {party} : {average}")
            print(f"💾 New agent saved.")
            print('\n' + '=' * 30 + '\n')
