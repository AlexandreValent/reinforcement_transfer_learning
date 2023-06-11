import os
import pandas as pd

def get_metrics_data(env, agt, run_number):
    """
    Run a given agent and get the metrics associated.
    """
    agt.train =False
    #We check whether or not we have already a file -> boolean
    file_path = f"saved_metrics_data/test2_metrics.csv"
    #file_path = f"saved_metrics_data/{agt.name}_metrics.csv"
    csvexiste=os.path.isfile(file_path) #csv_path à définir

    df = pd.DataFrame(columns = ['Agent',
                                 'Planet',
                                 'Number of party',
                                 'Time (s)',
                                 'Score',
                                 'Reward per decision',
                                 'Out of screen x',
                                 'Lander crashed',
                                 'Landing success',
                                 'Max distance from the center in x (m)',
                                 'Max high distance in y (m)',
                                 'Max angle (Beta)',
                                 'Max speed (m/s)',
                                 'Fuel Consumption (L)',
                                 'Main engine activation (nb)',
                                 'Side engines activation (nb)'
                                 ])
    party_count = 0

    for party in range(1, run_number+1):
        party_count += 1
        print(f":trophée: Playing game {party}")
        print(f"Run number: {run_number + 1}")
        frame = 0
        score = 0
        obs_old, info = env.reset()
        done = False
        full_consumed = 0
        main_engine_activated = 0
        side_engines_activated = 0
        landing_speed_max = 0
        landing_success = False
        max_distance_from_center = 0
        max_high = 0
        max_angle = 0
        while not done :
            # We can visually render the learning environment. We disable it for performance.
            env.render()
            # We request an action from the agent.
            #A changer avec les nouveaux réseaux de neuronnes
            act = agt.choose(obs_old)
            # We apply the action on the environment.
            obs_new, rwd, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            # We store the max speed in y
            if obs_new[3] < landing_speed_max :
                landing_speed_max = obs_new[3]
            # Update latest observation
            obs_old = obs_new
            #We calculate the metrics needed
            frame += 1
            score += rwd
            if act == 1 or act == 3 :
                side_engines_activated += 1
                full_consumed += 1.3
            if act == 2 :
                main_engine_activated += 1
                full_consumed += 5.2
            if not env.lander.awake:
                landing_success = True
            if abs(obs_new[0]) > max_distance_from_center :
                max_distance_from_center = abs(obs_new[0])
            if abs(obs_new[1]) > max_high :
                max_high = abs(obs_new[1])
            if abs(obs_new[4]) > max_angle :
                max_angle = abs(obs_new[4])
        ## We store the metrics for each game played ##
        ## Information metrics ##
        agent_type = agt.name #Which agent played
        planet_type = env #Planet name
        ## Learning metrics ##
        time = frame / 50 #Time duration (seconds)
        score_reward = score #Total reward
        score_per_move = score / frame #Reward per decision
        full_consumed_metric = full_consumed #Total Full Consumption
        main_engine_activation = main_engine_activated #How many times we activated the main engine
        side_engines_activation = side_engines_activated #How many times we activated the side engines
        landing_speed_max_metric = abs(landing_speed_max) #Maximum speed on y
        landing_success_metric = landing_success #Is the mission a success or not ?
        max_distance_from_cente_metric = max_distance_from_center #How far from the center the lander was
        max_angle_metric = max_angle #What was the max angle ?
        ## How did we loose metrics ##
        x_over = abs(obs_new[0]) >= 1.0 #Out of screen on x
        game_over = env.game_over #Lander crashed or not
        ## We add these data to our csv file ##
        data = [agent_type,
                planet_type,
                party_count,
                time,
                score_reward,
                score_per_move,
                x_over,
                game_over,
                landing_success_metric,
                max_distance_from_cente_metric,
                max_high,
                max_angle_metric,
                landing_speed_max_metric,
                full_consumed_metric,
                main_engine_activation,
                side_engines_activation]
        df.loc[party]=data
        #We load the data in a csv file or create if if it doesn't exist
        if csvexiste == False:
            df.reset_index()
            df.to_csv(file_path , index= False)
        if csvexiste == True :
            with open(file_path, "a") as fichier:
                writer = csv.writer(fichier, lineterminator='\n')
                writer.writerow(data)
    env.close()
    return





def preprocessing_data(data) :
    #Convertir les booleans en % pour les afficher
    #data['% landing Success'] = round((data['Landing success'].sum())/len(data['Landing success']),2)
    #data['% out of screen x'] = round((data['Out of screen x'].sum())/len(data['Out of screen x']),2)
    #data['% crash'] = round((data['Lander crashed'].sum())/len(data['Lander crashed']),2)
    #Regrouper les agents ensemble et faire la moyenne des scores
    df_grouped_by_agent = data.groupby(['Agent'])[['Time (s)',
                                   'Score',
                                   'Reward per decision',
                                   'Max distance from the center in x (m)',
                                   'Max high distance in y (m)',
                                   'Max angle (Beta)',
                                   'Max speed (m/s)',
                                   'Fuel Consumption (L)',
                                   'Main engine activation (nb)',
                                   'Side engines activation (nb)',
                                   'Out of screen x',
                                   'Landing success',
                                   'Lander crashed']].mean()
    df_grouped_by_agent = np.round(df_grouped_by_agent ,decimals = 2)
    #We convert the number in %
    data['Lander crashed'] = data['Lander crashed'] * 100
    data['Out of screen x'] = data['Out of screen x'] * 100
    data['Landing success'] = data['Landing success'] * 100
    df_grouped_by_agent.rename(columns = {'Out of screen x':'% out of screen x', 'Landing success':'% landing Success' ,'Lander crashed':'% crash'}, inplace = True)
    df_grouped_by_agent.sort_values(by=['Agent'])
    df2 = df_grouped_by_agent.reset_index().set_index('Agent')
    return df2

def extract_number(filename):
    number = ""
    found_first_digit = False
    for char in filename:
        if char.isdigit():
            if not found_first_digit:
                found_first_digit = True
            else:
                number += char
    if number:
        return int(number)
    else:
        return None

def to_rename (DQN_network , env , directory , run_number):
    '''
    Run a given number of agents, compile the key metrics per generation and
    return a DataFrame or load a csv
    '''
    for filename in os.listdir(directory):
        print(filename)
        f = os.path.join(directory, filename)
        print(f)
        if "info" in filename :
            print("One useless file has been found")
        if "info" not in filename and os.path.isfile(f):
            file_to_load = f
            number = extract_number(f)
            DQN1_Net = DQN_network
            DQNAgent = agent.DQNAgent(f"{number}",DQN1_Net)
            DQNAgent.net.load_state_dict(torch.load(file_to_load))
            print(f":muscle: Loading the data for generation number {number}")
            get_metrics_data(env,DQNAgent,run_number)
    #On ouvre le fichier et on récupère un df
    # Ici il faut s'assurer que le file path est le même que dans la fonction get_metrics
    file_path = "saved_metrics_data/test2_metrics.csv"
    data = pd.read_csv(file_path)
    #On récupère un df clean et regroupé par agent avec les bonnes metrics
    print(":yeux: Processing the data")
    df = preprocessing_data(data)
    #On enregistre le df clean en csv dans le bon dossier
    file_path = f"saved_metrics_data/generation_compilation2.csv"
    df.reset_index()
    #Vérifier si les index sortent ou non, dans ce cas mettre index = False
    df.to_csv(file_path)
    print(":coche_blanche: Data cleaned and process loaded, enjoy")
    return
