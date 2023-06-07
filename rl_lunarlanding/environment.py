import gym
import pandas as pd


#Terre , Lune , Mars, Jupiter


# Importer la classe
import gym

class Planet:
    def __init__(self, render_mode="human"):
        self.render_mode = render_mode

    def select_planet(self, planet):
        if planet == "Terre":
            return gym.make("LunarLander-v2",
                            render_mode=self.render_mode,
                            gravity=-8,
                            enable_wind=True,
                            wind_power=10)

        if planet == "Lune":
            return gym.make("LunarLander-v2",
                            render_mode=self.render_mode,
                            gravity=-1,
                            enable_wind=False)

        if planet == "Mars":
            return gym.make("LunarLander-v2",
                            render_mode=self.render_mode,
                            gravity=-3,
                            enable_wind=True,
                            wind_power=4)

        if planet == "Jupiter":
            return gym.make("LunarLander-v2",
                            render_mode=self.render_mode,
                            gravity=-11.5)

        if planet == "Terre_vent":
            return gym.make("LunarLander-v2",
                            render_mode=self.render_mode,
                            gravity=-8,
                            enable_wind=True,
                            wind_power=18)

        else:
            print("La planète sélectionnée n'est pas disponible")






def get_metrics():
    #We update our information
    frame += 1
    score += rwd

        #if terminated or truncated:
            #obs_end, info = env.reset()

    # We define the macro data
    agent_type = agt
    game_over = env.game_over
    time = frame / 50 #Comment récupérer la variable
    score_reward = score

    # We add the macro data to a Df
    data_for_df = [agent_type,game_over,time,score_reward]
    df = pd.DataFrame(data_for_df,columns=['Agent','Game Over','Time','Score'])



if __name__ == "__main__" :
    environnement = Planet()
    terre = environnement.select_planet('Terre')
    print(terre)
