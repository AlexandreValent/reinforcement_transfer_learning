import gym
import pandas as pd

class Planet:
    def __init__(self, render_mode="rgb_array"):
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
