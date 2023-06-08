"""
Configuration Module.

This module defines a singleton-type configuration class that can be used all across our project.
This class can contain any parameter that one may want to change from one simulation run to the other.
"""

class Configuration:
    def __init__(self):
        self.x_dim = 8
        self.y_dim = 4

        self.act_space = [0,1,2,3]

        self.epsilon = 0.25

        self.lr = 0.005
        self.gamma = 0.8

        self.nb_obs_init = 50_000
        self.nb_obs_run = 50_000

CFG = Configuration()
