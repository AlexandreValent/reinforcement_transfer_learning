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

        self.epsilon = 1
        self.decrease_eps = 0.995

        self.lr = 0.0001
        self.decrease_lr = 0.999

        self.gamma = 0.98

        self.nb_obs_init = 100_000
        self.nb_obs_run = 2500

CFG = Configuration()
