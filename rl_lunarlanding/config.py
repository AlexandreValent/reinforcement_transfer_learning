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
        self.decrease_eps = 0.99
        self.eps_min = 0.01

        self.lr = 0.0005
        self.decrease_lr = 1

        self.learn_every = 2

        self.gamma = 0.99

        self.nb_party_init = 1
        self.nb_party_run = 1

        self.tau = 0.005

        self.batch_size = 128
        self.memmory_len = 75_000

        self.AVERAGE_EVERY = 25

CFG = Configuration()
