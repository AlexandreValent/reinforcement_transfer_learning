"""
Agent module.

This module is used to train agents and obtain predictions.
"""

import random
import torch
import torch.nn
import tensorflow as tf

from rl_lunarlanding import network
from rl_lunarlanding.config import CFG


class Agent:
    """
    A learning agent parent class.
    """

    def __init__(self):
        pass

    def set(self):
        """
        Make the agent learn from a (s, a, r, s') tuple.
        """
        raise NotImplementedError

    def get(self):
        """
        Request a next action from the agent.
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """
    A random playing agent class.
    """

    def set(self, obs_old, act, rwd, obs_new):
        """
        A random agent doesn't learn.
        """
        return

    def get(self, obs_new):
        """
        Simply return a random action.
        """
        return CFG.act_space.sample()


class DQNAgent(Agent):
    """
    A basic pytorch Deep Q-learning agent.
    """

    def __init__(self):
        self.net = network.DQN()
        self.opt = torch.optim.Adam(self.net.parameters(), lr = CFG.lr)

    def set(self, obs_old, act, rwd, obs_new):
        """
        Learn from a single observation sample.
        """
        obs_new = torch.tensor(obs_new)

        # We get the network output
        out = self.net(torch.tensor(obs_old))[act]

        # We compute the target
        with torch.no_grad():
            exp = rwd + CFG.gamma * self.net(obs_new).max()

        # Compute the loss
        loss = torch.square(exp - out)

        # Perform a backward propagation.
        self.opt.zero_grad()
        loss.sum().backward()
        self.opt.step()

    def get(self, obs_new, train):
        """
        Run an epsilon-greedy policy for next actino selection.
        Set train = True for training purpose. False to get agent's metric.
        """
        # Return random action with probability epsilon
        if train and random.uniform(0, 1) < CFG.epsilon:
            return CFG.act_space.sample()

        # Else, return action with highest value
        with torch.no_grad():
            # Get the values of all possible actions
            val = self.net(torch.tensor(obs_new))
            # Choose the highest-values action
            return torch.argmax(val).numpy()
