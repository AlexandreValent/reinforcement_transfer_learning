"""
Agent module.

This module is used to train agents and obtain predictions.
"""

import random
import torch
import torch.nn
from random import sample
import os

from rl_lunarlanding import network
from rl_lunarlanding.config import CFG


class Agent():
    """
    A learning agent parent class.
    """

    def __init__(self,name):
        self.name = name
        self.gen = 0

    def learn(self):
        """
        Make the agent learn from a (s, a, r, s') tuple.
        """
        raise NotImplementedError

    def choose(self):
        """
        Request a next action from the agent.
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """
    A random playing agent class.
    """
    def __init__(self,name):
        super().__init__ (name)

    def learn(self, obs_old, act, rwd, obs_new):
        """
        A random agent doesn't learn.
        """
        return

    def choose(self, obs_new):
        """
        Simply return a random action.
        """
        act = sample(CFG.act_space,1)[0]
        return act


class DQNAgent(Agent):
    """
    A basic pytorch Deep Q-learning agent.
    """

    def __init__(self,name,DQN,DQN_target = False):
        super().__init__ (name)
        self.net = DQN
        self.net_target = DQN_target
        self.opt = torch.optim.Adam(self.net.parameters(), lr = CFG.lr)
        self.train = True
        self.nb_upd_target = 0

    def learn(self, obs_old, act, rwd, obs_new):
        """
        Learn from a single observation sample.
        """
        obs_new = torch.tensor(obs_new)

        # We choose the network output
        out = self.net.forward(torch.tensor(obs_old))[act]

        # Do we update the DQN_target
        if self.nb_upd_target % CFG.tau == 0 and self.net_target :
            torch.save(self.net.state_dict(),"temp.pth")
            self.net_target.load_state_dict(torch.load("temp.pth"))
            os.remove("temp.pth")
            print ("🎯 Succefully load target network")
            self.nb_upd_target += 1

        # We compute the target
        if not self.net_target:
            with torch.no_grad():
                exp = rwd + CFG.gamma * self.net.forward(obs_new).max()
        else :
            with torch.no_grad():
                exp = rwd + CFG.gamma * self.net_target.forward(obs_new).max()

        # Compute the loss
        loss = torch.square(exp - out)

        # Perform a backward propagation.
        self.opt.zero_grad()
        loss.sum().backward()
        self.opt.step()

    def choose(self, obs_new):
        """
        Run an epsilon-greedy policy for next actino selection.
        learn train = True for training purpose. False to choose agent's metric.
        """
        # Return random action with probability epsilon
        if self.train and random.uniform(0, 1) < CFG.epsilon:
            act = sample(CFG.act_space,1)[0]
            # print(act)
            return act

        # Else, return action with highest value
        with torch.no_grad():
            # choose the values of all possible actions
            val = self.net.forward(torch.tensor(obs_new))
            # Choose the highest-values action
            act = torch.argmax(val).numpy()
            # print(act)
            return act
