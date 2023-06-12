"""
Agent module.

This module is used to train agents and obtain predictions.
"""

import random
import torch
import torch.nn
from random import sample
import os
import numpy as np
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
        self.opt = torch.optim.Adam(self.net.parameters(), lr = CFG.lr, amsgrad= True)
        self.train = True
        self.DQN_target = DQN_target
        if self.DQN_target :
            self.net_target = network.DQN()

        self.net.eval()
        self.net_target.eval()

    def learn(self, batch):
        """
        Learn from observations sample.
        """
        # Calculate loss
        errors = []

        for observation in batch:
            state, action, reward, new_state, terminated = observation

            # Computing out
            out = self.net.forward(torch.tensor(state))[action]
            # Computing exp
            if self.DQN_target:
                with torch.no_grad():
                    exp = reward + CFG.gamma * self.net_target.forward(torch.tensor(new_state)).max() * (1 - terminated)
            else :
                with torch.no_grad():
                    exp = reward + CFG.gamma * self.net.forward(torch.tensor(new_state)).max() * (1 - terminated)

            errors.append((out-exp)**2)

        loss = torch.mean(torch.stack(errors))

        # Perform a backward propagation.
        self.net.train()
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=50, norm_type=2.0)
        self.opt.step()
        self.net.eval()

        # Update the DQN_target
        if self.net_target :
            for source_parameters, target_parameters in zip(self.net.parameters(), self.net_target.parameters()):
                target_parameters.data.copy_(CFG.tau * source_parameters.data + (1.0 - CFG.tau) * target_parameters.data)

    def choose(self, state):
        """
        Run an epsilon-greedy policy for next actino selection.
        learn train = True for training purpose. False to choose agent's metric.
        """
        # Return random action with probability epsilon
        if self.train and random.uniform(0, 1) < CFG.epsilon:
            act = sample(CFG.act_space,1)[0]
            return act

        # Else, return action with highest reward value
        with torch.no_grad():
            val = self.net.forward(torch.tensor(state)) # TODO env.action_space.sample()
            act = torch.argmax(val).numpy()
            return act
