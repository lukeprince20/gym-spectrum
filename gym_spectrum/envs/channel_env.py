#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from gym import spaces
from gym.utils import seeding

import math
import numpy as np

class DiscreteMarkov(spaces.Discrete):
    """
    {0,1,...,n-1} with support for non-uniform sampling.
    Example usage:
    self.observation_space = spaces.Discrete(2)
    self.sample(p=[0.3, 0.7])
    """
    def sample(self, *args, **kwargs):
        if args or kwargs:
            return spaces.np_random.choice(self.n, *args, **kwargs)
        else:
            return super().sample()

    def __repr__(self):
        return "DiscreteMarkov(%d)" % self.n


class ChannelEnv(gym.Env):
    metadata = {'render.modes': ['human', 'string']}
    reward_range = (-5, 1)
    spec = None

    action_space = spaces.Discrete(2)
    state_space = DiscreteMarkov(2)

    def __init__(self, alpha=0.5, beta=0.5, epochs=math.inf):
        self.observation_space = spaces.Discrete(2)
        self.channel_matrix = np.array([[1-alpha, alpha], [beta, 1-beta]])
        self.maxEpochs = epochs
        self.seed()
        self.reset()

    def step(self, action=None):
        action = action if action is not None else self.action_space.sample()
        self.epoch = self.epoch + 1
        self.update_state()
        reward = self.reward_range[0] if action is not self.state else self.reward_range[1]
        done = True if (self.epoch == self.maxEpochs) else False
        return self.get_observation(), reward, done, {}

    def seed(self):
        self.np_random, seed = seeding.np_random()
        return seed

    def update_state(self):
        pdf = self.channel_matrix[self.state, :]
        self.state = self.state_space.sample(p=pdf)

    def get_observation(self):
        return self.state

    def reset(self):
        self.epoch = 0
        self.state = self.state_space.sample()
        return self.epoch, self.state

    def render(self, mode='human'):
        renderStr = "Epoch: " + str(self.epoch) + "; Channel State: " + str(self.state)
        if mode is 'human':
            print(renderStr)
        elif mode is 'string':
            return renderStr
        else:
            raise ValueError("mode '{}' is invalid.", mode)


if __name__ == "__main__":
    env = ChannelEnv(epochs=50)
    done = False
    while not done:
        a = env.action_space.sample()
        (o, r, done, _) = env.step(a)
        print("Action Taken: ", a, "; ", env.render(mode="string"), "; Observation: ", o, "; Reward: ", r)
