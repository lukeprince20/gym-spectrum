#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from gym import spaces
from gym.utils import seeding
from gym_spectrum.envs.channel_env import ChannelEnv
import numpy as np

class SpectrumEnv(gym.Env):
    metadata = {'render.modes': ('human', 'string'), 'action.modes': ('access',)}
    spec = None

    def __init__(self, alphas=(0.5, 0.5), betas=(0.5, 0.5), epochs=50):
        self.maxEpochs = epochs

        # construct multiple channel environments depending on length of alphas/betas
        # along with composite multi-channel action, state, and observation spaces
        assert len(alphas) == len(betas), "alphas and betas must be equal length"
        self.channels = tuple(ChannelEnv(a,b) for a,b in zip(alphas,betas))
        self.action_space = spaces.Tuple([x.action_space for x in self.channels])
        self.state_space = spaces.Tuple([x.state_space for x in self.channels])
        self.observation_space = spaces.Tuple([x.observation_space for x in self.channels])

        # maintain transition probabilities in as numpy.array and dict for convenience
        self.transition_dict = {i:m for i,m in enumerate(c.transition_matrix for c in self.channels)}
        self.transition_matrix = np.array(1)
        for c in self.channels:
            self.transition_matrix = np.kron(self.transition_matrix, c.transition_matrix)
        self.seed()
        self.reset()

    def step(self, action=None, mode='access'):
        def isIterableCollection(x): return hasattr(x, "__iter__") and not isinstance(x, str)
        actions = action if isIterableCollection(action) else tuple(action for _ in self.channels)
        assert len(actions)==len(self.channels)
        modes = mode if isIterableCollection(mode) else tuple(mode for _ in self.channels)
        assert len(modes)==len(self.channels)
        self.epoch = self.epoch + 1

        # dispatch actions and retrieve channel observations/rewards
        observations, rewards, _, _ = zip(*tuple(map(
            lambda c,a,m:c.step(action=a,mode=m), self.channels, actions, modes)))

        done = True if (self.epoch == self.maxEpochs) else False
        return observations, rewards, done, {}

    def seed(self):
        self.np_random, seed = seeding.np_random()
        return seed

    def reset(self):
        self.epoch = 0
        self.state = tuple(s for _, s in map(lambda x:x.reset(), self.channels))
        return self.epoch, self.state

    def render(self, mode='human'):
        self.state = tuple(map(lambda x:x.state, self.channels))
        renderStr = "Epoch: " + str(self.epoch) + "; Channel State: " + str(self.state)
        if mode == 'human':
            print(renderStr)
        elif mode == 'string':
            return renderStr
        else:
            raise ValueError("mode '{}' is invalid.", mode)

if __name__ == "__main__":
    env = SpectrumEnv(alphas=(0.1, 0.1), betas=(0.2, 0.2), epochs=50)
    for i, c in enumerate(env.channels):
        print("channel {} transition matrix:\n{}".format(i,c.transition_matrix))
    print("transition dict:\n{}".format(env.transition_dict))
    print("transition matrix:\n{}".format(env.transition_matrix))
    done = False
    while not done:
        a = env.action_space.sample()
        (o, r, done, _) = env.step(a, 'access')
        print("Action Taken: ", a, "; ", env.render(mode="string"), "; Observation: ", o, "; Reward: ", r)
