#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from gym import spaces
from gym.utils import seeding

from channel_env import ChannelEnv

class SpectrumEnv(gym.Env):
    metadata = {'render.modes': ['human', 'string']}
    spec = None

    def __init__(self, channels=2, alphas=[0.5, 0.5], betas=[0.5, 0.5], epochs=50):
        self.numChannels = channels
        self.maxEpochs = epochs

        # construct multiple channel environments
        # along with composite multi-channel action, state, and observation spaces
        self.channels = tuple(ChannelEnv(a,b) for a,b in zip(alphas,betas))
        self.action_space = spaces.Tuple([x.action_space for x in self.channels])
        self.state_space = spaces.Tuple([x.state_space for x in self.channels])
        self.observation_space = spaces.Tuple([x.observation_space for x in self.channels])

        self.seed()
        self.reset()

    def step(self, action=None):
        action = action if action is not None else self.action_space.sample()
        self.epoch = self.epoch + 1

        # dispatch actions and retrieve channel observations/rewards
        observations, rewards, _, _ = zip(*tuple(map(lambda x,y:x.step(y), self.channels, action)))

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
        if mode is 'human':
            print(renderStr)
        elif mode is 'string':
            return renderStr
        else:
            raise ValueError("mode '{}' is invalid.", mode)

if __name__ == "__main__":
    env = SpectrumEnv(epochs=50)
    done = False
    while not done:
        a = env.action_space.sample()
        (o, r, done, _) = env.step(a)
        print("Action Taken: ", a, "; ", env.render(mode="string"), "; Observation: ", o, "; Reward: ", r)
