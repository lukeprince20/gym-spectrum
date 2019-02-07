#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from channel_env import ChannelEnv

class SpectrumEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = None

    def __init__(self, channels=2, alphas=[0.5, 0.5], betas=[0.5, 0.5], epochs=50):
        self.numChannels = channels
        self.maxEpochs = epochs

        # construct multiple channel environments
        # along with composite multi-channel action, state, and observation spaces
        self.channels = tuple(ChannelEnv(a,b) for a,b in zip(alphas,betas))
        self.action_space = spaces.Tuple([self.channels[x].action_space for x in range(channels)])
        self.state_space = spaces.Tuple([self.channels[x].state_space for x in range(channels)])
        self.observation_space = spaces.Tuple([self.channels[x].observation_space for x in range(channels)])

        self.seed()
        self.reset()

    def step(self, action=None):
        action = action if action is not None else self.action_space.sample()
        print(action)
        self.epoch = self.epoch + 1

        # dispatch actions and retrieve channel observations/rewards
        observations, rewards, _, _ = zip(*tuple(
            map(lambda x,y:x.step(y), self.channels, action)))

        done = True if (self.epoch == self.maxEpochs) else False
        return observations, rewards, done, {}

    def reset(self):
        self.epoch = 0
        observations = tuple(map(lambda x:x.reset(), self.channels))
        self.state = tuple(map(lambda x:x.state, self.channels))
        return observations

    def render(self, mode='human', close=False):
        self.state = tuple(map(lambda x:x.state, self.channels))
        print("Channel State: ", self.state, "; Epoch: ", self.epoch)

if __name__ == "__main__":
    env = SpectrumEnv()
    done = False
    while not done:
        (observation, reward, done, _) = env.step()
        env.render()
        print("Observation: ", observation, "; Reward: ", reward)
    print("done")
