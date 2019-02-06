#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
import numpy as np

class ChannelEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=0.5, beta=0.5, epochs=math.inf):
        self.action_space = spaces.Discrete(2)
        self.state_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)
        self.reward_space = (-5, 1)
        self.channel_matrix = np.array([[1-alpha, alpha], [beta, 1-beta]])
        self.maxEpochs = epochs
        self.seed()
        self.reset()

    def step(self, action=None):
        action = action if action is not None else self.action_space.sample()
        self.epoch = self.epoch + 1

        self.update_state()
        reward = 1 if (action == self.state) else -5
        done = True if (self.epoch == self.maxEpochs) else False
        return self.get_observation(), reward, done, {}

    def seed(self):
        self.np_random, seed = seeding.np_random()
        return seed

    def update_state(self):
        pdf = self.channel_matrix[self.state, :]
        self.state = self.np_random.choice(self.state_space.n, p=pdf)

    def get_observation(self):
        return (self.state, self.epoch)

    def reset(self):
        self.epoch = 0
        self.state = self.state_space.sample()
        return self.get_observation()

    def render(self, mode='human'):
        print("Channel State: ", self.state, "; Epoch: ", self.epoch)

if __name__ == "__main__":
    env = ChannelEnv(epochs=50)
    done = False
    while not done:
        (observation, reward, done, _) = env.step()
        env.render()
        print("Observation: ", observation, "; Reward: ", reward)
    print("done")
