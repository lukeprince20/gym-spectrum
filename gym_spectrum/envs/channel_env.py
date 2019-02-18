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
    metadata = {'render.modes': ('human', 'string'), 'action.modes': ('sense', 'predict', 'access')}
    reward_range = (-5, 1)
    spec = None

    action_space = spaces.Discrete(2)
    state_space = DiscreteMarkov(2)

    def __init__(self, alpha=0.5, beta=0.5, epochs=math.inf):
        self.observation_space = spaces.Discrete(2)
        self.transition_matrix = np.array([[1-alpha, alpha], [beta, 1-beta]])
        self.maxEpochs = epochs
        self.seed()
        self.reset()

    def step(self, action=None, mode='sense'):
        if (action is None) and (mode is not 'sense'):
            raise ValueError("action '{}' for mode '{}' is invalid", action, mode)
        self.epoch = self.epoch + 1
        self.update_state()
        if mode is 'sense':
            reward = None
        elif mode in ('predict', 'access'):
            if action not in self.action_space:
                raise AssertionError("action '{}' is invalid.", action)
            reward = self.reward_range[0] if action is not self.state else self.reward_range[1]
        else:
            raise ValueError("mode '{}' is invalid.", mode)
        done = True if (self.epoch == self.maxEpochs) else False
        return self.get_observation(), reward, done, {}

    def seed(self):
        self.np_random, seed = seeding.np_random()
        return seed

    def update_state(self):
        pdf = self.transition_matrix[self.state, :]
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
    env = ChannelEnv(alpha=0.1, beta=0.2, epochs=50)
    print("transition matrix:\n{}".format(env.transition_matrix))
    done = False
    while not done:
        a = env.action_space.sample()
        (o, r, done, _) = env.step(a, 'predict')
        print("Action Taken: ", a, "; ", env.render(mode="string"), "; Observation: ", o, "; Reward: ", r)
