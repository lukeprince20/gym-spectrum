import gym
from gym import error, spaces, utils
from gym.utils import seeding

class SpectrumEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    return NotImplementedError

  def step(self, action):
    return NotImplementedError

  def reset(self):
    return NotImplementedError

  def render(self, mode='human', close=False):
    return NotImplementedError
