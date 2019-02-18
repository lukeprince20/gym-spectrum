#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

from contextlib import redirect_stdout
import os
import numpy as np
from channel_env import ChannelEnv

def test_envs():
    envs = (ChannelEnv(),) 
    for env in envs:
        _test_env(env)

#TODO make compliant with SpectrumEnv
def _test_env(env):
    act_space = env.action_space
    ob_space = env.observation_space
    _, ob = env.reset()
    assert ob_space.contains(ob), 'Reset observation: {} not in space'.format(ob)
    for mode in env.metadata.get('action.modes', []):
        a = act_space.sample()
        observation, reward, done, _info = env.step(action=a, mode=mode)
        assert ob_space.contains(observation), 'Step observation: {} not in space'.format(observation)
        assert np.isscalar(reward) or (reward is None), "{} is not a scalar for {}".format(reward, env)
        assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    blackhole = open(os.devnull, 'w')
    for mode in env.metadata.get('render.modes', []):
        try:
            with redirect_stdout(blackhole):
                env.render(mode=mode)
        except:
            raise

    # Make sure we can render the environment after close.
    env.close()
    for mode in env.metadata.get('render.modes', []):
        try:
            with redirect_stdout(blackhole):
                env.render(mode=mode)
        except:
            raise

if __name__ == "__main__":
    test_envs()
