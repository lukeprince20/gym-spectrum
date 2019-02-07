#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import pytest
from channel_env import ChannelEnv

def test_channel_env():
    env = ChannelEnv(epochs=50)
    act_space = env.action_space
    ob_space = env.observation_space
    ob = env.reset()
    assert ob_space.contains(ob), 'Reset observation: {!r} not in space'.format(ob)
    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(observation), 'Step observation: {!r} not in space'.format(observation)
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)

    # Make sure we can render the environment after close.
    env.close()
    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)
