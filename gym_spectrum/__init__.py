from gym.envs.registration import register

register(
    id='spectrum-v0',
    entry_point='gym_spectrum.envs:SpectrumEnv',
)
