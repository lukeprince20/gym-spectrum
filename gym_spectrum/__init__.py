from gym.envs.registration import register

register(
    id='channel-v0',
    entry_point='gym_spectrum.envs:ChannelEnv',
    max_episode_steps=200,
    reward_threshold=25.0
)

register(
    id='spectrum-v0',
    entry_point='gym_spectrum.envs:SpectrumEnv',
    max_episode_steps=200,
    reward_threshold=25.0
)
