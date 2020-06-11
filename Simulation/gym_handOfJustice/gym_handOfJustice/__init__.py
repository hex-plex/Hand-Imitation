from gym.envs.registration import register

register(
    id = 'handOfJustice-v0',
    entry_point='gym_handOfJustice.envs:HandOfJusticeEnv'
    )
