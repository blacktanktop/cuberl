from gym.envs.registration import register

register(
    id='CubeEnv-v0',
    entry_point='cuberl.env:CubeEnv'
)
