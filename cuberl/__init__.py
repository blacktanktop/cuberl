from gym.envs.registration import register

register(
    id='CubeEnv3x3-v0',
    entry_point='cuberl.env:CubeEnv3x3'
)
