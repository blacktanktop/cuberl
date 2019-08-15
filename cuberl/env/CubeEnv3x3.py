import matplotlib.pyplot as plt
import numpy as np

import gym
from gym import spaces
from cuberl.env.cube import Actions, Cube


class CubeEnv3x3(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.cube = Cube(3, whiteplastic=False)
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.observation_space = spaces.Box(low=0, high=5, shape=(6, 3, 3), dtype=np.int32)
        self.status = self.cube.score()
        self.before_status = self.status
        self.fig = None
        self.scramble = []
        self.renderViews = True
        self.renderFlat = True
        self.renderCube = True
        self.scrambleSize = 1
        self.config()

    def config(self, views=True, flat=True, render=True, scramble_size=1):
        self.renderViews = views
        self.renderFlat = flat
        self.scrambleSize = scramble_size
        self.renderCube = render

        if self.renderCube:
            plt.ion()
            plt.show()

    def step(self, action):
        self.status = self.cube.score()
        reward = self._reward()
        observation = self._observe()
        done = self.cube.is_solved(self.status)
        if done:
            print("Solved !!!")
        return observation, reward, done, {}

    def reset(self):
        self.cube = Cube(3, whiteplastic=False)
        self.status = self.cube.score()
        self.before_status = self.status
        return self._observe()

    def render(self, mode='human', close=False):
        if self.renderCube:
            self.fig = self.cube.render(self.fig, views=self.renderViews, flat=self.renderFlat)
            plt.pause(0.1)

    @staticmethod

    def _reward(self):
        reward = self.status - self.before_status
        if reward > 0:
            self.before_status = self.status
            return reward - 1
        return -1

    def _observe(self):
        # get sticker
        return self.cube.get_state()


ACTION_LOOKUP = {
    0: Actions.U,
    1: Actions.U_1,
    2: Actions.D,
    3: Actions.D_1,
    4: Actions.F,
    5: Actions.F_1,
    6: Actions.B,
    7: Actions.B_1,
    8: Actions.R,
    9: Actions.R_1,
    10: Actions.L,
    11: Actions.L_1
    }

