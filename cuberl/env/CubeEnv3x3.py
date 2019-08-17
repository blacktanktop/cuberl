import matplotlib.pyplot as plt
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from cuberl.env.cube import Actions, Cube


class CubeEnv3x3(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.cube = Cube(3, whiteplastic=False)
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.observation_space = spaces.Box(low=0, high=5, shape=(6, 3, 3), dtype=np.int32)
        self.score = self.cube.score()
        self.before_score = self.score
        self.fig = None
        self.scramble = []
        self.renderViews = True
        self.renderFlat = True
        self.renderCube = True
        self.scrambleSize = 1
        self.config()
        self.seed()

    def config(self, views=True, flat=True, render=True, scramble_size=1):
        self.renderViews = views
        self.renderFlat = flat
        self.scrambleSize = scramble_size
        self.renderCube = render

        if self.renderCube:
            plt.ion()
            plt.show()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.score = self.cube.score()
        n_state = self._state()
        reward = self._reward()
        done = self.cube.is_solved(self.score)
        if done:
            print("Solved !!!")
        return n_state, reward, done, {}

    def reset(self):
        self.cube = Cube(3, whiteplastic=False)
        self.score = self.cube.score()
        self.before_score = self.score
        return self._state()

    def render(self, mode='human', close=False):
        if self.renderCube:
            self.fig = self.cube.render(self.fig, views=self.renderViews, flat=self.renderFlat)
            plt.pause(0.1)
    
    #@staticmethod

    def _reward(self):
        reward = self.score - self.before_score
        if reward > 0:
            self.before_score = self.score
            return reward - 1
        return -1

    def _state(self):
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

