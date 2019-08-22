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
        self.scramble_list = []
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
        self._action(action)
        self.score = self.cube.score()
        n_state = self._state()
        reward = self._reward()
        done = self.cube.is_solved(self.score)
        if done:
            print("Solved !!!")
        return n_state, reward, done, {}

    def reset(self):
        # initialize
        self.cube = Cube(3, whiteplastic=False)
        # scramble
        self.scramble_list = []
        if self.scrambleSize > 0:
            self.scramble(self.scrambleSize)
        self.score = self.cube.score()
        self.before_score = self.score
        return self._state()

    def render(self, mode='human', close=False):
        if self.renderCube:
            # Measure CubeEnv3x3 render function at add_axes
            self.fig = None
            self.fig = self.cube.render(self.fig, views=self.renderViews, flat=self.renderFlat)
            plt.pause(0.001)
            plt.close()
    
    def _action(self, action):
        self.cube.action2move(ACTION_LOOKUP[action])

    def scramble_check(self, action, before_actions):
        before_actions_count = len(before_actions)
        # check action and opposite action (ex:F, F_1)
        if before_actions_count > 1 \
                and self.cube.opposite_action(before_actions[before_actions_count - 1], action):
            return False
        # check action 3 same action (ex:F, F, F)
        if before_actions_count > 2 \
                and before_actions[before_actions_count - 1] == before_actions[before_actions_count - 2] \
                and action.name == before_actions[before_actions_count - 1]:
            return False
        return True

    def scramble(self, n):
        t = 0
        while t < n:
            action = ACTION_LOOKUP[np.random.randint(len(ACTION_LOOKUP.keys()))]
            if self.scramble_check(action, self.scramble_list):
                self.scramble_list.append(action.name)
                print(self.scramble_list)
                self.cube.action2move(action)
                t += 1
    
    @staticmethod
    def action2name(action):
        return ACTION_LOOKUP[action].name

    def _reward(self):
        reward = self.score - self.before_score
        if reward > 0:
            self.before_score = self.score
            return reward - 1
        return -1

    def _state(self):
        # get sticker
        sticker = self.cube.get_sticker()
        # from sticker to one-hot
        state = np.identity(6)[sticker.reshape(-1)].astype('uint8')
        return state


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

