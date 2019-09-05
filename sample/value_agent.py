import datetime

import random
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import gym
from nn_framework import nn_base_agent, Trainer, Observer

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cuberl

class ValueFunctionAgent(nn_base_agent):

    def save(self, model_path):
        joblib.dump(self.model, model_path)

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = joblib.load(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        scaler = StandardScaler()
        estimator = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1)
        #self.model = Pipeline([("scaler", scaler), ("estimator", estimator)])
        self.model = Pipeline([("estimator", estimator)])

        states = np.vstack([e.s for e in experiences])
        #self.model.named_steps["scaler"].fit(states)
        #self.model.named_steps.fit(states)

        # Avoid the predict before fit.
        self.update([experiences[0]], gamma=0)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def estimate(self, s):
        estimated = self.model.predict(s)[0]
        return estimated

    # return value of actions (batchsize, 12)
    def _predict(self, states):
        if self.initialized:
            predicteds = self.model.predict(states)
        else:
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
        return predicteds

    def update(self, experiences, gamma):
        states = np.vstack([e.s for e in experiences])
        n_states = np.vstack([e.n_s for e in experiences])
        estimateds = self._predict(states)
        future = self._predict(n_states)
        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward

        estimateds = np.array(estimateds)
        #states = self.model.named_steps["scaler"].transform(states)
        self.model.named_steps["estimator"].partial_fit(states, estimateds)


class CubeEnv3x3Observer(Observer):

    def transform(self, state):
        return np.array(state).reshape((1, -1))

class ValueFunctionTrainer(Trainer):
    
    def train(self, env, episode_count=220, epsilon=0.1, initial_count=-1,
              render=False):
        actions = list(range(env.action_space.n))
        agent = ValueFunctionAgent(epsilon, actions)
        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    def episode_begin(self, episode, agent, scramble_action):
        #print("episode : ", episode)
        print("scramble action : ", scramble_action)

    def begin_train(self, episode, agent):
        agent.initialize(self.experiences)

    def step(self, episode, action, step_count, agent, experience):
        #print("step_count : ", step_count)
        #print("action", action)
        #scramble_action(self)
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)

    def episode_end(self, episode, step_count, agent):
        self.get_recent(step_count)
        rewards = [e.r for e in self.get_recent(step_count)]
        self.reward_log.append(sum(rewards))

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)

    
dt_now = datetime.datetime.now()
now = dt_now.strftime('%Y%m%d%H%M%S')
env = CubeEnv3x3Observer(gym.make('CubeEnv3x3-v0'))
env.config(views=True, flat=True, render=False, scramble_size=2)
scramble_size = str(env.get_scramble_size())
trainer = ValueFunctionTrainer()
path = trainer.logger.path_of("value_function_agent.pkl")
trained = trainer.train(env, episode_count=1000)
#print(trainer.reward_log)
trainer.logger.plot("Rewards", trainer.reward_log,
                      trainer.report_interval, './sample/png/vf_reward' + '_scramsize_' + scramble_size + '_' + now + '.png')
