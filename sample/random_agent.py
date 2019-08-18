import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gym
import cuberl
import matplotlib.pyplot as plt
import numpy as np

# Configuration
solved = 0
episode_count = 10
print_every = 1
max_steps = 3
scramble_size = 1
render = True
#render = False
seed = 1

env = gym.make('CubeEnv3x3-v0')
np.random.seed(seed)
env.seed(seed)

x = []
y = []

env.config(views=True, flat=True, render=render, scramble_size=scramble_size)

for episode in range(episode_count):
    action_list = []
    print("Episode ", episode)
    s = env.reset()
    #print(s)
    #actions = []
    for step in range(max_steps):
        s = env.render()
        #print(s)
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        action_list.append(env.action2name(action))
        
        print("step ", step)
        print(action_list)
        #print(n_state)
        if done:
            solved += 1
            print("Episode solved after {0} step ; solved: {1}/{2}".format(step + 1, str(solved), str(episode_count + 1)))

            x.append(episode + 1)
            y.append(solved)
            break

    if (episode_count + 1) % print_every == 0:
        print("Episode not solved after {0} steps ; solved: {1}/{2}".format(max_steps, str(solved),
                                                                                str(episode_count + 1)))
        x.append(episode + 1)
        y.append(solved)

print(x, y)
#plt.plot(x, y)
#plt.show()
