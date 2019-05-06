from unityagents import UnityEnvironment
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from collections import namedtuple, deque

from time import sleep

# from model import QNetwork
from train import Agent

env = UnityEnvironment(file_name='./Reacher_2.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# environment information
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
n_agents = len(env_info.agents)
print('Number of agents:', n_agents)

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters Used

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
N_EPISODES = 1000

# All four networks structure: two hidden layers with size (256, 128)
fc1_units=256
fc2_units=128

CHECKPOINT_FOLDER = './Saved_Model/'


agent = Agent(
                DEVICE,
                state_size, n_agents, action_size, 4,
                BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                fc1_units=256, fc2_units=128, checkpoint_folder=CHECKPOINT_FOLDER
        )

def ddpg_train(n_episodes=N_EPISODES, train=True):
    scores = []
    scores_window = deque(maxlen=100)
    n_episodes = N_EPISODES

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]            # reset the environment
        states = env_info.vector_observations
        agent.reset()                                                # reset the agent noise
        score = np.zeros(n_agents)

        while True:
            actions = agent.act(states)

            env_info = env.step( actions )[brain_name]               # send the action to the environment
            next_states = env_info.vector_observations               # get the next state
            rewards = env_info.rewards                               # get the reward
            dones = env_info.local_done                              # see if episode has finished

            if train:
                agent.step(states, actions, rewards, next_states, dones)

            score += rewards                                         # update the score

            states = next_states                                     # roll over the state to next time step

            sleep(0.03) # Time in seconds.
            
            if np.any( dones ):                                      # exit loop if episode finished
                break

        scores.append(np.mean(score))
        scores_window.append(np.mean(score))

        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.mean(score), np.mean(scores_window)), end="")

    return scores


# train the agent
scores = ddpg_train(train=False)

env.close()
