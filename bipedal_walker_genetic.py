import gym
import Box2D
import mujoco_py

import numpy as np
from random import random
from statistics import mean
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

################# MODEL DEFINITION ###################

class Policy(nn.Module):
    drop = nn.Dropout(0.5) # dropout layer used to randomly select which weights aren't mutated from parent

    def __init__(self, is_child, parent_parameters, sigma):
        super(Policy, self).__init__()

        self.sigma = sigma
        self.affine1 = nn.Linear(24, 60, bias=False) # simple two layer linear model
        self.affine2 = nn.Linear(60, 4, bias=False)
        
        if is_child:
            for parent, child in zip(parent_parameters, self.parameters()):
                # init random tensor w/ same shape as parameter. mean 0, std sigma
                to_add = torch.normal(0.0, sigma, child.size())
                # randomly zero half of the to_add tensor
                Policy.drop(to_add)
                # in-place zero child's initial tensor, add the values from parent and mutating tensor
                child.data.zero_().add_(parent.data.clone()).add_(to_add)


    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        tanh = nn.Tanh() # tanh so output fits the environments action space values
        return tanh(action_scores)

def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    return probs.detach().numpy()[0]

##################### TRAINING #######################

def train():
    env = gym.make('BipedalWalker-v2')
    state = env.reset()

    policies = []
    policy_rewards = []
    best_all_time_reward = -999
    best = None
    successful_sigma = []

    render = False

    start = 0
    end = 0

    first = True
    for episode in range(10000):
        if first:
            # initial generation, all random policies
            for _ in range(100):
                policy = Policy(False, [], 0.0)
                running_reward = 0
                stopped_frames = 0
                while True:
                    action = select_action(policy, state)
                    state, reward, done, _ = env.step(action)
                    running_reward += reward

                    if abs(state[2]) < 0.03:
                        stopped_frames += 1
                    else:
                        stopped_frames = 0
                    if stopped_frames >= 100:
                        done = True
                        running_reward -= 100

                    if done:
                        policies.append(policy)
                        policy_rewards.append(running_reward)
                        state = env.reset()
                        break
            first = False
        else:
            # get best policy from previous episode, print results
            best_reward = max(policy_rewards)
            i_of_best = policy_rewards.index(best_reward)
            print('--------- GENERATION {} ---------'.format(episode))
            print('Mean Reward: {}\nBest Reward: {}'.format(mean(policy_rewards), best_reward))
            print('Time Elapsed: {} seconds\n'.format(end - start))
            print('All time best: {}'.format(best_all_time_reward))
            if len(successful_sigma) > 0: print('Mean Successful Sigma: {}\n'.format(mean(successful_sigma)))

            # render condition
            if best_reward > 50: render = True

            # get best policy from storage
            if best_reward > best_all_time_reward:
                best = policies[i_of_best]
                best_all_time_reward = best_reward
                successful_sigma.append(best.sigma)
            ep_first = True

            # clear previous generation
            del policies[:]
            del policy_rewards[:]
            start = time.time()

            # test new generation
            for _ in range(15):
                # construct policy
                if ep_first:
                    policy = best
                else:
                    policy = Policy(True, best.parameters(), random() * 0.7)

                # run episode
                running_reward = 0
                stopped_frames = 0
                while (True):
                    if ep_first and render: env.render()
                    action = select_action(policy, state)
                    state, reward, done, _ = env.step(action)
                    running_reward += reward

                    # kill stuck runners
                    if abs(state[2]) < 0.03:
                        stopped_frames += 1
                    else:
                        stopped_frames = 0
                    if stopped_frames >= 100:
                        done = True
                        running_reward -= 100

                    # store reward/policy
                    if done:
                        policies.append(policy)
                        policy_rewards.append(running_reward)
                        env.reset()
                        if ep_first:
                            ep_first = False
                        break
            end = time.time()
    env.close()
train()