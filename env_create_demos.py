import os
import gym
import numpy as np
import logging
import time
import sys
import scipy
from gym import wrappers
from collections import deque
import tensorflow as tf
from q2_linear import Linear
import csv

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from core.deep_q_learning import DQN
from resnet_dqn import ResnetQN
from q3_nature import NatureQN
from configs.q5_train_atari_nature import config


def playPoint(expert, state):
    experts_replay_buffer = ReplayBuffer(
        config.buffer_size, config.state_history)
    counter = 0
    initial_action = -1
    while True:
        idx = experts_replay_buffer.store_frame(state)
        q_input = experts_replay_buffer.encode_recent_observation()
        action, _ = expert.get_best_action(q_input)
        if counter == 0:
            initial_action = action
        # perform action in env
        new_state, reward, done, info = env.step(action)
        # store in replay memory
        state = new_state
        experts_replay_buffer.store_effect(idx, action, reward, done)
        # count reward
        if abs(reward) == 1:
            break
        counter += 1
    print("PLAY POINT ENDED")
    return (config.gamma**counter) * reward, initial_action


env = gym.make(config.env_name)
env = MaxAndSkipEnv(env, skip=config.skip_frame)
env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                    overwrite_render=config.overwrite_render)

rewards = []

head = 'longres'
experts_meta_lis = ['./checkpoints/q_learning/skip_connection/q5_train_atari_nature/longres_weights/.meta']#['./checkpoints/q_learning/skip_connection/q5_train_atari_nature/resnet_weights/.meta']#, './checkpoints/q_learning/skip_connection/q5_train_atari_nature/deepdqn_weights/.meta']
    #'./core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/deepdqn_weights/.meta', './core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/resnet_weights/.meta', './core/checkpoints/policy_gradients/policy_network.ckpt.meta']
experts_chkpt_lis = ['./checkpoints/q_learning/skip_connection/q5_train_atari_nature/longres_weights/']#['./checkpoints/q_learning/skip_connection/q5_train_atari_nature/resnet_weights/']#, './checkpoints/q_learning/skip_connection/q5_train_atari_nature/deepdqn_weights/']
    #'./core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/deepdqn_weights/', './core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/resnet_weights/', './core/checkpoints/policy_gradients/policy_network.ckpt']
experts = []

#temp_sess = None
for meta_path, chkpt_path in zip(experts_meta_lis, experts_chkpt_lis):
    if "dqn" in meta_path:
        model = NatureQN(env, config)
    if "res" in meta_path:
        model = ResnetQN(env, config)
    if "policy" in meta_path:
        continue
    # if temp_sess == None:
    #temp_sess = model.sess
    model.initialize(meta_path, chkpt_path)
    experts.append(model)
    # with model.graph.as_default():

print("LOADED ALL MODELS")

for i in range(len(experts)):
    guide = experts[i]
    guide_experience = [[]]
    env_history = [[]]
    num_points = 0
    num_games = 1
    for j in range(num_games):
        state = env.reset()
        guide_replay_buffer = ReplayBuffer(config.buffer_size, config.state_history)
        while True:
                # store last state in buffer
            idx = guide_replay_buffer.store_frame(state)
            q_input = guide_replay_buffer.encode_recent_observation()
            action, _ = guide.get_best_action(q_input)
            # perform action in env
            new_state, reward, done, info = env.step(action)
            # store in replay memory
            guide_replay_buffer.store_effect(idx, action, reward, done)
            if len(guide_experience) <= num_points:
                guide_experience.append([])
                env_history.append([])
            guide_experience[num_points].append((q_input, action, 0))
            env_history[num_points].append(env.unwrapped.clone_full_state())
            state = new_state
            if abs(reward) == 1:
                print("REWARD IS: " + str(reward))
                cur_point_lis = guide_experience[num_points]
                for k in range(len(cur_point_lis) - 1, -1, -1):
                    cur_point_lis[k] = (cur_point_lis[k][0], cur_point_lis[k][1], config.gamma**(len(cur_point_lis) - k - 1) * reward)
                #for k in range(len(cur_point_lis)):
                #    index = int(len(cur_point_lis) - k - 1)
                #    if k == 0:
                #        cur_point_lis[index] = (
                #            cur_point_lis[index][0], cur_point_lis[index][1], reward)
                #    else:
                #        cur_point_lis[index] = (cur_point_lis[index][0], cur_point_lis[index][1], float(cur_point_lis[index][2]) +
                #                                config.gamma * cur_point_lis[index+1][2])
                guide_experience[num_points] = cur_point_lis
                num_points += 1
                #if num_points == 20:
                #    break
            if done:
                break
            # updates to perform at the end of an episode
    rows = []
    for point_index, point in enumerate(guide_experience):
        for state_index, (state, guide_action, guide_reward) in enumerate(point):
            env_state = env_history[point_index][state_index]
            row = ['./state_images/' + str(head) + '_' + str(point_index) + '_' + str(
                state_index) + '.npz', guide_action, guide_reward]
            np.savez('./state_images/'+ str(head) + '_' + str(point_index) + '_' + str(state_index), state, env_hist = env_state)
            #for j in range(len(experts)):
            #    if j == i:
            #        continue
            #    expert = experts[j]
            #    expert_reward, expert_action = playPoint(expert, state)
            #    print("EXPERT REWARD IS: " + str(expert_reward))
            #    row.append(expert_action)
            #    row.append(expert_reward)
            rows.append(row)
    #rows = np.array(rows)
    np.savez(str(head)+'_demos.npz', demos=rows)
