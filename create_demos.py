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
from resnet_dqn import NatureQN
#from q3_nature import NatureQN
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

experts_meta_lis = [
    './core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/resnet_weights/.meta']
experts_chkpt_lis = [
    './core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/resnet_weights/']
experts = []

model = NatureQN(env, config)
model.initialize()
experts.append(model)

for meta_path, chkpt_path in zip(experts_meta_lis, experts_chkpt_lis):
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(model.sess, chkpt_path)

for i in range(len(experts)):
    guide = experts[i]
    guide_experience = [[]]
    num_points = 0
    state = env.reset()
    guide_replay_buffer = ReplayBuffer(
        config.buffer_size, config.state_history)
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
        guide_experience[num_points].append((state, action, 0))
        state = new_state
        if abs(reward) == 1:
            cur_point_lis = guide_experience[num_points]
            for k in range(len(cur_point_lis)):
                index = int(len(cur_point_lis) - k - 1)
                if k == 0:
                    cur_point_lis[index] = (
                        cur_point_lis[index][0], cur_point_lis[index][1], reward)
                else:
                    cur_point_lis[index] = (cur_point_lis[index][0], cur_point_lis[index][1], float(cur_point_lis[index][2]) +
                                            config.gamma * cur_point_lis[index+1][2])
            guide_experience[num_points] = cur_point_lis
            num_points += 1
            if num_points == 20:
                break
        if done:
            break
        # updates to perform at the end of an episode
    rows = []
    for point_index, point in enumerate(guide_experience):
        for state_index, (state, guide_action, guide_reward) in enumerate(point):
            row = ['./state_images/'+str(i) + '_' + str(point_index) + '_' + str(
                state_index) + '.npz', guide_action, guide_reward]
            np.savez('./state_images/'+str(i) + '_' +
                     str(point_index) + '_' + str(state_index), state)
            for j in range(len(experts)):
                if j == i:
                    continue
                expert = experts[j]
                expert_reward, expert_action = playPoint(expert, state)
                row.append(expert_action)
                row.append(expert_reward)
            rows.append(row)
    rows = np.array(rows)
    np.savez('demonstrations.npz', demos=rows)
