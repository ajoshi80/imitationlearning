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

env = gym.make(config.env_name)
env = MaxAndSkipEnv(env, skip=config.skip_frame)
env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                    overwrite_render=config.overwrite_render)

rewards = []

experts_meta_lis = ['./checkpoints/q_learning/skip_connection/q5_train_atari_nature/longres_weights/.meta']#['./checkpoints/q_learning/skip_connection/q5_train_atari_nature/deepdqn_weights/.meta']#, './checkpoints/q_learning/skip_connection/q5_train_atari_nature/deepdqn_weights/.meta']
    #'./core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/deepdqn_weights/.meta', './core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/resnet_weights/.meta', './core/checkpoints/policy_gradients/policy_network.ckpt.meta']
experts_chkpt_lis = ['./checkpoints/q_learning/skip_connection/q5_train_atari_nature/longres_weights/']#['./checkpoints/q_learning/skip_connection/q5_train_atari_nature/deepdqn_weights/']#, './checkpoints/q_learning/skip_connection/q5_train_atari_nature/deepdqn_weights/
    #'./core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/deepdqn_weights/', './core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/resnet_weights/', './core/checkpoints/policy_gradients/policy_network.ckpt']
head = 'res'
npz_file = './longres_demos.npz'

states_info = np.load(npz_file)['demos']
num_states, feats = states_info.shape
output_states_info = np.zeros((num_states, feats+2), dtype=states_info.dtype)
output_states_info[:,:feats] = states_info.copy()
#temp_sess = None
for meta_path, chkpt_path in zip(experts_meta_lis, experts_chkpt_lis):
    if "dqn" in meta_path:
        model = NatureQN(env, config)
    if "res" in meta_path:
        model = ResnetQN(env, config)
    if "policy" in meta_path:
        continue
    model.initialize(meta_path, chkpt_path)

print("LOADED ALL MODELS")
print("NUM STATES IS: " + str(num_states))
guide = model
for i in range(num_states):
    guide_replay_buffer = ReplayBuffer(config.buffer_size, config.state_history)
    data = np.load(str(states_info[i,0]))
    start_state = data['arr_0']
    env_state = data['env_hist']
    for j in range(3):
        _ = guide_replay_buffer.store_frame(start_state[:,:,j:j+1]) 
    state = env.reset()
    state = start_state[:,:,3:]
    env.unwrapped.restore_full_state(env_state)
    iter_num = 0
    init_action = -1
    reward_lis = []
    while True:
            # store last state in buffer
        idx = guide_replay_buffer.store_frame(state)
        q_input = guide_replay_buffer.encode_recent_observation()
        action, _ = guide.get_best_action(q_input)
        if iter_num == 0:
            init_action = action
        # perform action in env
        new_state, reward, done, info = env.step(action)
        reward_lis.append(reward)
        # store in replay memory
        guide_replay_buffer.store_effect(idx, action, reward, done)
        #guide_experience[num_points].append((state, action, 0))
        state = new_state
        iter_num += 1
        if abs(reward) == 1:
            #print("REWARD IS: " + str(reward))
            break
    weighted_reward = reward_lis[0] * config.gamma
    for j in range(1, len(reward_lis)):
        weighted_reward += (config.gamma**(j+1))*reward_lis[j]
    print("before action {} reward {}".format(output_states_info[i, 1], output_states_info[i, 2]))
    print('after action {} reward {}'.format(init_action, weighted_reward))
    output_states_info[i, feats] = init_action
    output_states_info[i, feats+1] = weighted_reward
#np.savez(npz_file, demos=output_states_info)
