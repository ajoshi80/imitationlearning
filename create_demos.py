import os
import gym
import numpy as np
import logging
import time
import sys
from gym import wrappers
from collections import deque
import tensorflow as tf
from q2_linear import Linear

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from core.deep_q_learning import DQN


class config():
    # env config
    render_train = False
    render_test = False
    env_name = "Pong-v0"
    overwrite_render = True
    record = True
    high = 255.

    # output config
    output_path = "results/q5_train_atari_nature/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    record_path = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50
    grad_clip = True
    clip_val = 10
    saving_freq = 250000
    log_freq = 50
    eval_freq = 250000
    record_freq = 250000
    soft_epsilon = 0.05

    # nature paper hyper params
    nsteps_train = 5000000
    batch_size = 32
    buffer_size = 1000000
    target_update_freq = 10000
    gamma = 0.99
    learning_freq = 4
    state_history = 4
    skip_frame = 4
    lr_begin = 0.00025
    lr_end = 0.00005
    lr_nsteps = nsteps_train/2
    eps_begin = 1
    eps_end = 0.1
    eps_nsteps = 1000000
    learning_start = 50000


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: 
            - You may find the following functions useful:
                - tf.layers.conv2d
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################

        with tf.variable_scope(scope, reuse=reuse):
            post_conv_1 = tf.layers.conv2d(
                state, filters=32, kernel_size=8, strides=4, padding='same', activation=tf.nn.relu)
            post_conv_2 = tf.layers.conv2d(
                post_conv_1, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
            post_conv_3 = tf.layers.conv2d(
                post_conv_2, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            skip_connection = tf.layers.conv2d(
                state, filters=64, kernel_size=3, strides=8, padding='same')

            post_conv_3 = post_conv_3 + skip_connection
            flat = tf.layers.flatten(post_conv_3)
            last = tf.layers.dense(flat, units=512, activation=tf.nn.relu)
            out = tf.layers.dense(last, units=num_actions)

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


num_episodes = 10

env = gym.make(config.env_name)
env = MaxAndSkipEnv(env, skip=config.skip_frame)
env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                    overwrite_render=config.overwrite_render)

rewards = []
replay_buffer = ReplayBuffer(config.buffer_size, config.state_history)
model = NatureQN(env, config)
model.initialize()
#saver = tf.train.Saver()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(
        './core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/model.weights/.meta')
    saver.restore(
        sess, './core/checkpoints/q_learning/skip_connection/q5_train_atari_nature/model.weights/')

for i in range(num_episodes):
    total_reward = 0
    state = env.reset()
    while True:
        #if self.config.render_test: env.render()

        # store last state in buffer
        idx = replay_buffer.store_frame(state)
        q_input = replay_buffer.encode_recent_observation()

        action, _ = model.get_best_action(q_input)
	print("STATE SHAPE: " + str(state.shape))
        # perform action in env
        new_state, reward, done, info = env.step(action)
        # store in replay memory
        replay_buffer.store_effect(idx, action, reward, done)
        state = new_state

        # count reward
        total_reward += reward
        if abs(reward) == 1:
            break

    # updates to perform at the end of an episode
    rewards.append(total_reward)

avg_reward = np.mean(rewards)
sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

if num_episodes > 1:
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
        avg_reward, sigma_reward)
    print(msg)
