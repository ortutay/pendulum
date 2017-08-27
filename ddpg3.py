import time
import sys
import random
import gym
import numpy as np
import tensorflow as tf
import pprint
import tempfile
import os
import shutil
import json

from collections import defaultdict
from gym import wrappers

MAX_STEPS_PER_EPISODE = 500
NUM_EPISODES = 150
OPEN_AI_KEY = os.environ.get('OPEN_AI_KEY')
INPUT_DIM = 3
ACTION_DIM = 1
DISCOUNT_RATE = .98

pp = pprint.PrettyPrinter()


class Actor:
    def __init__(self, sess):
        self.sess = sess

    def get_actions(self, states):
        # TODO: implement this
        return [[0] for i in range(len(states))]


class Critic:
    def __init__(self, sess, state_dim=3, action_dim=1, hidden_layers=[100, 50]):
        self.sess = sess
        input_dim = state_dim + action_dim
        layers = hidden_layers + [1]

        # TODO: use keras?

        self.weights = []
        self.biases = []

        minv = -.5
        maxv = .5
        nn_transform = tf.nn.relu

        # Define the critic NN which takes state + action pair, and outputs
        # an estimate for the reward.
        self.input_state_action = tf.placeholder(
            tf.float32, [None, input_dim], name='critic/input_state_action')
        prev_out = self.input_state_action
        prev_n = input_dim
        with tf.name_scope('critic'):
            for i, n in enumerate(layers):
                is_last_layer = (i == len(layers) - 1)
                wt = tf.Variable(
                    tf.random_uniform([prev_n, n], minval=minv, maxval=maxv),
                    name='weights/%s/%s' % (i, n))
                bias = tf.Variable(
                    tf.random_uniform([n], minval=minv, maxval=maxv),
                    name='bias/%s/%s' % (i, n))
                layer = tf.matmul(prev_out, wt) + bias
                if not is_last_layer:
                    layer = nn_transform(layer)
                self.weights.append(wt)
                self.biases.append(bias)
                prev_n = n
                prev_out = layer
        self.out = prev_out

        # Define another NN which is used for training. This shares weights with
        # the previous NN, but will have a separate input
        self.input_next_state_action = tf.placeholder(
            tf.float32, [None, input_dim], name='critic/input_next_state_action')
        prev_out = self.input_next_state_action

        # TODO: could somehow consolidate this code
        with tf.name_scope('critic_next'):
            for i, n in enumerate(layers):
                is_last_layer = (i == len(layers) - 1)
                wt = self.weights[i]
                bias = self.biases[i]
                layer = tf.matmul(prev_out, wt) + bias
                if not is_last_layer:
                    layer = nn_transform(layer)
                self.weights.append(wt)
                self.biases.append(bias)
                prev_n = n
                prev_out = layer
        self.out_next = prev_out

        self.input_reward = tf.placeholder(
            tf.float32, [None, 1], name='critic/loss/reward')
        self.predicted_total_reward = self.input_reward + DISCOUNT_RATE * self.out_next

        # TODO: in the paper this is written as:
        #
        #   self.predicted_total_reward - tf.pow(self.out, 2)
        #
        # which seems like a typo? but take another look at this...
        self.loss = tf.pow(self.predicted_total_reward - self.out, 2)
        # self.loss = self.predicted_total_reward - tf.pow(self.out, 2)

        self.train_step = tf.train.AdamOptimizer(.01).minimize(self.loss)
        # self.train_step = tf.train.GradientDescentOptimizer(.01).minimize(self.loss)

    def eval(self, states, actions):
        state_action = np.concatenate([states, actions], axis=0)
        return self.sess.run(self.out, feed_dict={
            self.input_state_action: [state_action],
        })

    def eval_next(self, rewards, next_states, next_actions):
        next_state_actions = np.concatenate([next_states, next_actions], axis=0)
        return self.sess.run(self.predicted_total_reward, feed_dict={
            self.input_next_state_action: [next_state_actions],
            self.input_reward: [rewards],
        })

    def eval_loss(self, states, actions, rewards, next_states, next_actions):
        state_action = np.concatenate([states, actions], axis=0)
        next_state_actions = np.concatenate([next_states, next_actions], axis=0)
        return self.sess.run(self.loss, feed_dict={
            self.input_state_action: [state_action],
            self.input_next_state_action: [next_state_actions],
            self.input_reward: [rewards],
        })

    def train(self, memory, actor, num):
        states = memory.get_column('state', num)
        actions = memory.get_column('action', num)
        rewards = memory.get_column('reward', num)
        next_states = memory.get_column('next_state', num)
        next_actions = actor.get_actions(next_states)

        state_actions = np.concatenate([states, actions], axis=1)
        next_state_actions = np.concatenate([next_states, next_actions], axis=1)

        return self.sess.run(self.train_step, feed_dict={
            self.input_state_action: state_actions,
            self.input_next_state_action: next_state_actions,
            self.input_reward: rewards,
        })


class Pendulum:
    def __init__(self, env, actor, critic, memory, render=True):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.render = render

    def run_episode(self, max_steps=MAX_STEPS_PER_EPISODE, debug=False):
        prev_obs = self.env.reset()
        total_reward = 0
        for i in range(max_steps):
            if self.render:
                self.env.render()

            action = self.actor.get_actions([prev_obs])[0]
            critic_value = self.critic.eval(prev_obs, action)[0][0]

            obs, reward, done, _ = self.env.step(action)
            self.memory.save(prev_obs, action, reward, obs)

            next_action = self.actor.get_actions([obs])[0]
            critic_next_value = self.critic.eval_next([reward], obs, next_action)[0][0]
            critic_loss = self.critic.eval_loss(prev_obs, action, [reward], obs, next_action)[0][0]

            print('obs: %s, action: %s, reward: %s, total_reward: %s, critic says: %s, next: %s, loss: %s' % (
                obs, action, reward, total_reward, critic_value, critic_next_value, critic_loss))
            prev_obs = obs
            total_reward += reward

            if i % 10 == 0:
                self.critic.train(self.memory, self.actor, 100)


class Memory:
    def __init__(self):
        self.memories = []

    def save(self, state, action, reward, next_state):
        # TODO: use numpy arrays/make this more efficient
        self.memories.append({
            'state': list(state),
            'action': list(action),
            'reward': [reward],
            'next_state': list(next_state),
        })

    def get_column(self, col_name, num):
        rows = np.random.choice(np.array(self.memories), num)
        return [row[col_name] for row in rows]


if __name__ == '__main__':
    sess = tf.Session()
    env = gym.make('Pendulum-v0')

    critic = Critic(sess)
    actor = Actor(sess)
    memory = Memory()
    pendulum = Pendulum(env, actor, critic, memory, render=False)

    sess.run(tf.global_variables_initializer())
    num_episodes = 50
    for i in range(num_episodes):
        pendulum.run_episode(max_steps=200)
