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
ACTIONS_DIM = 1

pp = pprint.PrettyPrinter()


class NeuralNetwork(object):
    def __init__(self, sess, input_dim, outputs):
        self.sess = sess
        self.input_dim = input_dim

        minv = -.001
        maxv = .001

        num_layers = 3
        self.layers = []
        # fully connected layers?
        # full, inputs, and N outputs

        # x, policy(x) -> q(x) elt. of Reals
        # x, a, r -> gradient(x, a, theta)
        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')

        prev_out = self.x
        prev_n = input_dim
        layers = [self.x]
        for i, n in enumerate(outputs):
            wt = tf.Variable(tf.random_uniform([prev_n, n], minval=minv, maxval=maxv))
            bias = tf.Variable(tf.random_uniform([n], minval=minv, maxval=maxv))
            layer = tf.matmul(prev_out, wt) + bias
            if i < len(outputs) - 1:
                layer = tf.nn.relu(layer)
            layers.append(layer)
            prev_n = n
            prev_out = layers[-1]
        self.out = prev_out

        self.targets = tf.placeholder(tf.float32, [None, outputs[-1]])
        self.loss = .5 * tf.pow(self.out - self.targets, 2)
        self.train_step = tf.train.GradientDescentOptimizer(.05).minimize(self.loss)

    def eval(self, x):
        return self.sess.run(self.out, feed_dict={self.x: x})


class Actor(object):
    def __init__(self, sess, state_dim=3, action_dim=1, hidden_layers=[100, 50]):
        self.nn = NeuralNetwork(sess, state_dim, hidden_layers + [action_dim])

    def eval(self, states):
        return self.nn.eval(states)


class Critic(object):
    def __init__(self, sess, state_dim=3, action_dim=1, hidden_layers=[100, 50]):
        self.nn = NeuralNetwork(sess, state_dim + action_dim, hidden_layers + [1])

    def eval(self, states, actions):
        #print('states actions', states, actions)
        return self.nn.eval(np.concatenate([states, actions], axis=1))

    def train(self, prev_states, next_states, actions, rewards, computed_actions):
        print('all args', prev_states, next_states, actions, rewards, computed_actions)
        #print(next_states, computed_actions)
        targets = np.sum([self.eval(next_states, computed_actions), rewards], axis=0)
        print('targets', targets)
        self.nn.sess.run(self.nn.train_step, feed_dict={
            self.nn.targets: [targets],
            self.nn.x: np.concatenate([prev_states, actions], axis=1)
        })


class Pendulum(object):
    def __init__(self, env, actor, critic, render=False):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.render = render

    def run_episode(self):
        obs = self.env.reset()
        for i in range(MAX_STEPS_PER_EPISODE):
            if self.render:
                self.env.render()

            actions = self.actor.eval([obs])
            value = self.critic.eval([obs], actions)
            action = actions[0]

            new_obs, reward, done, _ = self.env.step(action)
            real_reward = int(not done)
            # new_memory = ()
#            memories.append(new_memory[0:4])
            self.critic.train(
                prev_states=[obs],
                next_states=[new_obs],
                actions=[action],
                rewards=[float(real_reward)],
                computed_actions=actor.eval([new_obs]))
            obs = new_obs
        return i


class Trainer():
    def __init__(self, policy, value, env):
        self.policy = policy
        self.value = value
        self.env = env


if __name__ == '__main__':
    sess = tf.Session()
    env = gym.make('Pendulum-v0')
    actor = Actor(sess, state_dim=3, action_dim=1, hidden_layers=[100, 50])
    critic = Critic(sess, state_dim=3, action_dim=1, hidden_layers=[100, 50])
    sess.run(tf.global_variables_initializer())
    pend = Pendulum(env, actor, critic, render=True)
    pend.run_episode()
