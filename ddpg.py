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
INPUT_DIM = 4
ACTIONS_DIM = 1

pp = pprint.PrettyPrinter()

class Policy():
    def __init__(self, sess, input_dim=INPUT_DIM, actions_dim=ACTIONS_DIM):
        self.sess = sess
        self.input_dim = input_dim
        self.actions_dim = actions_dim

        minv = -.001
        maxv = .001




class Value():
    def __init__(self, sess, input_dim=INPUT_DIM, actions_dim=ACTIONS_DIM):
        self.sess = sess
        self.input_dim = input_dim
        self.actions_dim = actions_dim

        minv = -.001
        maxv = .001

        num_layers = 3
        layers = []
        # fully connected layers?
        # full, inputs, and N outputs
        outputs = [100, 50]

        # x, policy(x) -> q(x) elt. of Reals
        # x, a, r -> gradient(x, a, theta)
        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')
        self.a = tf.placeholder(tf.float32, [None, self.actions_dim], name='a')

        # TODO: check if S X A is the best way of doing this
        # produce layers

        self.input = tf.concat([self.x, self.a], 1)

        prev_out = self.input
        prev_n = input_dim + actions_dim
        layers = [self.input]
        for i, n in enumerate(outputs):
            wt = tf.Variable(tf.random_uniform([prev_n, n], minval=minv, maxval=maxv))
            bias = tf.Variable(tf.random_uniform([n], minval=minv, maxval=maxv))
            layer = tf.matmul(prev_out, wt) + bias
            if i < len(outputs) - 1:
                layer = tf.nn.relu(layer)
            layers.append(layer)
            prev_n = n
            prev_out = layers[-1]



class Pendulum():
    def __init__(self, env, model, render=False):
        self.env = env
        self.model = model
        self.render = render

    def run_episode(self):
        obs = self.env.reset()
        for i in range(MAX_STEPS_PER_EPISODE):
            if self.render:
                self.env.render()
            action = self.model.suggest([obs])[0]
            new_obs, reward, done, _ = self.env.step(action)
            real_reward = int(not done)
            new_memory = (obs, action, new_obs, float(real_reward))
            self.memory.append(new_memory)
            obs = new_obs
            if done:
                break
        return i


class Trainer():
    def __init__(self, policy, value, env):
        self.policy = policy
        self.value = value
        self.env = env


if __name__ == '__main__':
    sess = tf.Session()
    env = gym.make('Pendulum-v0')
    model = Policy(sess)
    value = Value(sess)
    
    pend = Pendulum(env, model, render=False)

