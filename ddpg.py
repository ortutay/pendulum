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
DISCOUNT_RATE = .4

pp = pprint.PrettyPrinter()


class NeuralNetwork(object):
    def __init__(self, sess, input_dim, outputs, transform='none'):
        self.sess = sess
        self.input_dim = input_dim

        minv = -.001
        maxv = .001

        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')

        # fully connected nn
        prev_out = self.x
        prev_n = input_dim
        self.weights = []
        self.biases = []
        self.layers = [self.x]
        for i, n in enumerate(outputs):
            wt = tf.Variable(tf.random_uniform([prev_n, n], minval=minv, maxval=maxv))
            bias = tf.Variable(tf.random_uniform([n], minval=minv, maxval=maxv))
            layer = tf.matmul(prev_out, wt) + bias
            if i < len(outputs) - 1:
                if (transform == 'relu'):
                    layer = tf.nn.relu(layer)
                if (transform == 'none'):
                    0
            self.weights.append(wt)
            self.biases.append(bias)
            self.layers.append(layer)
            prev_n = n
            prev_out = self.layers[-1]
        self.out = prev_out

        # self.targets = tf.placeholder(tf.float32, [None, outputs[-1]])
        # self.loss = .5 * tf.pow(self.out - self.targets, 2)
        # self.train_step = tf.train.GradientDescentOptimizer(.05).minimize(self.loss)

    def eval(self, x):
        return self.sess.run(self.out, feed_dict={self.x: x})


class Actor(object):
    def __init__(self, sess, state_dim=3, action_dim=1, hidden_layers=[100, 50]):
        self.nn = NeuralNetwork(sess, state_dim, hidden_layers + [action_dim])

    def eval(self, states):
        return self.nn.eval(states)

    def train(self, memory, actor, critic):
        (x, x_, a, r) = list(zip(*memory))
        # Steps:
        action_gradient = tf.placeholder(tf.float32, [None, ACTION_DIM])

        # (1) Get action gradient from critic
        action_gradients = critic.action_gradients()

        # (2) Get network gradients
        # (3) Combine the gradients using chain rule
        # (4) Apply using Adam optimizer


class Critic(object):
    def __init__(self, sess, state_dim=3, action_dim=1, hidden_layers=[100, 50]):
        self.outputs = 1
        self.nn = NeuralNetwork(sess, state_dim + action_dim, hidden_layers + [self.outputs])
        # x, policy(x) / a -> q(x) elt. of Reals
        # x, x_, a, r, (actor, critic) -> gradient(x, a, theta)
        self.targets = tf.placeholder(tf.float32, [None, self.outputs])
        self.loss = .5 * tf.pow(self.nn.out - self.targets, 2)
        self.train_step = tf.train.GradientDescentOptimizer(.05).minimize(self.loss)

    def eval(self, states, actions):
        state_action = np.concatenate([states, actions], axis=1)
        result = self.nn.eval(state_action)
        return result

    # np arrays
    def train(self, memory, actor, critic):
        (x, x_, a, r) = list(zip(*memory))
        # print('x, x_, a, r', x, x_, a, r)
        u = actor.eval(x_)
        q_ = critic.eval(x_, u)

        # TODO: math, convert to log space?
        discount_rate = .95
        targets = np.tanh(discount_rate * (r + q_))

        # calculate and apply gradient
        self.nn.sess.run(self.train_step, feed_dict={
            self.targets: targets,
            self.nn.x: np.concatenate([x, a], axis=1)
        })

    def action_gradients(self):
        raise NotImplemented('TODO')


class Pendulum(object):
    def __init__(self, env, actor, critic, memory, render=False):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.render = render

    def run_episode(self):
        obs = self.env.reset()
        for i in range(MAX_STEPS_PER_EPISODE):
            if self.render:
                self.env.render()

            action = self.actor.eval([obs])[0]
            new_obs, reward, done, _ = self.env.step(action)
            event = (obs, new_obs, action, reward)
            memory.store(event)

            target_actor = actor
            target_critic = critic

            print('expected q-value: %s, reward: %s' % (critic.eval([new_obs], [action]), reward))

            self.critic.train(
                memory=memory.retrieve(5),
                actor=target_actor,
                critic=target_critic)

            if i % 100 == 0:
                import pdb; pdb.set_trace()

            # self.critic.train(
            #     x=[obs],
            #     x_=[new_obs],
            #     a=[action],
            #     r=[float(real_reward)],
            #     actor=target_actor,
            #     critic=target_critic)
            obs = new_obs
        return i


# (x, a, x_, r)
class Memory():
    def __init__(self, state_dim, action_dim, maxlength=1000):
        self.maxlength = maxlength
        self.length = 0
        self.x = np.zeros((maxlength, state_dim))
        self.x_ = np.zeros((maxlength, state_dim))
        self.a = np.zeros((maxlength, action_dim))
        self.r = np.zeros((maxlength, 1))

    # single values
    def store(self, memory):
        (x, x_, a, r) = memory
        # print('storing:', x, x_, a, r)
        if (self.length < self.maxlength):
            self.x[self.length] = x
            self.x_[self.length] = x_
            self.a[self.length] = a
            self.r[self.length] = r
            self.length += 1
        else:
            # replace randomly for now
            i = np.random.randint(0, self.length)
            self.x[i] = x
            self.x_[i] = x_
            self.a[i] = a
            self.r[i] = r

    def retrieve(self, count):
        if (count > self.length):
            choices = np.arange(self.length)
        else:
            # replace?
            choices = np.random.choice(self.length, (count), replace=True)
        return zip(
            self.x[(choices), ],
            self.x_[(choices), ],
            self.a[(choices), ],
            self.r[(choices), ])


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

    states_data = [
        [1, 2, 3],
        [-2, 3, 4.5],
    ]
    # a = actor.eval(states_data)
    # print('the action is:', a)

    states = tf.placeholder(tf.float32, [None, 3])
    a_var = tf.get_variable("action", [1], dtype=tf.float32)
    loss = tf.reduce_sum(states + a_var)
    a_grad = tf.gradients(loss, [a_var])

    sess.run(tf.global_variables_initializer())

    print('A gradient', a_grad)
    print('A gradient runn:', sess.run(a_grad, {
        states: states_data,
    }))

    # result = sess.run({
    #     states: states_data,
    # })

    # memory = Memory(state_dim=3, action_dim=1, maxlength=100)
    # sess.run(tf.global_variables_initializer())
    # pend = Pendulum(env, actor, critic, memory, render=True)
    # pend.run_episode()
