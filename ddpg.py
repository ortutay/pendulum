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
DISCOUNT_RATE = .4

pp = pprint.PrettyPrinter()


class NeuralNetwork(object):
    def __init__(self, sess, input_dim, outputs, transform='none', name=None):
        self.sess = sess
        self.input_dim = input_dim

        minv = -.5
        maxv = .5

        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')

        # fully connected nn
        prev_out = self.x
        prev_n = input_dim
        self.weights = []
        self.biases = []
        self.layers = [self.x]
        with tf.name_scope(name):
            for i, n in enumerate(outputs):
                print('Layer %s %s %s' % (name, i, n))
                is_last_layer = (i == len(outputs) - 1)
                wt = tf.Variable(
                    tf.random_uniform([prev_n, n], minval=minv, maxval=maxv),
                    name='weights/%s/%s' % (i, n))
                if is_last_layer:
                    bias = tf.Variable(
                        tf.random_uniform([n], minval=100*minv, maxval=0),
                        name='bias/%s/%s' % (i, n))
                else:
                    bias = tf.Variable(
                        tf.random_uniform([n], minval=minv, maxval=maxv),
                        name='bias/%s/%s' % (i, n))
                layer = tf.matmul(prev_out, wt) + bias
                if not is_last_layer:
                    print('Applying transform: %s' % transform)
                    if (transform == 'relu'):
                        layer = tf.nn.relu(layer)
                    if (transform == 'none'):
                        pass
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

    NOISE_RATIO = .3

    def __init__(self, sess, state_dim=3, action_dim=1, hidden_layers=[100, 50]):
        self.sess = sess
        self.nn = NeuralNetwork(
            sess,
            state_dim,
            hidden_layers + [action_dim],
            transform='relu',
            name='actor')

    def eval(self, states):
        # # FOR DEBUG
        if True:
            size = (len(states), 1)
            x = np.full(size, -2) + 2.1
            # print(x)
            return x

        if random.random() < self.NOISE_RATIO:
            size = (len(states), 1)
            return np.full(size, -2) + 4 * np.random.rand(*size)
        a = self.nn.eval(states)
        return a

    def grad_weights(self, x_data, grad_ys=None):
        actor_trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')

        # TODO: Figure out how to do this without iterating!!
        for x_data_iter in x_data:
            grad = tf.gradients(self.nn.out, actor_trainable_variables, grad_ys=grad_ys)
            return sess.run(grad, {self.nn.x: [x_data_iter]}), actor_trainable_variables

    def train(self, memories, actor, critic):
        (x, x_, a, r) = memories
        critic_data = np.concatenate([x, a], axis=1)
        grad_action = critic.grad_action(critic_data)
        grad_weights, grad_vars = self.grad_weights(x, -grad_action)
        train_step = tf.train.GradientDescentOptimizer(.001).apply_gradients(
            zip(grad_weights, grad_vars))
        self.sess.run(train_step)

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

    DISCOUNT_RATE = 0

    def __init__(self, sess, state_dim=3, action_dim=1, hidden_layers=[100, 50]):
        self.outputs_dim = 1
        self.nn = NeuralNetwork(
            sess,
            state_dim + action_dim,
            hidden_layers + [self.outputs_dim],
            transform='tanh',
            name='critic')
        self.state_dim = state_dim

        self.targets = tf.placeholder(tf.float32, [None, self.outputs_dim])
        self.loss = tf.pow(self.nn.out - self.targets, 2)
        self.train_step = tf.train.GradientDescentOptimizer(.05).minimize(self.loss)
        # self.train_step = tf.train.AdamOptimizer(.005).minimize(self.loss)

    def eval(self, states, actions):
        state_action = np.concatenate([states, actions], axis=1)
        result = self.nn.eval(state_action)
        return result

    def print_debug(self, memories, actor, critic):
        loss = self.get_loss(
            memories=memories,
            actor=actor,
            critic=critic)
        print('Loss', loss)

        targets = self._targets(memories, actor, critic)
        (x, x_, a, r) = memories
        predictions = self.eval(x, a)
        print('targets', targets[:3])
        print('predictions', predictions[:3])
        # import pdb; pdb.set_trace()

    def _targets(self, memories, actor, critic):
        (x, x_, a, r) = memories
        a_ = actor.eval(x_)
        q_ = critic.eval(x_, a_)
        return r + self.DISCOUNT_RATE * (q_)

        # TODO: convert to log space?
        # return np.tanh(r + self.DISCOUNT_RATE * (q_))

    def get_loss(self, memories, actor, critic):
        (x, x_, a, r) = memories
        targets = self._targets(memories, actor, critic)

        return sum(self.nn.sess.run(self.loss, feed_dict={
            self.targets: targets,
            self.nn.x: np.concatenate([x, a], axis=1)
        })) / len(memories)

    # np arrays
    def train(self, memories, actor, critic):
        (x, x_, a, r) = memories
        targets = self._targets(memories, actor, critic)

        # calculate and apply gradient
        self.nn.sess.run(self.train_step, feed_dict={
            self.targets: targets,
            self.nn.x: np.concatenate([x, a], axis=1)
        })

    def grad_action(self, x_data):
        grad = tf.gradients(self.nn.out, self.nn.x)
        grad_action = tf.slice(grad, [0, 0, self.state_dim], [1, len(x_data), 1])
        return sess.run(grad_action, {self.nn.x: x_data})


class Pendulum(object):
    def __init__(self, env, actor, critic, memory, render=True):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.render = render

    def run_episode(self, max_steps=MAX_STEPS_PER_EPISODE, debug=False):
        obs = self.env.reset()
        total_reward = 0
        for i in range(max_steps):
            if self.render:
                self.env.render()

            action = self.actor.eval([obs])[0]
            new_obs, reward, done, _ = self.env.step(action)
            event = (obs, new_obs, action, reward)
            total_reward += reward
            self.memory.store(event)

            # print('Stored', event)

            target_actor = actor
            target_critic = critic

            # print('expected q-value: %s, reward: %s' % (critic.eval([new_obs], [action]), reward))
            # print('event', i, action, reward, total_reward)

            if debug and (i and i % 10 == 0):
                # print('Training critic')
                memories = self.memory.retrieve(500)
                self.critic.train(
                    memories=memories,
                    actor=target_actor,
                    critic=target_critic)

                self.critic.print_debug(
                    memories=memories,
                    actor=target_actor,
                    critic=target_critic)

                q = self.critic.eval([obs], [action])
                print('Q is:', q)

                # print('Training actor')
                # self.actor.train(
                #     memories=memories,
                #     actor=target_actor,
                #     critic=target_critic)

            # if i % 100 == 0:
            #     import pdb; pdb.set_trace()

            # self.critic.train(
            #     x=[obs],
            #     x_=[new_obs],
            #     a=[action],
            #     r=[float(real_reward)],
            #     actor=target_actor,
            #     critic=target_critic)
            obs = new_obs
        if debug:
            print('Total reward was', total_reward)
        return total_reward


# (x, a, x_, r)
class Memory():
    def __init__(self, state_dim=3, action_dim=1, maxlength=5):
        self.maxlength = maxlength
        self.length = 0
        self.x = np.zeros((maxlength, state_dim))
        self.x_ = np.zeros((maxlength, state_dim))
        self.a = np.zeros((maxlength, action_dim))
        self.r = np.zeros((maxlength, 1))

    # single values
    def store(self, memory):
        if (self.length >= self.maxlength):
            return

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
        return [
            self.x[(choices), ],
            self.x_[(choices), ],
            self.a[(choices), ],
            self.r[(choices), ]]


# class Trainer():
#     def __init__(self, policy, value, env):
#         self.policy = policy
#         self.value = value
#         self.env = env


if __name__ == '__main__':
    sess = tf.Session()
    env = gym.make('Pendulum-v0')
    actor = Actor(sess, state_dim=3, action_dim=1, hidden_layers=[10, 5])
    critic = Critic(sess, state_dim=3, action_dim=1, hidden_layers=[100])
    memory = Memory(state_dim=3, action_dim=1)

    count = 0.
    for i in range(10000):
        pendulum = Pendulum(env, actor, critic, memory)
        sess.run(tf.global_variables_initializer())
        total_reward = pendulum.run_episode(200, debug=True) #bool(i % 5 == 0))
        count += total_reward

        # n = 20
        # if i and i % n == 0:
        #     avg_reward = count / n
        #     print('avg_reward over last %s: %.02f' % (n, avg_reward))
        #     count = 0.

    # states_data = [
    #     # [100, 20, 30],
    #     [100, 20, 30],
    # ]
    # a_data = [
    #     # [10],
    #     [30],
    # ]

    # sess.run(tf.global_variables_initializer())

    # critic_data = np.concatenate([states_data, a_data], axis=1)
    # grad_action = critic.grad_action(critic_data)
    # print('Gradient of Critic over Action', grad_action.shape)

    # grad_weights = actor.grad_weights(states_data)
    # print('grad_weights', grad_weights)
    # print('Gradient of Actor over weights', [x.shape for x in grad_weights])

    # import pdb; pdb.set_trace()

    # grad_weights = actor.grad_weights(states_data, -grad_action)
    # print('grad_weights', grad_weights)
    # print('Gradient of Actor over weights with chain rule', [x.shape for x in grad_weights])
