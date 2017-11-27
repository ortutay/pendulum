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
DISCOUNT_RATE = .9
TAU = .05
LEARNING_RATE = .001

init_hack = False

pp = pprint.PrettyPrinter()


class Actor:
    def __init__(self, sess, state_dim=3, action_dim=1, hidden_layers=[100, 50]):
        self.sess = sess
        self.optimizer = tf.train.GradientDescentOptimizer(.05)
        # self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        # sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='beta1_power')))

        input_dim = state_dim
        layers = hidden_layers + [action_dim]

        self.weights = []
        self.biases = []

        minv = -.01
        maxv = .01
        nn_transform = tf.nn.tanh

        # Define the actor NN which takes state, and outputs what it
        # thinks is the best action to take. Action space is continuous.
        self.input_state = tf.placeholder(tf.float32, [None, input_dim], name='input_state')
        prev_out = self.input_state
        prev_n = input_dim
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
        # Action should range from -2 to 2
        self.out = 2 * tf.nn.tanh(prev_out / 10)

    def get_actions(self, states):
        return self.sess.run(self.out, feed_dict={
            self.input_state: states,
        })

    def train(self, memory, target_critic, num, var_list):
        states = memory.get_column('state', num)
        loss = -target_critic.out
        train_step = self.optimizer.minimize(loss, var_list=var_list)
        self.sess.run(train_step, feed_dict={
            self.input_state: states,
            # target_critic.input_state: states,
        })


class Critic:
    def __init__(self, sess, input_state=None, input_action=None, state_dim=3, action_dim=1, hidden_layers=[100, 50]):
        self.sess = sess
        # self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        # sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='beta1_power')))
        self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

        input_dim = state_dim + action_dim
        layers = hidden_layers + [1]

        # TODO: use keras?

        self.weights = []
        self.biases = []

        minv = -.1
        maxv = .01
        nn_transform = tf.nn.tanh

        # Define the critic NN which takes state + action pair, and outputs
        # an estimate for the reward.
        if input_state is None:
            self.input_state = tf.placeholder(tf.float32, [None, state_dim], name='input_state')
        else:
            self.input_state = input_state
        if input_action is None:
            self.input_action = tf.placeholder(tf.float32, [None, action_dim], name='input_action')
        else:
            self.input_action = input_action
        input_state_action = tf.concat([self.input_state, self.input_action], axis=1)
        prev_out = input_state_action
        prev_n = input_dim
        for i, n in enumerate(layers):
            is_last_layer = (i == len(layers) - 1)
            wt = tf.Variable(
                tf.random_uniform([prev_n, n]),
                name='weights/%s/%s' % (i, n))
            bias = tf.Variable(
                tf.random_uniform([n]),
                name='bias/%s/%s' % (i, n))
            layer = tf.matmul(prev_out, wt) + bias
            if not is_last_layer:
                layer = nn_transform(layer)
            self.weights.append(wt)
            self.biases.append(bias)
            prev_n = n
            prev_out = layer
        self.out = prev_out

    def eval(self, states, actions):
        return self.sess.run(self.out, feed_dict={
            self.input_state: states,
            self.input_action: actions,
        })

    def train(self, memory, target_critic, target_actor, num, var_list):
        states = memory.get_column('state', num)
        rewards = memory.get_column('reward', num)
        actions = actor.get_actions(states)
        next_states = memory.get_column('next_state', num)
        next_actions = actor.get_actions(next_states)

        target_q_ph = tf.placeholder(tf.float32, [None, 1])
        target_q = rewards + DISCOUNT_RATE * self.eval(next_states, next_actions)

        # target_critic_out = target_critic.eval(next_states, next_actions)
        # out_val = self.sess.run(self.out, feed_dict={
        #     self.input_state: states,
        #     self.input_action: actions,
        # })
        # for i in range(len(out_val[:5])):
        #     print('out %s, target_q %s, target_critic_out %s' % (out_val[i], target_q[i], target_critic_out[i]))

        loss = tf.reduce_mean(tf.square(self.out - target_q_ph))
        train_step = self.optimizer.minimize(loss, var_list=var_list)
        self.sess.run(train_step, feed_dict={
            self.input_state: states,
            self.input_action: actions,
            target_q_ph: target_q,
        })


class Pendulum:
    def __init__(self, sess, env, actor, critic, target_actor, target_critic,
                 memory, render=False):
        self.sess = sess
        self.env = env
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.memory = memory
        self.render = render

    def run_episode(self, max_steps=MAX_STEPS_PER_EPISODE, debug=False, test_run=False):
        prev_obs = self.env.reset()
        total_reward = 0

        # Run the episode
        noise_rate = 0.5
        for i in range(max_steps):
            # if test_run:
            #     self.env.render()

            action = self.actor.get_actions([prev_obs])[0]
            if not test_run:
                # noise_rate *= noise_rate
                action = noise_rate * np.random.normal(-2, 2) + (1 - noise_rate) * action

            critic_value = self.critic.eval([prev_obs], [action])[0][0]
            obs, reward, done, _ = self.env.step(action)
            self.memory.save(prev_obs, action, reward, obs)
            if test_run:
                print('obs: %s, action: %s, reward: %s, total_reward: %s, critic says: %s' % (
                    obs, action, reward, total_reward, critic_value))
            prev_obs = obs
            total_reward += reward
            if done:
                break

        if test_run:
            print('total reward:', total_reward)
            return

        # Train
        for _ in range(min(10, int(self.memory.size() / 100))):
            n = 100
            critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_critic')
            actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_actor')
            target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')
            target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')

            # Train the actor
            self.actor.train(self.memory, self.target_critic, n, actor_vars)

            # Train the critic
            self.critic.train(self.memory, self.target_critic, self.target_actor, n, critic_vars)

            # Upate the target actor/critic
            update_ops = []
            for i in range(len(actor_vars)):
                update_ops.append(
                    target_actor_vars[i].assign((TAU * actor_vars[i] + (1 - TAU) * target_actor_vars[i])))
            for i in range(len(critic_vars)):
                update_ops.append(
                    target_critic_vars[i].assign((TAU * critic_vars[i] + (1 - TAU) * target_critic_vars[i])))
            self.sess.run(update_ops)


class Memory:
    def __init__(self):
        self.memories = []

    def save(self, state, action, reward, next_state):
        # TODO: use numpy arrays/make this more efficient
        item = {
            'state': list(state),
            'action': list(action),
            'reward': [reward],
            'next_state': list(next_state),
        }
        if len(self.memories) > 100000:
            self.memories[np.random.randint(len(self.memories))] = item
        else:
            self.memories.append(item)

    def get_column(self, col_name, num):
        rows = np.random.choice(np.array(self.memories), num)
        return [row[col_name] for row in rows]

    def size(self):
        return len(self.memories)


if __name__ == '__main__':
    sess = tf.Session()
    env = gym.make('Pendulum-v0')

    hidden_layers = [100, 50]

    with tf.name_scope('main_critic'):
        critic = Critic(sess, hidden_layers=hidden_layers)
    with tf.name_scope('main_actor'):
        actor = Actor(sess, hidden_layers=hidden_layers)

    with tf.name_scope('target_critic'):
        target_critic = Critic(
            sess, input_state=actor.input_state, input_action=actor.out, hidden_layers=hidden_layers)
    with tf.name_scope('target_actor'):
        target_actor = Actor(sess, hidden_layers=hidden_layers)

    # TODO: refactor this!
    critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_critic')
    actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_actor')
    target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')
    target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')

    memory = Memory()
    pendulum = Pendulum(
        sess, env, actor, critic, target_actor, target_critic,
        memory, render=False)

    sess.run(tf.global_variables_initializer())
    num_episodes = 1000
    for i in range(num_episodes):
        print(i)
        pendulum.run_episode(max_steps=200)
        if i % 2 == 0:
            pendulum.run_episode(max_steps=200, test_run=True)
