'''
author: Thyrix Yang
github: https://github.com/ThyrixYang
'''

# Deep DPG (DDPG)

# I didn't do much experiment on the model and hyper-parameters,
# and this is a implementation without some tricks mentioned in the paper.
# The result on this env is well, but worse than my CEM implementation...

# paper:
#        https://arxiv.org/pdf/1509.02971.pdf


import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.slim as slim
import gym
import random
from collections import deque
from numpy.random import normal

discount_factor = 0.9

ob_len = 3
action_len = 1
hidden_size = 16

tau = 0.05
learn_rate = 1e-3

replay_memory = deque(maxlen=1000000)

def sample_from_memory(batch_size):
    return random.sample(replay_memory, batch_size)

def build_actor(state_input):
    actor_fc_1 = slim.fully_connected(state_input, hidden_size, activation_fn=tf.nn.relu)
    actor_fc_2 = slim.fully_connected(actor_fc_1, hidden_size, activation_fn=tf.nn.relu)
    actor_fc_3 = slim.fully_connected(actor_fc_2, hidden_size, activation_fn=tf.nn.relu)
    actor_fc_4 = slim.fully_connected(actor_fc_3, hidden_size, activation_fn=tf.nn.tanh)
    actor_output = slim.fully_connected(actor_fc_4, action_len, activation_fn=tf.nn.tanh) * 2
    return actor_output

def build_critic(state_input, action_input):
    critic_input = slim.flatten(tf.concat([state_input, action_input], axis=1))
    critic_fc_1 = slim.fully_connected(critic_input, hidden_size, activation_fn=tf.nn.relu)
    critic_fc_2 = slim.fully_connected(critic_fc_1, hidden_size, activation_fn=tf.nn.relu)
    critic_fc_3 = slim.fully_connected(critic_fc_2, hidden_size, activation_fn=tf.nn.tanh)
    critic_fc_4 = slim.fully_connected(critic_fc_3, hidden_size, activation_fn=tf.nn.tanh)
    critic_output = slim.fully_connected(critic_fc_4, 1, activation_fn=None)
    return critic_output

state_input_ph = tf.placeholder(tf.float32, shape=(None, ob_len))
action_input_ph = tf.placeholder(tf.float32, shape=(None, action_len))
target_q_ph = tf.placeholder(tf.float32, shape=(None, 1))

with tf.variable_scope("actor"):
    actor_output = build_actor(state_input_ph)

with tf.variable_scope("critic"):
    critic_output = build_critic(state_input_ph, action_input_ph)

with tf.variable_scope("target_actor"):
    target_actor_output = build_actor(state_input_ph)

with tf.variable_scope("target_critic"):
    target_critic_output = build_critic(state_input_ph, actor_output)

actor_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
critic_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
target_actor_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
target_critic_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')

update_target_ops = []
for i in range(len(actor_weights)):
    update_target_op = target_actor_weights[i].assign(tau*actor_weights[i] + (1-tau)*target_actor_weights[i])
    update_target_ops.append(update_target_op)
for i in range(len(critic_weights)):
    update_target_op = target_critic_weights[i].assign(tau*critic_weights[i] + (1-tau)*target_critic_weights[i])
    update_target_ops.append(update_target_op)

critic_lose = tf.reduce_mean(tf.square(target_q_ph-critic_output))
critic_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(critic_lose, var_list=critic_weights)

actor_lose = tf.reduce_mean(-target_critic_output)
actor_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(actor_lose, var_list=actor_weights)



def train():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def select_action(state):
        state = np.reshape(np.array(state), (-1, 3))
        action = sess.run(actor_output, {state_input_ph: state})
        return action

    def batch_updade(batch_size):
        batch = sample_from_memory(batch_size)
        state_0 = np.reshape(np.vstack([b[0] for b in batch]), (-1, 3))
        action_0 = np.reshape(np.vstack([b[1] for b in batch]), (-1, 1))
        reward_0 = np.reshape(np.vstack([b[2] for b in batch]), (-1, 1))
        state_1 = np.reshape(np.vstack([b[3] for b in batch]), (-1, 3))
        action_1 = sess.run(actor_output, {state_input_ph:state_1})
        q = sess.run(critic_output, {state_input_ph:state_1, action_input_ph:action_1})
        target_q = reward_0 + discount_factor*q
        lose, _ = sess.run([critic_lose, critic_optimizer],
                           {state_input_ph:state_0, 
                            action_input_ph:action_0,
                            target_q_ph:target_q})

        lose, _ = sess.run([actor_lose, actor_optimizer],
                           {state_input_ph:state_0})

        sess.run(update_target_ops)


    env = gym.make("Pendulum-v0")
    env = gym.wrappers.Monitor(env, '/tmp/experiment-4', force=True)
    epoch = 2000

    def test():
        observation = env.reset()
        accreward = 0
        while True:
            env.render()
            action = select_action(observation)
            observation, reward, done, info = env.step(action)
            accreward += reward
            if done:
                print("test end with reward: {}".format(accreward))
                break

    noise_std = 4
    noise_rate = 0.995
    for ep in range(epoch):
        observation = env.reset()
        print("at ep: {}".format(ep))
        noise_std *= noise_rate
        while True:
            action = select_action(observation) + normal(0, noise_std)
            new_observation, reward, done, info = env.step(action)
            new_observation = np.reshape(new_observation, (-1, 3))
            replay_memory.append([observation, action, reward, new_observation])
            observation = new_observation
            if done:
                break

        for _ in range(min(100, len(replay_memory) // 256)):
            batch_updade(128)

        if (ep % 10 == 0):
            print("start test at ep: {}".format(ep))
            test()



if __name__ == '__main__':
    train()
