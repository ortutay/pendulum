import tensorflow as tf
import gym
import numpy as np

import tflearn
from tflearn.models.dnn import DNN

MAX_STEPS_PER_EPISODE = 100


class Critic(object):

    def __init__(self, state_dim=3, action_dim=1):
        tflearn.init_graph()

        # TODO: Copy directly from tutorial?
        self.inputs = tflearn.input_data(shape=[None, state_dim])
        nn = tflearn.fully_connected(self.inputs, 128, activation='relu')
        nn = tflearn.fully_connected(nn, 64, activation='relu')
        nn = tflearn.fully_connected(nn, 32)
        self.out = tflearn.fully_connected(nn, 1)
        self.nn = nn

    def eval(self, sess, states, actions):
        dnn = DNN(self.nn)
        print('dnn', dnn)
        raise 'x'


class Pendulum(object):
    def __init__(self, env, sess, actor, critic, memory, render=True):
        self.env = env
        self.sess = sess
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

            # Take a step
            action = [.1]
            new_obs, reward, done, _ = self.env.step(action)
            event = (obs, new_obs, action, reward)
            total_reward += reward
            self.memory.store(event)
            obs = new_obs

            # Train the critic
            critic_out = self.critic.eval(self.sess, [obs])
            print('critic_out', critic_out)

        print('Total reward was', total_reward)
        return total_reward


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


if __name__ == '__main__':
    sess = tf.Session()
    env = gym.make('Pendulum-v0')
    memory = Memory(state_dim=3, action_dim=1)
    critic = Critic()

    for i in range(10000):
        pendulum = Pendulum(env, sess, None, None, memory)
        total_reward = pendulum.run_episode(200, debug=True)
