from pickle import load
from random import random
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import sigmoid, linear, relu
from keras.optimizers import Adam, SGD


class Learner:
    '''
    This agent jumps randomly.
    '''

    def __init__(self, scaler, epsilon=1, eps_decay=0.999, gamma=0.1):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gamma = 1
        self.eps = epsilon
        self.eps_decay = eps_decay
        self.scaler = scaler
        self.optimizer = None
        self.q_network = self._build_q_network()
        self.g = 2.5
        self.t = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.g = 2.5
        self.t += 1

    def _build_q_network(self):
        activ = 'relu'
        # simple linear regression
        q_network = Sequential()
        q_network.add(Dense(256, activation=activ))
        q_network.add(Dense(256, activation=activ))
        q_network.add(Dense(256, activation=activ))
        q_network.add(Dense(256, activation=activ))
        q_network.add(Dense(256, activation=activ))
        q_network.add(Dense(256, activation=activ))
        q_network.add(Dense(1))


        q_network.compile(loss="mse", optimizer=Adam(0.0000005))
        return q_network

    def _gravity(self, state):
        if self.last_action == 0:
            self.g = state['monkey']['vel'] - self.last_state['monkey']['vel']
        return self.g


    def _process_state(self, state):
        _state = np.array([state["tree"]["dist"],
                          state["tree"]["top"],
                          state["tree"]["bot"],
                          state["monkey"]["vel"],
                          state["monkey"]["top"],
                          state["monkey"]["bot"]], ndmin=2)

        _state = list(self.scaler.transform(_state)[0])
        _state.append(self._gravity(state))
        return _state


    def epsilon_greedy(self, q_vals):
        self.eps *= self.eps_decay
        if random() < self.eps:
            return npr.randint(0, 1)
        else:
            return np.argmax(q_vals)


    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        if self.last_state is not None:
            A = [self.last_action]
            S = self._process_state(self.last_state)
            r = self.last_reward

            S_ = self._process_state(state)
            q_vals = [self.q_network.predict(np.array(S_ + [0]*7, ndmin=2)),
                      self.q_network.predict(np.array([0]*7 + S_, ndmin=2))]

            if A == 0:
                S_p = S + [0] * 7
            else:
                S_p = [0] * 7 + S
            pred = self.q_network.predict(np.array(S_p, ndmin=2))
            target = np.array([r + self.gamma * np.max(q_vals)])


            self.q_network.fit(np.array(S_p, ndmin=2), target, verbose=0, batch_size=16)
            print(r, pred, target)

            self.last_action = self.epsilon_greedy(q_vals)
            self.last_state  = state
        else:
            self.last_action = 1
            self.last_state  = state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):

        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    scaler = load(open("scaler.pkl", "rb"))
    agent = Learner(scaler)

    # Empty list to save history.
    hist = []
    # Run games.
    run_games(agent, hist, 5000, 1)
    np.save('NN_256_units_lr_5e-7_hist_batchsize_16', np.array(hist))
    print(max(hist))
