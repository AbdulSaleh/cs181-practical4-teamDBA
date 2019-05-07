# Imports.
import numpy as np
import numpy.random as npr
#npr.seed(1984)
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.t = 0

        self.Q = {}
        self.alpha = 0.3
        self.eps = 0.0
        self.eps_min = 0.0
        self.eps_max = 0.5
        self.eps_decay = 100
        self.gamma = 1
        self.g = 0

        self.bins = 50

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.g = 0
        self.t += 1


    def discretize(self, state):
        B = self.bins
        return (int(state['tree']['dist'] / B),
                int((state['tree']['top']-state['monkey']['top']) / B),
                int(state['monkey']['vel'] / B),
                int((state['monkey']['bot'] - state['tree']['bot']) / B),
                self.g)

        #
        # return (int(state['tree']['dist'] / B),
        #         int(state['tree']['bot'] / B),
        #         int(state['monkey']['vel'] / B),
        #         int(state['monkey']['bot'] / B),
        #         self.g)


    def init_Q(self, S, A):
        if (S, A) not in self.Q:
            self.Q[(S, A)] = 0
        return


    def epsilon_greedy(self, S):
        self.eps = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * self.t/self.eps_decay)
        if npr.rand() < self.eps:
            return npr.choice([0, 1])
        else:
            return int(self.Q[(S, 0)] < self.Q[(S, 1)])


    def update_g(self, state):
        if self.g == 0:
            v = abs(state['monkey']['vel'])
            if v > 2:
                self.g = 4
            else:
                self.g = 1
        return


    def action_callback(self, state):
        A_ = 0

        if self.last_state is not None and self.last_action is not None:
            self.update_g(state)
            S = self.discretize(self.last_state)
            A = self.last_action
            r = self.last_reward

            S_ = self.discretize(state)
            self.init_Q(S, A)
            self.init_Q(S_, 0)
            self.init_Q(S_, 1)

            target_Q = r + self.gamma * max(self.Q[(S_, 0)], self.Q[(S_, 1)])
            self.Q[(S, A)] = self.Q[(S, A)] + self.alpha * (target_Q - self.Q[(S, A)])
            A_ = self.epsilon_greedy(S_)

        self.last_action = A_
        self.last_state = state

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
    all_hist = []

    for i in range(5):
        # Select agent.
        agent = Learner()
        # Empty list to save history.
        hist = []
        # Run games.
        run_games(agent, hist, 1000, 1)

        # Save history.
        all_hist.append(hist)

    np.save('output/avg/discrete_{}bins_{}gamma_{}alpha_{}epsmin_{}epsmax_{}decay_{}avgscore_newfeats_5runs'.format(
                agent.bins,
                agent.gamma,
                agent.alpha,
                agent.eps_min,
                agent.eps_max,
                agent.eps_decay,
                max(hist)),np.array(all_hist))






    #
    # # Select agent.
    # agent = Learner()
    # # Empty list to save history.
    # hist = []
    # # Run games.
    # run_games(agent, hist, 1000, 1)
    #
    # # Save history.
    #
    # np.save('output/avg/discrete_{}bins_{}gamma_{}alpha_{}epsmin_{}epsmax_{}decay_{}avgscore_newfeats'.format(
    #             agent.bins,
    #             agent.gamma,
    #             agent.alpha,
    #             agent.eps_min,
    #             agent.eps_max,
    #             agent.eps_decay,
    #             max(hist)),np.array(hist))
