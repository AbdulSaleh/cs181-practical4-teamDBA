# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.Q = {}
        self.alpha = 1e-12
        self.eps = 1
        self.gamma = 1
        self.g = 0


        self.weights = np.array([0.0 for i in range(12)])

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.g = 0
        self.eps *= .999


    def _featurize(self, state, action):


      vec = [1.0,
       state['tree']['dist'],
       state['tree']['bot'],
       state['monkey']['vel'],
       state['monkey']['bot'],
       self.g]


      if action == 0:
        vec = vec + [0] * len(vec)
      else:
        vec = [0] * len(vec) + vec

      return(np.array(vec))

    def _Q(self, state, action):

      features = self._featurize(state, action)
      return np.sum(self.weights * features)


    def action_callback(self, state):

        rand = npr.rand()
        if rand < self.eps:
          new_action = npr.choice([0, 1])
        else:
          new_action = int(self._Q(state, 0) < self._Q(state, 1))


        if self.last_state is not None and self.last_action is not None:
          if self.g == 0:
            v = abs(state['monkey']['vel'])
            if v > 2:
              self.g = 4
            else:
              self.g = 1

          Q = self._Q(self.last_state, self.last_action)
          U = self.last_reward + self.gamma * max(self._Q(state, 0), self._Q(state, 1))
          features = self._featurize(self.last_state, self.last_action)

          self.weights += self.alpha * (U - Q) * features

        self.last_action = new_action
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
        print(learner.weights)
    pg.quit()
    return


if __name__ == '__main__':

  # Select agent.
  agent = Learner()

  # Empty list to save history.
  hist = []

  # Run games.
  run_games(agent, hist, 5000, 1)

  # Save history.
  np.save('linear_approximation',np.array(hist))
  print(max(hist), np.mean(hist))
