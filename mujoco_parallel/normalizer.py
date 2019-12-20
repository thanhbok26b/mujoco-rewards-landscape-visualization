import time
import numpy as np

class Normalizer(object):

  def __init__(self, n):
    self.count     = np.zeros(n)
    self.mean      = np.zeros(n)
    self.mean_diff = np.zeros(n)
    self.var       = np.zeros(n)   

  def observe_many(self, states):
    for state in states:
      self.observe(state)

  def observe(self, state):
    self.count     += 1.
    last_mean       = self.mean.copy()
    self.mean      += (state - self.mean) / self.count
    self.mean_diff += (state - last_mean) * (state - self.mean)
    self.var        = (self.mean_diff / self.count).clip(min = 1e-2)
 
  def normalize(self, state):
    obs_mean = self.mean
    obs_std  = np.sqrt(self.var)
    if obs_std.any() == 0:
      return state - obs_mean
    else:
      return (state - obs_mean) / obs_std
