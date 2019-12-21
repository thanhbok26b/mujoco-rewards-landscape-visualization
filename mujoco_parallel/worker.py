import os, sys, time, redis, pickle, gym
import numpy as np
from gym_extensions.continuous import mujoco

BASEDIR = os.path.dirname(__file__)
sys.path.append(BASEDIR)

from config import *

class Worker(object):

  def __init__(self):
    '''A class used to represent a Worker to run Mujoco Simulation

    Attributes
    ----------
    db : redis.Redis
      connection to redis database
    envs : dict
      dictionary with key is environment name, value is openai gym 
      environment object to run simulation
    normalizer_hash : str
      a string to represent the hash of state normalizer. If it is 
      changed by master, all worker update its true normalizer. 
      Equivalent to a environment switch
    cache: tuple
      (environment name, gym environment object, normalizer object)
    '''
    self.db = redis.Redis()
    self.envs = {}
    self.initialize_envs()
    self.normalizer_hash = None
    self.cache = (None, None, None)

  def initialize_envs(self):
    '''Get the updated list of ENV_NAMES from redis. Then make 
    the gym environment according to those names. The created 
    environments are stand by.'''
    names = [_.decode('ascii') for _ in self.db.lrange(ENV_NAMES, 0, -1)] 
    self.envs = {}
    for name in names:
      self.envs[name] = gym.make(name)

  def get_next_job(self):
    '''Get index the next policy to be evaluated'''
    if self.db.llen(JOB) > 0:
      i = self.db.lpop(JOB)
      if i is not None:
        return int(i)

  def serialize(self, obj):
    return pickle.dumps(obj, protocol=4)

  def deserialize(self, obj):
    return pickle.loads(obj)

  def get_policy(self, i):
    '''Decode from index to the parameter of policy to be evaluated'''
    return self.deserialize(self.db.lindex(POLICY, i))

  def get_env(self):
    '''Check if environment to be evaluated is changed. Not change use
    cache, else update cache'''
    normalizer_hash = self.db.get(NORMALIZER_HASH).decode('ascii')
    if normalizer_hash != self.normalizer_hash:
      # update hash
      self.normalizer_hash = normalizer_hash
      # get environment name
      name = self.db.get(ENV_NAME).decode('ascii')
      if name not in self.envs.keys():
        self.initialize_envs()
      # get environment
      env = self.envs[name]
      # get normalizer
      normalizer = self.deserialize(self.db.get(NORMALIZER))
      self.cache = (name, env, normalizer)
    return self.cache

  def close_envs(self):
    '''Gratefully close all environments'''
    for env in self.envs.values():
      env.close()

  def mainloop(self):
    '''Wait for new job (new policy) arrive. Compute its trajectory
    including (state, next_state, action, reward) over time of an
    episode
    '''
    try:
      while 1:
        i = self.get_next_job()
        if i is not None:
          trajectory = {
            'state': [],
            'action': [],
            'reward': [],
          }
          policy = self.get_policy(i)
          name, env, normalizer = self.get_env()
#           env.seed(0)
          state = env.reset()
          while 1:
            state = normalizer.normalize(state)
            action = policy @ state
            state, reward, done, _ = env.step(action)
            trajectory['state'].append(state)
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)
            if done:
              break
          for k in trajectory:
            trajectory[k] = np.array(trajectory[k])
          self.db.lset(TRAJECTORY, i, self.serialize(trajectory))
        else:
          time.sleep(0.03)
    except KeyboardInterrupt:
      self.close_envs()

def main():
  w = Worker()
  w.mainloop()

if __name__ == '__main__':
  main()
