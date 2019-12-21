import os, sys, time, redis, pickle, hashlib
import numpy as np

BASEDIR = os.path.dirname(__file__)
sys.path.append(BASEDIR)

from normalizer import Normalizer
from config import *

class MujocoParallel(object):

  def __init__(self, benchmark):
    '''
    Mainly used:
      - evaluate_parallel()
      - extract_train_fitness()
      - extract_test_fitness()
    '''
    # initialize connection
    self.db = redis.Redis()
    # clear database
    self.db.flushdb()
    # push list of environment to db
    self.db.lpush(ENV_NAMES, *benchmark['names'])
    # push list placeholder of policy & fitness to db
    self.db.lpush(POLICY, *[0 for _ in range(CAPACITY)])
    self.db.lpush(TRAJECTORY, *[0 for _ in range(CAPACITY)])
    # normalizer
    self.initialize_normalizers(benchmark)
    self.dim_p    = benchmark['p']
    self.dim_n    = benchmark['n']
    self.dim      = self.dim_p * self.dim_n
    self.names    = benchmark['names']
    self.instance = benchmark['instance'] 

  def initialize_normalizers(self, benchmark):
    self.normalizers = {}
    for name in benchmark['names']:
      self.normalizers[name] = Normalizer(benchmark['n'])

  def serialize(self, obj):
    return pickle.dumps(obj, protocol=4)

  def deserialize(self, obj):
    return pickle.loads(obj)

  def set_policies(self, policies):
    # policies: 3D ndarray, first dimension is N
    for i, policy in enumerate(policies):
      self.db.lset(POLICY, i, self.serialize(policy))
      self.db.lset(TRAJECTORY, i, 0)
      self.db.lpush(JOB, i)

  def wait(self):
    while 1:
      if self.db.llen(JOB) > 0:
        time.sleep(0.03)
      else:
        break

  def get_trajectories(self, n_eval):
    while 1:
      try: trajectories = [self.deserialize(trajectory) for trajectory in self.db.lrange(TRAJECTORY, 0, n_eval-1)]
      except: time.sleep(0.01)
      else: break
    return trajectories

  def set_env_name(self, name):
    self.db.set(ENV_NAME, name)

  def set_normalizer(self, name):
    normalizer = self.serialize(self.normalizers[name])
    normalizer_hash = hashlib.md5(normalizer).hexdigest()
    self.db.set(NORMALIZER, normalizer)
    self.db.set(NORMALIZER_HASH, normalizer_hash)

  def update_normalizer(self, name, trajectories):
    for trajectory in trajectories:
      self.normalizers[name].observe_many(trajectory['state'])

  def evaluate_parallel(self, policies, name, update_normalizer_flag=True):
    '''Send request to evaluate 'policies' on environment 'name'

    Parameters
    ----------
    policies : list
      List of nd.array with shape p x n, which is a linear mapping
      from state space to action space
    name : str
      Name of the environment to run simulation
    update_normalizer_flag : bool
      Specify whether normalizer is updated or not, not update when 
      run testing, because when testing, the learning is not occur, so
      cannot update it
    '''
    n_eval = policies.shape[0]
    self.set_env_name(name)
    self.set_normalizer(name)
    self.set_policies(policies)
    self.wait()
    trajectories = self.get_trajectories(n_eval)
    if update_normalizer_flag:
      self.update_normalizer(name, trajectories)
    return trajectories

  def extract_train_fitness(self, trajectories):
    '''Reduce the reward in trajectories to a scalar fitness.
    Per each step, reward = reward - 1 to encourage agent not
    to stay in one place
    '''
    fitness = []
    for trajectory in trajectories:
      fitness.append(np.sum(trajectory['reward']) - len(trajectory['reward']))
    return np.array(fitness)

  def extract_test_fitness(self, trajectories):
    '''Reduce the reward in trajectories to a scalar fitness.
    Unlike train fitness, this is the true reward given by the 
    environment
    '''
    fitness = []
    for trajectory in trajectories:
      fitness.append(np.sum(trajectory['reward']))
    return np.array(fitness)
