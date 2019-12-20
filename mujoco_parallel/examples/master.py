import time
import yaml
import numpy as np
from mujoco_parallel import WorkerManager
from mujoco_parallel import MujocoParallel, benchmarks

def main():
  config = yaml.load(open('config.yaml').read())
  benchmark = benchmarks['sanity-check']

  # Start workers
  wm = WorkerManager()
  wm.start_redis()
  wm.create_workers()

  # Start master
  mp = MujocoParallel(benchmark)

  # Generate random policies
  n = benchmark['n'] # dimension of state space
  p = benchmark['p'] # dimension of action space
  N = config['pop_size']
  policies = np.random.rand(N, p, n)

  # Evaluate
  trajectories  = mp.evaluate_parallel(policies, benchmark['names'][0])
  train_fitness = mp.extract_train_fitness(trajectories)

  print(train_fitness)

if __name__ == '__main__':
  main()
