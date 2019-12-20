import os
import time
import signal
import subprocess

BASEDIR = os.path.dirname(__file__)

class WorkerManager(object):

  def __init__(self):
    '''
    Attributes
    ----------
    redis_pid : object
      subprocess of redis server
    worker_pids : list
      List of subprocess of workers
    '''
    self.redis   = None
    self.workers = None    

  def start_redis(self):
    self.redis = subprocess.Popen(['redis-server'])

  def stop_redis(self):
    self.redis.send_signal(signal.SIGINT)

  def create_workers(self, n_worker=None):
    if n_worker is None:
      n_worker = os.cpu_count()
    self.workers = [subprocess.Popen(['python3', '-W', 'ignore', os.path.join(BASEDIR, 'worker.py')]) for _ in range(n_worker)]

  def kill_workers(self):
    for i, worker in enumerate(self.workers):
      print('[+] kill worker', i)
      worker.send_signal(signal.SIGINT)

  def __del__(self):
    self.stop_redis()
    self.kill_workers()