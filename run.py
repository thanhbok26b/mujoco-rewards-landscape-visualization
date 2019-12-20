import os
import yaml
import pickle
from ars import ars
from mujoco_parallel import WorkerManager
from mujoco_parallel import MujocoParallel, benchmarks

results = []

def callback(res):
    global results
    results.append(res)

def main():
    global results

    config    = yaml.load(open('config.yaml').read())
    instance  = config['instance']
    benchmark = benchmarks[instance]

    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/%s' % instance):
        os.mkdir('data/%s' % instance)

    # Start workers
    # wm = WorkerManager()
    # wm.start_redis()
    # wm.create_workers()

    # Start master
    mp = MujocoParallel(benchmark)

    for i in range(config['repeat']):
        results = []
        ars(mp, config, callback)
        obj = pickle.dumps(results, protocol=4)
        with open('data/%s/%d.pkl' % (instance, i), 'wb') as fp:
            fp.write(obj)

if __name__ == '__main__':
    main()
