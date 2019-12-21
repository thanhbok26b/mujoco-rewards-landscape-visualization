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

def load_results(path, iteration=-1):
    with open(path, 'rb') as fp:
        results = pickle.load(fp)
    print('[+] Loading', results[iteration][0].fun)
    return results[iteration][0].x, results[iteration][0].message

def main():
    global results

    config    = yaml.load(open('config.yaml').read())
    instance  = config['instance']
    benchmark = benchmarks[instance]
    vis_iteration = config['vis_iteration']

    theta, normalizers = load_results('data/%s/0.pkl' % instance, vis_iteration)

    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/%s' % instance):
        os.mkdir('data/%s' % instance)

#    # Start workers
#    wm = WorkerManager()
#    wm.start_redis()
#    wm.create_workers()

    # Start master
    mp = MujocoParallel(benchmark)
    mp.normalizers = normalizers

    for i in range(config['repeat']):
        results = []
        ars(mp, config, callback)
        obj = pickle.dumps(results, protocol=4)
#        with open('data/%s/%d.pkl' % (instance, i), 'wb') as fp:
#            fp.write(obj)

if __name__ == '__main__':
    main()
