from mujoco_parallel import MujocoParallel, benchmarks
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy.io import savemat
from matplotlib import cm
from copy import deepcopy
from tqdm import trange
import numpy as np
import pickle
import yaml

def load_results(path, iteration=-1):
    with open(path, 'rb') as fp:
        results = pickle.load(fp)
    print('[+] Loading', results[iteration][0].fun)
    return results[iteration][0].x, results[iteration][0].message

def evaluate(mp, theta):
    trials = np.array([theta for _ in range(30)]).reshape(-1, mp.dim_p, mp.dim_n)
    # trials = np.array([theta]).reshape(-1, mp.dim_p, mp.dim_n)
    trajectories = mp.evaluate_parallel(trials, mp.names[0], update_normalizer_flag=False)
    fitnesses = mp.extract_test_fitness(trajectories)
    return np.mean(fitnesses)

def main():
    # load hyper-parameter
    config    = yaml.load(open('config.yaml').read())
    instance  = config['instance']
    benchmark = benchmarks[instance]
    vis_iteration = config['vis_iteration']

    # load parallel evaluator
    mp = MujocoParallel(benchmark)

    # load data
    theta, normalizers = load_results('data/%s/0.pkl' % instance, vis_iteration)
    mp.normalizers = normalizers
    fitness = evaluate(mp, theta)
    print('    Actual performance', fitness)
    print('    Parameter', theta)

    # generate vis data
    x1 = np.linspace(-1, 1, 10)
    x2 = np.linspace(-1, 1, 10)
    X1, X2 = np.meshgrid(x1, x2)

    # render Y
    d1, d2 = X1.shape
    Y = np.empty([d1, d2])
    iterator = trange(d1)
    for i in iterator:
        for j in range(d2):
            new_theta = deepcopy(theta)
            new_theta[22] = new_theta[22] + X1[i, j]
            new_theta[25] = new_theta[25] + X2[i, j]
            Y[i, j] = evaluate(mp, theta)
            desc = 'Y[%d, %d] = %0.4f' % (i, j, Y[i, j])
            iterator.set_description(desc)

    # save data for ploting
    savemat('data/vis/reward-landscape.mat', {'X1':X1, 'X2':X2, 'Y':Y})

    # actual ploting
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.plot_surface(X1, X2, Y, rstride=8, cstride=8, alpha=0.3)
#    plt.show()

if __name__ == '__main__':
    main()
