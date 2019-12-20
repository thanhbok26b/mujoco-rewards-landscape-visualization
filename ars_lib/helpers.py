import yaml
from scipy.optimize import OptimizeResult

def load_config(path='config.yaml'):
    with open(path) as fp:
        config = yaml.load(fp)
    return config

def get_optimization_results(t, N, factors):
    K = len(factors)
    results = []
    for k in range(K):
        factor = factors[k]
        result         = OptimizeResult()
        result.x       = factor.theta
        result.fun     = factor.f_opt
        result.nit     = t
        result.nfev    = (t + 1) * 2 * N
        results.append(result)
    return results
