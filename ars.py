from ars_lib import *

def ars(taskset, config, callback=None):
    # Unpack hyperparameters
    N = config['num_direction']
    T = config['num_iteration']
    alpha = config['alpha']
    sigma = config['sigma']

    # Initialize
    K = len(taskset.names)
    factors = [Factor(taskset, taskset.names[k], sigma, alpha) for k in range(K)]

    # Mainloop
    iterator = trange(T)
    for t in iterator:
        for k in range(K):
            # generate
            noise, x_pos, x_neg = factors[k].generate(N)
            # evaluate
            y_pos, y_neg = factors[k].evaluate(x_pos, x_neg)
            # rescale
            y_pos, y_neg = factors[k].rescale(y_pos, y_neg)
            # select
            noise, y_pos, y_neg = factors[k].select(noise, y_pos, y_neg)
            # update
            factors[k].update(noise, y_pos, y_neg)

        # info
        desc = 'ARS generation=%d %s' % (t, ' '.join(['%0.4f' % factors[k].f_opt for k in range(K)]))
        iterator.set_description(desc)
        results = get_optimization_results(t, N, factors)
        if callback:
            callback(results)
    return np.array(stats)
