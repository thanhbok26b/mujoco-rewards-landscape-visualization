import numpy as np

class Factor(object):

    def __init__(self, taskset, name, sigma, alpha):
        # Objective function
        self.taskset = taskset
        self.name    = name

        # Distribution mean and deviation
        self.theta = np.zeros([taskset.dim])
        self.sigma = sigma
        self.alpha = alpha

        # Statistic
        self.f_opt = -np.inf

    def generate(self, N):
        D = self.theta.shape[0]
        noise = np.random.randn(N, D)
        x_pos = []; x_neg = []
        for i in range(N):
          x_pos.append(self.theta + noise[i] * self.sigma)
          x_neg.append(self.theta - noise[i] * self.sigma)
        x_pos = np.array(x_pos); x_neg = np.array(x_neg)
        return noise, x_pos, x_neg

    def evaluate(self, x_pos, x_neg):
        N = len(x_pos)
        trials = np.concatenate([x_pos, x_neg]).reshape(-1, self.taskset.dim_p, self.taskset.dim_n)
        trajectories = self.taskset.evaluate_parallel(trials, self.name)
        y = self.taskset.extract_train_fitness(trajectories)
        y_pos, y_neg = y[:N], y[N:]
        return y_pos, y_neg

    def evaluate_one(self, theta):
        trials = np.array([theta]).reshape(1, self.taskset.dim_p, self.taskset.dim_n)
        trajectories = self.taskset.evaluate_parallel(trials, self.name, update_normalizer_flag=False)
        true_fitness = self.taskset.extract_test_fitness(trajectories)
        return true_fitness[0]

    def rescale(self, y_pos, y_neg):
        sigma_r = np.std(np.concatenate([y_pos, y_neg]))
        return y_pos / sigma_r, y_neg / sigma_r

    def select(self, noise, y_pos, y_neg):
        N = len(y_pos)

        y_max = np.array([np.max([y_pos[i], y_neg[i]]) for i in range(N)])
        idx = np.argsort(y_max)[::-1][:N//2]

        y_pos = y_pos[idx]
        y_neg = y_neg[idx]
        noise = noise[idx, :]

        return noise, y_pos, y_neg

    def update(self, noise, y_pos, y_neg):
        N, D = noise.shape
        grad = np.zeros([D])
        for i in range(N):
            grad += (y_pos[i] - y_neg[i]) * noise[i]
        self.theta += self.alpha / N * grad
        trials = np.array([self.theta for _ in range(30)]).reshape(-1, self.taskset.dim_p, self.taskset.dim_n)
        trajectories = self.taskset.evaluate_parallel(trials, self.name, update_normalizer_flag=False)
        true_fitness = self.taskset.extract_test_fitness(trajectories)
        self.f_opt = np.mean(true_fitness)

