import contextlib
import json
import os

import numpy as np
import pandas as pd

import main

default_args = {'max_episodes': 200,
                'environment': 'gridworld',
                'algorithm': 'n-step',
                'alpha': 0.1,
                'beta': 0.1,
                'n': 1,
                'lambda': 0,
                'off_policy': False,
                'cv': False,
                'full_rbar': False,
                'cv_rbar': False,
                'seed': 1}

experiments_path = 'Experiments/'


def n_step_on_policy_prediction():
    path = experiments_path + 'n_step_on_policy_prediction/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['max_episodes'] = 1000

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    n_exp = 8 * 10 * 10 * 100
    column_list = ['n', 'alpha', 'beta', 'seed', 'rmse', 'rmse_rbar']
    results = np.zeros((n_exp, len(column_list)))

    i = 0
    for n in range(1, 9):
        for alpha in np.arange(0.1, 1.1, 0.1):
            for beta in np.arange(0.1, 1.1, 0.1):
                for seed in range(100):
                    config['n'] = n
                    config['alpha'] = alpha
                    config['beta'] = beta
                    config['seed'] = seed

                    with open(path + str(i) + '.txt', 'w') as f:
                        with contextlib.redirect_stdout(f):
                            metrics = main.run(config)

                    results[i] = [n, alpha, beta, seed, metrics['rmse'], metrics['rmse_rbar']]
                    i += 1

    results = pd.DataFrame(results, columns=column_list)
    results.to_csv(path + 'results.csv')


if __name__ == '__main__':
    n_step_on_policy_prediction()
