from subprocess import run

import numpy as np

arglist = ['--max-episodes', '--environment', '--algorithm', '--alpha', '--beta', '--n', '--lambda',
           '--off-policy', '--cv', '--full-rbar', '--cv-rbar', '--seed']


def n_step_on_policy_prediction():
    for n in range(1, 5):
        for alpha in np.arange(0.1, 0.3, 0.1):
            for beta in np.arange(0.1, 0.3, 0.1):
                for seed in range(1):
                    args = ['python', 'main.py', '--max-episodes', 1000, '--environment', 'gridworld',
                            '--algorithm', 'n-step', '--alpha', alpha, '--beta', beta, '--n', n, '--seed', seed]

                    args = [str(x) for x in args]
                    run(args)


if __name__ == '__main__':
    n_step_on_policy_prediction()
