import json
import os
import time
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

import main

default_args = {'max_episodes': 200,
                'environment': 'gridworld',
                'algorithm': 'n-step',
                'alpha': 0.1,
                'beta': 0.1,
                'n': 1,
                'lam': 0,
                'off_policy': False,
                'cv': False,
                'full_rbar': False,
                'cv_rbar': False,
                'seed': 1}

experiments_path = 'Experiments/'


def run_ab(config):
    n_cols = 6 if config['environment'] == 'gridworld' else 4
    seeds = 30
    n_exp = 6 * 6
    results = np.zeros((n_exp, n_cols))
    i = 0
    for a in range(4, 10):
        alpha = 2 ** (-a)
        for b in range(5, 11):
            beta = 2 ** (-b)
            config['alpha'] = alpha
            config['beta'] = beta
            print('Running', [alpha, beta])
            metrics_list = run_seeds(config, seeds)

            metrics_df = pd.DataFrame(metrics_list)
            mean = metrics_df.mean()
            sem = metrics_df.sem()

            if config['environment'] == 'gridworld':
                results[i] = [alpha, beta, mean['rmse'], sem['rmse'], mean['rmse_rbar'], sem['rmse_rbar']]
            else:
                results[i] = [alpha, beta, mean['reward'], sem['reward']]
            i += 1
    return results


def run_seeds(config, seeds):
    metrics_list = []
    for seed in range(seeds):
        config['seed'] = seed
        metrics = main.run(config)
        metrics_list.append(metrics)
    return metrics_list


def n_step_on_policy_prediction(full_rbar):
    path = experiments_path + 'n_step_on_policy_prediction/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['max_episodes'] = 1000
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['n', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
    ns = [1, 2, 4, 8]
    configs = [config.copy() for _ in range(4)]
    for i in range(4):
        configs[i]['n'] = ns[i]
    with Pool(4) as p:
        results = p.map(run_ab, configs)

    for i, n in enumerate(ns):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * n, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def lambda_on_policy_prediction(full_rbar):
    path = experiments_path + 'lambda_on_policy_prediction/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['max_episodes'] = 1000
    config['algorithm'] = 'lambda'
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['lambda', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
    lams = [1 - 2 ** (-l) for l in range(4)]
    configs = [config.copy() for _ in range(4)]
    for i in range(4):
        configs[i]['lambda'] = lams[i]
    with Pool(4) as p:
        results = p.map(run_ab, configs)

    for i, l in enumerate(lams):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * l, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def n_step_off_policy_prediction(full_rbar):
    path = experiments_path + 'n_step_off_policy_prediction/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['max_episodes'] = 1000
    config['off_policy'] = True
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['n', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
    ns = [1, 2, 4, 8]
    configs = [config.copy() for _ in range(4)]
    for i in range(4):
        configs[i]['n'] = ns[i]
    with Pool(4) as p:
        results = p.map(run_ab, configs)

    for i, n in enumerate(ns):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * n, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def lambda_off_policy_prediction(full_rbar):
    path = experiments_path + 'lambda_off_policy_prediction/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['max_episodes'] = 1000
    config['algorithm'] = 'lambda'
    config['off_policy'] = True
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['lambda', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
    lams = [1 - 2 ** (-l) for l in range(4)]
    configs = [config.copy() for _ in range(4)]
    for i in range(4):
        configs[i]['lambda'] = lams[i]
    with Pool(4) as p:
        results = p.map(run_ab, configs)

    for i, l in enumerate(lams):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * l, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def n_step_cv_prediction(full_rbar, cv_rbar):
    path = experiments_path + 'n_step_cv_prediction/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['max_episodes'] = 1000
    config['off_policy'] = True
    config['cv'] = True
    config['full_rbar'] = full_rbar
    config['cv_rbar'] = cv_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['n', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
    ns = [1, 2, 4, 8]
    configs = [config.copy() for _ in range(4)]
    for i in range(4):
        configs[i]['n'] = ns[i]
    with Pool(4) as p:
        results = p.map(run_ab, configs)

    for i, n in enumerate(ns):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * n, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def lambda_cv_prediction(full_rbar, cv_rbar):
    path = experiments_path + 'lambda_cv_prediction/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['max_episodes'] = 1000
    config['algorithm'] = 'lambda'
    config['off_policy'] = True
    config['cv'] = True
    config['full_rbar'] = full_rbar
    config['cv_rbar'] = cv_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['lambda', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
    lams = [1 - 2 ** (-l) for l in range(4)]
    configs = [config.copy() for _ in range(4)]
    for i in range(4):
        configs[i]['lambda'] = lams[i]
    with Pool(4) as p:
        results = p.map(run_ab, configs)

    for i, l in enumerate(lams):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * l, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def n_step_on_policy_control():
    path = experiments_path + 'n_step_on_policy_control/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['environment'] = 'mountain_car'
    # config['full_rbar'] = True

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['n', 'alpha', 'beta', 'reward', 'sem_reward']
    ns = [1, 2, 4, 8]
    configs = [config.copy() for _ in range(4)]
    for i in range(4):
        configs[i]['n'] = ns[i]
    with Pool(4) as p:
        results = p.map(run_ab, configs)

    for i, n in enumerate(ns):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * n, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


if __name__ == '__main__':
    n_step_on_policy_prediction(False)
    n_step_on_policy_prediction(True)
    lambda_on_policy_prediction(False)
    lambda_on_policy_prediction(True)

    n_step_off_policy_prediction(False)
    n_step_off_policy_prediction(True)
    lambda_off_policy_prediction(False)
    lambda_off_policy_prediction(True)

    n_step_cv_prediction(False, False)
    n_step_cv_prediction(True, False)
    n_step_cv_prediction(True, True)
    n_step_cv_prediction(False, True)

    lambda_cv_prediction(False, False)
    lambda_cv_prediction(True, False)
    lambda_cv_prediction(True, True)
    lambda_cv_prediction(False, True)

    # n_step_on_policy_control()
