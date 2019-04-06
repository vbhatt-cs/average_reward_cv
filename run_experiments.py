import json
import os
import time
from multiprocessing import Process, Pool

import numpy as np
import pandas as pd

import main

default_args = {'max_t': 20000,
                'environment': 'gridworld',
                'algorithm': 'n-step',
                'alpha': 0.1,
                'beta': 0.1,
                'init_rbar': 1 / 50,
                'n': 1,
                'lam': 0,
                'off_policy': False,
                'cv': False,
                'full_rbar': False,
                'cv_rbar': False,
                'seed': 1}

experiments_path = 'Experiments/'


def run_ab(config):
    n_cols = 7 if config['environment'] == 'gridworld' else 5
    seeds = 5
    n_exp = 7 * 9 * 3
    results = np.zeros((n_exp, n_cols))
    i = 0
    for ir in [0.01, 0.1, 1]:
        for a in range(1, 8):
            alpha = 2 ** (-a)
            for b in range(1, 10):
                beta = 2 ** (-b)
                config['alpha'] = alpha
                config['beta'] = beta
                config['init_rbar'] = ir
                print('Running', [ir, alpha, beta])
                metrics_list = run_seeds(config, seeds)

                metrics_df = pd.DataFrame(metrics_list)
                mean = metrics_df.mean()
                sem = metrics_df.sem()

                if config['environment'] == 'gridworld':
                    results[i] = [ir, alpha, beta, mean['rmse'], sem['rmse'], mean['rmse_rbar'], sem['rmse_rbar']]
                else:
                    results[i] = [ir, alpha, beta, mean['reward'], sem['reward']]
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
    config['max_t'] = 1000
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['n', 'init_rbar', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
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
    config['max_t'] = 1000
    config['algorithm'] = 'lambda'
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['lambda', 'init_rbar', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
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
    config['max_t'] = 1000
    config['off_policy'] = True
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['n', 'init_rbar', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
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
    config['max_t'] = 1000
    config['algorithm'] = 'lambda'
    config['off_policy'] = True
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['lambda', 'init_rbar', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
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
    config['max_t'] = 1000
    config['off_policy'] = True
    config['cv'] = True
    config['full_rbar'] = full_rbar
    config['cv_rbar'] = cv_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['n', 'init_rbar', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
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
    config['max_t'] = 1000
    config['algorithm'] = 'lambda'
    config['off_policy'] = True
    config['cv'] = True
    config['full_rbar'] = full_rbar
    config['cv_rbar'] = cv_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['lambda', 'init_rbar', 'alpha', 'beta', 'rmse', 'sem_rmse', 'rmse_rbar', 'sem_rmse_rbar']
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


def n_step_on_policy_control(full_rbar):
    path = experiments_path + 'n_step_on_policy_control/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['environment'] = 'mountain_car'
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['n', 'init_rbar', 'alpha', 'beta', 'reward', 'sem_reward']
    ns = [1, 2, 4]
    configs = [config.copy() for _ in range(len(ns))]
    for i in range(len(ns)):
        configs[i]['n'] = ns[i]
    with Pool(len(ns)) as p:
        results = p.map(run_ab, configs)

    for i, n in enumerate(ns):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * n, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def lambda_on_policy_control(full_rbar):
    path = experiments_path + 'lambda_on_policy_control/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['environment'] = 'mountain_car'
    config['algorithm'] = 'lambda'
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['lambda', 'init_rbar', 'alpha', 'beta', 'reward', 'sem_reward']
    lams = [1 - 2 ** (-l) for l in range(3)]
    configs = [config.copy() for _ in range(len(lams))]
    for i in range(len(lams)):
        configs[i]['lambda'] = lams[i]
    with Pool(len(lams)) as p:
        results = p.map(run_ab, configs)

    for i, l in enumerate(lams):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * l, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def n_step_off_policy_control(full_rbar):
    path = experiments_path + 'n_step_off_policy_control/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['environment'] = 'mountain_car'
    config['off_policy'] = True
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['n', 'init_rbar', 'alpha', 'beta', 'reward', 'sem_reward']
    ns = [1, 2, 4]
    configs = [config.copy() for _ in range(len(ns))]
    for i in range(len(ns)):
        configs[i]['n'] = ns[i]
    with Pool(len(ns)) as p:
        results = p.map(run_ab, configs)

    for i, n in enumerate(ns):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * n, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def lambda_off_policy_control(full_rbar):
    path = experiments_path + 'lambda_off_policy_control/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['environment'] = 'mountain_car'
    config['algorithm'] = 'lambda'
    config['off_policy'] = True
    config['full_rbar'] = full_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['lambda', 'init_rbar', 'alpha', 'beta', 'reward', 'sem_reward']
    lams = [1 - 2 ** (-l) for l in range(3)]
    configs = [config.copy() for _ in range(len(lams))]
    for i in range(len(lams)):
        configs[i]['lambda'] = lams[i]
    with Pool(len(lams)) as p:
        results = p.map(run_ab, configs)

    for i, l in enumerate(lams):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * l, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def n_step_cv_control(full_rbar, cv_rbar):
    path = experiments_path + 'n_step_cv_control/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['environment'] = 'mountain_car'
    config['off_policy'] = True
    config['cv'] = True
    config['full_rbar'] = full_rbar
    config['cv_rbar'] = cv_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['n', 'init_rbar', 'alpha', 'beta', 'reward', 'sem_reward']
    ns = [1, 2, 4]
    configs = [config.copy() for _ in range(len(ns))]
    for i in range(len(ns)):
        configs[i]['n'] = ns[i]
    with Pool(len(ns)) as p:
        results = p.map(run_ab, configs)

    for i, n in enumerate(ns):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * n, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


def lambda_cv_control(full_rbar, cv_rbar):
    path = experiments_path + 'lambda_cv_control/' + str(int(time.time())) + '/'
    os.makedirs(path, exist_ok=True)

    config = default_args.copy()
    config['environment'] = 'mountain_car'
    config['algorithm'] = 'lambda'
    config['off_policy'] = True
    config['cv'] = True
    config['full_rbar'] = full_rbar
    config['cv_rbar'] = cv_rbar

    with open(path + 'config.json', 'w') as f:
        json.dump(config, f)

    column_list = ['lambda', 'init_rbar', 'alpha', 'beta', 'reward', 'sem_reward']
    lams = [1 - 2 ** (-l) for l in range(3)]
    configs = [config.copy() for _ in range(len(lams))]
    for i in range(len(lams)):
        configs[i]['lambda'] = lams[i]
    with Pool(len(lams)) as p:
        results = p.map(run_ab, configs)

    for i, l in enumerate(lams):
        n_rows = results[i].shape[0]
        results[i] = np.append(np.ones((n_rows, 1)) * l, results[i], axis=1)

    results_df = pd.DataFrame(np.concatenate(results), columns=column_list)
    results_df.to_csv(path + 'results.csv')


if __name__ == '__main__':
    # print('Experiment: n-step on policy')
    # n_step_on_policy_prediction(False)
    # print('Experiment: n-step on policy full_rbar')
    # n_step_on_policy_prediction(True)
    # print('Experiment: lambda on policy')
    # lambda_on_policy_prediction(False)
    # print('Experiment: lambda on policy full_rbar')
    # lambda_on_policy_prediction(True)
    #
    # print('Experiment: n-step off policy')
    # n_step_off_policy_prediction(False)
    # print('Experiment: n-step off policy full_rbar')
    # n_step_off_policy_prediction(True)
    # print('Experiment: lambda off policy')
    # lambda_off_policy_prediction(False)
    # print('Experiment: lambda off policy full_rbar')
    # lambda_off_policy_prediction(True)
    #
    # print('Experiment: n-step cv')
    # n_step_cv_prediction(False, False)
    # print('Experiment: n-step cv full_rbar')
    # n_step_cv_prediction(True, False)
    # print('Experiment: n-step cv full_rbar cv_rbar')
    # n_step_cv_prediction(True, True)
    # print('Experiment: n-step cv cv_rbar')
    # n_step_cv_prediction(False, True)
    #
    # print('Experiment: lambda cv')
    # lambda_cv_prediction(False, False)
    # print('Experiment: lambda cv full_rbar')
    # lambda_cv_prediction(True, False)
    # print('Experiment: lambda cv full_rbar cv_rbar')
    # lambda_cv_prediction(True, True)
    # print('Experiment: lambda cv cv_rbar')
    # lambda_cv_prediction(False, True)

    # print('Experiment: n-step on policy')
    # p1 = Process(target=n_step_on_policy_control, args=(False,))
    # p1.start()
    # time.sleep(2)
    # print('Experiment: n-step on policy full_rbar')
    # p2 = Process(target=n_step_on_policy_control, args=(True,))
    # p2.start()
    # print('Experiment: lambda on policy')
    # p1 = Process(target=lambda_on_policy_control, args=(False,))
    # p1.start()
    # time.sleep(2)
    # print('Experiment: lambda on policy full_rbar')
    # p2 = Process(target=lambda_on_policy_control, args=(True,))
    # p2.start()
    #
    # print('Experiment: n-step off policy')
    # p1 = Process(target=n_step_off_policy_control, args=(False,))
    # p1.start()
    # time.sleep(2)
    # print('Experiment: n-step off policy full_rbar')
    # p2 = Process(target=n_step_off_policy_control, args=(True,))
    # p2.start()
    # print('Experiment: lambda off policy')
    # p1 = Process(target=lambda_off_policy_control, args=(False,))
    # p1.start()
    # time.sleep(2)
    # print('Experiment: lambda off policy full_rbar')
    # p2 = Process(target=lambda_off_policy_control, args=(True,))
    # p2.start()
    #
    # print('Experiment: n-step cv')
    # p1 = Process(target=n_step_cv_control, args=(False, False))
    # p1.start()
    # time.sleep(2)
    # print('Experiment: n-step cv full_rbar')
    # p2 = Process(target=n_step_cv_control, args=(True, False))
    # p2.start()
    # print('Experiment: n-step cv full_rbar cv_rbar')
    # p1 = Process(target=n_step_cv_control, args=(True, True))
    # p1.start()
    # time.sleep(2)
    # print('Experiment: n-step cv cv_rbar')
    # p2 = Process(target=n_step_cv_control, args=(False, True))
    # p2.start()
    #
    # print('Experiment: lambda cv')
    # p1 = Process(target=lambda_cv_control, args=(False, False))
    # p1.start()
    # time.sleep(2)
    # print('Experiment: lambda cv full_rbar')
    # p2 = Process(target=lambda_cv_control, args=(True, False))
    # p2.start()
    # print('Experiment: lambda cv full_rbar cv_rbar')
    # p1 = Process(target=lambda_cv_control, args=(True, True))
    # p1.start()
    # time.sleep(2)
    # print('Experiment: lambda cv cv_rbar')
    # p2 = Process(target=lambda_cv_control, args=(False, True))
    # p2.start()
    pass
