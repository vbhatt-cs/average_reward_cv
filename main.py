import argparse
import os

import numpy as np

from algs import RLearning, NStepPrediction, NStepControl, LambdaPrediction, LambdaControl
from features import TileCoding, OneHot, Identity
from envs import GridWorld, MountainCar, RandomWalk
from policies import EpsGreedy, BiasedRandom

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Control variates for average reward')
    parser.add_argument('--max-t', type=int, default=20000, metavar='N',
                        help='number of episodes to repeat (default: 20000)')
    parser.add_argument('--environment', type=str, default='gridworld', metavar='E',
                        choices=['gridworld', 'random_walk' 'mountain_car'],
                        help="environment to use:\n"
                             "'gridworld' (default)\n"
                             "'random_walk'\n"
                             "'mountain_car'")
    parser.add_argument('--algorithm', type=str, default='n-step', metavar='A',
                        choices=['n-step', 'n-step', 'r-learning'],
                        help="algorithm to use:\n"
                             "'n-step' - N-step method\n"
                             "'lambda' - Lambda method\n"
                             "'r-learning' - RLearning (default: 'n-step')")

    parser.add_argument('--alpha', type=float, default=0.1, metavar='LR',
                        help='learning rate for Q/V (default: 0.1)')
    parser.add_argument('--beta', type=float, default=0.1, metavar='LR',
                        help='learning rate for Rbar (default: 0.1)')
    parser.add_argument('--init-rbar', type=float, default=0.1, metavar='LR',
                        help='initial value for Rbar (default: 0.1)')
    parser.add_argument('--n', type=int, default=1, metavar='N',
                        help='Steps for n-step (default: 1)')
    parser.add_argument('--lambda', type=float, default=0, metavar='L',
                        help='lambda for lambda methods (default: 0)')
    parser.add_argument('--off-policy', action='store_true', default=False,
                        help='True if learning is off-policy (default: False)')
    parser.add_argument('--cv', action='store_true', default=False,
                        help='True if control variates are used (default: False)')
    parser.add_argument('--full-rbar', action='store_true', default=False,
                        help='True if Rbar also uses n-step/lambda updates (default: False)')
    parser.add_argument('--cv-rbar', action='store_true', default=False,
                        help='True if Rbar uses control variates (default: False)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    return parser.parse_args()


def run(config):
    """
    Args:
        config (dict): Dictionary containing all arguments added in parse_args() and their values

    Returns:
        Dictionary containing metrics corresponding to the experiment that is run
    """
    rng = np.random.RandomState(config['seed'])
    # np.seterr(all='raise')

    if config['environment'] == 'mountain_car':
        env = MountainCar(rng)
        features = TileCoding(np.array([8, 8]), 16, [[-1.2, 0.6], [-0.07, 0.07]])

        behaviour_policy = EpsGreedy(0.3, env.action_space.n, rng)
        target_policy = EpsGreedy(0, env.action_space.n, rng)  # Greedy policy
    elif config['environment'] == 'random_walk':
        rw_state_size = 19
        env = RandomWalk(rw_state_size)
        features = Identity(rw_state_size)
        behaviour_policy = EpsGreedy(1, env.action_space.n, rng)  # Equiprobable random
        target_policy = EpsGreedy(1, env.action_space.n, rng)
    else:  # Grid world
        env = GridWorld()
        features = OneHot(25)
        if config['off_policy']:
            behaviour_policy = EpsGreedy(1, env.action_space.n, rng)  # Equiprobable random
        else:
            behaviour_policy = BiasedRandom(0.5, 0, env.action_space.n, rng)
            # behaviour_policy = ScriptedPolicy()
        target_policy = BiasedRandom(0.5, 0, env.action_space.n, rng)
        # target_policy = ScriptedPolicy()
        # target_policy = EpsGreedy(0, env.action_space.n, rng)  # Greedy policy

    if config['environment'] == 'random_walk':
        true_values = np.linspace(-1, 1, rw_state_size + 2)[1:-1]
        true_rbar = 0
    else:  # Grid world
        # True values for BiasedRandom(0.5, 0, ...) in gridworld
        true_values = np.array([[0, 0.42290273, 0.00798559, -0.26424724, -0.39868317],
                                [0.85637394, 0.40017918, 0.00475787, -0.26258757, -0.39366252],
                                [0.73389499, 0.36524459, 0, -0.25110309, -0.36017762],
                                [0.62960723, 0.32662237, 0.00152571, -0.19625266, -0.16237105],
                                [0.55060987, 0.29507967, 0.02129253, -0.01420391, 0]]).flatten()
        true_rbar = 0.01743207298180489

    state_size = features.state_size
    action_size = env.action_space.n

    if config['algorithm'] == 'r-learning':
        alg = RLearning(behaviour_policy, config['alpha'], config['beta'], config['init_rbar'], state_size, action_size)
    elif config['algorithm'] == 'n-step':
        if config['environment'] in ['gridworld', 'random_walk']:  # Prediction
            alg = NStepPrediction(behaviour_policy, target_policy, config['alpha'], config['beta'], config['init_rbar'],
                                  config['off_policy'], config['cv'], config['full_rbar'], config['cv_rbar'],
                                  config['n'], state_size)
            # alg = NStepControl(behaviour_policy, target_policy, config['alpha'], config['beta'], config['init_rbar'],
            #                    config['off_policy'], config['cv'], config['full_rbar'], config['cv_rbar'],
            #                    config['n'], state_size, action_size)
        else:  # Control
            alg = NStepControl(behaviour_policy, target_policy, config['alpha'] / 16, config['beta'] / 16,
                               config['init_rbar'], config['off_policy'], config['cv'], config['full_rbar'],
                               config['cv_rbar'], config['n'], state_size, action_size)
    else:  # Lambda
        if config['environment'] in ['gridworld', 'random_walk']:  # Prediction
            alg = LambdaPrediction(behaviour_policy, target_policy, config['alpha'], config['beta'],
                                   config['init_rbar'], config['off_policy'], config['cv'], config['full_rbar'],
                                   config['cv_rbar'], config['lambda'], state_size)
            # alg = LambdaControl(behaviour_policy, target_policy, config['alpha'], config['beta'],
            #                     config['init_rbar'], config['off_policy'], config['cv'], config['full_rbar'],
            #                     config['cv_rbar'], config['lambda'], state_size, action_size)
        else:  # Control
            alg = LambdaControl(behaviour_policy, target_policy, config['alpha'] / 16, config['beta'] / 16,
                                config['init_rbar'], config['off_policy'], config['cv'], config['full_rbar'],
                                config['cv_rbar'], config['lambda'], state_size, action_size)

    tot_reward = 0
    tot_t = 0
    e = 0
    while tot_t < config['max_t']:
        obs = env.reset()
        state = features.extract(obs)
        if tot_t == 0:
            alg.reset(state)
        done = False
        while not done:
            # env.render()
            action = alg.act(state)
            obs, reward, done, _ = env.step(action)
            state = features.extract(obs)
            alg.train(reward, state)
            if tot_t > 0 * config['max_t']:
                tot_reward += reward
            tot_t += 1
            if tot_t == config['max_t']:
                break

        if config['environment'] == 'gridworld':
            # print("Episode: {}, Weights: {}, Rbar: {}".format(e, alg.weights.reshape((5, 5)), alg.rbar))
            # print("Episode: {}, Reward: {}, Actions: {} Rbar: {}".format(e, avg_reward,
            #                                                              alg.weights.argmax(axis=1).reshape((5, 5)),
            #                                                              alg.rbar))
            pass
        elif config['environment'] == 'random_walk':
            # print("Episode: {}, Weights: {}, Rbar: {}".format(e, alg.weights, alg.rbar))
            pass
        else:
            # print("Episode: {}, Reward: {}, Rbar: {}".format(e, tot_reward, alg.rbar))
            pass

        e += 1

    if config['environment'] == 'gridworld':
        values = alg.weights - alg.weights[12]  # Since true value of initial state is set to zero
        rmse = np.sqrt(np.mean(np.square(values - true_values)))
        rmse_rbar = np.abs(alg.rbar - true_rbar)

        metrics = {'rmse': rmse, 'rmse_rbar': rmse_rbar}
    elif config['environment'] == 'random_walk':
        values = alg.weights - alg.weights[int(rw_state_size / 2)]  # Since true value of initial state is set to zero
        rmse = np.sqrt(np.mean(np.square(values - true_values)))
        rmse_rbar = np.abs(alg.rbar - true_rbar)

        metrics = {'rmse': rmse, 'rmse_rbar': rmse_rbar}
    else:
        metrics = {'reward': tot_t / (tot_reward + 1)}

    return metrics


if __name__ == '__main__':
    args = parse_args()
    run(vars(args))
