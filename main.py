import argparse

import gym
import numpy as np

from algs import RLearning, NStepPrediction, NStepControl, LambdaPrediction, LambdaControl
from features import TileCoding, OneHot
from gridworld import GridWorld
from policies import EpsGreedy, BiasedRandom, ScriptedPolicy


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Control variates for average reward')
    parser.add_argument('--max-episodes', type=int, default=1000, metavar='N',
                        help='number of episodes to repeat (default: 200)')
    parser.add_argument('--environment', type=str, default='gridworld', metavar='E',
                        choices=['gridworld', 'mountain_car'],
                        help="environment to use:\n"
                             "'gridworld' - GridWorld environment\n"
                             "'mountain_car' - Mountain Car environment (default: 'gridworld')")
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
    parser.add_argument('--n', type=int, default=1, metavar='N',
                        help='Steps for n-step (default: 1)')
    parser.add_argument('--lambda', type=float, default=0, metavar='L', dest='lam',
                        help='lambda for lambda methods (default: 0)')
    parser.add_argument('--off-policy', action='store_true', default=True,
                        help='True if learning is off-policy (default: True)')
    parser.add_argument('--cv', action='store_true', default=True,
                        help='True if control variates are used (for off-policy) (default: True)')
    parser.add_argument('--full-rbar', action='store_true', default=False,
                        help='True if Rbar also uses n-step/lambda updates (default: False)')
    parser.add_argument('--cv-rbar', action='store_true', default=True,
                        help='True if Rbar uses control variates (default: True)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    return parser.parse_args()


def main():
    args = parse_args()
    print(args.full_rbar)
    rng = np.random.RandomState(args.seed)
    # np.seterr(all='raise')

    if args.environment == 'mountain_car':
        env = gym.make('MountainCar-v0')
        env.seed(args.seed)
        env._max_episode_steps = 1000
        features = TileCoding(np.array([8, 8]), 16)

        # Scale alpha, beta with n_tilings
        args.alpha = args.alpha / 16
        args.beta = args.beta / 16

        behaviour_policy = EpsGreedy(0.1, env.action_space.n, rng)
        target_policy = EpsGreedy(0, env.action_space.n, rng)  # Greedy policy
    else:  # Grid world
        env = GridWorld()
        features = OneHot(25)
        if args.off_policy:
            behaviour_policy = EpsGreedy(1, env.action_space.n, rng)  # Equiprobable random
        else:
            behaviour_policy = BiasedRandom(0.5, 0, env.action_space.n, rng)
            # behaviour_policy = ScriptedPolicy()
        target_policy = BiasedRandom(0.5, 0, env.action_space.n, rng)
        # target_policy = ScriptedPolicy()
        # target_policy = EpsGreedy(0, env.action_space.n, rng)  # Greedy policy

    # True values for BiasedRandom(0.5, 0, ...) in gridworld
    true_values = np.array([[0, 0.42290273, 0.00798559, -0.26424724, -0.39868317],
                            [0.85637394, 0.40017918, 0.00475787, -0.26258757, -0.39366252],
                            [0.73389499, 0.36524459, 0, -0.25110309, -0.36017762],
                            [0.62960723, 0.32662237, 0.00152571, -0.19625266, -0.16237105],
                            [0.55060987, 0.29507967, 0.02129253, -0.01420391, 0]])
    true_rbar = -0.9825679270181953

    state_size = features.state_size
    action_size = env.action_space.n

    if args.algorithm == 'r-learning':
        alg = RLearning(behaviour_policy, args.alpha, args.beta, state_size, action_size)
    elif args.algorithm == 'n-step':
        if args.environment == 'gridworld':  # Prediction
            alg = NStepPrediction(behaviour_policy, target_policy, args.alpha, args.beta, args.off_policy,
                                  args.cv, args.full_rbar, args.cv_rbar, args.n, state_size)
            # alg = NStepControl(behaviour_policy, target_policy, args.alpha, args.beta, args.off_policy,
            #                    args.cv, args.full_rbar, args.cv_rbar, args.n, state_size, action_size)
        else:  # Control
            alg = NStepControl(behaviour_policy, target_policy, args.alpha, args.beta, args.off_policy,
                               args.cv, args.full_rbar, args.cv_rbar, args.n, state_size, action_size)
    else:  # Lambda
        if args.environment == 'gridworld':  # Prediction
            alg = LambdaPrediction(behaviour_policy, target_policy, args.alpha, args.beta, args.off_policy,
                                   args.cv, args.full_rbar, args.cv_rbar, args.lam, state_size)
            # alg = LambdaControl(behaviour_policy, target_policy, args.alpha, args.beta, args.off_policy,
            #                     args.cv, args.full_rbar, args.cv_rbar, args.lam, state_size, action_size)
        else:  # Control
            alg = LambdaControl(behaviour_policy, target_policy, args.alpha, args.beta, args.off_policy,
                                args.cv, args.full_rbar, args.cv_rbar, args.lam, state_size, action_size)

    for e in range(args.max_episodes):
        obs = env.reset()
        state = features.extract(obs)
        if e == 0:
            alg.reset(state)
        done = False
        avg_reward = 0
        # action_count = np.zeros(action_size)
        while not done:
            # env.render()
            action = alg.act(state)
            # action_count[action] += 1
            obs, reward, done, _ = env.step(action)
            state = features.extract(obs)
            alg.train(reward, state)
            avg_reward += reward

        # # print(action_count / sum(action_count))
        # print(avg_reward)
        #
        # # Testing
        # # avg_reward = 0
        # # action_count = np.zeros(action_size)
        # obs = env.reset()
        # state = features.extract(obs)
        # done = False
        # while not done:
        #     action = alg.act(state, True)
        #     # action_count[action] += 1
        #     obs, reward, done, _ = env.step(action)
        #     state = features.extract(obs)
        #     avg_reward -= reward

        print("Episode: {}, Weights: {}, Rbar: {}".format(e, alg.weights.reshape((5, 5)), alg.rbar))
        # print("Episode: {}, Reward: {}, Actions: {} Rbar: {}".format(e, avg_reward,
        #                                                              alg.weights.argmax(axis=1).reshape((5, 5)),
        #                                                              alg.rbar))
        # print("Episode: {}, Reward: {}, Rbar: {}".format(e, avg_reward, alg.rbar))


if __name__ == '__main__':
    main()
