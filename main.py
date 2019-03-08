import argparse

import gym

from gridworld import GridWorld


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Control variates for average reward')
    parser.add_argument('--max_episodes', type=int, default=10000, metavar='N',
                        help='number of episodes to repeat (default: 10000)')
    parser.add_argument('--environment', type=str, default='mountain_car', metavar='E',
                        choices=['gridworld', 'mountain_car'],
                        help="environment to use: "
                             "'gridworld' - GridWorld environment\n"
                             "'mountain_car' - Mountain Car environment (default: 'gridworld')")
    parser.add_argument('--algorithm', type=str, default='', metavar='A',
                        help="algorithm to use: <><><><> (default: '')")

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many episodes to wait before logging training status (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model (default: False)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.environment == 'mountain_car':
        env = gym.make('MountainCar-v0')
    elif args.environment == 'gridworld':
        env = GridWorld()
    else:  # Will never occur, just added to remove warnings in editor
        raise ValueError('Invalid choice for environment')


if __name__ == '__main__':
    main()
