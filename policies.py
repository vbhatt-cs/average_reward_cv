"""
All policies need to take state, weights as arguments. Use act method of classes as policies
"""

import numpy as np


class EpsGreedy:
    """
    Epsilon Greedy
    """
    def __init__(self, eps, action_size, rng=None):
        """
        Args:
            eps (float): Epsilon
            action_size (int): Size of action space
            rng (np.RandomState): Numpy random state
        """
        self.eps = eps
        self.action_size = action_size

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

    def act(self, state, weights):
        """
        Calculate the action
        Args:
            state (1D array): Current state
            weights (array): Current weights

        Returns:
            Action
        """
        if self.rng.rand() < self.eps:
            return self.rng.randint(self.action_size)
        else:
            q = state.dot(weights)
            return q.argmax()


class BiasedRandom:
    """
    Take bias_action w.p (1-eps) and random action otherwise
    """
    def __init__(self, eps, bias_action, action_size, rng=None):
        """
        Args:
            eps (float): Epsilon
            bias_action (int): Biased action
            action_size (int): Size of action space
            rng (np.RandomState): Numpy random state
        """
        self.eps = eps
        self.action_size = action_size
        self.bias_action = bias_action

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

    def act(self, state, weights):
        """
        Calculate the action. Args are just for syntax
        Returns:
            Action
        """
        if self.rng.rand() < self.eps:
            return self.rng.randint(self.action_size)
        else:
            return self.bias_action


def scripted_policy(state, weights):
    """
    A scripted policy specific to 5x5 gridworld for testing. Goes N, N, W, W from initial state.
    Args:
        state (1D array): Current state
        weights: Just for syntax

    Returns:
        Action
    """
    if state[12] == 1:
        return 0
    elif state[7] == 1:
        return 0
    elif state[2] == 1:
        return 3
    elif state[1] == 1:
        return 3
    else:
        raise ValueError

