"""
All policies need to take state, weights as arguments. Use act method of classes as policies
"""

import numpy as np


class BasePolicy:
    """
    Base class for policies
    """
    def act(self, state, weights):
        """
        To sample an action
        Args:
            state (1D array): Current state
            weights (array): Current weights
        """
        raise NotImplementedError

    def prob(self, state, action, weights):
        """
        To calculate probability of taking the given action
        Args:
            state (1D array): Current state
            action (int): Action taken
            weights (array): Current weights
        """
        raise NotImplementedError

    def act_prob(self, state, weights):
        """
        Sample an action and calculate the probability of it
        Args:
            state (1D array): Current state
            weights (array): Current weights
        """
        raise NotImplementedError

    def expected_value(self, q):
        """
        Calculate the expected Q value
        Args:
            q (1D array): Q values
        """
        raise NotImplementedError

    def probs(self, state, weights):
        """
        Calculate the probabilities of all actions
        Args:
            state (1D array): Current state
            weights (array): Current weights
        """
        raise NotImplementedError


class EpsGreedy(BasePolicy):
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

    def prob(self, state, action, weights):
        """
        Args:
            state (1D array): Current state
            action (int): Action taken
            weights (array): Current weights

        Returns:
            Probability of taking the action under the policy
        """
        if action == np.argmax(state.dot(weights)):
            return 1 - self.eps + self.eps / self.action_size
        else:
            return self.eps / self.action_size

    def act_prob(self, state, weights):
        """
        Sample an action and calculate the probability of it
        Args:
            state (1D array): Current state
            weights (array): Current weights

        Returns:
            Action, probability
        """
        q = state.dot(weights)
        best_action = q.argmax()
        if self.rng.rand() < self.eps:
            action = self.rng.randint(self.action_size)
        else:
            action = best_action

        if action == best_action:
            return action, 1 - self.eps + self.eps / self.action_size
        else:
            return action, self.eps / self.action_size

    def expected_value(self, q):
        """
        Calculate the expected Q value
        Args:
            q (1D array): Q values

        Returns:
            Expected Q value
        """
        axis = 0 if len(q.shape) == 1 else 1
        ev = q.sum(axis=axis) * self.eps
        ev += q.max(axis=axis) * (1 - self.eps)
        return ev

    def probs(self, state, weights):
        """
        Calculate the probabilities of all actions
        Args:
            state (1D array): Current state
            weights (array): Current weights

        Returns:
            Action probabilities
        """
        q = state.dot(weights)
        best_action = q.argmax()
        probs = np.ones(self.action_size) * self.eps / self.action_size
        probs[best_action] += 1 - self.eps
        return probs


class BiasedRandom(BasePolicy):
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

    def prob(self, state, action, weights):
        """
        Args:
            state (1D array): Current state
            action (int): Action taken
            weights (array): Current weights

        Returns:
            Probability of taking the action under the policy
        """
        if action == self.bias_action:
            return 1 - self.eps + self.eps / self.action_size
        else:
            return self.eps / self.action_size

    def act_prob(self, state, weights):
        """
        Sample an action and calculate the probability of it
        Args:
            state (1D array): Current state
            weights (array): Current weights

        Returns:
            Action, probability
        """
        if self.rng.rand() < self.eps:
            action = self.rng.randint(self.action_size)
        else:
            action = self.bias_action

        if action == self.bias_action:
            return action, 1 - self.eps + self.eps / self.action_size
        else:
            return action, self.eps / self.action_size

    def expected_value(self, q):
        """
        Calculate the expected Q value
        Args:
            q (1D array): Q values

        Returns:
            Expected Q value
        """
        axis = 0 if len(q.shape) == 1 else 1
        ev = q.sum(axis=axis) * self.eps
        ev += np.take(q, self.bias_action, axis=axis) * (1 - self.eps)
        return ev

    def probs(self, state, weights):
        """
        Calculate the probabilities of all actions
        Args:
            state (1D array): Current state
            weights (array): Current weights

        Returns:
            Action probabilities
        """
        probs = np.ones(self.action_size) * self.eps / self.action_size
        probs[self.bias_action] += 1 - self.eps
        return probs


class ScriptedPolicy(BasePolicy):
    """
    A scripted policy specific to 5x5 gridworld for testing. Optimal deterministic policy
    """
    def act(self, state, weights):
        """
        Calculate the action. Args are just for syntax
        Returns:
            Action
        """
        optimal_policy = np.array([0, 3, 3, 3, 3, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 0, 2, 2, 2, 2, 1, 1, 1, 1, 0])
        return int(optimal_policy[state.astype(np.bool)])

    def prob(self, state, action, weights):
        """
        Args:
            state (1D array): Current state
            action (int): Action taken
            weights (array): Current weights

        Returns:
            Probability of taking the action under the policy
        """
        scripted_action = self.act(state, weights)
        if action == scripted_action:
            return 1
        else:
            return 0

    def act_prob(self, state, weights):
        """
        Sample an action and calculate the probability of it
        Args:
            state (1D array): Current state
            weights (array): Current weights

        Returns:
            Action, probability
        """
        return self.act(state, weights), 1

    def expected_value(self, state, weights):
        """
        Calculate the expected Q value
        Args:
            state (1D array): Current state
            weights (array): Current weights

        Returns:
            Expected Q value
        """
        q = state.dot(weights)
        action = self.act(state, weights)
        return q[action]

    def probs(self, state, weights):
        """
        Calculate the probabilities of all actions
        Args:
            state (1D array): Current state
            weights (array): Current weights

        Returns:
            Action probabilities
        """
        action = self.act(state, weights)
        probs = np.zeros(4)
        probs[action] = 1
        return probs

