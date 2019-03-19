from collections import deque

import numpy as np


class BaseAlg:
    """
    Base class for algorithms
    """
    # Policy is fixed for prediction, function of Q value for control
    def __init__(self, behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar):
        """
        Args:
            behaviour_policy (function): Policy to follow to generate experience. Should take state, weights as args
            target_policy (function): Policy to learn about. Should take state, weights as args
            alpha (float): Learning rate for weights
            beta (float): Learning rate for average reward
            off_policy (bool): If learning is off-policy
            cv (bool): If control variates should be used
            full_rbar (bool): If update for average reward should use n-step/lambda method instead of one step TD error
            cv_rbar (bool): If control variates should be used in average reward update
        """
        self.behaviour_policy = behaviour_policy
        self.target_policy = target_policy if off_policy else behaviour_policy
        self.rbar = 0
        self.alpha = alpha
        self.beta = beta
        self.weights = None

        self.rho = None if off_policy else 1
        self.cv = cv
        self.full_rbar = full_rbar
        self.cv_rbar = cv_rbar

    def act(self, state):
        """
        Takes one step using behaviour policy
        Args:
            state (1D array): Current state

        Returns:
            Action
        """
        return self.behaviour_policy(state, self.weights)

    def reset(self, state):
        """
        Called when the experiment begins
        Args:
            state (1D array): Initial state

        Returns:
            None
        """
        raise NotImplementedError

    def train(self, reward, next_state):
        """
        Called on each iteration for updating weights
        Args:
            reward (float): Reward obtained in this step
            next_state (1D array): Next state

        Returns:
            None
        """
        raise NotImplementedError


class RLearning:
    def __init__(self, state_size):
        pass

    def train(self, reward, next_state):
        raise NotImplementedError


class NStepPrediction(BaseAlg):
    """
    N-step TD
    """
    def __init__(self, behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar, n, state_size):
        """
        Args:
            behaviour_policy (function): Policy to follow to generate experience. Should take state, weights as args
            target_policy (function): Policy to learn about. Should take state, weights as args
            alpha (float): Learning rate for weights
            beta (float): Learning rate for average reward
            off_policy (bool): If learning is off-policy
            cv (bool): If control variates should be used
            full_rbar (bool): If update for average reward should use n-step/lambda method instead of one step TD error
            cv_rbar (bool): If control variates should be used in average reward update
            n (int): Number of time steps before updating
            state_size (int): Length of state feature vector
        """
        super().__init__(behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar)
        self.weights = np.zeros(state_size)
        self.n = n
        self.r_history = deque(maxlen=n)
        self.s_history = deque(maxlen=n + 1)

    def reset(self, state):
        """
        Store initial state in the history
        Args:
            state (1D array): Initial state

        Returns:
            None
        """
        self.s_history.append(state)

    def train(self, reward, next_state):
        """
        Do n-step TD update
        Args:
            reward (float): Reward obtained in this step
            next_state (1D array): Next state

        Returns:
            None
        """
        self.r_history.append(reward)
        self.s_history.append(next_state)

        if len(self.r_history) == self.n:
            # TODO: Off-policy, CV
            # s_history[-1] = s_(tau+n), s_history[0] = s_tau
            delta = sum(self.r_history) - self.n * self.rbar + self.s_history[-1].dot(self.weights) - self.s_history[
                0].dot(self.weights)
            if self.full_rbar:
                self.rbar += self.beta * delta
            else:
                # r_history[0] = r_(tau+1)
                delta_one_step = self.r_history[0] - self.rbar + self.s_history[1].dot(self.weights) - self.s_history[
                    0].dot(self.weights)
                self.rbar += self.beta * delta_one_step

            self.weights += self.alpha * delta * self.s_history[0]


class LambdaPrediction(BaseAlg):
    """
    TD(lambda)
    """
    def __init__(self, behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar, lam,
                 state_size):
        """
        Args:
            behaviour_policy (function): Policy to follow to generate experience. Should take state, weights as args
            target_policy (function): Policy to learn about. Should take state, weights as args
            alpha (float): Learning rate for weights
            beta (float): Learning rate for average reward
            off_policy (bool): If learning is off-policy
            cv (bool): If control variates should be used
            full_rbar (bool): If update for average reward should use n-step/lambda method instead of one step TD error
            cv_rbar (bool): If control variates should be used in average reward update
            lam (float): Lambda
            state_size (int): Length of state feature vector
        """
        super().__init__(behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar)
        self.weights = np.zeros(state_size)
        self.lam = lam
        self.z = np.zeros_like(self.weights)
        self.z_rbar = 0
        self.state = None

    def reset(self, state):
        """
        Store initial state
        Args:
            state (1D array): Initial state

        Returns:
            None
        """
        self.state = state

    def train(self, reward, next_state):
        """
        Do TD(lambda) update
        Args:
            reward (float): Reward obtained in this step
            next_state (1D array): Next state

        Returns:
            None
        """
        # TODO: Off-policy, CV
        delta = reward - self.rbar + next_state.dot(self.weights) - self.state.dot(self.weights)
        self.z = self.lam * self.z + self.state
        if self.full_rbar:
            self.z_rbar = self.lam * self.z_rbar + 1
            self.rbar += self.beta * delta * self.z_rbar
        else:
            self.rbar += self.beta * delta

        self.weights += self.alpha * delta * self.z
        self.state = next_state


class NStepControl(BaseAlg):
    """
    N-step SARSA / expected SARSA
    """
    def __init__(self, behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar, n, state_size,
                 action_size):
        """
        Args:
            behaviour_policy (function): Policy to follow to generate experience. Should take state, weights as args
            target_policy (function): Policy to learn about. Should take state, weights as args
            alpha (float): Learning rate for weights
            beta (float): Learning rate for average reward
            off_policy (bool): If learning is off-policy
            cv (bool): If control variates should be used
            full_rbar (bool): If update for average reward should use n-step/lambda method instead of one step TD error
            cv_rbar (bool): If control variates should be used in average reward update
            n (int): Number of time steps before updating
            state_size (int): Length of state feature vector
            action_size (int): Size of action space
        """
        super().__init__(behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar)
        self.weights = np.zeros((state_size, action_size))
        self.n = n
        self.r_history = deque(maxlen=n)
        self.s_history = deque(maxlen=n + 1)
        self.a_history = deque(maxlen=n + 1)

    def reset(self, state):
        """
        Store initial state and calculate the first action
        Args:
            state (1D array): Initial state

        Returns:
            None
        """
        self.s_history.append(state)
        action = self.behaviour_policy(state, self.weights)
        self.a_history.append(action)

    def act(self, state):
        """
        Return a_t which is stored in prev step as a_(t+1)
        Args:
            state (1D array): Current state

        Returns:
            Action
        """
        return self.a_history[-1]

    def train(self, reward, next_state):
        """
        Do n-step SARSA / expected SARSA update
        Args:
            reward (float): Reward obtained in this step
            next_state (1D array): Next state

        Returns:
            None
        """
        self.r_history.append(reward)
        self.s_history.append(next_state)
        next_action = self.behaviour_policy(next_state, self.weights)
        self.a_history.append(next_action)

        if len(self.r_history) == self.n:
            # TODO: Off-policy, CV
            # s_history[-1] = s_(tau+n), s_history[0] = s_tau
            delta = sum(self.r_history) - self.n * self.rbar + self.s_history[-1].dot(self.weights)[
                self.a_history[-1]] - self.s_history[0].dot(self.weights)[self.a_history[0]]
            if self.full_rbar:
                self.rbar += self.beta * delta
            else:
                # r_history[0] = r_(tau+1)
                delta_one_step = self.r_history[0] - self.rbar + self.s_history[1].dot(self.weights)[
                    self.a_history[1]] - self.s_history[0].dot(self.weights)[self.a_history[0]]
                self.rbar += self.beta * delta_one_step

            self.weights[:, self.a_history[0]] += self.alpha * delta * self.s_history[0]


class LambdaControl(BaseAlg):
    """
    SARSA(lambda)
    """
    def __init__(self, behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar, lam,
                 state_size, action_size):
        """
        Args:
            behaviour_policy (function): Policy to follow to generate experience. Should take state, weights as args
            target_policy (function): Policy to learn about. Should take state, weights as args
            alpha (float): Learning rate for weights
            beta (float): Learning rate for average reward
            off_policy (bool): If learning is off-policy
            cv (bool): If control variates should be used
            full_rbar (bool): If update for average reward should use n-step/lambda method instead of one step TD error
            cv_rbar (bool): If control variates should be used in average reward update
            lam (int): Lambda
            state_size (int): Length of state feature vector
            action_size (int): Size of action space
        """
        super().__init__(behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar)
        self.weights = np.zeros((state_size, action_size))
        self.lam = lam
        self.z = np.zeros_like(self.weights)
        self.z_rbar = 0
        self.state = None
        self.action = None

    def reset(self, state):
        """
        Store initial state and calculate the first action
        Args:
            state (1D array): Initial state

        Returns:
            None
        """
        self.state = state
        self.action = self.behaviour_policy(state, self.weights)

    def act(self, state):
        """
        Return a_t which is stored in prev step as a_(t+1)
        Args:
            state (1D array): Current state

        Returns:
            Action
        """
        return self.action

    def train(self, reward, next_state):
        """
        Do SARSA(lambda)
        Args:
            reward (float): Reward obtained in this step
            next_state (1D array): Next state

        Returns:
            None
        """
        next_action = self.behaviour_policy(next_state, self.weights)

        # TODO: Off-policy, CV
        delta = reward - self.rbar + next_state.dot(self.weights)[next_action] - self.state.dot(self.weights)[
            self.action]
        self.z = self.lam * self.z
        self.z[:, self.action] += self.state
        if self.full_rbar:
            self.z_rbar = self.lam * self.z_rbar + 1
            self.rbar += self.beta * delta * self.z_rbar
        else:
            self.rbar += self.beta * delta

        self.weights += self.alpha * delta * self.z
        self.state = next_state
        self.action = next_action
