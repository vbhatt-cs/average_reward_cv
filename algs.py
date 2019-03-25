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
            behaviour_policy (BasePolicy): Policy to follow to generate experience. Should take state, weights as args
            target_policy (BasePolicy): Policy to learn about. Should take state, weights as args
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

        self.cv = cv
        self.full_rbar = full_rbar
        self.cv_rbar = cv_rbar

    def act(self, state, test=False):
        """
        Takes one step using behaviour policy
        Args:
            state (1D array): Current state
            test (bool): If the mode is testing (target policy is used if true)

        Returns:
            Action
        """
        raise NotImplementedError

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
    """
    R-Learning algorithm as given in Schwartz, Anton. "A reinforcement learning method for maximizing undiscounted
    rewards." Proceedings of the tenth international conference on machine learning. Vol. 298. 1993.
    """

    def __init__(self, behaviour_policy, alpha, beta, state_size, action_size):
        """
        Args:
            behaviour_policy (BasePolicy): Policy to follow to generate experience. Should take state, weights as args
            alpha (float): Learning rate for weights
            beta (float): Learning rate for average reward
            state_size (int): Length of state feature vector
            action_size (int): Size of action space
        """
        self.behaviour_policy = behaviour_policy
        self.alpha = alpha
        self.beta = beta
        self.weights = np.zeros((state_size, action_size))
        self.rbar = 0

        self.state = None
        self.action = None

    def act(self, state, test=False):
        """
        Takes one step using behaviour policy
        Args:
            state (1D array): Current state
            test (bool): If the mode is testing (target policy is used if true)

        Returns:
            Action
        """
        if test:
            return state.dot(self.weights).argmax()
        else:
            self.action = self.behaviour_policy.act(state, self.weights)
            return self.action

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
        Do R-Learning update
        Args:
            reward (float): Reward obtained in this step
            next_state (1D array): Next state

        Returns:
            None
        """
        q = self.state.dot(self.weights)
        next_q = next_state.dot(self.weights)
        delta = reward - self.rbar + np.max(next_q) - q[self.action]
        self.weights[:, self.action] += self.alpha * delta * self.state
        if self.action == np.argmax(q):
            self.rbar += self.beta * delta
        self.state = next_state


class NStepPrediction(BaseAlg):
    """
    N-step TD
    """

    def __init__(self, behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar, n, state_size):
        """
        Args:
            behaviour_policy (BasePolicy): Policy to follow to generate experience. Should take state, weights as args
            target_policy (BasePolicy): Policy to learn about. Should take state, weights as args
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
        self.rho_history = deque(maxlen=n)

    def reset(self, state):
        """
        Store initial state in the history
        Args:
            state (1D array): Initial state

        Returns:
            None
        """
        self.s_history.append(state)

    def act(self, state, test=False):
        """
        Takes one step using behaviour policy
        Args:
            state (1D array): Current state
            test (bool): If the mode is testing (target policy is used if true)

        Returns:
            Action
        """
        if test:
            return self.target_policy.act(state, self.weights)
        else:
            action, prob = self.behaviour_policy.act_prob(state, self.weights)
            rho = self.target_policy.prob(state, action, self.weights) / prob
            self.rho_history.append(rho)
            return action

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

        # s_history[-1] = s_(tau+n), s_history[0] = s_tau, r_history[0] = r_(tau+1), rho_history[0] = rho_(tau)
        if len(self.r_history) == self.n:
            g = self.s_history[-1].dot(self.weights)
            g_cv = self.s_history[-1].dot(self.weights)
            for i in range(1, self.n + 1):
                g = self.rho_history[-i] * (self.r_history[-i] - self.rbar + g)
                g_cv = self.rho_history[-i] * (self.r_history[-i] - self.rbar + g_cv) + (1 - self.rho_history[-i]) * \
                       self.s_history[-i - 1].dot(self.weights)

            delta_cv = g_cv - self.s_history[0].dot(self.weights)
            delta = g - self.s_history[0].dot(self.weights)

            if self.full_rbar:
                if self.cv_rbar:
                    self.rbar += self.beta * delta_cv
                else:
                    self.rbar += self.beta * delta
            else:
                if self.cv_rbar:
                    delta_one_step = self.rho_history[0] * (
                            self.r_history[0] - self.rbar + self.s_history[1].dot(self.weights) - self.s_history[
                        0].dot(self.weights))
                else:
                    delta_one_step = self.rho_history[0] * (
                            self.r_history[0] - self.rbar + self.s_history[1].dot(self.weights)) - self.s_history[
                                         0].dot(self.weights)
                self.rbar += self.beta * delta_one_step

            if self.cv:
                self.weights += self.alpha * delta_cv * self.s_history[0]
            else:
                self.weights += self.alpha * delta * self.s_history[0]


class LambdaPrediction(BaseAlg):
    """
    TD(lambda)
    """

    def __init__(self, behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar, lam,
                 state_size):
        """
        Args:
            behaviour_policy (BasePolicy): Policy to follow to generate experience. Should take state, weights as args
            target_policy (BasePolicy): Policy to learn about. Should take state, weights as args
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
        self.rho = 1
        self.prev_rho = 1

    def reset(self, state):
        """
        Store initial state
        Args:
            state (1D array): Initial state

        Returns:
            None
        """
        self.state = state

    def act(self, state, test=False):
        """
        Takes one step using behaviour policy
        Args:
            state (1D array): Current state
            test (bool): If the mode is testing (target policy is used if true)

        Returns:
            Action
        """
        if test:
            return self.target_policy.act(state, self.weights)
        else:
            action, prob = self.behaviour_policy.act_prob(state, self.weights)
            self.rho = self.target_policy.prob(state, action, self.weights) / prob
            return action

    def train(self, reward, next_state):
        """
        Do TD(lambda) update
        Args:
            reward (float): Reward obtained in this step
            next_state (1D array): Next state

        Returns:
            None
        """
        delta_cv = reward - self.rbar + next_state.dot(self.weights) - self.state.dot(self.weights)
        delta = self.rho * (reward - self.rbar + next_state.dot(self.weights)) - self.state.dot(self.weights)
        if self.cv:
            self.z = self.rho * (self.lam * self.z + self.state)
        else:
            self.z = self.lam * self.prev_rho * self.z + self.state

        if self.full_rbar:
            if self.cv_rbar:
                self.z_rbar = self.rho * (self.lam * self.z_rbar + 1)
                self.rbar += self.beta * delta_cv * self.z_rbar
            else:
                self.z_rbar = self.lam * self.prev_rho * self.z_rbar + 1
                self.rbar += self.beta * delta * self.z_rbar
        else:
            if self.cv_rbar:
                self.rbar += self.beta * self.rho * delta_cv
            else:
                self.rbar += self.beta * delta

        if self.cv:
            self.weights += self.alpha * delta_cv * self.z
        else:
            self.weights += self.alpha * delta * self.z
        self.state = next_state
        self.prev_rho = self.rho


class NStepControl(BaseAlg):
    """
    N-step SARSA / expected SARSA
    """

    def __init__(self, behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar, n, state_size,
                 action_size):
        """
        Args:
            behaviour_policy (BasePolicy): Policy to follow to generate experience. Should take state, weights as args
            target_policy (BasePolicy): Policy to learn about. Should take state, weights as args
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
        self.rho_history = deque(maxlen=n)

    def reset(self, state):
        """
        Store initial state and calculate the first action
        Args:
            state (1D array): Initial state

        Returns:
            None
        """
        self.s_history.append(state)
        action, prob = self.behaviour_policy.act_prob(state, self.weights)
        self.a_history.append(action)
        rho = self.target_policy.prob(state, action, self.weights) / prob
        self.rho_history.append(rho)

    def act(self, state, test=False):
        """
        Return a_t which is stored in prev step as a_(t+1)
        Args:
            state (1D array): Current state
            test (bool): If the mode is testing (target policy is used if true)

        Returns:
            Action
        """
        if test:
            return self.target_policy.act(state, self.weights)
        else:
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
        next_action, prob = self.behaviour_policy.act_prob(next_state, self.weights)
        self.a_history.append(next_action)
        rho = self.target_policy.prob(next_state, next_action, self.weights) / prob
        self.rho_history.append(rho)

        # s_history[-1] = s_(tau+n), s_history[0] = s_tau, r_history[0] = r_(tau+1), rho_history[0] = rho_(tau+1)
        if len(self.r_history) == self.n:
            g = self.s_history[-1].dot(self.weights)[self.a_history[-1]]
            g_cv = self.s_history[-1].dot(self.weights)[self.a_history[-1]]
            for i in range(1, self.n + 1):
                g = self.r_history[-i] - self.rbar + self.rho_history[-i] * g
                expected_v = self.target_policy.expected_value(self.s_history[-i], self.weights)
                g_cv = self.r_history[-i] - self.rbar + self.rho_history[-i] * (
                        g_cv - self.s_history[-i].dot(self.weights)[self.a_history[-i]]) + expected_v

            delta_cv = g_cv - self.s_history[0].dot(self.weights)[self.a_history[0]]
            delta = g - self.s_history[0].dot(self.weights)[self.a_history[0]]

            if self.full_rbar:
                if self.cv_rbar:
                    self.rbar += self.beta * delta_cv
                else:
                    self.rbar += self.beta * delta
            else:
                if self.cv_rbar:
                    expected_v = self.target_policy.expected_value(self.s_history[1], self.weights)
                    delta_one_step = self.r_history[0] - self.rbar + expected_v - \
                                     self.s_history[0].dot(self.weights)[self.a_history[0]]
                else:
                    delta_one_step = self.r_history[0] - self.rbar + self.rho_history[0] * \
                                     self.s_history[1].dot(self.weights)[self.a_history[1]] - \
                                     self.s_history[0].dot(self.weights)[self.a_history[0]]
                self.rbar += self.beta * delta_one_step

            if self.cv:
                self.weights[:, self.a_history[0]] += self.alpha * delta_cv * self.s_history[0]
            else:
                self.weights[:, self.a_history[0]] += self.alpha * delta * self.s_history[0]


class LambdaControl(BaseAlg):
    """
    SARSA(lambda)
    """

    def __init__(self, behaviour_policy, target_policy, alpha, beta, off_policy, cv, full_rbar, cv_rbar, lam,
                 state_size, action_size):
        """
        Args:
            behaviour_policy (BasePolicy): Policy to follow to generate experience. Should take state, weights as args
            target_policy (BasePolicy): Policy to learn about. Should take state, weights as args
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
        self.rho = 1

    def reset(self, state):
        """
        Store initial state and calculate the first action
        Args:
            state (1D array): Initial state

        Returns:
            None
        """
        self.state = state
        self.action, prob = self.behaviour_policy.act_prob(state, self.weights)
        self.rho = self.target_policy.prob(state, self.action, self.weights) / prob

    def act(self, state, test=False):
        """
        Return a_t which is stored in prev step as a_(t+1)
        Args:
            state (1D array): Current state
            test (bool): If the mode is testing (target policy is used if true)

        Returns:
            Action
        """
        if test:
            return self.target_policy.act(state, self.weights)
        else:
            return self.action

    def train(self, reward, next_state):
        """
        Do SARSA(lambda) update
        Args:
            reward (float): Reward obtained in this step
            next_state (1D array): Next state

        Returns:
            None
        """
        next_action, prob = self.behaviour_policy.act_prob(next_state, self.weights)
        # rho = rho_(t+1), self.rho = rho_t
        rho = self.target_policy.prob(next_state, next_action, self.weights) / prob

        expected_v = self.target_policy.expected_value(next_state, self.weights)
        delta_cv = reward - self.rbar + expected_v - self.state.dot(self.weights)[self.action]
        delta = reward - self.rbar + rho * next_state.dot(self.weights)[next_action] - self.state.dot(self.weights)[
            self.action]
        self.z = self.lam * self.rho * self.z
        self.z[:, self.action] += self.state
        if self.full_rbar:
            self.z_rbar = self.lam * self.rho * self.z_rbar + 1
            if self.cv_rbar:
                self.rbar += self.beta * delta_cv * self.z_rbar
            else:
                self.rbar += self.beta * delta * self.z_rbar
        else:
            if self.cv_rbar:
                self.rbar += self.beta * delta_cv
            else:
                self.rbar += self.beta * delta

        if self.cv:
            self.weights += self.alpha * delta_cv * self.z
        else:
            self.weights += self.alpha * delta * self.z

        self.state = next_state
        self.action = next_action
        self.rho = rho
