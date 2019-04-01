import numpy as np

from tiles3 import IHT, tiles


class BaseFeature:
    """
    Base class for features
    """
    def __init__(self):
        """
        All classes need to define state_size
        """
        self.state_size = 0

    def extract(self, obs):
        """
        Extract state from observation
        """
        return NotImplementedError


class OneHot(BaseFeature):
    """
    One hot feature vector or tabular case (specialized for square gridworld)
    """
    def __init__(self, state_size):
        """
        Args:
            state_size (int): Number of states
        """
        super().__init__()
        self.state_size = state_size
        self.n = int(np.sqrt(self.state_size))

    def extract(self, obs):
        """
        Calculate the index based on the coordinates
        Args:
            obs (tuple): Co-ordinates

        Returns:
            One-hot vector with index corresponding to the coordinates equal to 1
        """
        x, y = obs
        state = np.zeros(self.state_size)
        state[x * self.n + y] = 1
        return state


class TileCoding(BaseFeature):
    """
    Wrapper class for tile coding
    """
    def __init__(self, n_tiles, n_tilings, limits):
        """
        Args:
            n_tiles (list or 1D array): Number of tiles in each dimension
            n_tilings (int): Number of tilings
            limits (list): List of (min, max) tuples for each dimension
        """
        super().__init__()
        self.n_tiles = np.array(n_tiles)
        self.n_tilings = n_tilings
        self.state_size = n_tilings * np.prod(self.n_tiles)
        self.iht = IHT(self.state_size)
        self.limits = np.array(limits)
        self.scaling = self.n_tiles / (self.limits[:, 1] - self.limits[:, 0])

    def extract(self, obs):
        """
        Set a list of indices to 1 based on observation
        Args:
            obs (list): List of floats

        Returns:
            Vector with all zeros except the list of indices set to 1
        """
        obs = np.array(obs)
        state = np.zeros(self.state_size)
        idx = tiles(self.iht, self.n_tilings, (obs - self.limits[:, 0]) * self.scaling)
        state[idx] = 1
        return state
