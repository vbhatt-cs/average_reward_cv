# import matplotlib.pyplot as plt
from gym.spaces import Discrete
import numpy as np
# from matplotlib.table import Table


class GridWorld:
    """
    Gridworld environment as specified in De Asis, Kristopher, and Richard S. Sutton. "Per-decision Multi-step
    Temporal Difference Learning with Control Variates." arXiv preprint arXiv:1807.01830 (2018).
    """

    def __init__(self):
        self.size = 5  # 5x5 grid
        self.state = (2, 2)  # Position of the agent
        self.prev_state = self.state  # For efficient rendering
        self.goal_states = [(0, 0), (self.size - 1, self.size - 1)]  # Terminal states
        self.action_space = Discrete(4)
        self.t = 0

        # # Rendering init
        # fig, ax = plt.subplots()
        # ax.set_axis_off()
        # self.grid = Table(ax, bbox=[0, 0, 1, 1])
        # width, height = 1.0 / self.size, 1.0 / self.size
        #
        # # Add cells
        # for i in range(self.size):
        #     for j in range(self.size):
        #         color = 'black' if (i, j) in self.goal_states else 'white'
        #         val = 'O' if self.state == (i, j) else ''
        #         self.grid.add_cell(i, j, width, height, text=val, loc='center', facecolor=color)
        #
        # ax.add_table(self.grid)

    def reset(self):
        """
        Reset the environment and return the starting state
        Returns:
            Starting state
        """
        self.state = (2, 2)  # Start at the center of the grid
        self.t = 0
        return self.state

    # def render(self):
    #     """
    #     Render the environment for visualization
    #     Adapted from https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/grid_world.py
    #     """
    #     self.grid.get_celld()[self.prev_state].get_text().set_text('')
    #     self.grid.get_celld()[self.state].get_text().set_text('O')
    #     plt.pause(0.1)
    #     plt.draw()

    def step(self, action):
        """
        Advance the environment by one step
        Args:
            action (int): Action to take

        Returns:
            next state, reward, if the episode is done, None
        """
        x, y = self.state

        if action == 0:  # North
            x = max(0, x - 1)
        elif action == 1:  # East
            y = min(self.size - 1, y + 1)
        elif action == 2:  # South
            x = min(self.size - 1, x + 1)
        elif action == 3:  # West
            y = max(0, y - 1)

        self.prev_state = self.state
        self.state = (x, y)

        if self.state in self.goal_states:
            reward = 1
            done = True
            self.state = (2, 2)  # Since in continuing case env is reset and it doesn't matter in episodic case
        else:
            reward = 0
            done = False

        self.t += 1
        # if self.t == 1000:
        #     reward = -1
        #     done = True
        #     self.state = (2, 2)  # Since in continuing case env is reset and it doesn't matter in episodic case

        return self.state, reward, done, None  # No info returned (kept to make it consistent with gym)


class MountainCar:
    """
    Mountain car environment
    """
    def __init__(self, rng):
        """
        Args:
            rng: Random state
        """
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng
        self.position = self.rng.uniform(-0.6, -0.4)
        self.velocity = 0.0
        self.velocity_limit = [-0.07, 0.07]
        self.position_limit = [-1.2, 0.6]
        self.action_space = Discrete(3)
        self.t = 0

    def reset(self):
        """
        Reset the environment and return the starting state
        Returns:
            Starting state
        """
        self.position = self.rng.uniform(-0.6, -0.4)
        self.velocity = 0.0
        self.t = 0
        return [self.position, self.velocity]

    def step(self, action):
        """
        Advance the environment by one step. Adapted from https://github.com/ShangtongZhang/
        reinforcement-learning-an-introduction/blob/master/chapter10/mountain_car.py
        Args:
            action (int): Action to take

        Returns:
            next state, reward, if the episode is done, None
        """
        done = False
        action -= 1  # Since action is 0, 1 or 2 but the equations assume -1, 0, 1
        new_velocity = self.velocity + 0.001 * action - 0.0025 * np.cos(3 * self.position)
        new_velocity = min(max(self.velocity_limit[0], new_velocity), self.velocity_limit[1])
        new_position = self.position + new_velocity
        new_position = min(max(self.position_limit[0], new_position), self.position_limit[1])
        reward = 0
        if new_position == self.position_limit[0]:
            new_velocity = 0.0

        self.position = new_position
        self.velocity = new_velocity

        if new_position == self.position_limit[1]:
            done = True
            reward = 1.0
            self.reset()

        self.t += 1
        # if self.t == 1000:
        #     done = True
        #     self.reset()

        return [self.position, self.velocity], reward, done, None


class RandomWalk:
    """
    Environment for random walk. Set behaviour policy to equiprobable random for random walk.
    Moving to right end puts the agent back in middle with a +1 reward. Similar for left end but with -1 reward.
    """

    def __init__(self, n_states):
        """
        Args:
            n_states (int): Number of states
        """
        self.n_states = n_states
        self.position = int(n_states / 2)
        self.action_space = Discrete(2)
        self.t = 0

    def reset(self):
        """
        Reset the environment and return the starting state
        Returns:
            Starting state
        """
        self.position = int(self.n_states / 2)
        self.t = 0
        state = np.zeros(self.n_states)
        state[self.position] = 1
        return state

    def step(self, action):
        """
        Advance the environment by one step. Adapted from https://github.com/ShangtongZhang/
        reinforcement-learning-an-introduction/blob/master/chapter10/mountain_car.py
        Args:
            action (int): Action to take

        Returns:
            next state, reward, if the episode is done, None
        """
        done = False
        action = action * 2 - 1  # Make the action +-1 from 0, 1
        self.position += action
        reward = 0
        if self.position == -1:
            reward = -1
            done = True
            self.reset()

        if self.position == self.n_states:
            reward = 1
            done = True
            self.reset()

        state = np.zeros(self.n_states)
        state[self.position] = 1

        self.t += 1
        # if self.t == 1000:
        #     done = True
        #     self.reset()

        return state, reward, done, None


def test_gridworld():
    """
    Testing if gridworld dynamics are correct
    """
    env = GridWorld()

    assert env.size == 5
    assert env.action_space == Discrete(4)

    true_state = (2, 2)
    obs = env.reset()
    assert obs == env.state == true_state

    # Test actions
    acts = [0, 1, 2, 3]
    true_states = [(1, 2), (1, 3), (2, 3), (2, 2)]

    for act, true_state in zip(acts, true_states):
        obs, reward, done, _ = env.step(act)
        assert obs == env.state == true_state
        assert reward == 0
        assert not done

    # Test goal states, transitions on edges
    env.reset()
    acts = [2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 1, 1, 1, 3, 3, 3, 3, 3, 0]
    for act in acts:
        obs, reward, done, _ = env.step(act)
        assert reward == 0
        assert not done

    obs, reward, done, _ = env.step(0)
    assert obs == env.state == (2, 2)
    assert reward == 1
    assert done

    env.reset()
    acts = [2, 2, 1, 1]
    for act in acts:
        obs, reward, done, _ = env.step(act)
    assert obs == env.state == (2, 2)
    assert reward == 1
    assert done


def test_gridworld_render():
    """
    Testing if the rendering works.
    A window should open with a 5x5 grid, with top left and bottom right corner marked black.
    Center cell should have a mark 'O' which moves to right and then bottom before terminating
    """
    env = GridWorld()
    env.reset()
    acts = [1, 1, 2, 2]
    for act in acts:
        env.render()
        env.step(act)


def test_randomwalk():
    rw = RandomWalk(5)

    actions = [1, 1, 1]
    true_pos = [3, 4, 2]

    obs = rw.reset()
    assert rw.position == int(np.where(obs == 1)[0]) == 2

    i = 0
    for i in range(len(actions) - 1):
        obs, reward, done, _ = rw.step(actions[i])
        assert rw.position == int(np.where(obs == 1)[0]) == true_pos[i]
        assert reward == 0
        assert not done

    i += 1
    obs, reward, done, _ = rw.step(actions[i])
    assert rw.position == int(np.where(obs == 1)[0]) == true_pos[i]
    assert reward == 1
    assert done

    actions = [0, 0, 0]
    true_pos = [1, 0, 2]

    rw.reset()
    i = 0
    for i in range(len(actions) - 1):
        obs, reward, done, _ = rw.step(actions[i])
        assert rw.position == int(np.where(obs == 1)[0]) == true_pos[i]
        assert reward == 0
        assert not done

    i += 1
    obs, reward, done, _ = rw.step(actions[i])
    assert rw.position == int(np.where(obs == 1)[0]) == true_pos[i]
    assert reward == -1
    assert done
