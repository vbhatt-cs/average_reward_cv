"""
This code is specific to the environment setting used in this project.
"""

import numpy as np

from features import OneHot
from envs import GridWorld
from policies import BiasedRandom, ScriptedPolicy

env = GridWorld()
features = OneHot(25)
rng = np.random.RandomState(0)
policy = BiasedRandom(0.5, 0, env.action_space.n, rng)
# policy = ScriptedPolicy()
value = np.zeros(25)

# Coeffs for linear system. 23 states (2 goal states are not considered) and Rbar are variables
a = np.zeros((24, 24))
b = np.zeros(24)

# Calculate a, b using Bellman equations
k = 0
for i in range(5):
    for j in range(5):
        if (i, j) == (0, 0) or (i, j) == (4, 4):
            continue
        else:
            env.reset()
            env.state = (i, j)
            state = features.extract(env.state)
            probs = policy.probs(state, None)

            # Since the environment is deterministic, next state and reward are fixed given state and action
            for action in range(env.action_space.n):
                env.reset()
                env.state = (i, j)
                state = features.extract(env.state)
                obs, reward, done, _ = env.step(action)
                next_state = features.extract(obs)

                # Coeff corresponding to next state.
                # -1 when indexing next state since state[0] is goal state and not a variable
                a[k, int(np.where(next_state == 1)[0]) - 1] += -1 * probs[action]
                b[k] += reward * probs[action]

            # Coeff for current state
            a[k, k] += 1
            # Coeff for Rbar
            a[k, -1] = 1
            k += 1

# Last constraint sets value of initial state to 0 (the condition can be arbitrary)
a[k, 11] = 1
b[k] = 0

x = np.linalg.solve(a, b)
value[1:24] = x[:-1]
rbar = x[-1]

print("Values: {}, Rbar: {}".format(value.reshape((5, 5)), rbar))
