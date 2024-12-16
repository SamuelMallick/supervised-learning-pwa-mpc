import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from examples.simple_3D.model import Model
from slpwampc.misc.regions import Polytope

N = 5
system = Model.get_system()

with open(
    f"examples/simple_3D/results/optimal_policy_N_{N}_samples.pkl",
    "rb",
) as file:
    data = pickle.load(file)
    states_opt = data["x"]
    actions_opt = data["a"]

fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection="3d")
p = Polytope(system.D, system.E)
for o in np.unique(actions_opt):
    idx = np.where(np.array(actions_opt) == o)
    ax.scatter(
        np.array(states_opt)[idx, 0],
        np.array(states_opt)[idx, 1],
        np.array(states_opt)[idx, 2],
        label=f"Region {o}",
        s=0.5,
    )
p.plot(ax)
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_zlim(-12, 12)
plt.show()
