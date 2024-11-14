import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from slpwampc.agents.agent import Agent

sys.path.append(os.getcwd())
from examples.paper_2024.model import Model
from examples.paper_2024.mpc import MixedIntegerMpc, TimeVaryingAffineMpc

N = 12  # prediction horizon

nx, nu = Model.nx, Model.nu
system = Model.get_system()
system_dict = Model.get_system_dict()

mixed_integer_mpc = MixedIntegerMpc(system_dict, N, X_f=Model.X_f)
time_varying_affine_mpc = TimeVaryingAffineMpc(system_dict, N, X_f=Model.X_f)
agent = Agent(
    system,
    mixed_integer_mpc=mixed_integer_mpc,
    time_varying_affine_mpc=time_varying_affine_mpc,
    N=N,
    learn_infeasible_regions=True,
)
agent.load(f"examples/paper_2024/results/parc_agent_N_{N}")

regions = agent.get_regions()

print(f"Number of regions: {len(regions)}")

# label each region using the trained tree # TODO get labels directly from c code
fig, ax = plt.subplots()
labels = np.unique([region.label for region in regions])
for i, label in enumerate(labels):
    for region in regions:
        if region.label == label:
            region.plot(ax=ax, color=f"C{i}" if label != -1 else "black")

plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.show()
