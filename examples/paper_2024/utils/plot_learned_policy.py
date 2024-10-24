import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from slpwampc.agents.parc_agent import ParcAgent
sys.path.append(os.getcwd())
from examples.paper_2024.model import Model
from examples.paper_2024.mpc_mld import ThisMpcMld

N = 12  # prediction horizon

nx, nu = Model.nx, Model.nu
system = Model.get_system()
system_dict = Model.get_system_dict()

mpc = ThisMpcMld(system_dict, N, nx, nu, X_f=Model.X_f, verbose=False)
agent = ParcAgent(
    system,
    mpc,
    N,
    learn_infeasible_regions=True,
)
agent.load(f"examples/paper_2024/results/parc_agent_N_{N}")

regions = agent.get_regions()

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
