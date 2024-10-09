import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from rollout.agents.svm_agent import SvmAgent
from rollout.misc.action_mapping import PwaActionMapper

sys.path.append(os.getcwd())
from examples.simple_2D.model import Model
from examples.simple_2D.mpc_mld import ThisMpcMld

N = 8
d = 3
first_region_from_policy = False

check_feasability = True
plot = True
count_differences = True

with open(
    f"examples/simple_2D/results/samples/optimal_policy_N_{N}_samples.pkl",
    "rb",
) as file:
    data = pickle.load(file)
    states_opt = data["x"]
    actions_opt = data["a"]

name = f"svm_policy_N_{N}"
with open(
    f"examples/simple_2D/results/samples/{name}_d_{d}_samples.pkl",
    "rb",
) as file:
    data = pickle.load(file)
    states_l = data["x"]
    actions_l = data["a"]
with open(f"examples/simple_2D/results/models/{name}_d_{d}.pkl", "rb") as file:
    svm = pickle.load(file)

# veryify feasability of svm
if check_feasability:
    nx, nu = Model.nx, Model.nu
    system = Model.get_system()
    system_dict = Model.get_system_dict()
    mpc = ThisMpcMld(system_dict, N, nx, nu, verbose=False)
    agent = SvmAgent(
        system,
        mpc,
        N,
        first_region_from_policy,
    )
    agent.set_svm(svm)
    action_mapper = PwaActionMapper(
        len(system.A), N if first_region_from_policy else N - 1
    )
    infeasible_states = np.empty((0, nx, 1))
    for i, (state, action) in enumerate(zip(states_l, actions_l)):
        if i % 100 == 0:
            print(f"Checking state {i} of {len(states_l)}")
        switching_sequence = np.asarray(
            action_mapper.get_action_from_label(np.asarray(action).reshape(1, 1))
        ).reshape(-1)
        _, sol = mpc.solve_mpc_with_switching_sequence(
            state, switching_sequence, raises=False
        )
        if sol["cost"] == np.inf:
            raise ValueError(f"State {state} is infeasible")
            infeasible_states = np.vstack((infeasible_states, state.reshape(1, -1, 1)))
    # fig, ax = plt.subplots()
    # ax.plot(np.asarray(states_l)[:, 0, 0], np.asarray(states_l)[:, 1, 0], "o")
    # ax.plot(infeasible_states[:, 0, 0], infeasible_states[:, 1, 0], "x")
    # regions = agent.get_svm_regions_all()
    # for region in regions:
    #     plot_polytope(ax, region[0], region[1], color="blue", alpha=0.5)

if count_differences:
    # count differences in svm and optimal policy
    assert np.array_equal(np.asarray(states_opt), np.asarray(states_l))
    print(
        f"Number of differences: {np.sum(np.asarray(actions_opt) != np.asarray(actions_l))}"
    )

# num_points = len(states_opt)
# fig = plt.figure()
# ax_scat = fig.add_subplot(111, projection="3d")
# fig = plt.figure()
# ax_surf = fig.add_subplot(111, projection="3d")
# fig = plt.figure()
# ax_diff = fig.add_subplot(111, projection="3d")
# x_o = np.asarray([states_opt[i][0, 0] for i in range(num_points)])
# y_o = np.asarray([states_opt[i][1, 0] for i in range(num_points)])
# z_o = np.asarray([actions_opt[i] for i in range(num_points)])
# ax_scat.scatter(x_o, y_o, z_o, color="blue", alpha=1, antialiased=True)
# ax_surf.plot_trisurf(x_o, y_o, z_o, color="blue", linewidth=0)

# x_l = np.asarray([states_l[i][0, 0] for i in range(num_points)])
# y_l = np.asarray([states_l[i][1, 0] for i in range(num_points)])
# z_l = np.asarray([actions_l[i] for i in range(num_points)]).squeeze()
# ax_scat.scatter(x_l, y_l, z_l, color="red", alpha=1, antialiased=True)
# ax_surf.plot_trisurf(x_l, y_l, z_l, color="red", linewidth=0)

# ax_diff.plot_trisurf(x_o, y_o, z_o - z_l, color="red", linewidth=0)

plt.show()
