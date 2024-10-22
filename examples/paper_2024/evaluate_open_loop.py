import pickle

import matplotlib.pyplot as plt
import numpy as np
from model import Model

from mpc import TimeVaryingAffineMpc, MixedIntegerMpc
from slpwampc.agents.parc_agent import ParcAgent

np_random = np.random.default_rng(0)
np.random.seed(2)

GENERATE = True  # if false we just plot from already saved data

N = 10  # prediction horizon

nx, nu = Model.nx, Model.nu
system = Model.get_system()
system_dict = Model.get_system_dict()

mixed_integer_mpc = MixedIntegerMpc(system_dict, N, X_f=Model.X_f)
time_varying_affine_mpc = TimeVaryingAffineMpc(system_dict, N, X_f=Model.X_f)
agent = ParcAgent(
    system,
    mixed_integer_mpc,  # not important which mpc is passed here
    N,
    learn_infeasible_regions=True,
)
agent.load(f"examples/paper_2024/results/parc_agent_N_{N}")

if GENERATE:
    initial_state_samples = Model.sample_state_space(
        d=0.1, np_random=np_random, sample_strategy="grid", num_points=2000
    )
    states = []
    costs_opt = []
    costs_subopt = []
    for i, x0 in enumerate(initial_state_samples):
        print(f"{i}/{len(initial_state_samples)}")
        x0 = x0.reshape(-1, 1)
        sol = mixed_integer_mpc.solve({"x_0": x0})
        if sol.success:
            states.append(x0)
            costs_opt.append(sol.f)
            delta = agent.get_switching_sequence(x0)
            if delta is None:
                costs_subopt.append(-1)
            else:
                time_varying_affine_mpc.set_sequence(delta.flatten().tolist())
                sol = time_varying_affine_mpc.solve({"x_0": x0})
                if not sol.success:
                    raise ValueError("Infeasible problem for the given delta.")
                costs_subopt.append(sol.f)
    with open(f"examples/paper_2024/results/evaluate_open_loop_N_{N}.pkl", "wb") as f:
        pickle.dump(
            {"states": states, "costs_opt": costs_opt, "costs_subopt": costs_subopt}, f
        )

else:
    with open(f"examples/paper_2024/results/evaluate_open_loop_N_{N}.pkl", "rb") as f:
        data = pickle.load(f)
        states = data["states"]
        costs_opt = data["costs_opt"]
        costs_subopt = data["costs_subopt"]

    f = np.array(
        [
            100 * (costs_subopt[i] - costs_opt[i]) / costs_opt[i]
            for i in range(len(costs_opt))
            if costs_subopt[i] != -1 and costs_opt[i] != 0
        ]
    )
    x = np.array(
        [
            states[i][0]
            for i in range(len(states))
            if costs_subopt[i] != -1 and costs_opt[i] != 0
        ]
    ).reshape(-1)
    y = np.array(
        [
            states[i][1]
            for i in range(len(states))
            if costs_subopt[i] != -1 and costs_opt[i] != 0
        ]
    ).reshape(-1)
    x_o = np.array([states[i][0] for i in range(len(states)) if costs_subopt[i] == -1])
    y_o = np.array([states[i][1] for i in range(len(states)) if costs_subopt[i] == -1])
    contour = plt.tripcolor(x, y, f, cmap="coolwarm")
    plt.colorbar(contour)
    contour.set_rasterized(True)
    plt.axis("off")
    plt.show()
