import pickle
import warnings

import numpy as np
from model import Model
from mpc_mld import ThisMpcMld, ThisTightenedMpcMld

from slpwampc.agents.parc_agent import ParcAgent

warnings.filterwarnings("ignore")

np_random = np.random.default_rng(1)
np.random.seed(1)

SAVE = True

N = 5  # prediction horizon
d = 2  # spacing for initial grid sampling

nx, nu = Model.nx, Model.nu
system = Model.get_system()
system_dict = Model.get_system_dict()
# system.B = [np.array([[1], [1]]), np.array([[1], [1]])]
# system_dict["B"] = system.B

mpc = ThisMpcMld(system_dict, N, nx, nu, X_f=Model.X_f, verbose=False)
tighened_mpc = ThisTightenedMpcMld(
    system_dict, N, nx, nu, 0.01, X_f=Model.X_f, verbose=False
)
# initial_state_samples = Model.sample_state_space(
#     d=d, np_random=np_random, sample_strategy="grid"
# )
initial_state_samples = [
    Model.sample_state_space(
        num_points=15, np_random=np_random, sample_strategy="random", region=i
    )
    for i in range(2)
]

agent = ParcAgent(
    system,
    mpc,
    N,
    tightened_mpc=tighened_mpc,
    learn_infeasible_regions=True,
)
x, y, info = agent.train(initial_state_samples, plot=True, interactive=True)

if SAVE:
    agent.save(f"parc_agent_N_{N}")
    with open(f"training_N_{N}.pkl", "wb") as f:
        pickle.dump(
            {
                "x": x,
                "y": y,
                "iters": info["iters"],
                "num_regions": info["num_regions"],
            },
            f,
        )
