import pickle
import warnings

import numpy as np
from model import Model
from mpc import MixedIntegerMpc, TightenedMixedIntegerMpc, TimeVaryingAffineMpc

from slpwampc.agents.parc_agent import ParcAgent

warnings.filterwarnings("ignore")

np_random = np.random.default_rng(0)
np.random.seed(2)

SAVE = True
first_region_from_policy = False

N = 11  # prediction horizon
d = 2  # spacing for initial grid sampling

nx, nu = Model.nx, Model.nu
system = Model.get_system()
system_dict = Model.get_system_dict()
# system.B = [np.array([[1], [1]]), np.array([[1], [1]])]
# system_dict["B"] = system.B

mixed_integer_mpc = MixedIntegerMpc(system_dict, N, X_f=Model.X_f)
time_varying_affine_mpc = TimeVaryingAffineMpc(system_dict, N, X_f=Model.X_f)
tighened_mpc = TightenedMixedIntegerMpc(system_dict, N, eps=0.1, X_f=Model.X_f)

# initial_state_samples = Model.sample_state_space(
#     d=d, np_random=np_random, sample_strategy="grid"
# )
initial_state_samples = Model.sample_state_space(
    num_points=30, np_random=np_random, sample_strategy="random"
)

agent = ParcAgent(
    system,
    mixed_integer_mpc=mixed_integer_mpc,
    time_varying_affine_mpc=time_varying_affine_mpc,
    N=N,
    first_region_from_policy=first_region_from_policy,
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
