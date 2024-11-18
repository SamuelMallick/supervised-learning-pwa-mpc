import pickle
import warnings

import numpy as np
from model import Model
from mpc import MixedIntegerMpc, TightenedMixedIntegerMpc, TimeVaryingAffineMpc

from slpwampc.agents.agent import Agent
from slpwampc.core.classifiers.parc import ParcEnsemble
from slpwampc.core.classifiers.pwl_sep import PwlSep

warnings.filterwarnings("ignore")

np_random = np.random.default_rng(1)
np.random.seed(1)

SAVE = False

N = 5  # prediction horizon
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
initial_state_samples = [
    Model.sample_state_space(
        num_points=45, np_random=np_random, sample_strategy="random", region=i
    )
    for i in range(2)
]

# classifiers = [
#     PwlSep(A=np.vstack([S, system.D]), b=np.vstack([T, system.E]))
#     for S, T in zip(system.S, system.T)
# ]

classifiers = [
    Parc(
        A=np.vstack([S, system.D]),
        b=np.vstack([T, system.E]),
        K=15,
        alpha=1.0e2,
        maxiter=150,
        sigma=15,
        separation="Softmax",
        verbose=0,
        min_number=1,
    )
    for S, T in zip(system.S, system.T)
]

agent = Agent(
    system=system,
    time_varying_affine_mpc=time_varying_affine_mpc,
    classifiers=classifiers,
)
x, y, info = agent.train(
    initial_state_samples,
    mixed_integer_mpc=mixed_integer_mpc,
    learn_infeasible_regions=True,
    tightened_mpc=tighened_mpc,
    plot=True,
    interactive=True,
)

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
