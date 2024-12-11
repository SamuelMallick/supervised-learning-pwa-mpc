import os
import pickle
import sys

import numpy as np

from slpwampc.agents.agent import Agent

sys.path.append(os.getcwd())
from examples.paper_2024.model import Model
from examples.paper_2024.mpc import MixedIntegerMpc, TimeVaryingAffineMpc
from slpwampc.misc.action_mapping import PwaActionMapper

np_random = np.random.default_rng(0)

SAVE = True

N = 11
nx, nu = Model.nx, Model.nu
system = Model.get_system()
system_dict = Model.get_system_dict()

mixed_integer_mpc = MixedIntegerMpc(system_dict, N, X_f=Model.X_f)
time_varying_affine_mpc = TimeVaryingAffineMpc(system_dict, N, X_f=Model.X_f)
agent = Agent(system, time_varying_affine_mpc=time_varying_affine_mpc, classifiers=[])

action_mapper = PwaActionMapper(len(system.A), N)
validation_samples = Model.sample_state_space(
    d=0.1, np_random=np_random, sample_strategy="grid"
)
valid_validation_states: list[np.ndarray] = []
optimal_validation_actions: list[int] = []
for idx, state in enumerate(validation_samples):
    print(f"evaluating optimal action for state {idx} of {len(validation_samples)}")
    optimal_action = agent.get_switching_sequence_from_mpc(state, mixed_integer_mpc)
    if optimal_action is not None:
        valid_validation_states.append(state)
        optimal_validation_actions.append(
            action_mapper.get_label_from_action(optimal_action)
        )
if SAVE:
    with open(
        f"examples/paper_2024/results/optimal_policy_N_{N}_samples.pkl",
        "wb",
    ) as file:
        pickle.dump(
            {"x": valid_validation_states, "a": optimal_validation_actions}, file
        )
