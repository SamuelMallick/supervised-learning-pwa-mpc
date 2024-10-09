import os
import pickle
import sys

import numpy as np

from rollout.agents.svm_agent import SvmAgent

sys.path.append(os.getcwd())
from examples.Mayne_2003.simple_2D.model import Model
from examples.Mayne_2003.simple_2D.mpc_mld import ThisMpcMld
from rollout.misc.action_mapping import PwaActionMapper

np_random = np.random.default_rng(0)

SAVE = True
first_region_from_policy = False

N = 8
nx, nu = Model.nx, Model.nu
system = Model.get_system()
system_dict = Model.get_system_dict()

mpc = ThisMpcMld(system_dict, N, nx, nu, X_f=Model.X_f, verbose=False)
agent = SvmAgent(
    system,
    mpc,
    N,
    first_region_from_policy,
)

action_mapper = PwaActionMapper(len(system.A), N if first_region_from_policy else N - 1)
validation_samples = Model.sample_state_space(
    d=0.05, np_random=np_random, sample_strategy="grid"
)
valid_validation_states: list[np.ndarray] = []
optimal_validation_actions: list[int] = []
for idx, state in enumerate(validation_samples):
    print(f"evaluating optimal action for state {idx} of {len(validation_samples)}")
    optimal_action = agent.get_switching_sequence_from_state(state)
    if optimal_action is not None:
        valid_validation_states.append(state)
        optimal_validation_actions.append(
            action_mapper.get_label_from_action(optimal_action)
        )
if SAVE:
    with open(
        f"optimal_policy_N_{N}_samples_Tset.pkl",
        "wb",
    ) as file:
        pickle.dump(
            {"x": valid_validation_states, "a": optimal_validation_actions}, file
        )
