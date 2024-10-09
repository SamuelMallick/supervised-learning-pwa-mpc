import os
import pickle
import sys

import numpy as np

from rollout.agents.svm_agent import SvmAgent

sys.path.append(os.getcwd())
from examples.simple_2D.model import Model
from examples.simple_2D.mpc_mld import ThisMpcMld
from rollout.misc.action_mapping import PwaActionMapper

np_random = np.random.default_rng(0)

SAVE = True
first_region_from_policy = False
validation_points = 10000

N = 8
d = 3
with open(
    f"examples/simple_2D/results/models/svm_policy_N_{N}_d_{d}.pkl", "rb"
) as file:
    svm = pickle.load(file)

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

action_mapper = PwaActionMapper(len(system.A), N if first_region_from_policy else N - 1)
validation_samples = Model.sample_state_space(
    num_points=validation_points, np_random=np_random, sample_strategy="random"
)
valid_validation_states: list[np.ndarray] = []
validation_actions: list[int] = []
for idx, state in enumerate(validation_samples):
    if idx % 100 == 0:
        print(f"evaluating svm action for state {idx} of {len(validation_samples)}")
    optimal_action = agent.get_switching_sequence_from_state(state)
    if optimal_action is not None:
        valid_validation_states.append(state)
        svm_action = agent.get_svm_action(state)
        validation_actions.append(svm_action.item())
if SAVE:
    with open(
        f"svm_policy_N_{N}_d_{d}_samples.pkl",
        "wb",
    ) as file:
        pickle.dump({"x": valid_validation_states, "a": validation_actions}, file)
