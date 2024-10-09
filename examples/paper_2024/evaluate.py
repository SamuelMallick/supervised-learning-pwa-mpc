import pickle

import numpy as np
import pandas as pd
import torch
from env import ThisParallelPwaEnv
from gymnasium.wrappers import TimeLimit
from model import get_system, get_system_details
from mpc_mld import ThisPwaMpcCvx
from mpcrl.wrappers.envs import MonitorEpisodes

from rollout.agents.pwa_agent import PwaAgent
from rollout.core.classifiers import FeedForwardPolicy

np.random.seed(0)
SAVE = True

sim_length = 20

nx, nu, l = get_system_details()
system = get_system()
env = MonitorEpisodes(TimeLimit(ThisParallelPwaEnv(system), sim_length))

N = 3
mpc = ThisPwaMpcCvx(N, nx, nu, l, system, soft_regions=False)
policy = FeedForwardPolicy()
name = "default_2"
policy.load_model(f"examples/simple_2D/data/models/{name}")
agent = PwaAgent(policy, mpc, system, first_region_from_policy=False)

df = pd.read_csv("examples/simple_2D/ICs.csv", header=None)
X0s = df.values
for i in range(X0s.shape[0]):
    x0 = X0s[[i], :].T
    x0 = torch.tensor(x0, dtype=torch.float32)[None]
    agent.evaluate(
        env,
        policy_trained=policy.evaluate_greedy,
        x0=x0,
        num_actions_attempts=1,
        seed=0,
    )

X = [env.observations[i].squeeze() for i in range(X0s.shape[0])]
U = [env.actions[i].squeeze() for i in range(X0s.shape[0])]
R = [env.rewards[i].squeeze() for i in range(X0s.shape[0])]

if SAVE:
    with open(
        f"eval_{name}.pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
