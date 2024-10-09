import pickle

import gurobipy as gp
import numpy as np
import pandas as pd
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from env import PwaEnv
from gymnasium.wrappers import TimeLimit
from model import get_system, get_system_details, get_system_dict
from mpcrl.wrappers.envs import MonitorEpisodes

np.random.seed(0)

SAVE = True

sim_length = 20

N = 3
nx, nu, l = get_system_details()
system = get_system()
system_dict = get_system_dict()
env = MonitorEpisodes(TimeLimit(PwaEnv(nx, nu, system), sim_length))


class Cent_MPC(MpcMld):
    Q_x = np.eye(nx)
    Q_u = np.eye(nu)

    def __init__(self, system: dict, N: int) -> None:
        # dynamics, state, and input constraints built in here with MLD model conversion
        super().__init__(system, N, verbose=True)

        obj = 0
        for k in range(N):
            obj += (
                self.x[:, k] @ self.Q_x @ self.x[:, [k]]
                + self.u[:, k] @ self.Q_u @ self.u[:, [k]]
            )
        obj += self.x[:, N] @ self.Q_x @ self.x[:, [N]]
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)


# controller
mpc = Cent_MPC(system_dict, N)
# agent
agent = MldAgent(mpc)

df = pd.read_csv("examples/simple_2D/ICs.csv", header=None)
X0s = df.values
for i in range(X0s.shape[0]):
    x0 = X0s[[i], :].T

    agent.evaluate(env=env, episodes=1, seed=1, env_reset_options={"x0": x0})


X = [env.observations[i].squeeze() for i in range(X0s.shape[0])]
U = [env.actions[i].squeeze() for i in range(X0s.shape[0])]
R = [env.rewards[i].squeeze() for i in range(X0s.shape[0])]

if SAVE:
    with open(
        f"MLD.pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
