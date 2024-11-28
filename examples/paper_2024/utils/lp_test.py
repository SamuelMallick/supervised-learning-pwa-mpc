import numpy as np
from slpwampc.utils.conversions import mpc_as_linear_program
import os
import sys
import casadi as cs

sys.path.append(os.getcwd())
from examples.paper_2024.model import Model
from examples.paper_2024.mpc import MixedIntegerMpc, TimeVaryingAffineMpc


np_random = np.random.default_rng(1)

N = 2
for _ in range(1000):
    x0 = np_random.random((Model.nx, 1))
    system_dict = Model.get_system_dict()
    mixed_integer_mpc = MixedIntegerMpc(system_dict, N, X_f=Model.X_f)
    sol = mixed_integer_mpc.solve({"x_0": x0})
    if sol.success:
        delta = sol.vals["delta"]  # binary vars that represent PWA regions
        switching_sequence = np.argmax(delta, axis=0).reshape(-1, 1)
    else:
        raise ValueError("MPC solve failed")


    # set up LP
    problem_data = mpc_as_linear_program(
        {
            "Q": np.eye(Model.nx),
            "R": np.eye(Model.nu),
            "A": [system_dict["A"][i.item()] for i in switching_sequence],
            "B": [system_dict["B"][i.item()] for i in switching_sequence],
            "c": [system_dict["c"][i.item()] for i in switching_sequence],
            "D": system_dict["D"],
            "E": system_dict["E"],
            "F": system_dict["F"],
            "G": system_dict["G"],
            "A_f": Model.X_f[0],
            "b_f": Model.X_f[1],
        },
        N=N,
    )
    f, G, W, S, D, E = (
        problem_data["f"],
        problem_data["G"],
        problem_data["W"],
        problem_data["S"],
        problem_data["D"],
        problem_data["E"],
    )
    A = np.vstack([G, D])
    b = np.vstack([W + S @ x0, E])
    lp = {}
    lp["a"] = cs.DM(A).sparsity()
    prob = cs.conic("S", "gurobi", lp, {"gurobi.DualReductions": 1,"gurobi.OutputFlag": 1, "error_on_fail": False})
    # prob = cs.conic("S", "clp", lp, {"error_on_fail": False})
    result = prob(g=f.T, a=A, uba=b)
    if np.fabs(sol.f - result["cost"]) > 1e-6:
        raise ValueError("Costs do not match")

