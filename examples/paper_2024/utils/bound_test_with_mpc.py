import os
import sys

import casadi as cs
import numpy as np

from slpwampc.utils.conversions import mpc_as_linear_program

sys.path.append(os.getcwd())
from examples.paper_2024.model import Model

np_random = np.random.default_rng(12)

N = 5
system_dict = Model.get_system_dict()
nx = system_dict["A"][0].shape[0]
nu = system_dict["B"][0].shape[1]
nz = nx * (N + 1) + 2 * nu * N
switching_sequence = [0] * N
Q = np.eye(nx)
R = np.eye(nu)
# set up LP
problem_data = mpc_as_linear_program(
    {
        "Q": np.eye(Model.nx),
        "R": np.eye(Model.nu),
        "A": [system_dict["A"][i] for i in switching_sequence],
        "B": [system_dict["B"][i] for i in switching_sequence],
        "c": [system_dict["c"][i] for i in switching_sequence],
        "D": [
            np.vstack([system_dict["D"], system_dict["S"][i]])
            for i in switching_sequence
        ]
        + [system_dict["D"]],
        "E": [
            np.vstack([system_dict["E"], system_dict["T"][i]])
            for i in switching_sequence
        ]
        + [system_dict["E"]],
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
nc = G.shape[0]

# start_time = time.time()
# for _ in range(1000):
#     A = np_random.random((10, 10))
#     b = np_random.random((10, 1))
#     np.linalg.inv(A) @ b
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Time per operation: {elapsed_time / 1000:.6f} seconds")
# print(f"total est: {700000000*elapsed_time}")
# exit()

A_x, b_x = np.vstack(
    [system_dict["D"], system_dict["S"][switching_sequence[0]]]
), np.vstack(
    [system_dict["E"], system_dict["T"][switching_sequence[0]]]
)  # X bounds
A_u, b_u = system_dict["F"], system_dict["G"]  # U bounds

# add bounds to eps_x and eps_u
ncx = A_x.shape[0]
big_lim = 5000
for i in range(N + 1):
    D = np.vstack(
        [
            D,
            np.hstack(
                [
                    # multiplying eps_x
                    np.zeros((nx, nx * (i))),
                    np.eye(nx),
                    # A_x@np.linalg.inv(Q),
                    np.zeros((nx, nx * (N - i))),
                    # multiplying eps_u
                    np.zeros((nx, nu * N)),
                    # multiplying u
                    np.zeros((nx, nu * N)),
                ]
            ),
        ]
    )
    # E = np.vstack([E, b_x])
    E = np.vstack([E, big_lim * np.ones((nx, 1))])

    D = np.vstack(
        [
            D,
            np.hstack(
                [
                    # multiplying eps_x
                    np.zeros((nx, nx * (i))),
                    -np.eye(nx),
                    np.zeros((nx, nx * (N - i))),
                    # multiplying eps_u
                    np.zeros((nx, nu * N)),
                    # multiplying u
                    np.zeros((nx, nu * N)),
                ]
            ),
        ]
    )
    E = np.vstack([E, np.zeros((nx, 1))])

ncu = A_u.shape[0]
for i in range(N):
    D = np.vstack(
        [
            D,
            np.hstack(
                [
                    # multiplying eps_x
                    np.zeros((nu, nx * (N + 1))),
                    # multiplying eps_u
                    np.zeros((nu, nu * i)),
                    # A_u@np.linalg.inv(R),
                    np.eye(nu),
                    np.zeros((nu, nu * (N - i - 1))),
                    # multiplying u
                    np.zeros((nu, nu * N)),
                ]
            ),
        ]
    )
    # E = np.vstack([E, b_u])
    E = np.vstack([E, big_lim * np.ones((nu, 1))])

    D = np.vstack(
        [
            D,
            np.hstack(
                [
                    # multiplying eps_x
                    np.zeros((nu, nx * (N + 1))),
                    # multiplying eps_u
                    np.zeros((nu, nu * i)),
                    -np.eye(nu),
                    np.zeros((nu, nu * (N - i - 1))),
                    # multiplying u
                    np.zeros((nu, nu * N)),
                ]
            ),
        ]
    )
    E = np.vstack([E, np.zeros((nu, 1))])

# generate slater point with robust optimization
qp = {}
qp["a"] = cs.DM(A_x).sparsity()
prob = cs.conic("S", "gurobi", qp, {"gurobi.OutputFlag": 0, "error_on_fail": True})
results = [prob(g=S[j, :], a=A_x, uba=b_x) for j in range(nc)]
min_x_vals = [result["cost"].full().item() for result in results]

qp = {}
qp["a"] = cs.DM(np.vstack([G, D])).sparsity()
prob = cs.conic("S", "gurobi", qp, {"gurobi.OutputFlag": 0, "error_on_fail": False})
result = prob(
    g=-f.T, a=np.vstack([G, D]), uba=np.vstack([W + np.vstack(min_x_vals) - 1e-6, E])
)
z_slater = result["x"].full().reshape(-1, 1)
# z_slater = np.vstack(
#     [5000 * np.ones((nx * (N + 1) + nu * N, 1)), np.zeros((nu * N, 1))]
# )
lmda = 0 * np.ones((nc, 1))

if not np.all(D @ z_slater - E <= 0):
    pass
    # raise ValueError("Slater vector not correct - not even feasible bro")

# ubz = np.vstack([5*np.ones((nx*(N+1), 1)), 3*np.ones((nu*N, 1)), 3*np.ones((nu*N, 1))])
# lbz = np.vstack([0*np.ones((nx*(N+1) + nu*N, 1)), -3*np.ones((nu*N, 1))])

# demoninator of bound
# start_time = time.time()
qp = {}
qp["a"] = cs.DM(A_x).sparsity()
prob = cs.conic("S", "gurobi", qp, {"gurobi.OutputFlag": 0, "error_on_fail": True})
results = [prob(g=S[j, :], a=A_x, uba=b_x) for j in range(nc)]
vals = [
    result["cost"] + W[j, :] - G[j, :] @ z_slater for j, result in enumerate(results)
]
min_val = np.min(vals)
if min_val < 0:
    pass
    # raise ValueError("Slater vector not correct as min_j g < 0")

# numerator of bound
result = prob(g=lmda.T @ S, a=A_x, uba=b_x)
val_x_max = -result["cost"] - lmda.T @ W

qp = {}
qp["a"] = cs.DM(D).sparsity()
prob = cs.conic("S", "gurobi", qp, {"gurobi.OutputFlag": 0, "error_on_fail": False})
result = prob(g=(f.T + lmda.T @ G), a=D, uba=E)  # , lbx=lbz, ubx=ubz)
val_z_min = result["cost"]

L = (f.T @ z_slater - val_x_max - val_z_min) / min_val  # max norm of dual var
ub_dV = np.linalg.norm(S, ord=2) * L
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Time per operation: {elapsed_time / 1:.6f} seconds")
# print(f"total est: {90112*elapsed_time}")
# exit()

num_samples = 100
dual_norms: list[float] = []
grads: list[float] = []
d = 0.001  # distance between samples for calculating gradient
qp = {}
qp["a"] = cs.DM(np.vstack([G, D])).sparsity()
# qp["a"] = cs.DM(G).sparsity()
mpc_prob = cs.conic("S", "gurobi", qp, {"gurobi.OutputFlag": 0, "error_on_fail": False})
xs = Model.sample_state_space(
    sample_strategy="random",
    num_points=num_samples,
    np_random=np_random,
    region=switching_sequence[0],
)
for i in range(num_samples):
    x = xs[i]
    if not np.all(G @ z_slater - W - S @ x < 0):
        pass
        # raise ValueError("Slater vector not correct")

    # original bound for this x
    vals = [-G[j, :] @ z_slater + W[j, :] + S[j, :] @ x for j in range(nc)]
    min_val = np.min(vals)
    if min_val < 0:
        pass
        # raise ValueError("Slater vector not correct as min_j g < 0")
    result = prob(g=f.T + lmda.T @ G, a=D, uba=E)  # , lbx=lbz, ubx=ubz)
    val = result["cost"] - lmda.T @ (W + S @ x)
    L_ = (f.T @ z_slater - val) / min_val

    mpc_result = mpc_prob(g=f.T, a=np.vstack([G, D]), uba=np.vstack([W + S @ x, E]))
    # mpc_result = mpc_prob(g=f.T, a=G, uba=W+S@x, lbx=lbz, ubx=ubz)
    if not np.isnan(mpc_result["cost"].full()).item():
        dual = mpc_result["lam_a"][: G.shape[0]]
        cost = mpc_result["cost"]
        dual_norm = np.linalg.norm(dual, ord=2).item()
        if dual_norm > L_:
            print(f"L_ {L_}, dual_norm {dual_norm}")
            # raise ValueError(
            #     f"Original bound {L_} is not correct, dual norm = {dual_norm}"
            # )
        dual_norms.append(dual_norm)

        theta = np_random.uniform(0, 2 * np.pi)
        x_pert = x + d * np.array([[np.cos(theta)], [np.sin(theta)]])
        mpc_result = mpc_prob(
            g=f.T, a=np.vstack([G, D]), uba=np.vstack([W + S @ x_pert, E])
        )
        # mpc_result = mpc_prob(g=f.T, a=G, uba=W+S@x_pert, lbx=lbz, ubx=ubz)
        if not np.isnan(mpc_result["cost"].full()).item():
            dual_pert = mpc_result["lam_a"][: G.shape[0]]
            cost_pert = mpc_result["cost"]
            dual_norms.append(np.linalg.norm(dual_pert, ord=2).item())

            grads.append(np.abs((cost_pert - cost) / d).item())
print(f"dual ub = {L}, max dual norm = {np.max(dual_norms)}")
print(f"grad ub = {ub_dV}, max grad = {np.max(grads)}")
