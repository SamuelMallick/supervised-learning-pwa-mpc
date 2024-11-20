import numpy as np
import casadi as cs


np_random = np.random.default_rng(12)

nx = 2
nz = 2
nc = 6

# box constraints
x_lim = 1
z_lim = 10

f = np_random.uniform(-10, 10, (nz, 1))
# G = np_random.uniform(-10, 10, (nc, nz))
G = np.array([[-1, 1], [-3, -1], [0.2, 1], [-1, 0], [1, 0], [0, -1]])
# G = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
# W = np_random.uniform(-10, 10, (nc, 1))
W = np.array([[15], [25], [9], [6], [8], [10]])
# lim = 1
# W = np.array([[lim], [lim], [lim], [lim]])
S = np_random.uniform(-1, 1, (nc, nx))

z_slater = np.zeros((nz, 1))
lmda = 0.1*np.ones((nc, 1))

# demoninator of bound
qp = {}
qp["a"] = cs.Sparsity_dense(nc, nz)
prob = cs.conic("S", "gurobi", qp, {"gurobi.OutputFlag": 0, "error_on_fail": True})
results = [prob(g=S[j, :], lbx=-x_lim, ubx=x_lim) for j in range(nc)]
vals = [result["cost"] + W[j, :] - G[j, :] @ z_slater for j, result in enumerate(results)]
min_val = np.min(vals)
if min_val < 0:
    raise ValueError("Slater vector not correct as min_j g < 0")

# numerator of bound
result = prob(g=lmda.T @ S, lbx=-x_lim, ubx=x_lim)
val_x_max = -result["cost"] - lmda.T @ W

qp = {}
qp["a"] = cs.Sparsity_dense(nc, nz)
prob = cs.conic("S", "gurobi", qp, {"gurobi.OutputFlag": 0, "error_on_fail": False})
result = prob(g=(f.T + lmda.T @ G), lbx=-z_lim, ubx=z_lim)
val_z_min = result["cost"]

L = (f.T @ z_slater - val_x_max - val_z_min) / min_val  # max norm of dual var
ub_dV = np.linalg.norm(S, ord=2) * L

num_samples = 1000
dual_norms: list[float] = []
grads: list[float] = []
d = 0.001  # distance between samples for calculating gradient
for i in range(num_samples):
    x = np_random.uniform(-x_lim, x_lim, (nx, 1))
    if not np.all(G @ z_slater - W - S @ x < 0):
        raise ValueError("Slater vector not correct")

    # original bound for this x
    vals = [-G[j, :] @ z_slater + W[j, :] + S[j, :]@x for j in range(nc)]
    min_val = np.min(vals)
    if min_val < 0:
        raise ValueError("Slater vector not correct as min_j g < 0")
    result = prob(g=f.T + lmda.T@G, lbx=-z_lim, ubx=z_lim)
    val = result["cost"] - lmda.T @ (W + S@x)
    L_ = (f.T@z_slater - val) / min_val

    result = prob(g=f.T, a=G, uba=W + S @ x, lbx=-z_lim, ubx=z_lim)
    if not np.isnan(result["cost"].full()).item():
        dual = result["lam_a"]
        cost = result["cost"]
        dual_norm = np.linalg.norm(dual, ord=2).item()
        if dual_norm > L_:
            raise ValueError(f"Original bound {L_} is not correct, dual norm = {dual_norm}")
        dual_norms.append(dual_norm)

        theta = np_random.uniform(0, 2 * np.pi)
        x_pert = x + d * np.array([[np.cos(theta)], [np.sin(theta)]])
        result = prob(g=f.T, a=G, uba=W + S @ x_pert, lbx=-z_lim, ubx=z_lim)
        if not np.isnan(result["cost"].full()).item():
            dual_pert = result["lam_a"]
            cost_pert = result["cost"]
            dual_norms.append(np.linalg.norm(dual_pert, ord=2).item())

            grads.append(np.abs((cost_pert - cost) / d).item())
print(f"dual ub = {L}, max dual norm = {np.max(dual_norms)}")
print(f"grad ub = {ub_dV}, max grad = {np.max(grads)}")
