import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from slpwampc.misc.regions import Polytope

f = np.array([[1], [2]])  # cost function
G = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1], [0, -1]])
W = np.array([[0], [1], [2], [0], [2]])
S = np.array([[1], [1], [1], [1], [-0.5]])

# x = np.array([[0]])

# plot solution as function of x
num_points = 100
x = []
J_x = []
fig, ax = plt.subplots()
# plt.ioff()
for x_ in np.linspace(0, 2, num_points):
    qp = {}
    qp["a"] = cs.DM(G).sparsity()
    prob = cs.conic("S", "gurobi", qp, {"gurobi.OutputFlag": 0, "error_on_fail": False})
    result = prob(g=f, a=G, uba=W + S * x_)
    if not np.isnan(result["cost"].full().item()):
        x.append(x_)
        J_x.append(result["cost"].full().item())
        P = Polytope(G, W + S * x_)
        ax.clear()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        P.plot(ax, alpha=0.2, color="blue")
        plt.pause(0.1)

# plt.ion()
fig, ax = plt.subplots()
ax.plot(x, J_x)
plt.show()


# fig, ax = plt.subplots()
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)

# plot feasible region in x space
# x = 0
# P = Polytope(G, W+S*x)
# P.plot(ax)

# # plot level sets
# for a in np.linspace(-10, 10, 20):
#     ax.plot([-10, 10], [(a + f[0]*10)/f[1], (a - f[0]*10)/f[1]], 'k--')
# plt.show()
