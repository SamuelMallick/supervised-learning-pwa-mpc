import matplotlib.pyplot as plt
import numpy as np

from slpwampc.core.classifiers.pwl_sep import PwlSep
from slpwampc.misc.regions import Polytope

np_random = np.random.default_rng(1)

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b = np.array([[1], [1], [1], [1]])
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_axis_off()

pwl_sep = PwlSep(A, b)

S_1 = np.array([[1, 0], [1, -1]])
T_1 = np.array([[0], [0]])
p_1 = Polytope(np.vstack([A, S_1]), np.vstack([b, T_1]), 1)
p_1.plot(ax=ax, color="C1")

S_2 = np.array([[1, 0], [-1, 1]])
T_2 = np.array([[0], [0]])
p_2 = Polytope(np.vstack([A, S_2]), np.vstack([b, T_2]), 2)
p_2.plot(ax=ax, color="C2")

S_3 = np.array([[-1, 0], [1, -1], [1, 0]])
T_3 = np.array([[0], [0], [0.5]])
p_3 = Polytope(np.vstack([A, S_3]), np.vstack([b, T_3]), 3)
p_3.plot(ax=ax, color="C3")

S_4 = np.array([[-1, 0], [-1, -1]])
T_4 = np.array([[-0.5], [-1]])
p_4 = Polytope(np.vstack([A, S_4]), np.vstack([b, T_4]), 3)
p_4.plot(ax=ax, color="C3")

S_5 = np.array([[-1, 0], [1, 1], [-1, 1]])
T_5 = np.array([[0], [1], [0]])
p_5 = Polytope(np.vstack([A, S_5]), np.vstack([b, T_5]), 4)
p_5.plot(ax=ax, color="C4")

polys = [p_1, p_2, p_3, p_4, p_5]

num_points = 10000
X = np_random.uniform(-1, 1, (num_points, 2, 1))
Y = np.zeros(num_points, dtype=int)
for i in range(num_points):
    for p in polys:
        if np.all(p.A @ X[[i], :, :] <= p.b):
            Y[i] = p.label

fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_axis_off()

for p in polys:
    p.plot(ax=ax, alpha=0.5, color=f"C{p.label}")
    for i in range(num_points):
        if Y[i] == p.label:
            ax.scatter(X[i, 0], X[i, 1], c=f"C{p.label}", s=10)

fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_axis_off()

pwl_sep.fit(X.squeeze(), Y, max_iters=1)
regions = pwl_sep.get_partition()
for region in regions:
    region.plot(ax=ax, color=f"C{region.label}")
for p in polys:
    for i in range(num_points):
        if Y[i] == p.label:
            ax.scatter(X[i, 0], X[i, 1], c=f"C{p.label}", s=10)

fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_axis_off()

pwl_sep.fit(X.squeeze(), Y, max_iters=2)
regions = pwl_sep.get_partition()
for region in regions:
    region.plot(ax=ax, color=f"C{region.label}")
for p in polys:
    for i in range(num_points):
        if Y[i] == p.label:
            ax.scatter(X[i, 0], X[i, 1], c=f"C{p.label}", s=10)
plt.show()
