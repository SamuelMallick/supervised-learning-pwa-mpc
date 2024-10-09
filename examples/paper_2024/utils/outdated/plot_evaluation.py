import pickle

import matplotlib.pyplot as plt

from rollout.utils.tikz import save2tikz

name = "default"
with open(
    "examples/simple_2D/data/MLD.pkl",
    "rb",
) as file:
    X_mld = pickle.load(file)
    U_mld = pickle.load(file)
    R_mld = pickle.load(file)

with open(
    f"examples/simple_2D/data/evals/eval_{name}.pkl",
    "rb",
) as file:
    X_r = pickle.load(file)
    U_r = pickle.load(file)
    R_r = pickle.load(file)

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot([sum(R_mld[i]) for i in range(len(R_mld))], "o", color="red")
axs.plot([sum(R_r[i]) for i in range(len(R_r))], "x", color="blue")
axs.set_ylabel(r"\sum L")

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for i in range(len(X_mld)):
    axs.plot(X_mld[i][:, 0], X_mld[i][:, 1], "--o", color="red")
    axs.plot(X_r[i][:, 0], X_r[i][:, 1], "--x", color="blue")
axs.set_xlabel("x1")
axs.set_xlabel("x2")
axs.legend(["Optimal", "Learned"])

save2tikz(plt.gcf())
plt.show()
