import pickle

import numpy as np

for N in range(5, 13):
    with open(f"examples/paper_2024/results/training_N_{N}.pkl", "rb") as f:
        data = pickle.load(f)
        if N == 5:
            base_line_samples = len(data["x"])
        print(f"Results for N={N}")
        print(f"Num samples: {len(data['x'])}")
        print(f"Num samples relative to N=5: {len(data['x']) / base_line_samples}")
        print(f"Num iters: {data['iters']}")

N = 12
with open(
    f"examples/paper_2024/results/evaluate_closed_loop_N_{N}_learned_True_clp.pkl", "rb"
) as f:
    data = pickle.load(f)
    X_l = data["X"]
    U_l = data["U"]
    R_l = data["R"]
    t_l = np.concatenate(data["t"])

with open(
    f"examples/paper_2024/results/evaluate_closed_loop_N_{N}_learned_False.pkl", "rb"
) as f:
    data = pickle.load(f)
    X_o = data["X"]
    U_o = data["U"]
    R_o = data["R"]
    t_o = np.concatenate(data["t"])

print(f"Results for closed loop evaluation")
print(f"Num episodes: {len(X_l)}")
f = [100 * (sum(R_l[i]) - sum(R_o[i])) / sum(R_o[i]) for i in range(len(X_l))]
print(f"mean {np.mean(f)}")
print(f"std {np.std(f)}")
print(f"min {min(f)}")
print(f"max {max(f)}")
print(f"median {np.median(f)}")

print(f"Times our approach")
print(f"mean {np.mean(t_l)}")
print(f"std {np.std(t_l)}")
print(f"min {min(t_l)}")
print(f"max {max(t_l)}")
print(f"median {np.median(t_l)}")

print(f"Times MILP")
print(f"mean {np.mean(t_o)}")
print(f"std {np.std(t_o)}")
print(f"min {min(t_o)}")
print(f"max {max(t_o)}")
print(f"median {np.median(t_o)}")
