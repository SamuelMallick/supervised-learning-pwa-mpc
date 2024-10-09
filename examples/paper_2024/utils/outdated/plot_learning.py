import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch


def plot_learning(
    Q_sums: list[float],
    nn_loss: list[float],
    feasible_set: list[torch.Tensor],
    sample_points: torch.Tensor,
    d: float | None = None,
):
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(Q_sums, "o")
    axs[0].set_ylabel(r"$\sum Q$")

    axs[1].plot(nn_loss, "o")
    axs[1].set_ylabel(r"$nn loss$")

    _, axs = plt.subplots(1, len(feasible_set), constrained_layout=True, sharex=True)
    for i in range(len(feasible_set)):
        axs[i].plot(sample_points[:, 0, 0], sample_points[:, 1, 0], "x")
        axs[i].plot(feasible_set[i][:, 0, 0], feasible_set[i][:, 1, 0], "o")
        if d is not None:
            for j in range(feasible_set[i].shape[0]):
                rect = patches.Rectangle(
                    (
                        feasible_set[i][j, 0, 0].item() - d / 2,
                        feasible_set[i][j, 1, 0].item() - d / 2,
                    ),
                    d,
                    d,
                    edgecolor="r",
                    facecolor=(1, 0, 0, 0.3),
                )
                axs[i].add_patch(rect)
    axs[-1].set_ylabel(r"$feas set$")

    plt.show()


if __name__ == "__main__":
    name = "one_step_default_d_0.1"
    # file_name = f"examples/simple_2D/results/models/{name}.pkl"
    file_name = f"{name}.pkl"

    with open(
        file_name,
        "rb",
    ) as file:
        data = pickle.load(file)
        Q = data["Q"]
        nn_loss = data["nn_loss"]
        F = data["F"]
        X_sam = data["X_sam"]

    plot_learning(Q, nn_loss, F, X_sam, 0.1)
