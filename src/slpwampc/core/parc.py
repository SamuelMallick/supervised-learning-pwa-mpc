import numpy as np
from pyparc.parc import PARC

from slpwampc.misc.regions import Polytope


class Parc(PARC):
    def get_partition(self, A: np.ndarray, b: np.ndarray) -> list[Polytope]:
        """Return the partition as a collection of regions within the space Ax <= b.

        Parameters
        ----------
        A : np.ndarray
            The matrix A in the inequality Ax <= b
        b : np.ndarray
            The vector b in the inequality Ax <= b

        Returns
        -------
        list[Polytope]
            The regions in the partition."""
        nx = self.nx
        ind = np.arange(2, nx, dtype=int)
        values = np.zeros(nx - 2)

        omega = self.omega
        gamma = self.gamma + (omega[:, ind] @ values).ravel()
        omega = np.delete(omega, ind, axis=1)
        xbar = np.delete(self.xbar, ind, axis=1)
        K = self.K

        # Plot PWL partition
        A_ = np.vstack((A, np.zeros((K - 1, 2))))
        b_ = np.vstack((b, np.zeros((K - 1, 1))))
        regions = list()

        for j in range(0, K):
            i = b.shape[0]
            for h in range(0, K):
                if h != j:
                    A_[i, :] = omega[h, :] - omega[j, :]
                    b_[i] = -gamma[h] + gamma[j]
                    i += 1
            regions.append(Polytope(A_, b_))
        return regions


class ParcEnsemble:
    def __init__(
        self,
        num_classifiers: int,
        regions: list[tuple[np.ndarray, np.ndarray]],
        sigma: float = 5,
        alpha: float = 1.0e2,
        K: int = 15,
    ):
        # TODO check dimensions match for num classifiers and regions - or maybe just use regions
        self.num_classifiers = num_classifiers
        self.classifiers = [
            Parc(
                K=K,
                alpha=alpha,
                maxiter=150,
                sigma=sigma,
                separation="Softmax",
                verbose=0,
                min_number=1,
            )
            for _ in range(num_classifiers)
        ]
        self.regions = regions

    def predict(self, x: np.ndarray) -> np.ndarray:
        for i, region in enumerate(self.regions):
            A, b = region
            if (A @ x.T <= b).all():
                return self.classifiers[i].predict(x)
        raise ValueError(f"No region found for state {x}")

    def fit(self, i, X, Y, categorical=None):
        self.classifiers[i].fit(X, Y, categorical=categorical)

    def get_partition(self, i: int, A: np.ndarray, b: np.ndarray) -> list[Polytope]:
        return self.classifiers[i].get_partition(
            np.vstack((A, self.regions[i][0])), np.vstack((b, self.regions[i][1]))
        )
    
    def save(self, filename: str):
        for i, classifier in enumerate(self.classifiers):
            classifier.save(f"{filename}_{i}")

    def load(self, filename: str):
        for i, classifier in enumerate(self.classifiers):
            classifier.load(f"{filename}_{i}")
