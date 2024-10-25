import numpy as np
from pyparc.parc import PARC

from slpwampc.misc.regions import Polytope


class Parc(PARC):
    regions: list[Polytope] = []

    def fit(self, X: np.ndarray, Y: np.ndarray, A: np.ndarray, b: np.ndarray, categorical=None):
        # TODO docstring
        super().fit(X, Y, categorical=categorical)
        self._set_partition(A, b)

    def get_partition(self) -> list[Polytope]:
        return self.regions

    def _set_partition(self, A: np.ndarray, b: np.ndarray) -> None:
        """Return the partition as a collection of regions within the space Ax <= b.

        Parameters
        ----------
        A : np.ndarray
            The matrix A in the inequality Ax <= b
        b : np.ndarray
            The vector b in the inequality Ax <= b"""
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

        for region in regions:
            region.set_label(lambda x: self.predict(x.T)[0].item())
        self.regions = regions


class ParcEnsemble:
    def __init__(
        self,
        num_classifiers: int,
        regions: list[tuple[np.ndarray, np.ndarray]],
        sigma: float = 10,
        alpha: float = 1.0e2,
        K: int = 15,
    ):
        # TODO check dimensions match for num classifiers and regions - or maybe just use regions
        self.num_classifiers = num_classifiers
        self.classifiers: list[Parc] = [
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

    def fit(self, i, X, Y, A, b, categorical=None):
        self.classifiers[i].fit(X, Y, np.vstack((A, self.regions[i][0])), np.vstack((b, self.regions[i][1])), categorical=categorical)

    def get_partition(self, i: int) -> list[Polytope]:
        return self.classifiers[i].get_partition()
       
    def save(self, filename: str):
        for i, classifier in enumerate(self.classifiers):
            classifier.save(f"{filename}_{i}")

    def load(self, filename: str, A: np.ndarray, b: np.ndarray):
        for i, classifier in enumerate(self.classifiers):
            classifier.load(f"{filename}_{i}")
            classifier._set_partition(np.vstack((A, self.regions[i][0])), np.vstack((b, self.regions[i][1])))
