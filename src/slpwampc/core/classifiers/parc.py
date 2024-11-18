import numpy as np
from pyparc.parc import PARC

from slpwampc.core.classifiers.classifier import PartitionClassifier
from slpwampc.misc.regions import Polytope


class Parc(PARC, PartitionClassifier):
    """A classifier that creates a convex partition via
    clustering a softmax regression, as presented in Bemporad (2024)."""

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        K=15,
        alpha=1.0e2,
        maxiter=150,
        sigma=15,
        separation="Softmax",
        verbose=0,
        min_number=1,
    ):
        """Initialize the classifier. The inequality Ax <= b defines
        the region over which the classifier partitions.

        Parameters
        ----------
        A : np.ndarray
            The matrix A in the inequality Ax <= b.
        b : np.ndarray
            The vector b in the inequality Ax <= b.
        K : int
            number of linear affine regressor/classifiers in PWA predictor.
        alpha : float
            L2-regularization term.
        maxiter : int
            maximum number of block-coordinate descent iterations.
        sigma : float
            tradeoff coefficient between PWL separability and quality of target fit.
        separation : str
            type of PWL separation used, either 'Voronoi' or 'Softmax'.
        verbose : int
            verbosity level (0 = none).
        min_number : int
            minimum number of points allowed per cluster. At the end
            of the procedure, points in excessively small clusters
            are reassigned to cluster of closest point (default: nx+1).
        """
        self.regions: list[Polytope] = []
        self.A, self.b = A, b
        PARC.__init__(
            self,
            K=K,
            alpha=alpha,
            maxiter=maxiter,
            sigma=sigma,
            separation=separation,
            verbose=verbose,
            min_number=min_number,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the classifier to the data.

        Parameters
        ----------
        X : np.ndarray
            The states.
        Y : np.ndarray
            The labels.
        """
        # TODO confirm all points are within the region
        super().fit(
            X, Y, categorical=[True]
        )  # TODO add comment explaining why categorical is set to True
        self._set_partition()

    def get_partition(self) -> list[Polytope]:
        return self.regions

    def predict(self, x: np.ndarray) -> int:
        """Predict the label for a given state.

        Parameters
        ----------
        x : np.ndarray
            The state.

        Returns
        -------
        int
            The label.
        """
        if (
            len(self.regions) == 0
        ):  # if no partition has been set, use the underlying classifier
            return int(PARC.predict(self, x)[0].item())
        for i, region in enumerate(self.regions):
            if not region.is_empty and np.all(region.A @ x <= region.b):
                return region.label
        raise ValueError("No region found for the given state.")

    def _set_partition(self) -> None:
        """Return the partition as a collection of regions within the space
        self.Ax <= self.b."""
        nx = self.nx
        ind = np.arange(2, nx, dtype=int)
        values = np.zeros(nx - 2)

        omega = self.omega
        gamma = self.gamma + (omega[:, ind] @ values).ravel()
        omega = np.delete(omega, ind, axis=1)
        xbar = np.delete(self.xbar, ind, axis=1)
        K = self.K

        A_ = np.vstack((self.A, np.zeros((K - 1, 2))))
        b_ = np.vstack((self.b, np.zeros((K - 1, 1))))
        regions = list()

        for j in range(0, K):
            i = self.b.shape[0]
            for h in range(0, K):
                if h != j:
                    A_[i, :] = omega[h, :] - omega[j, :]
                    b_[i] = -gamma[h] + gamma[j]
                    i += 1
            regions.append(Polytope(A_, b_))

        for region in regions:
            region.set_label(lambda x: self.predict(x))
        self.regions = regions
