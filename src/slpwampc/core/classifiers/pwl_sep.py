import numpy as np
import casadi as cs
from scipy.sparse import csr_matrix
from slpwampc.core.classifiers.classifier import PartitionClassifier
from slpwampc.misc.regions import Polytope


class PwlSep(PartitionClassifier):
    """A classifier that performs multi-class discrimination via linear
    programming, as presented in Bennett and Mangasarian (1992)."""

    def __init__(self):
        """Initialize the classifier."""
        self.regions: list[Polytope] = []
        self.labels: list[int] = []

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
        for i, region in enumerate(self.regions):
            if np.all(region.A @ x <= region.b):
                return self.labels[i]
        raise ValueError("No region found for the given state.")

    def fit(self, X: np.ndarray, Y: np.ndarray, D: np.ndarray, E: np.ndarray) -> None:
        """Fit the classifier to the data, creating a partition of the polytope Dx <= E.

        Parameters
        ----------
        X : np.ndarray
            The states.
        Y : np.ndarray
            The labels.
        D : np.ndarray
            The matrix D in the inequality Dx <= E.
        E : np.ndarray
            The vector E in the inequality Dx <= E.
        """
        self.labels = np.unique(Y)
        k = self.labels.size  # number of clusters
        X = [
            X[Y.squeeze() == int(i), :] for i in np.unique(Y)
        ]  # group data points by cluster
        n = X[0].shape[1]  # dimension of the data
        m = [X[i].shape[0] for i in range(k)]  # number of points in each cluster

        f = cs.vertcat(
            cs.DM(k * n, 1),  # zeroing out the omega
            cs.DM(k, 1),  # zeroing out the gamma
            *[
                (1 / m[i]) * cs.DM.ones((m[i] * (k - 1), 1)) for i in range(k)
            ],  # adding the y
        )

        Ashape = (2 * sum(m[i] * (k - 1) for i in range(k)), f.shape[0])
        b = cs.vertcat(
            -np.ones((sum(m[i] * (k - 1) for i in range(k)), 1)),
            cs.DM(sum(m[i] * (k - 1) for i in range(k)), 1),
        )
        row = 0

        data = np.empty(sum(m) * (k - 1) * ((3 + 2 * n) + 1), dtype=float)
        row_ind = np.empty_like(data, dtype=int)
        col_ind = np.empty_like(data, dtype=int)
        cnt = 0
        for i in range(k):  # TODO loop only over constraints, rather than k,k etc.
            row_ = 0
            X_i = X[i]
            m_i = m[i]
            sum_m_i = sum(m[z] for z in range(i))
            for j in range(k):
                if i != j:
                    for l in range(m_i):
                        for d in range(n):
                            row_ind[cnt] = row
                            col_ind[cnt] = n * i + d
                            data[cnt] = -X_i[l, d]
                            cnt += 1
                            row_ind[cnt] = row
                            col_ind[cnt] = n * j + d
                            data[cnt] = X_i[l, d]
                            cnt += 1

                        row_ind[cnt] = row
                        col_ind[cnt] = n * k + i
                        data[cnt] = 1
                        cnt += 1
                        row_ind[cnt] = row
                        col_ind[cnt] = n * k + j
                        data[cnt] = -1
                        cnt += 1

                        row_ind[cnt] = row
                        col_ind[cnt] = n * k + k + (k - 1) * sum_m_i + row_ * m_i + l
                        data[cnt] = -1
                        cnt += 1

                        row += 1

                    row_ += 1

        idx = np.arange((n + 1) * k, Ashape[1])
        row_ind[cnt:] = row + idx - (n * k + k)
        col_ind[cnt:] = idx
        data[cnt:] = -1

        A = csr_matrix((data, (row_ind, col_ind)), shape=Ashape, dtype=float)
        A = cs.DM(A)
        qp = {}
        qp["a"] = A.sparsity()
        S = cs.conic("S", "gurobi", qp, {"gurobi.OutputFlag": 0})
        result = S(g=f, a=A, uba=b)

        # TODO check if the optimization was successful
        omega = result["x"][: n * k].reshape((n, k)).T
        gamma = result["x"][n * k : n * k + k].reshape((1, k)).T
        self.regions = []  # TODO pre allocate, dont just empty them
        for i in range(k):
            A_ = D
            b_ = E
            for j in range(k):
                if j != i:
                    A_ = np.vstack([A_, omega[j, :] - omega[i, :]])
                    b_ = np.vstack([b_, gamma[j] - gamma[i]])
            self.regions.append(Polytope(A_, b_, label=self.labels[i]))
