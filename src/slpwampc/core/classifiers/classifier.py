import numpy as np

from slpwampc.misc.regions import Polytope


class PartitionClassifier:
    """A generic class for a classifier that partitions the state space into
    regions and assigns a label to each region."""

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
        raise NotImplementedError

    def get_partition(self) -> list[Polytope]:
        """Get the partition of the state space.

        Returns
        -------
        list[Polytope]
            The partition.
        """
        raise NotImplementedError

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the classifier to the data.

        Parameters
        ----------
        X : np.ndarray
            The states.
        Y : np.ndarray
            The labels.
        """
        raise NotImplementedError
