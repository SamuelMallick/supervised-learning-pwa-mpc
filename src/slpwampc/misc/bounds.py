from itertools import product
from typing import Literal

import numpy as np


def compute_gradient_bound(
    system_dict: dict, N: int, norm_type: Literal["1", "inf"] = "1"
) -> float:
    """Compute a bound for the gradient of the PWA system defined
    by the system dictionary. The gradient |\nabla_x J(x)| is bounded
    by computing a bound on |\nabla_x J(x, \delta)| for every
    switching sequence \delta.

    Parameters
    ----------
    system_dict : dict
        A dictionary containing the lists PWA system matrices A, B,
        and c for the dynamics x^+ = A[i]x + B[i]u + c[i]. The lists
        of region matrices S and T such that region i is active when
        S[i]x <= T[i]. The state constrains Dx <= E, and control
        constraints Fu <= G. Furthermore, the cost matrices Q_x
        and Q_u for the stage l(x, u) = norm(Q_x*x) + norm(Q_u*u).
        Can optionally contain A_f, b_f for a terminal region
        Ax[N] <= b[N]. Can optionally contain a terminal cost
        matrix P for the terminal cost V_f(x) = norm(Px), other
        wise the terminal cost is Q_x.
    N : int
        The horizon length.
    norm_type : Literal["1", "inf"], optional
        The type of norm to use for the cost function. Either the
        1-norm or the infinity-norm."""
    if norm_type == "inf":
        raise NotImplementedError("The infinity-norm is not yet supported.")

    l = len(system_dict["A"])  # number of regions
    sequences = np.asarray(list(product(range(l), repeat=N)), dtype=int)[
        :, :, None
    ]  # all possible switching sequences. Shape: (l^N, N, 1)

    return 0.0
