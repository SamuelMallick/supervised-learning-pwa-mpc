import numpy as np

from slpwampc.core.systems import PwaSystem


class Model:
    nx = 2  # state dimension
    nu = 1  # control dimension
    num_ineqs = 1  # number of inequalities defining the pwa regions
    l = 2  # number of regions

    # pwa regions defined by Sx + Ru <= T
    S = [np.array([[1, 0]]), np.array([[-1, 0]])]
    R = [np.zeros((1, nu)), np.zeros((1, nu))]
    T = [np.array([[1]]), np.array([[-1]])]

    # x^+ = Ax + Bu + c
    A = [np.array([[1, 0.2], [0, 1]]), np.array([[0.5, 0.2], [0, 1]])]
    B = [np.array([[0.1], [1]]), np.array([[0.1], [1]])]
    c = [np.zeros((nx, 1)), np.array([[0.5], [0]])]

    # state constraints Dx <= E
    D = np.array([[-1, 1], [-3, -1], [0.2, 1], [-1, 0], [1, 0], [0, -1]])
    E = np.array([[15], [25], [9], [6], [8], [10]])

    # constrol constraints Fu <= G
    F = np.array([[1], [-1]])
    u_lim = 3
    G = u_lim * np.array([[1], [1]])

    system = {
        "S": S,
        "R": R,
        "T": T,
        "A": A,
        "B": B,
        "c": c,
        "D": D,
        "E": E,
        "F": F,
        "G": G,
    }

    X_f: tuple[np.ndarray, np.ndarray] = (
        np.array(
            [
                [0.943554152340661, 0.126216752413879],
                [-0.564458476593388, -0.737832475861210],
                [0.564458476593388, 0.737832475861210],
                [1, 0],
                [-1, 0],
            ]
        ),
        np.array([[1, 2, 2, 1, 6]]).T,
    )

    K_term = np.array([[-0.564458476593388, -0.737832475861210]])

    @staticmethod
    def get_system_dict():
        return Model.system

    @staticmethod
    def get_system():
        return PwaSystem(Model.system)
