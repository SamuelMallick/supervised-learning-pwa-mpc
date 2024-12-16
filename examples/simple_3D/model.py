import numpy as np

from slpwampc.core.systems import PwaSystem


class Model:
    nx = 3  # state dimension
    nu = 1  # control dimension
    num_ineqs = 1  # number of inequalities defining the pwa regions
    l = 2  # number of regions

    # pwa regions defined by Sx + Ru <= T
    S = [np.array([[0, 1, 0]]), np.array([[0, -1, 0]])]
    R = [np.zeros((1, nu)), np.zeros((1, nu))]
    T = [np.array([[1]]), np.array([[-1]])]

    # x^+ = Ax + Bu + c
    A = [
        np.array([[1, 0.5, 0.3], [0, 1, 1], [0, 0, 1]]),
        np.array([[1, 0.2, 0.3], [0, 0.5, 1], [0, 0, 1]]),
    ]
    B = [np.array([[0], [0.1], [1]]), np.array([[0], [0.1], [1]])]
    c = [np.zeros((nx, 1)), np.array([[0.3], [0.5], [0]])]

    # state constraints Dx <= E
    D = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    E = np.array([[10], [10], [5], [5], [10], [10]])

    # constrol constraints Fu <= G
    F = np.array([[1], [-1]])
    u_lim = 1
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

    @staticmethod
    def get_system_dict():
        return Model.system

    @staticmethod
    def get_system():
        return PwaSystem(Model.system)
