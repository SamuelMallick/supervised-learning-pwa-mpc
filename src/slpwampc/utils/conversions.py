from functools import reduce
from typing import Literal

import numpy as np


def mpc_as_linear_program(
    mpc_data: dict, N: int, norm_type: Literal["1", "inf"] = "1"
) -> dict[str, np.ndarray]:
    # TODO doc string
    if not all(key in mpc_data for key in ["A", "B", "c", "Q", "R"]):
        raise ValueError(
            "mpc_data must at least contain the keys 'A', 'B', 'c', 'Q', 'R'"
        )
    if norm_type == "inf":
        raise NotImplementedError("Only 1-norm is currently supported")

    A, B, c, Q, R = (
        mpc_data["A"],
        mpc_data["B"],
        mpc_data["c"],
        mpc_data["Q"],
        mpc_data["R"],
    )
    if "P" not in mpc_data:
        P = Q
    else:
        P = mpc_data["P"]

    if not isinstance(A, list):
        A = [A] * N
        B = [B] * N
        c = [c] * N
    nx = Q.shape[0]
    nu = R.shape[0]
    if not all(a.shape == (nx, nx) for a in A):
        raise ValueError("All A matrices must have shape (nx, nx)")
    if not all(b.shape == (nx, nu) for b in B):
        raise ValueError("All B matrices must have shape (nx, nu)")
    if not all(c.shape == (nx, 1) for c in c):
        raise ValueError("All c vectors must have shape (nx, 1)")

    # calculate dimension of optimization vector z = [eps, u] where eps are the cost aux vars
    nz = nx * (N + 1) + 2 * nu * N

    # cost
    f = np.vstack([np.ones((nx * (N + 1) + nu * N, 1)), np.zeros((nu * N, 1))])

    # constraints Gz <= W + Sx

    # converting dynamics
    G = np.zeros((0, nz))
    W = np.zeros((0, 1))
    S = np.zeros((0, nx))
    for i in range(N + 1):
        if i == N:
            Q = P
        G = np.vstack(
            [
                G,
                np.hstack(
                    [
                        # multiplying eps_x
                        np.zeros((nx, nx * (i))),
                        -np.eye(nx),
                        np.zeros((nx, nx * (N - i))),
                        # multiplying eps_u
                        np.zeros((nx, nu * N)),
                        # multliplying u
                        # A[1:(i-j])] takes the appropriate A matrices and the multiplication order is reversed with [::-1]
                        (
                            np.zeros((nx, nu))
                            if i == 0
                            else np.hstack(
                                [
                                    Q
                                    @ reduce(
                                        np.matmul, [np.eye(nx), *(A[(j + 1) : i][::-1])]
                                    )
                                    @ B[j]
                                    for j in range(i)
                                ]
                            )
                        ),
                        (
                            np.zeros((nx, nu * (N - 1)))
                            if i == 0
                            else np.zeros((nx, nu * (N - i)))
                        ),
                    ]
                ),
            ]
        )
        W = np.vstack(
            [
                W,
                (
                    np.zeros((nx, 1))
                    if i == 0
                    else -sum(
                        [
                            Q
                            @ reduce(np.matmul, [np.eye(nx), *(A[(j + 1) : i][::-1])])
                            @ c[j]
                            for j in range(i)
                        ]
                    )
                ),
            ]
        )
        S = np.vstack([S, -(Q @ reduce(np.matmul, [np.eye(nx), *(A[:i][::-1])]))])

        G = np.vstack(
            [
                G,
                np.hstack(
                    [
                        # multiplying eps_x
                        np.zeros((nx, nx * (i))),
                        -np.eye(nx),
                        np.zeros((nx, nx * (N - i))),
                        # multiplying eps_u
                        np.zeros((nx, nu * N)),
                        # multliplying u
                        (
                            np.zeros((nx, nu))
                            if i == 0
                            else -np.hstack(
                                [
                                    Q
                                    @ reduce(
                                        np.matmul, [np.eye(nx), *(A[(j + 1) : i][::-1])]
                                    )
                                    @ B[j]
                                    for j in range(i)
                                ]
                            )
                        ),
                        (
                            np.zeros((nx, nu * (N - 1)))
                            if i == 0
                            else np.zeros((nx, nu * (N - i)))
                        ),
                    ]
                ),
            ]
        )
        W = np.vstack(
            [
                W,
                (
                    np.zeros((nx, 1))
                    if i == 0
                    else sum(
                        [
                            Q
                            @ reduce(np.matmul, [np.eye(nx), *(A[(j + 1) : i][::-1])])
                            @ c[j]
                            for j in range(i)
                        ]
                    )
                ),
            ]
        )
        S = np.vstack([S, (Q @ reduce(np.matmul, [np.eye(nx), *(A[:i][::-1])]))])

    # converting state constraints Dx <= E
    if "D" in mpc_data and "E" in mpc_data:
        D_x, E_x = mpc_data["D"], mpc_data["E"]
        if not isinstance(D_x, list):
            D_x = [D_x] * (N + 1)
            E_x = [E_x] * (N + 1)
        if not D_x[0].shape[1] == nx:
            raise ValueError("D must have the same number of columns as x")

        for i in range(N + 1):
            nc = D_x[i].shape[0]
            G = np.vstack(
                [
                    G,
                    np.hstack(
                        [
                            # multiplying eps_x
                            np.zeros((nc, nx * (N + 1))),
                            # multiplying eps_u
                            np.zeros((nc, nu * N)),
                            # multliplying u
                            # A[1:(i-j])] takes the appropriate A matrices and the multiplication order is reversed with [::-1]
                            (
                                np.zeros((nc, nu))
                                if i == 0
                                else np.hstack(
                                    [
                                        D_x[i]
                                        @ reduce(
                                            np.matmul,
                                            [np.eye(nx), *(A[(j + 1) : i][::-1])],
                                        )
                                        @ B[j]
                                        for j in range(i)
                                    ]
                                )
                            ),
                            (
                                np.zeros((nc, nu * (N - 1)))
                                if i == 0
                                else np.zeros((nc, nu * (N - i)))
                            ),
                        ]
                    ),
                ]
            )
            W = np.vstack(
                [
                    W,
                    (
                        E_x[i]
                        if i == 0
                        else E_x[i]
                        - sum(
                            [
                                D_x[i]
                                @ reduce(
                                    np.matmul, [np.eye(nx), *(A[(j + 1) : i][::-1])]
                                )
                                @ c[j]
                                for j in range(i)
                            ]
                        )
                    ),
                ]
            )
            S = np.vstack(
                [S, -(D_x[i] @ reduce(np.matmul, [np.eye(nx), *(A[:i][::-1])]))]
            )

    # converting terminal constraints A_f x <= b_f
    if "A_f" in mpc_data and "b_f" in mpc_data:
        A_f, b_f = mpc_data["A_f"], mpc_data["b_f"]
        if not A_f.shape[1] == nx:
            raise ValueError("A_f must have the same number of columns as x")

        G = np.vstack(
            [
                G,
                np.hstack(
                    [
                        # multiplying eps_x
                        np.zeros((A_f.shape[0], nx * (N + 1))),
                        # multiplying eps_u
                        np.zeros((A_f.shape[0], nu * N)),
                        # multliplying u
                        np.hstack(
                            [
                                A_f
                                @ reduce(
                                    np.matmul,
                                    [np.eye(nx), *(A[(j + 1) : N][::-1])],
                                )
                                @ B[j]
                                for j in range(N)
                            ]
                        ),
                    ]
                ),
            ]
        )
        W = np.vstack(
            [
                W,
                b_f
                - sum(
                    [
                        A_f
                        @ reduce(np.matmul, [np.eye(nx), *(A[(j + 1) : N][::-1])])
                        @ c[j]
                        for j in range(N)
                    ]
                ),
            ]
        )
        S = np.vstack([S, -(A_f @ reduce(np.matmul, [np.eye(nx), *(A[:N][::-1])]))])

    # constraints Dz <= E

    # converting control penalty
    D = np.zeros((0, nz))
    E = np.zeros((0, 1))
    # # TODO shrink the following code
    for i in range(N):
        D = np.vstack(
            [
                D,
                np.hstack(
                    [
                        # multiplying eps_x
                        np.zeros((nu, nx * (N + 1))),
                        # multiplying eps_u
                        np.zeros((nu, nu * i)),
                        -np.eye(nu),
                        np.zeros((nu, nu * (N - i - 1))),
                        # multiplying u
                        np.zeros((nu, nu * i)),
                        R,
                        np.zeros((nu, nu * (N - i - 1))),
                    ]
                ),
            ]
        )
        E = np.vstack([E, np.zeros((nu, 1))])

        D = np.vstack(
            [
                D,
                np.hstack(
                    [
                        # multiplying eps_x
                        np.zeros((nu, nx * (N + 1))),
                        # multiplying eps_u
                        np.zeros((nu, nu * i)),
                        -np.eye(nu),
                        np.zeros((nu, nu * (N - i - 1))),
                        # multiplying u
                        np.zeros((nu, nu * i)),
                        -R,
                        np.zeros((nu, nu * (N - i - 1))),
                    ]
                ),
            ]
        )
        E = np.vstack([E, np.zeros((nu, 1))])

    # converting control constraints: constraints F_u u <= G_u
    if "F" in mpc_data and "G" in mpc_data:
        F_u, G_u = mpc_data["F"], mpc_data["G"]
        if not F_u.shape[1] == nu:
            raise ValueError("F must have the same number of columns as u")

        nc = F_u.shape[0]
        for i in range(N):
            D = np.vstack(
                [
                    D,
                    np.hstack(
                        [
                            # multiplying eps_x
                            np.zeros((nc, nx * (N + 1))),
                            # multiplying eps_u
                            np.zeros((nc, nu * N)),
                            # multiplying u
                            np.zeros((nc, nu * i)),
                            F_u,
                            np.zeros((nc, nu * (N - i - 1))),
                        ]
                    ),
                ]
            )
            E = np.vstack([E, G_u])

    return {"f": f, "G": G, "W": W, "S": S, "D": D, "E": E}


# N = 5
# Q = 2 * np.eye(2)
# R = np.eye(1)
# # A = np.array([[1, 1], [0, 1]])
# # B = np.array([[0], [1]])
# # c = np.array([[0.1], [0.1]])
# A = [np.random.rand(2, 2) for _ in range(N)]
# B = [np.random.rand(2, 1) for _ in range(N)]
# c = [np.random.rand(2, 1) for _ in range(N)]
# F = np.array([[1], [-1]])
# G = np.array([[1], [1]])
# D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
# E = np.array([[1], [1], [1], [1]])
# A_f = np.array([[1, 1], [0, 1]])
# b_f = np.array([[1], [1]])
# mpc_data = {
#     "A": A,
#     "B": B,
#     "c": c,
#     "Q": Q,
#     "R": R,
#     "F": F,
#     "G": G,
#     "D": D,
#     "E": E,
#     "A_f": A_f,
#     "b_f": b_f,
# }
# mpc_as_linear_program(mpc_data, N)
