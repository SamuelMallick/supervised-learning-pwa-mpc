import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers.mpc.pwa_mpc import PwaMpc, PwaRegion

solver_options = {
    "ipopt": {
        "expand": True,
        "show_eval_warnings": True,
        "warn_initial_bounds": True,
        "print_time": False,
        "record_time": True,
        "bound_consistency": True,
        "calc_lam_x": True,
        "calc_lam_p": False,
        "ipopt": {
            "max_iter": 500,
            "sb": "yes",
            "print_level": 0,
        },
    },
    "qrqp": {
        "expand": True,
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "print_info": False,
        "print_iter": False,
        "print_header": False,
        "max_iter": 2000,
    },
    "qpoases": {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "printLevel": "none",
        "jit": True,
    },
    "gurobi": {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "gurobi": {
            "OutputFlag": 0,
            "LogToConsole": 0,
        },
    },
    "bonmin": {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "bonmin": {
            "print_level": 0,
            "max_iter": 1000,
        },
    },
    "knitro": {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "knitro": {
            "outlev": 0,
            "maxit": 1000,
        },
    },
    "clp": {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
    },
}


class TimeVaryingAffineMpc(PwaMpc):
    def __init__(
        self,
        system: dict,
        prediction_horizon: int,
        X_f: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        nlp = Nlp[cs.SX](sym_type="SX")
        super().__init__(nlp, prediction_horizon)

        nx = system["A"][0].shape[0]
        nu = system["B"][0].shape[1]
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu)

        # convert dictionary form to for for csnlp pwa_mpc
        pwa_system = [
            PwaRegion(
                A=system["A"][i],
                B=system["B"][i],
                c=system["c"][i][:, 0],  # TODO request to avoid this in csnlp
                S=cs.horzcat(system["S"][i], system["R"][i]),
                T=system["T"][i][:, 0],
            )
            for i in range(len(system["A"]))
        ]

        self.set_time_varying_affine_dynamics(pwa_system)
        self.constraint("state_constraints", system["D"] @ x - system["E"], "<=", 0)
        self.constraint("input_constraints", system["F"] @ u - system["G"], "<=", 0)
        if X_f is not None:
            A, b = X_f
            self.constraint("terminal", A @ x[:, -1] - b, "<=", 0)
        self.minimize(self.norm_1("x", x) + self.norm_1("u", u))
        self.init_solver(solver_options["clp"], solver="clp")  # clp


class MixedIntegerMpc(PwaMpc):
    def __init__(
        self,
        system: dict,
        prediction_horizon: int,
        X_f: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        nlp = Nlp[cs.SX](sym_type="SX")
        super().__init__(nlp, prediction_horizon)

        nx = system["A"][0].shape[0]
        nu = system["B"][0].shape[1]
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu)

        # convert dictionary form to for for csnlp pwa_mpc
        pwa_system = [
            PwaRegion(
                A=system["A"][i],
                B=system["B"][i],
                c=system["c"][i][:, 0],
                S=cs.horzcat(system["S"][i], system["R"][i]),
                T=system["T"][i][:, 0],
            )
            for i in range(len(system["A"]))
        ]
        D = D = cs.diagcat(system["D"], system["F"]).sparse()
        E = np.concatenate((system["E"][:, 0], system["G"][:, 0]))

        self.set_pwa_dynamics(pwa_system, D, E)
        self.constraint("state_constraints", system["D"] @ x - system["E"], "<=", 0)
        self.constraint("input_constraints", system["F"] @ u - system["G"], "<=", 0)
        if X_f is not None:
            A, b = X_f
            self.constraint("terminal", A @ x[:, -1] - b, "<=", 0)
        self.minimize(self.norm_1("x", x) + self.norm_1("u", u))
        self.init_solver(solver_options["knitro"], solver="knitro")


class TightenedMixedIntegerMpc(PwaMpc):
    def __init__(
        self,
        system: dict,
        prediction_horizon: int,
        eps: float,
        X_f: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        nlp = Nlp[cs.SX](sym_type="SX")
        super().__init__(nlp, prediction_horizon)

        nx = system["A"][0].shape[0]
        nu = system["B"][0].shape[1]
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu)

        # convert dictionary form to for for csnlp pwa_mpc
        pwa_system = [
            PwaRegion(
                A=system["A"][i],
                B=system["B"][i],
                c=system["c"][i][:, 0],
                S=cs.horzcat(system["S"][i], system["R"][i]),
                T=system["T"][i][:, 0],
            )
            for i in range(len(system["A"]))
        ]
        D = D = cs.diagcat(system["D"], system["F"]).sparse()
        E = np.concatenate((system["E"][:, 0], system["G"][:, 0]))

        self.set_pwa_dynamics(pwa_system, D, E)
        self.constraint(
            "state_constraints",
            system["D"] @ x - system["E"],
            "<=",
            -eps * np.linalg.norm(system["D"], ord=1, axis=1, keepdims=True),
        )
        self.constraint("input_constraints", system["F"] @ u - system["G"], "<=", 0)
        if X_f is not None:
            A, b = X_f
            self.constraint(
                "terminal",
                A @ x[:, -1] - b,
                "<=",
                -eps * np.linalg.norm(A, ord=1, axis=1, keepdims=True),
            )
        self.minimize(self.norm_1("x", x) + self.norm_1("u", u))
        self.init_solver(solver_options["gurobi"], solver="gurobi")
