import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from examples.paper_2024.model import Model

N = 2
Q_x = np.eye(Model.nx)
Q_u = np.eye(Model.nu)
system_dict = Model.get_system_dict()
system_dict["Q_x"] = Q_x
system_dict["Q_u"] = Q_u
