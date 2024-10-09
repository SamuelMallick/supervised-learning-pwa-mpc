from typing import Literal


class Test:
    sample_strategy: Literal["random", "grid", "focused"] = "grid"
    rollout_length = 1
    policy_class: Literal["feed_forward"] = "feed_forward"
    lr = 0.001
    T_train = 2
    batch_size = 64
    hidden_layers = 64

    # MPC
    N = 3
    first_region_from_policy = False
    d = 1

    ID = f"one_step_default_d_{d}"
