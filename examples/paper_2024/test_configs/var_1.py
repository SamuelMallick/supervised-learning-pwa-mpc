from test_configs.default import Test


class Var1(Test):
    def __init__(
        self,
        num_state_samples: int,
        lr: float,
        batch_size: int,
        hidden_layers: int,
        ID: str,
    ) -> None:
        self.num_state_samples = num_state_samples
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.ID = ID

        self.T_train = 20
        super().__init__()
