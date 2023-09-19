from .metric import Metric


class MeanSquaredError(Metric):
    def __init__(self):
        super().__init__("MSE")

    def compute(self, y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()


class MeanAbsoluteError(Metric):
    def __init__(self):
        super().__init__("MAE")

    def compute(self, y_pred, y_true):
        return (y_pred - y_true).abs().mean()
