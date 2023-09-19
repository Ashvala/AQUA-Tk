from abc import ABC, abstractmethod
from numpy import ndarray


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name

    def compute(self, y_pred:ndarray, y_true:ndarray):
        raise NotImplementedError

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)
