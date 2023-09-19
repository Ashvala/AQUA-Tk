import unittest
from aquatk.functions.errors import *
import numpy as np


class MetricUnitTests(unittest.TestCase):

    def test_mse(self):
        y_pred = np.array([1, 2, 3])
        y_true = np.array([2, 3, 4])
        self.assertEqual(mean_squared_error(y_pred, y_true), 1)

    def test_mae(self):
        y_pred = np.array([1, 2, 3])
        y_true = np.array([2, 3, 4])
        self.assertEqual(mean_absolute_error(y_pred, y_true), 1)


if __name__ == '__main__':
    unittest.main()
