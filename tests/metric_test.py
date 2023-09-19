import unittest
from aquatk.distance import *
import numpy as np


class MetricUnitTests(unittest.TestCase):

    def test_mse(self):
        y_pred = np.array([1, 2, 3])
        y_true = np.array([2, 3, 4])
        mse = MeanSquaredError()
        self.assertEqual(mse.compute(y_pred, y_true), 1)


if __name__ == '__main__':
    unittest.main()
