import unittest
from aquatk.metrics.errors import *
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

    def test_kl_divergence(self):
        p = np.array([0.1, 0.2, 0.7])
        q = np.array([0.1, 0.3, 0.6])
        result = kl_divergence(p, q)
        self.assertTrue(np.isfinite(result))

    def test_snr(self):
        reference = np.array([1, 2, 3, 4, 5])
        generated = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = snr(reference, generated)
        # self.assertTrue(np.isfinite(result))
        self.assertTrue(np.isclose(result, 110))

    def test_si_sdr(self):
        reference = np.array([1, 2, 3, 4, 5])
        generated = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = si_sdr(reference, generated)
        self.assertTrue(np.isfinite(result))
        # self.assertTrue(np.isclose(result, 107))
    def test_rms(self):
        signal = np.array([1, 2, 3, 4, 5])
        result = rms(signal)
        self.assertTrue(np.isclose(result, 3.31662479))

    def test_adjusted_rms(self):
        clean_rms = 0.5
        snr_level = 20
        result = adjusted_rms(clean_rms, snr_level)
        print(result)
        self.assertEqual(result, 0.05)


if __name__ == '__main__':
    unittest.main()

