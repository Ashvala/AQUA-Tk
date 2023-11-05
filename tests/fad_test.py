import unittest
from aquatk.metrics.frechet_distance import frechet_audio_distance
import numpy as np

class FrechetUnitTests(unittest.TestCase):
    def test_frechet_finiteness(self):
        ref_feats = np.array([[1, 2, 3], [2, 3, 4]])
        recon_feats = np.array([[1, 2, 3], [2, 3, 4]])
        result = frechet_audio_distance(ref_feats, recon_feats)
        self.assertTrue(np.isfinite(result))

    def test_frechet_sine_noise(self):
        f = 440
        fs = 16000
        t = np.arange(0, 1, 1/fs)
        sine = np.sin(2 * np.pi * f * t)
        noise = np.random.normal(0, 1, len(sine))
        sine_noise = sine + noise

    def


if __name__ == "__main__":
    unittest.main()