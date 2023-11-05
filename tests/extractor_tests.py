import unittest
from aquatk.metrics.frechet_distance import frechet_audio_distance

class FrechetUnitTests(unittest.TestCase):
    def test_frechet_finiteness(self):
        ref_feats = np.array([[1, 2, 3], [2, 3, 4]])
        recon_feats = np.array([[1, 2, 3], [2, 3, 4]])
        result = frechet_audio_distance(ref_feats, recon_feats)
        self.assertTrue(np.isfinite(result))

if __name__ == "__main__": 
    unittest.main()