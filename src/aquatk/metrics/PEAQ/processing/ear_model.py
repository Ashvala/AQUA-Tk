import numpy as np
from ..core.types import FFTParams, EarModelOutput, PEAQError

class EarModelProcessor:
    """FFT-based ear model processor"""
    
    def __init__(self, params: FFTParams):
        self.params = params
        self.window = self._create_window()
        
    def _create_window(self) -> np.ndarray:
        """Create Hann window with normalization"""
        N = self.params.size
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
        return np.sqrt(8.0/3.0) * window
        
    def _apply_ear_weighting(self, freqs: np.ndarray, fft: np.ndarray) -> np.ndarray:
        """Apply outer/middle ear frequency weighting
        
        Implementation of equation 7 from the spec"""
        f_khz = freqs / 1000.0
        W = -0.6 * 3.64 * (f_khz ** -0.8) + \
            6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) - \
            0.001 * f_khz ** 3.6
        
        return fft * np.power(10.0, W / 20.0)

    def process(self, audio: np.ndarray, playback_level: float = 92.0) -> EarModelOutput:
        """Process audio block through ear model
        
        Args:
            audio: Input audio samples
            playback_level: Reference playback level in dB SPL
            
        Returns:
            EarModelOutput containing FFT and weighted results
        """
        if len(audio) != self.params.size:
            raise PEAQError(f"Expected {self.params.size} samples, got {len(audio)}")
            
        # Apply window and FFT
        windowed = audio * self.window
        fft = np.fft.rfft(windowed)
        
        # Calculate frequency points
        freqs = np.fft.rfftfreq(self.params.size, 1/self.params.sample_rate)
        
        # Scale for playback level
        scale = np.power(10.0, playback_level/20.0) / self.params.size
        fft *= scale
        
        # Apply ear weighting
        weighted = self._apply_ear_weighting(freqs, np.abs(fft))
        
        return EarModelOutput(
            fft_magnitudes=np.abs(fft),
            weighted_fft=weighted,
            frequencies=freqs,
            sample_rate=self.params.sample_rate
        )