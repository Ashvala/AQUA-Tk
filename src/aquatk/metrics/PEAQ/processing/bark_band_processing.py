import numpy as np
from aquatk.metrics.PEAQ.core.types import EarModelOutput, BarkBandOutput, PEAQError
from aquatk.metrics.PEAQ.core.bark import create_bark_bands
from aquatk.metrics.PEAQ.constants.basic_version import MIN_FREQ, MAX_FREQ, BARK_SPACING
from typing import Tuple

class BarkProcessor:
    """Critical band processing"""
    
    def __init__(self):
        # Initialize bark band boundaries
        self.lower_freqs, self.center_freqs, self.upper_freqs = \
            create_bark_bands(MIN_FREQ, MAX_FREQ, dz=BARK_SPACING)
            
    def _init_bark_bands(self, f_low: float, f_high: float, 
                        dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize bark band boundaries
        
        Implementation of critical band calculation from spec Table 6"""
        # Bark scale conversion
        B = lambda f: 7 * np.arcsinh(f / 650.0)
        BI = lambda z: 650 * np.sinh(z / 7.0)
        
        zL = B(f_low)
        zU = B(f_high)
        num_bands = int(np.ceil((zU - zL) / dz))
        
        # Calculate band frequencies
        z_centers = np.linspace(zL + dz/2, zU - dz/2, num_bands) 
        z_lower = z_centers - dz/2
        z_upper = z_centers + dz/2
        
        return (BI(z_centers), BI(z_lower), BI(z_upper))
        
    def process(self, ear_output: EarModelOutput) -> BarkBandOutput:
        """Group FFT output into critical bands"""
        energies = np.zeros(len(self.center_freqs))
        
        # Frequency resolution of FFT
        df = ear_output.sample_rate / (2 * len(ear_output.weighted_fft))
        
        # Group into bands
        for band_idx, (f_low, f_up) in enumerate(zip(self.lower_freqs, 
                                                    self.upper_freqs)):
            # Find FFT bins in this band
            bin_low = int(np.floor(f_low / df))
            bin_high = int(np.ceil(f_up / df))
            
            # Sum energy in band
            energies[band_idx] = np.sum(
                ear_output.weighted_fft[bin_low:bin_high] ** 2
            )
            
            # Handle partial bins at edges
            if bin_low > 0:
                frac_low = (bin_low * df - f_low) / df
                energies[band_idx] += frac_low * \
                    ear_output.weighted_fft[bin_low-1] ** 2
                    
            if bin_high < len(ear_output.weighted_fft):
                frac_high = (f_up - (bin_high-1) * df) / df
                energies[band_idx] += frac_high * \
                    ear_output.weighted_fft[bin_high] ** 2
                    
        return BarkBandOutput(
            energies=energies,
            center_frequencies=self.center_freqs,
            lower_frequencies=self.lower_freqs,
            upper_frequencies=self.upper_freqs
        )
    
if __name__ == "__main__":
    b = BarkProcessor()
    print(b.lower_freqs)