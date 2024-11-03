from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional
from enum import Enum


class SpreadingError(Exception):
    """Base exception for spreading-related errors"""
    pass


class SpreadingMethod(Enum):
    """Available methods for computing spreading function"""
    BASIC = "basic"  # Basic version spreading
    ADVANCED = "advanced"  # Advanced version spreading


@dataclass
class SpreadingConfig:
    """Configuration parameters for spreading calculations"""
    bark_spacing: float = 0.25  # Spacing between bark bands
    lower_slope: float = 27.0  # Lower spreading slope in dB/Bark
    upper_slope_min: float = -24.0  # Minimum upper spreading slope
    upper_slope_max: float = -4.0   # Maximum upper spreading slope
    method: SpreadingMethod = SpreadingMethod.BASIC


class SpreadingProcessor:
    """
    Handles spectral spreading calculations for PEAQ Basic and Advanced versions.
    
    The spreading function models the frequency spread of excitation along the basilar 
    membrane. It uses a two-sided exponential function with level-dependent slopes.
    """
    
    def __init__(self, config: Optional[SpreadingConfig] = None):
        """Initialize spreading processor with given configuration"""
        self.config = config or SpreadingConfig()
        self._validate_config()
        
        # Cache for spreading matrices to avoid recomputation
        self._spreading_matrix = None
        self._last_levels = None
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.config.bark_spacing <= 0:
            raise SpreadingError("Bark spacing must be positive")
        if self.config.lower_slope <= 0:
            raise SpreadingError("Lower slope must be positive")
        if self.config.upper_slope_max > 0:
            raise SpreadingError("Upper slope maximum must be negative")
        if self.config.upper_slope_min < self.config.upper_slope_max:
            raise SpreadingError("Upper slope minimum must be less than maximum")

    def _compute_level_dependent_slopes(self, levels: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
        """
        Compute level-dependent upper slopes for each frequency band.
        
        Args:
            levels: Power levels in dB for each frequency band
            freq_hz: Center frequencies in Hz
        
        Returns:
            Array of upper slopes in dB/Bark for each band
        """
        # Base slope calculation from ITU spec
        slopes = (-24.0 - 230.0/freq_hz + 0.2 * levels)
        
        # Limit slopes to configured range
        slopes = np.clip(slopes, self.config.upper_slope_min, self.config.upper_slope_max)
        
        return slopes

    def compute_spreading_matrix(self, 
                               bark_bands: int, 
                               levels: np.ndarray, 
                               freq_hz: np.ndarray) -> np.ndarray:
        """
        Compute the spreading matrix for given levels and frequencies.
        
        Args:
            bark_bands: Number of bark bands
            levels: Power levels in dB for each band
            freq_hz: Center frequencies in Hz for each band
            
        Returns:
            2D spreading matrix of shape (bark_bands, bark_bands)
        """
        if bark_bands <= 0:
            raise SpreadingError("Number of bark bands must be positive")
        if levels.shape[0] != bark_bands or freq_hz.shape[0] != bark_bands:
            raise SpreadingError("Levels and frequencies must match bark bands length")
            
        # Generate bark difference matrix
        bark_indices = np.arange(bark_bands)
        bark_diffs = (bark_indices[:, None] - bark_indices) * self.config.bark_spacing
        
        # Get level-dependent upper slopes
        upper_slopes = self._compute_level_dependent_slopes(levels, freq_hz)
        
        # Build spreading matrix
        spreading = np.zeros((bark_bands, bark_bands))
        for i in range(bark_bands):
            mask = bark_diffs[i] >= 0
            spreading[i, mask] = np.exp(-self.config.lower_slope * bark_diffs[i, mask])
            mask = bark_diffs[i] < 0 
            spreading[i, mask] = np.exp(upper_slopes[i] * bark_diffs[i, mask])
            
        # Normalize each row
        spreading /= spreading.sum(axis=1, keepdims=True)
        
        return spreading

    def process(self, power_spectrum: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
        """
        Apply spreading to input power spectrum.
        
        Args:
            power_spectrum: Power spectrum values for each frequency band
            freq_hz: Center frequencies in Hz for each band
            
        Returns:
            Spread power spectrum
        """
        if power_spectrum.shape != freq_hz.shape:
            raise SpreadingError("Power spectrum and frequency arrays must have same shape")
            
        bark_bands = power_spectrum.shape[0]
        levels = 10 * np.log10(np.maximum(power_spectrum, 1e-12))
        
        # Check if we can reuse cached spreading matrix
        if (self._spreading_matrix is None or 
            self._last_levels is None or
            not np.allclose(levels, self._last_levels)):
            
            self._spreading_matrix = self.compute_spreading_matrix(
                bark_bands, levels, freq_hz)
            self._last_levels = levels.copy()
        
        # Apply spreading
        spread_power = np.dot(self._spreading_matrix, power_spectrum)
        
        return spread_power
