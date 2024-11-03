from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class ModulationConfig:
    """Configuration parameters for modulation processing"""
    t_min: float = 0.008  # Minimum time constant (seconds)
    t_100: float = 0.030  # Time constant at 100 Hz (seconds)
    alpha_power: float = 0.3  # Power for envelope compression
    frame_rate: int = 48000  # Audio frame rate
    frame_size: int = 2048  # Frame size for processing


class ModulationError(Exception):
    """Base exception for modulation-related errors"""
    pass


@dataclass
class ModulationState:
    """Internal state for modulation processing"""
    e2_tmp: np.ndarray  # Previous frame energy
    etilde_tmp: np.ndarray  # Smoothed energy
    eder_tmp: np.ndarray  # Energy derivative

    @classmethod
    def create(cls, n_bands: int) -> 'ModulationState':
        """Create initial state for given number of frequency bands"""
        return cls(
            e2_tmp=np.zeros(n_bands),
            etilde_tmp=np.zeros(n_bands),
            eder_tmp=np.zeros(n_bands)
        )


class ModulationProcessor:
    """
    Processes temporal modulation patterns in audio signals.
    
    This processor analyzes temporal envelope fluctuations in different frequency bands,
    which is important for detecting issues like pre-echoes and temporal smearing.
    """

    def __init__(self, config: Optional[ModulationConfig] = None):
        """Initialize modulation processor with given configuration"""
        self.config = config or ModulationConfig()
        self._validate_config()
        
        # Calculate derived constants
        self.frame_time = self.config.frame_size / self.config.frame_rate
        
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.config.t_min <= 0 or self.config.t_100 <= 0:
            raise ModulationError("Time constants must be positive")
        if self.config.alpha_power <= 0 or self.config.alpha_power >= 1:
            raise ModulationError("Alpha power must be between 0 and 1")
        if self.config.frame_rate <= 0 or self.config.frame_size <= 0:
            raise ModulationError("Frame rate and size must be positive")

    def _compute_time_constants(self, center_freqs: np.ndarray) -> np.ndarray:
        """
        Compute frequency-dependent time constants.
        
        Args:
            center_freqs: Array of center frequencies in Hz
            
        Returns:
            Array of time constants for each frequency band
        """
        return (self.config.t_min + 
                (100.0 / center_freqs) * (self.config.t_100 - self.config.t_min))

    def _compute_smoothing_coeffs(self, time_constants: np.ndarray) -> np.ndarray:
        """
        Compute smoothing coefficients from time constants.
        
        Args:
            time_constants: Array of time constants in seconds
            
        Returns:
            Array of smoothing coefficients
        """
        return np.exp(-self.frame_time / (2.0 * time_constants))

    def process(self, 
               energies: np.ndarray,
               center_freqs: np.ndarray,
               state: Optional[ModulationState] = None) -> Tuple[np.ndarray, ModulationState]:
        """
        Process frame energies to compute modulation patterns.
        
        Args:
            energies: Current frame energy values for each frequency band
            center_freqs: Center frequencies for each band in Hz
            state: Optional previous frame state
            
        Returns:
            Tuple of (modulation_values, updated_state)
        """
        if energies.shape != center_freqs.shape:
            raise ModulationError("Energy and frequency arrays must match")
            
        n_bands = len(energies)
        if state is None:
            state = ModulationState.create(n_bands)
            
        # Compute time-varying coefficients
        time_constants = self._compute_time_constants(center_freqs)
        alphas = self._compute_smoothing_coeffs(time_constants)
        
        # Compress energies
        compressed = energies ** self.config.alpha_power
        prev_compressed = state.e2_tmp ** self.config.alpha_power
        
        # Update modulation state
        state.eder_tmp = (alphas * state.eder_tmp + 
                         (1 - alphas) * (self.config.frame_rate / self.config.frame_size) * 
                         np.abs(compressed - prev_compressed))
        
        state.e2_tmp = energies.copy()
        state.etilde_tmp = (alphas * state.etilde_tmp + 
                          (1 - alphas) * compressed)
        
        # Compute modulation values
        modulation = state.eder_tmp / (1 + (state.etilde_tmp / 0.3))
        
        return modulation, state

    def process_frames(self, 
                      frame_energies: np.ndarray,
                      center_freqs: np.ndarray) -> np.ndarray:
        """
        Process multiple frames of energy values.
        
        Args:
            frame_energies: Array of shape (n_frames, n_bands) containing energy values
            center_freqs: Center frequencies for each band
            
        Returns:
            Array of shape (n_frames, n_bands) containing modulation values
        """
        n_frames, n_bands = frame_energies.shape
        state = ModulationState.create(n_bands)
        modulation_frames = np.zeros_like(frame_energies)
        
        for i in range(n_frames):
            modulation_frames[i], state = self.process(
                frame_energies[i], center_freqs, state)
            
        return modulation_frames