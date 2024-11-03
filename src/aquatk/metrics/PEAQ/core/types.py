from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict
import numpy as np
from enum import Enum

class PEAQError(Exception):
    """Base class for PEAQ-related errors"""
    pass

@dataclass
class FFTParams:
    """FFT processing parameters"""
    size: int = 2048
    hop_size: int = 1024
    sample_rate: int = 48000

@dataclass
class EarModelOutput:
    """Output from ear model stage"""
    fft_magnitudes: np.ndarray
    weighted_fft: np.ndarray  
    frequencies: np.ndarray
    sample_rate: int

@dataclass
class BarkBandOutput:
    """Output from bark band processing"""
    energies: np.ndarray
    center_frequencies: np.ndarray
    lower_frequencies: np.ndarray
    upper_frequencies: np.ndarray

@dataclass
class MovValues:
    """Model Output Variables"""
    bandwidth_ref: float = 0.0
    bandwidth_test: float = 0.0
    total_nmr: float = 0.0
    win_mod_diff1: float = 0.0
    avg_mod_diff1: float = 0.0
    avg_mod_diff2: float = 0.0
    rms_noise_loud: float = 0.0
    rel_dist_frames: float = 0.0
    adb: float = 0.0
    mfpd: float = 0.0
    ehs: float = 0.0

    def as_array(self) -> np.ndarray:
        """Convert MOVs to array for neural net input"""
        return np.array([
            self.bandwidth_ref, self.bandwidth_test,
            self.total_nmr, self.win_mod_diff1,
            self.adb, self.ehs, self.avg_mod_diff1,
            self.avg_mod_diff2, self.rms_noise_loud,
            self.mfpd, self.rel_dist_frames
        ])