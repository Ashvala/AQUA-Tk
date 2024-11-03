from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import numpy as np
from enum import Enum


class MovType(Enum):
    """Types of Model Output Variables"""
    BANDWIDTH_REF = "BandwidthRef"
    BANDWIDTH_TEST = "BandwidthTest"
    TOTAL_NMR = "TotalNMR"
    WIN_MOD_DIFF1 = "WinModDiff1"
    ADB = "ADB"
    EHS = "EHS"
    AVG_MOD_DIFF1 = "AvgModDiff1"
    AVG_MOD_DIFF2 = "AvgModDiff2"
    RMS_NOISE_LOUD = "RmsNoiseLoud"
    MFPD = "MFPD"
    REL_DIST_FRAMES = "RelDistFrames"


@dataclass
class MovConfig:
    """Configuration for MOV calculations"""
    frame_size: int = 2048
    sample_rate: int = 48000
    energy_threshold: float = 8000.0  # For frame validation
    boundary_threshold: float = 200.0  # For signal boundary detection
    bark_spacing: float = 0.25
    fft_size: int = 2048


@dataclass
class MovResults:
    """Container for MOV calculation results"""
    bandwidth_ref: float = 0.0
    bandwidth_test: float = 0.0
    total_nmr: float = 0.0
    win_mod_diff1: float = 0.0
    adb: float = 0.0
    ehs: float = 0.0
    avg_mod_diff1: float = 0.0
    avg_mod_diff2: float = 0.0
    rms_noise_loud: float = 0.0
    mfpd: float = 0.0
    rel_dist_frames: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert results to dictionary"""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'MovResults':
        """Create results from dictionary"""
        return cls(**data)


@dataclass
class MovState:
    """State information for MOV calculations"""
    frame_count: int = 0
    nmr_sum: float = 0.0
    distorted_frames: int = 0
    noise_loud_sum: float = 0.0
    mod_diff1_sum: float = 0.0
    mod_diff2_sum: float = 0.0
    bandwidth_ref_sum: float = 0.0
    bandwidth_test_sum: float = 0.0
    valid_frames: int = 0
    prev_pmb: float = 0.0
    prev_ptilde: float = 0.0
    ehs_sum: float = 0.0
    
    def reset(self):
        """Reset all state variables"""
        for key in self.__dict__:
            if isinstance(self.__dict__[key], (int, float)):
                self.__dict__[key] = 0.0


class MovProcessor:
    """
    Processor for calculating Model Output Variables (MOVs).
    
    This class handles the computation of various perceptual metrics used
    in the PEAQ algorithm to assess audio quality.
    """
    
    def __init__(self, config: Optional[MovConfig] = None):
        """Initialize MOV processor with configuration"""
        self.config = config or MovConfig()
        self.state = MovState()
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.frame_size <= 0 or self.config.sample_rate <= 0:
            raise ValueError("Frame size and sample rate must be positive")
        if self.config.energy_threshold <= 0:
            raise ValueError("Energy threshold must be positive")

    def _calc_bandwidth(self, fft_ref: np.ndarray, fft_test: np.ndarray) -> Tuple[float, float]:
        """Calculate bandwidth differences between reference and test signals"""
        ZERO_THRESHOLD = 921  # FFT bin corresponding to bandwidth threshold
        
        # Calculate zero threshold level
        threshold_level = 20.0 * np.log10(np.abs(fft_test[ZERO_THRESHOLD:]).max())
        
        # Find bandwidth for reference signal
        bw_ref = 0
        for k in range(ZERO_THRESHOLD-1, -1, -1):
            level_ref = 20.0 * np.log10(np.abs(fft_ref[k]))
            if level_ref >= threshold_level + 10.0:
                bw_ref = k + 1
                break
                
        # Find bandwidth for test signal
        bw_test = 0
        for k in range(bw_ref-1, -1, -1):
            level_test = 20.0 * np.log10(np.abs(fft_test[k]))
            if level_test >= threshold_level + 5.0:
                bw_test = k + 1
                break
                
        return bw_ref, bw_test

    def _calc_nmr(self, noise_pattern: np.ndarray, mask_pattern: np.ndarray) -> float:
        """Calculate Noise-to-Mask Ratio"""
        return 10.0 * np.log10(np.mean(noise_pattern / mask_pattern))

    def _calc_modulation_differences(self,
                                   mod_test: np.ndarray,
                                   mod_ref: np.ndarray,
                                   energy_ref: np.ndarray) -> Tuple[float, float]:
        """Calculate modulation difference metrics"""
        diff1 = np.mean(np.abs(mod_test - mod_ref) / (1.0 + mod_ref))
        
        # Asymmetric difference calculation for AvgModDiff2
        diff2_pos = np.abs(mod_test - mod_ref) / (0.01 + mod_ref)
        diff2_neg = 0.1 * np.abs(mod_test - mod_ref) / (0.01 + mod_ref)
        diff2 = np.mean(np.where(mod_test > mod_ref, diff2_pos, diff2_neg))
        
        return 100.0 * diff1, 100.0 * diff2

    def _calc_detection_probability(self, 
                                  energy_test: np.ndarray,
                                  energy_ref: np.ndarray) -> Tuple[float, float]:
        """Calculate detection probability and related metrics"""
        # Convert to dB
        e_test_db = 10.0 * np.log10(energy_test + 1e-12)
        e_ref_db = 10.0 * np.log10(energy_ref + 1e-12)
        
        # Calculate level
        level = 0.3 * np.maximum(e_ref_db, e_test_db) + 0.7 * e_test_db
        
        # Calculate slopes and detection probability
        mask = level > 0
        s = np.zeros_like(level)
        s[mask] = 5.95072 * (6.39468/level[mask])**1.71332 + 9.01033e-11 * level[mask]**4 \
                  + 5.05622e-6 * level[mask]**3 - 0.00102438 * level[mask]**2 \
                  + 0.0550197 * level[mask] - 0.198719
        s[~mask] = 1e30
        
        error = e_ref_db - e_test_db
        b = np.where(e_ref_db > e_test_db, 4.0, 6.0)
        
        a = 10.0**(np.log10(np.log10(2.0))/b) / s
        p = 1.0 - 10.0**(-np.power(a * error, b))
        
        return np.prod(1.0 - p), np.sum(np.abs(error)/s)

    def process_frame(self,
                     fft_ref: np.ndarray,
                     fft_test: np.ndarray,
                     noise_pattern: np.ndarray,
                     mask_pattern: np.ndarray,
                     mod_test: np.ndarray,
                     mod_ref: np.ndarray,
                     energy_test: np.ndarray,
                     energy_ref: np.ndarray) -> MovResults:
        """
        Process a single frame to update MOV calculations
        
        Args:
            fft_ref: FFT of reference signal
            fft_test: FFT of test signal
            noise_pattern: Noise pattern
            mask_pattern: Masking pattern
            mod_test: Modulation pattern of test signal
            mod_ref: Modulation pattern of reference signal
            energy_test: Energy pattern of test signal
            energy_ref: Energy pattern of reference signal
        
        Returns:
            MovResults containing updated MOV values
        """
        # Calculate bandwidth
        bw_ref, bw_test = self._calc_bandwidth(fft_ref, fft_test)
        self.state.bandwidth_ref_sum += bw_ref
        self.state.bandwidth_test_sum += bw_test
        
        # Calculate NMR
        nmr = self._calc_nmr(noise_pattern, mask_pattern)
        self.state.nmr_sum += nmr
        
        # Calculate modulation differences
        mod_diff1, mod_diff2 = self._calc_modulation_differences(
            mod_test, mod_ref, energy_ref)
        self.state.mod_diff1_sum += mod_diff1
        self.state.mod_diff2_sum += mod_diff2
        
        # Calculate detection probability
        prob, dist = self._calc_detection_probability(energy_test, energy_ref)
        
        # Update state
        self.state.frame_count += 1
        if prob > 0.5:
            self.state.distorted_frames += 1
        
        # Calculate current results
        results = MovResults(
            bandwidth_ref=self.state.bandwidth_ref_sum / max(1, self.state.frame_count),
            bandwidth_test=self.state.bandwidth_test_sum / max(1, self.state.frame_count),
            total_nmr=self.state.nmr_sum / max(1, self.state.frame_count),
            avg_mod_diff1=self.state.mod_diff1_sum / max(1, self.state.frame_count),
            avg_mod_diff2=self.state.mod_diff2_sum / max(1, self.state.frame_count),
            rel_dist_frames=self.state.distorted_frames / max(1, self.state.frame_count),
            mfpd=prob,
            adb=dist
        )
        
        return results

    def get_final_results(self) -> MovResults:
        """Get final MOV results after processing all frames"""
        if self.state.frame_count == 0:
            return MovResults()
            
        return MovResults(
            bandwidth_ref=self.state.bandwidth_ref_sum / self.state.frame_count,
            bandwidth_test=self.state.bandwidth_test_sum / self.state.frame_count,
            total_nmr=self.state.nmr_sum / self.state.frame_count,
            avg_mod_diff1=self.state.mod_diff1_sum / self.state.frame_count,
            avg_mod_diff2=self.state.mod_diff2_sum / self.state.frame_count,
            rel_dist_frames=self.state.distorted_frames / self.state.frame_count,
            mfpd=self.state.prev_pmb,
            adb=np.log10(self.state.prev_ptilde) if self.state.prev_ptilde > 0 else -0.5
        )

    def reset(self):
        """Reset processor state"""
        self.state.reset()