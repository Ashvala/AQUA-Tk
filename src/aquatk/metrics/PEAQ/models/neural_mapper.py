from dataclasses import dataclass
from typing import Dict
import numpy as np
from aquatk.metrics.PEAQ.constants.tables import (
    NEURAL_INPUT_SCALING_MIN, 
    NEURAL_INPUT_SCALING_MAX, 
    NEURAL_HIDDEN_WEIGHTS,
    NEURAL_OUTPUT_WEIGHTS,
    ODG_SCALE_MIN,
    ODG_SCALE_MAX
)

@dataclass
class NetworkWeights:
    """Neural network weight parameters"""
    input_weights: np.ndarray
    output_weights: np.ndarray

class NeuralMapper:
    """
    Maps Model Output Variables (MOVs) to Objective Difference Grade (ODG)
    using the Basic version neural network from ITU-R BS.1387-2.
    """

    def __init__(self):
        """Initialize neural mapper with Basic PEAQ weights"""
        self.input_mins = NEURAL_INPUT_SCALING_MIN
        self.input_maxs = NEURAL_INPUT_SCALING_MAX
        self.output_min = ODG_SCALE_MIN
        self.output_max = ODG_SCALE_MAX
        self.weights = NetworkWeights(
            input_weights=NEURAL_HIDDEN_WEIGHTS,
            output_weights=NEURAL_OUTPUT_WEIGHTS
        )

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def process(self, processed: Dict[str, float]) -> Dict[str, float]:
        """
        Process Basic PEAQ MOV values to produce ODG and DI values.
        
        Args:
            processed: Dictionary containing the 11 Basic PEAQ MOV values with 'b' suffix
            
        Returns:
            Dictionary containing 'ODG' and 'DI' values
        """
        x = np.array([
            processed["BandwidthRefb"],
            processed["BandwidthTestb"],
            processed["TotalNMRb"],
            processed["WinModDiff1b"],
            processed["ADBb"],
            processed["EHSb"],
            processed["AvgModDiff1b"],
            processed["AvgModDiff2b"],
            processed["RmsNoiseLoudb"],
            processed["MFPDb"],
            processed["RelDistFramesb"]
        ])
        
        # Normalize inputs
        x_norm = (x - self.input_mins) / (self.input_maxs - self.input_mins)
        
        # Forward pass
        hidden_output = self._sigmoid(np.dot(x_norm, self.weights.input_weights))
        DI = self.weights.output_weights[-1] + np.dot(self.weights.output_weights[:-1], hidden_output)
        ODG = self.output_min + (self.output_max - self.output_min) * self._sigmoid(DI)
        
        return {
            'DI': float(DI),
            'ODG': float(ODG)
        }