from typing import Final
import numpy as np

# Neural Network Parameters for Basic Version
NEURAL_INPUT_SCALING_MIN: Final = np.array([
    393.916656,  # BandwidthRef
    361.965332,  # BandwidthTest
    -24.045116,  # Total NMR
    1.110661,    # WinModDiff1
    -0.206623,   # ADB
    0.074318,    # EHS
    1.113683,    # AvgModDiff1
    0.950345,    # AvgModDiff2
    0.029985,    # RmsNoiseLoud
    0.000101,    # MFPD
    0.0          # RelDistFrames
])

NEURAL_INPUT_SCALING_MAX: Final = np.array([
    921.0,        # BandwidthRef
    881.131226,   # BandwidthTest
    16.212030,    # Total NMR
    107.137772,   # WinModDiff1
    2.886017,     # ADB
    13.933351,    # EHS
    63.257874,    # AvgModDiff1
    1145.018555,  # AvgModDiff2
    14.819740,    # RmsNoiseLoud
    1.0,          # MFPD
    1.0           # RelDistFrames
])

# Hidden Layer Weights
NEURAL_HIDDEN_WEIGHTS: Final = np.array([
    [-0.502657, 0.436333, 1.219602],
    [4.307481, 3.246017, 1.123743],
    [4.984241, -2.211189, -0.192096],
    [0.051056, -1.762424, 4.331315],
    [2.321580, 1.789971, -0.754560],
    [-5.303901, -3.452257, -10.814982],
    [2.730991, -6.111805, 1.519223],
    [0.624950, -1.331523, -5.955151],
    [3.102889, 0.871260, -5.922878],
    [-1.051468, -0.939882, -0.142913],
    [-1.804679, -0.503610, -0.620456]
])

# Output Layer Weights
NEURAL_OUTPUT_WEIGHTS: Final = np.array([
    -3.817048,   # First hidden node
    4.107138,    # Second hidden node
    4.629582,    # Third hidden node
    -0.307594    # Bias
])

# ODG Scaling
ODG_SCALE_MIN: Final = -3.98
ODG_SCALE_MAX: Final = 0.22


# Time Constants for Different Processing Stages
TIME_CONSTANTS: Final = {
    'modulation': {
        'min': 0.008,
        '100': 0.030
    },
    'time_spreading': {
        'min': 0.008,
        '100': 0.030
    },
    'level_adaptation': {
        'min': 0.008,
        '100': 0.050
    }
}

# Threshold Parameters
THRESHOLD_PARAMS: Final = {
    'offset': 0.4,
    'factor': 0.364,
    'power': -0.8
}