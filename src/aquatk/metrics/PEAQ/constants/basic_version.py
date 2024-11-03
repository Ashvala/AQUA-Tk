from typing import Final

# FFT and Processing Parameters
FFT_SIZE: Final = 2048
HOP_SIZE: Final = 1024
SAMPLE_RATE: Final = 48000
NORM_FACTOR: Final = 11361.301063573899  # From spec calibration
FREQ_ADAPTATION: Final = 23.4375  # freq resolution = sample_rate/fft_size

# Bark Scale Parameters (for create_bark_bands)
BARK_SPACING: Final = 0.25  # Basic Version resolution
MIN_FREQ: Final = 80.0  # Hz
MAX_FREQ: Final = 18000.0  # Hz 

# Time Constants
TIME_CONSTANT_MIN: Final = 0.008  # seconds, τmin from spec
TIME_CONSTANT_100: Final = 0.030  # seconds, τ100 from spec

# Spreading Function Constants
LOWER_SLOPE: Final = 27.0  # dB/Bark for Basic Version
UPPER_SLOPE_BASE: Final = -24.0  # dB/Bark
UPPER_SLOPE_FACTOR: Final = 230.0  # Frequency adjustment factor
UPPER_SLOPE_LEVEL: Final = 0.2  # Level dependence factor

# Threshold Constants
THRESHOLD_FACTOR: Final = 0.364
THRESHOLD_POWER: Final = -0.8
THRESHOLD_OFFSET: Final = 0.4

# Energy Thresholds
ENERGY_THRESHOLD_LIMIT: Final = 8000
BOUNDARY_THRESHOLD: Final = 200
BOUNDARY_WINDOW: Final = 5

# Averaging Windows
MOD_DIFF_WINDOW: Final = 4  # frames
TIME_AVERAGING_WINDOW: Final = 0.1  # seconds

# Default Parameters
DEFAULT_REFERENCE_LEVEL: Final = 92.0  # dB SPL