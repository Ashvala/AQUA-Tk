import numpy as np
import pytest
from aquatk.metrics.PEAQ.movs.movs import MovProcessor, MovConfig, MovResults, MovState


def test_mov_initialization():
    """Test MOV processor initialization"""
    processor = MovProcessor()
    assert processor.state.frame_count == 0
    
    config = MovConfig(frame_size=1024)
    processor = MovProcessor(config)
    assert processor.config.frame_size == 1024


def test_bandwidth_calculation():
    """Test bandwidth calculation"""
    processor = MovProcessor()
    
    # Create test signals
    n_fft = 2048
    freq = np.linspace(0, 1, n_fft//2)
    ref_signal = np.exp(-10 * freq)
    test_signal = np.exp(-12 * freq)
    
    bw_ref, bw_test = processor._calc_bandwidth(ref_signal, test_signal)
    
    assert bw_ref > 0
    assert bw_test > 0
    assert bw_ref >= bw_test  # Test signal has faster rolloff


def test_nmr_calculation():
    """Test Noise-to-Mask Ratio calculation"""
    processor = MovProcessor()
    
    # Create test patterns
    noise = np.ones(100) * 0.1
    mask = np.ones(100)
    
    nmr = processor._calc_nmr(noise, mask)
    
    assert nmr < 0  # Noise below mask
    assert np.isfinite(nmr)


# def test_modulation_differences():
#     """Test modulation difference calculations"""
#     processor = MovProcessor()
    
#     # Create test patterns
#     mod_test = np.random.random(100)
#     mod_ref = np.random.random(100)
#     energy = np.ones(100)
    
#     diff1, diff2 = processor._calc_modulation_differences(mod_test, mod_ref, energy)
    
#     assert 0 <= diff1 <= 100
#     assert 0 <= diff2 <= 100


def test_frame_processing():
    """Test processing of a complete frame"""
    processor = MovProcessor()
    
    # Create test data
    n_bands = 100
    fft_ref = np.random.random(2048)
    fft_test = np.random.random(2048)
    noise = np.random.random(n_bands) * 0.1
    mask = np.random.random(n_bands)
    mod_test = np.random.random(n_bands)
    mod_ref = np.random.random(n_bands)
    energy_test = np.random.random(n_bands)
    energy_ref = np.random.random(n_bands)
    
    results = processor.process_frame(
        fft_ref, fft_test, noise, mask,
        mod_test, mod_ref, energy_test, energy_ref
    )
    
    assert isinstance(results, MovResults)
    assert processor.state.frame_count == 1
    assert np.all([hasattr(results, attr) for attr in MovResults.__annotations__])

