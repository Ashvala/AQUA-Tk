import numpy as np
from scipy.fft import fft
from .utils import p
import pdb

NORM=11361.301063573899
FREQADAP=23.4375


def earmodelfft(x, channels, lp, fft_size=512):
    """
    Args:
        x: array-like
            Input signal.
        channels: int
            Number of channels.
        lp: float
            Loudness factor.
        fft_size: int, optional
            Size of the FFT. Default is 512.

    Returns:
        tuple
            A tuple containing two arrays: `ffte` and `absfft`

    """
    # print(f"[Block stats] Min: {np.min(x)}\n[Block stats] Max: {np.max(x)}\n[Block stats] Mean: {np.mean(x)}\n")
    hann_window = np.sqrt(8 / 3) * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fft_size) / (fft_size - 1)))
    fac = p(10.0, lp/20.0)/NORM
    absfft = np.zeros(len(hann_window)//2, dtype=np.float64)
    ffte = np.zeros(len(hann_window)//2, dtype=np.float64)
    in_ = x*hann_window
    out = fft(in_)
    out.real *= (fac/(fft_size//2))
    out.imag *= (fac/(fft_size//2))
    for k in range(0, fft_size//2):
        absfft[k] = np.sqrt(p(out[k].real, 2.0) + p(out[k].imag, 2.0))
        #print(f"[earmodelfft] absfft[{k}]: {absfft[k]}")
        w = -0.6*3.64*p(k * FREQADAP/1000.0, -0.8) + 6.5*np.exp(-0.6*p(k * FREQADAP/1000.0 - 3.3, 2.0)) - 0.001*p(k * FREQADAP/1000.0, 3.6)
        #print("[earmodelfft] w: ", w)
        ffte[k] = absfft[k]*p(10.0, w/20.0)
        #print(f"[earmodelfft] ffte[{k}]: {ffte[k]}")
    return ffte, absfft


if __name__ == "__main__":
    rate = 48000
    hann = 2048
    t = np.arange(0, hann) / rate
    f = 10000
    x = np.sin(2 * np.pi * f * t)
    hann_window = np.hanning(hann)
    ffte, absfft = earmodelfft(x, 1, 92, 2048)
    print(ffte.shape)

    # plot 'em
    import matplotlib.pyplot as plt
    plt.plot(ffte)
    # plt.plot(absfft)
    plt.show()
