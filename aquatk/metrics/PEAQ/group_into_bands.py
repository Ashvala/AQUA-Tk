import numpy as np
from .fft_ear_model import earmodelfft
from .create_bark import calculate_bark_bands
from .utils import p, BARK, HANN
from numba import njit


def critbandgroup(ffte, rate, hann=HANN, bark=87, bark_table=[]):
    """
    Args:
        ffte: The array of FFT coefficients.
        rate: The sampling rate of the audio signal.
        hann: The length of the Hann window used for FFT.
        bark: The number of critical bands to calculate.
        bark_table: A tuple of three arrays (fC, fL, fU) representing the center frequencies, lower frequency bounds, and upper frequency bounds of each critical band.

    Returns:
        pe: An array of the calculated critical band energy values.

    """
    p = lambda x, y: x ** y
    fC, fL, fU = bark_table
    fres = rate / hann
    pe = np.zeros(bark)

    for i in range(bark):
        for k in range(hann // 2):
            k_lower = (k - 0.5) * fres
            k_upper = (k + 0.5) * fres
            if k_lower >= fL[i] and k_upper <= fU[i]:
                pe[i] += p(ffte[k], 2.0)
            elif k_lower < fL[i] and k_upper > fU[i]:
                pe[i] += p(ffte[k], 2.0) * (fU[i] - fL[i]) / fres
            elif k_lower < fL[i] and k_upper > fL[i]:
                pe[i] += p(ffte[k], 2.0) * (k_upper - fL[i]) / fres
            elif k_lower < fU[i] and k_upper > fU[i]:
                pe[i] += p(ffte[k], 2.0) * (fU[i] - k_lower) / fres

        pe[i] = max(pe[i], p(10.0, -12.0))

    return pe

def AddIntNoise(pe, fC):
    """
    Args:
        pe: List[float] - A list of values representing the input data.
        fC: List[float] - A list of values representing frequency coefficients.

    Returns:
        List[float] - A list of values after adding noise to each element in pe.

    Description:
        This method takes in a list of input data (pe) and a list of frequency coefficients (fC). It then adds noise to each element in pe based on the corresponding frequency coefficient
    * in fC.

    Example:
        pe = [0.5, 0.7, 0.9]
        fC = [1000, 2000, 3000]
        result = AddIntNoise(pe, fC)
        # result will be [0.655, 0.753, 0.857]
    """
    bark = BARK
    for k in range(bark):
        Pthres = p(10.0, 0.4 * 0.364 * p(fC[k] / 1000.0, -0.8))
        pe[k] += Pthres
    return pe

if __name__ == "__main__":
    fL, fU, fC = calculate_bark_bands(80, 18000)
    # create sine wave
    rate = 48000
    hann = 512
    t = np.arange(0, hann) / rate
    f = 1000
    x = np.sin(2 * np.pi * f * t)
    # calculate fft
    ffte, _ = earmodelfft(x, 1, hann)
    print(ffte.shape)
    # calculate pe
    pe = critbandgroup(ffte, rate, hann, bark_table=[fC, fL, fU])
    # plot
    import matplotlib.pyplot as plt
    plt.plot(pe)
    plt.show()
