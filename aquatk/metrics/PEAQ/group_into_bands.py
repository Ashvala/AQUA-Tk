import numpy as np
from fft_ear_model import earmodelfft
from create_bark import calculate_bark_bands
from utils import p, BARK, HANN
from numba import njit


def critbandgroup(ffte, rate, hann=HANN, bark=87, bark_table=None):
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
    # calculate pe
    pe = critbandgroup(ffte, rate, hann, (fC, fL, fU))
    print(pe)
    # plot
    import matplotlib.pyplot as plt
    plt.plot(pe)
    plt.show()
