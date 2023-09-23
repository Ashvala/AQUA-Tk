import numpy as np
from scipy.fft import fft


NORM=11361.301063573899
FREQADAP=23.4375

p = lambda x, y: x**y


def earmodelfft(x, channels, hann):
    fac = p(10.0, channels/20.0)/NORM    
    in_ = np.zeros_like(hann)
    out = np.zeros_like(hann)
    absfft = np.zeros(len(hann)//2, dtype=np.float64)
    ffte = np.zeros(len(hann)//2, dtype=np.float64)
    for k, _ in enumerate(hann):
        in_[k] = hann[k] * x[k]
    out = fft(in_)
    
    for k in range(1, len(hann)//2):
        print(hann[k])
        out[k] *= (fac/hann[k])
        absfft[k] = np.sqrt(p(out[k].real, 2.0) + p(out[k].imag, 2.0))
        w = -0.6*3.64*p(k * FREQADAP/1000.0, -0.8) + 6.5*np.exp(-0.6*p(k * FREQADAP/1000.0 - 3.3, 2.0)) - 0.001*p(k * FREQADAP/1000.0, 3.6)
        ffte[k] = absfft[k]*p(10.0, w/20.0)
    return ffte, absfft


if __name__ == "__main__":
    signal = np.random.rand(2048)
    hann = np.hanning(2048)
    ffte, absfft = earmodelfft(signal, 1, hann)
    print(ffte)
    print(absfft)
    
    
    
    
