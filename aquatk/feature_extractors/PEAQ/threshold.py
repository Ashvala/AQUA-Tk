from utils import BARK as bark
import numpy as np

def threshold(E, dz=0.25):
    M = np.zeros(bark)    
    for k in range(bark):
        m = k * dz
        if m <= 12:
            m = 3.0
        else:
            m *= 0.25
        M[k] = E[k] / (10 ** (m / 10))
    return M
