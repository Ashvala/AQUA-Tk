import numpy as np
from utils import BARK, HANN


T100 = 0.03
Tmin = 0.008


def time_spreading(E2, rate, fC):
    E = np.zeros(BARK)
    Etmp = np.zeros(BARK)
    
    for k in range(BARK):
        T = Tmin + (100.0 / fC[k]) * (T100 - Tmin)
        a = np.exp(-HANN / (T * 2.0 * rate))
        
        Etmp[k] = Etmp[k] * a + (1.0 - a) * E2[k]
        
        if Etmp[k] >= E2[k]:
            E[k] = Etmp[k]
        else:
            E[k] = E2[k]
            
    return E, Etmp
    
