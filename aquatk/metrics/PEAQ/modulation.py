import numpy as np
from utils import BARK as bark
from utils import module, HANN
from dataclasses import dataclass
from numba import njit

T100 = 0.03
Tmin = 0.008

@dataclass
class ModulationIn:
    def __init__(self, e2_tmp, etilde_tmp, eder_tmp):
        self.E2tmp = e2_tmp
        self.Etildetmp = etilde_tmp
        self.Edertmp = eder_tmp

    def __repr__(self):
        return f"ModulationIn(E2tmp={self.E2tmp}, Etildetmp={self.Etildetmp}, Edertmp={self.Edertmp})"


def modulation(E2, rate, in_struct, fC):
    """
    Compute ear modulation
    :param E2: power spectrum
    :param rate: sampling rate
    :param in_struct: modulation input structure
    :param fC: center frequency for each bark band
    """
    Mod = np.zeros(bark)
    for k in range(bark):
        T = Tmin + (100.0 / fC[k]) * (T100 - Tmin)
        a = np.exp(-HANN / (2 * rate * T))
        in_struct.Edertmp[k] = (
            in_struct.Edertmp[k] * a
            + (1 - a) * (rate / (HANN / 2)) * module(E2[k]**0.3 - in_struct.E2tmp[k]**0.3)
        )
        in_struct.E2tmp[k] = E2[k]
        in_struct.Etildetmp[k] = a * in_struct.Etildetmp[k] + (1 - a) * E2[k]**0.3

        Mod[k] = in_struct.Edertmp[k] / (1 + (in_struct.Etildetmp[k] / 0.3))

    return Mod, in_struct
