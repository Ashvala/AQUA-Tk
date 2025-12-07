import numpy as np
from .utils import BARK, HANN


T100 = 0.03
Tmin = 0.008


def time_spreading(E2, rate, fC):
    """
    Args:
        E2: numpy array of shape (BARK,), representing input energy values.
        rate: float, representing the time rate.
        fC: numpy array of shape (BARK,), representing the center frequency values.

    Returns:
        Tuple containing two numpy arrays:
        - E: numpy array of shape (BARK,), representing the output energy values.
        - Etmp: numpy array of shape (BARK,), representing the intermediate energy values.
    """
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


def time_spreading_with_state(E2, rate, fC, Etmp_prev=None):
    """
    Time spreading with persistent state across frames.

    Args:
        E2: numpy array of shape (BARK,), representing input energy values.
        rate: float, representing the time rate.
        fC: numpy array of shape (BARK,), representing the center frequency values.
        Etmp_prev: Previous Etmp state, or None for first frame.

    Returns:
        Tuple containing two numpy arrays:
        - E: numpy array of shape (BARK,), representing the output energy values.
        - Etmp: numpy array of shape (BARK,), representing the intermediate energy values.
    """
    E = np.zeros(BARK)

    if Etmp_prev is None:
        Etmp = np.zeros(BARK)
    else:
        Etmp = Etmp_prev.copy()

    for k in range(BARK):
        T = Tmin + (100.0 / fC[k]) * (T100 - Tmin)
        a = np.exp(-HANN / (T * 2.0 * rate))

        Etmp[k] = Etmp[k] * a + (1.0 - a) * E2[k]

        if Etmp[k] >= E2[k]:
            E[k] = Etmp[k]
        else:
            E[k] = E2[k]

    return E, Etmp
    
