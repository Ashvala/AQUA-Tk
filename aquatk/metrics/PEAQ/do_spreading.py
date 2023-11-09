import numpy as np
from utils import p
from utils import BARK as bark


def spreading(pp, fC, dz=0.25):
    """ 
    Spreading function
    :param pp: power spectrum
    :param fC: center frequency for each bark band
    """
    
    Sl = 27.0  # Lower spreading constant

    # Initialize arrays
    pplog = np.zeros(bark)
    su1 = np.zeros(bark)
    su2 = np.zeros(bark)
    el = np.zeros(bark)
    denom1 = np.zeros(bark)
    denom2 = np.zeros(bark)
    e2 = np.zeros(bark)

    # Calculate initial variables
    for j in range(bark):
        pplog[j] = 10.0 * np.log10(pp[j])
        su1[j] = -24.0 - 230.0 / fC[j] + 0.2 * pplog[j]
        su2[j] = -24.0 - 230.0 / fC[j]
        el[j] = p(10.0, pplog[j] / 10.0)
        denom1[j] = np.sum(p(10.0, -dz * (j - np.arange(j)) * Sl / 10.0))
        denom1[j] += np.sum(p(10.0, dz * (np.arange(j, bark) - j) * su1[j] / 10.0))
        denom2[j] = np.sum(p(10.0, -dz * (j - np.arange(j)) * Sl / 10.0))
        denom2[j] += np.sum(p(10.0, dz * (np.arange(j, bark) - j) * su2[j] / 10.0))

    # Perform spreading
    for k in range(bark):
        sum1 = sum2 = 0.0
        for j in range(bark):
            L = pplog[j]
            Su = su1[j]
            Eline = el[j]
            if k < j:
                Eline *= p(10.0, -dz * (j - k) * Sl / 10.0)
            else:
                Eline *= p(10.0, dz * (k - j) * Su / 10.0)
            Eline /= denom1[j]
            sum1 += p(Eline, 0.4)

            Su = su2[j]
            if k < j:
                Eline = p(10.0, -dz * (j - k) * Sl / 10.0)
            else:
                Eline = p(10.0, dz * (k - j) * Su / 10.0)
            Eline /= denom2[j]
            sum2 += p(Eline, 0.4)

        sum1 = p(sum1, 1.0 / 0.4)
        sum2 = p(sum2, 1.0 / 0.4)
        e2[k] = sum1 / sum2

    return e2        
