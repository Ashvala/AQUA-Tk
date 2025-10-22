import numpy as np
from .utils import p
import pdb
from .utils import BARK as bark

# @numba.njit(fastmath=True, cache=True)
def spreading(pp, fC, dz=0.25):
    """
    Spreading function
    :param pp: power spectrum
    :param fC: center frequency for each bark band

    Shoutout to @ijc8 for helping optimize this function
    """

    Sl = 27.0  # Lower spreading constant
    pdb.set_trace()
    # Initialize arrays
    pplog = np.zeros(bark)
    su1 = np.zeros(bark)
    su2 = np.zeros(bark)
    el = np.zeros(bark)
    denom1 = np.zeros(bark)
    denom2 = np.zeros(bark)
    e2 = np.zeros(bark)

    # Calculate initial variables
    pplog = 10.0 * np.log10(pp)
    su1 = -24.0 - 230.0 / fC + 0.2 * pplog
    su2 = -24.0 - 230.0 / fC
    el = 10.0 ** (pplog / 10.0)
    bax = np.mgrid[0:-bark:-1, 0:bark].sum(axis=0)
    # bax = np.meshgrid(np.arange(0, -bark, -1), np.arange(0, bark)).sum(axis=0)

    Su1 = np.repeat(su1.reshape((1, -1)), bark, axis=0).T
    Su1[bax < 0] = Sl
    denom1 = np.sum(np.power(10.0, dz * bax * Su1 / 10.0), axis=1)

    Su2 = np.repeat(su2.reshape((1, -1)), bark, axis=0).T
    Su2[bax < 0] = Sl
    denom2 = np.sum(np.power(10.0, dz * bax * Su2 / 10.0), axis=1)

    # Perform spreading
    Su1 = np.repeat(su1.reshape((1, -1)), bark, axis=0).T
    Su1[bax > 0] = Sl
    Su2 = np.repeat(su2.reshape((1, -1)), bark, axis=0).T
    Su2[bax > 0] = Sl

    sum1 = np.sum((el[None, :] * 10.0 ** (dz * -bax * Su1 / 10.0) / denom1[None, :]) ** 0.4, axis=1) ** (1.0 / 0.4)
    sum2 = np.sum((10.0 ** (dz * -bax * Su2 / 10.0) / denom2[None, :]) ** 0.4, axis=1) ** (1.0 / 0.4)
    return sum1 / sum2


if __name__ == "__main__":
    x = np.random.rand(bark)
