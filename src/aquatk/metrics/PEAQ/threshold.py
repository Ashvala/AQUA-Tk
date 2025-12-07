from .utils import BARK as bark
import numpy as np

def threshold(E, dz=0.25):
    """
    Args:
        E: Array-like
            The array containing the energy values for each frequency bin.
        dz: float, optional
            The step size for computing the threshold. Default is 0.25.

    Returns:
        Array-like
            The array containing the threshold values for each frequency bin.

    Raises:
        None

    Example:
        E = [1, 2, 3, 4, 5]  # Energy values
        dz = 0.5  # Step size for computing threshold
        threshold(E, dz)

    """
    M = np.zeros(bark)    
    for k in range(bark):
        m = k * dz
        if m <= 12:
            m = 3.0
        else:
            m *= 0.25
        M[k] = E[k] / (10 ** (m / 10))
    return M
