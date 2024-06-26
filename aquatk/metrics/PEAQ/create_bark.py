import numpy as np

B = lambda f: 7 * np.arcsinh(f / 650.0)
BI = lambda z: 650 * np.sinh(z / 7.0)


def calculate_bark_bands(f_low, f_high, dz=0.25):
    """
    Calculates the bark bands for a given frequency range.

    Args:
        f_low (float): The lower frequency boundary.
        f_high (float): The upper frequency boundary.
        dz (float, optional): The step size in Bark scale. Defaults to 0.25.

    Returns:
        tuple: A tuple containing three arrays:
            - fL: Array of lower frequency boundaries for each bark band.
            - fC: Array of center frequency for each bark band.
            - fU: Array of upper frequency boundaries for each bark band.
    """
    zL = B(f_low)
    zU = B(f_high)
    bark = int(np.ceil((zU - zL) / dz))

    fL = np.zeros(bark)
    fC = np.zeros(bark)
    fU = np.zeros(bark)

    zl = np.zeros(bark)
    zu = np.zeros(bark)
    zc = np.zeros(bark)

    for k in range(bark):
        zl[k] = zL + k * dz
        zu[k] = zL + (k + 1) * dz
        zc[k] = 0.5 * (zl[k] + zu[k])

    zu[-1] = zU
    zc[-1] = 0.5 * (zl[-1] + zu[-1])

    for k in range(bark):
        fL[k] = BI(zl[k])
        fU[k] = BI(zu[k])
        fC[k] = BI(zc[k])

    return fL, fC, fU

if __name__ == "__main__":
    fL, fC, fU = calculate_bark_bands(80, 18000)
    
    import matplotlib.pyplot as plt
    # start at fl, move to fc and end at fu for each index
    for i in range(len(fL)):
        plt.plot([fL[i], fC[i], fU[i]], [i, i, i], 'r')
    plt.show()
    
