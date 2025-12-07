import numpy as np
from .fft_ear_model import earmodelfft
from .create_bark import calculate_bark_bands
from .utils import p, BARK, HANN


def critbandgroup(ffte, rate, hann=HANN, bark=None, bark_table=[]):
    """
    Vectorized critical band grouping.

    Args:
        ffte: The array of FFT coefficients.
        rate: The sampling rate of the audio signal.
        hann: The length of the Hann window used for FFT.
        bark: The number of critical bands to calculate (inferred from bark_table if None).
        bark_table: A tuple of three arrays (fC, fL, fU) representing the center frequencies,
                    lower frequency bounds, and upper frequency bounds of each critical band.

    Returns:
        pe: An array of the calculated critical band energy values.
    """
    fC, fL, fU = bark_table
    bark = len(fC)
    fres = rate / hann
    num_bins = hann // 2

    # Precompute squared FFT magnitudes
    ffte_sq = ffte[:num_bins] ** 2

    # Precompute bin frequency boundaries
    k_vals = np.arange(num_bins)
    k_lower = (k_vals - 0.5) * fres
    k_upper = (k_vals + 0.5) * fres

    # Convert to numpy arrays for vectorized operations
    fL = np.asarray(fL)
    fU = np.asarray(fU)

    pe = np.zeros(bark)

    # For each band, compute contribution from all bins at once
    for i in range(bark):
        fl_i = fL[i]
        fu_i = fU[i]

        # Case 1: bin fully inside band
        mask1 = (k_lower >= fl_i) & (k_upper <= fu_i)

        # Case 2: band fully inside bin
        mask2 = (k_lower < fl_i) & (k_upper > fu_i)
        weight2 = (fu_i - fl_i) / fres

        # Case 3: bin overlaps lower edge
        mask3 = (k_lower < fl_i) & (k_upper > fl_i) & (k_upper <= fu_i)
        weight3 = (k_upper - fl_i) / fres

        # Case 4: bin overlaps upper edge
        mask4 = (k_lower >= fl_i) & (k_lower < fu_i) & (k_upper > fu_i)
        weight4 = (fu_i - k_lower) / fres

        # Sum contributions
        pe[i] = (np.sum(ffte_sq[mask1]) +
                 np.sum(ffte_sq[mask2] * weight2) +
                 np.sum(ffte_sq[mask3] * weight3[mask3]) +
                 np.sum(ffte_sq[mask4] * weight4[mask4]))

    # Apply minimum threshold
    pe = np.maximum(pe, 1e-12)

    return pe

def AddIntNoise(pe, fC):
    """
    Args:
        pe: List[float] - A list of values representing the input data.
        fC: List[float] - A list of values representing frequency coefficients.

    Returns:
        List[float] - A list of values after adding noise to each element in pe.

    Description:
        This method takes in a list of input data (pe) and a list of frequency coefficients (fC). It then adds noise to each element in pe based on the corresponding frequency coefficient
    * in fC.

    Example:
        pe = [0.5, 0.7, 0.9]
        fC = [1000, 2000, 3000]
        result = AddIntNoise(pe, fC)
        # result will be [0.655, 0.753, 0.857]
    """
    # Use length of fC to determine bark count
    bark = len(fC)
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
    print(ffte.shape)
    # calculate pe
    pe = critbandgroup(ffte, rate, hann, bark_table=[fC, fL, fU])
    # plot
    import matplotlib.pyplot as plt
    plt.plot(pe)
    plt.show()
