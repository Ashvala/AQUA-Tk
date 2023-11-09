import numpy as np


def mean_squared_error(x, y):
    """
    Compute the Mean Squared Error
    :param x: A vector of predictions
    :param y: A vector of true values
    :return:
    """
    return ((y - x) ** 2).mean()


def mean_absolute_error(x, y):
    """
    Compute the Mean Absolute Error
    :param x: A vector of predictions
    :param y: A vector of true values
    """
    return np.abs(x - y).mean()


def lp_distance(x, y, p=1):
    """
    $l_p$ norm
    :param x: A vector of predictions
    :param y: A vector of true values
    :param p: p=1: Manhattan distance, p=2: Euclidean distance
    """
    return ((y - x) ** p).mean() ** (1 / p)



def cosine_similarity(x, y):
    """
    Cosine similarity
    :param x: A vector of predictions
    :param y: A vector of true values
    :return:
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def kl_divergence(p, q):
    """
    The Kullback–Leibler divergence.
    Defined only if q != 0 whenever p != 0.

    Remember it's not symmetric!
    """
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(q))
    assert not np.any(np.logical_and(p != 0, q == 0))
    
    p_pos = (p > 0)
    return np.sum(p[p_pos] * np.log(p[p_pos] / q[p_pos]))


def snr(reference: np.ndarray, generated: np.ndarray):
    """
        SNR is Signal to Noise Ratio
        :param reference: A vector of reference values
        :param generated: A vector of generated values

    """
    eps = 1e-10
    reference = reference - np.mean(reference)
    generated = generated - np.mean(generated)

    noise = reference - generated

    snr = (np.sum(reference**2) + eps) / (np.sum(noise**2) + eps)
    snr = 10 * np.log10(snr)
    return snr

def si_sdr(reference: np.ndarray, generated: np.ndarray):
    """
        SISDR is Scale-Invariant Signal to Distortion Ratio
        :param reference: A vector of reference values
        :param generated: A vector of generated values

    """
    eps = 1e-10
    reference = reference - np.mean(reference)
    generated = generated - np.mean(generated)

    # compute the scale factor
    scale = np.sum(reference * generated) / (np.sum(generated * generated) + eps)

    # scale the generated signal
    generated = scale * generated

    # compute the SDR
    s_target = np.sum(reference * generated)
    e_noise = np.sum(reference * reference) - s_target
    sdr = (s_target + eps) / (e_noise + eps)
    sdr = 10 * np.log10(sdr)
    return sdr

def rms(signal: np.ndarray):
    return np.sqrt(np.mean(signal**2))


def adjusted_rms(clean_rms, snr_level) -> float:
    a = snr_level / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms