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
    :return:
    """
    return np.abs(x - y).mean()


def lp_distance(x, y, p=1):
    """
    $l_p$ norm
    :param x: A vector of predictions
    :param y: A vector of true values
    :param p: p=1: Manhattan distance, p=2: Euclidean distance
    :return:
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



