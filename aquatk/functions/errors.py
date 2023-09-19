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
    :param x:
    :param y:
    :param p:
    :return:
    """
    return ((y - x) ** p).mean() ** (1 / p)



def cosine_similarity(x, y):
    """
    Cosine similarity
    :param x:
    :param y:
    :return:
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

