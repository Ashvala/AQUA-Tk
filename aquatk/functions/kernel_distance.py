import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.gaussian_process.kernels import ExpSineSquared


def mmd(k_xx, k_yy, k_xy):
    # from KID score on TorchMetrics
    m = k_xx.shape[0]

    diag_x = np.diag(k_xx)
    diag_y = np.diag(k_yy)

    kt_xx_sums = k_xx.sum(axis=-1) - diag_x
    kt_yy_sums = k_yy.sum(axis=-1) - diag_y
    k_xy_sums = k_xy.sum(axis=0)

    kt_xx_sum = kt_xx_sums.sum()
    kt_yy_sum = kt_yy_sums.sum()
    k_xy_sum = k_xy_sums.sum()

    value = (kt_xx_sum + kt_yy_sum) / (m * (m - 1))
    value -= 2 * k_xy_sum / (m ** 2)
    return value


def poly_mmd(x, y, degree=3, gamma=None, coeff=1):
    x_kernel = polynomial_kernel(x, x, degree, gamma, coeff)
    y_kernel = polynomial_kernel(y, y, degree, gamma, coeff)
    xy_kernel = polynomial_kernel(x, y, degree, gamma, coeff)
    # compute mmd
    mmd_value = mmd(x_kernel, y_kernel, xy_kernel)
    return mmd_value


def periodic_mmd(x, y):
    x_kernel = ExpSineSquared()(x, x)
    y_kernel = ExpSineSquared()(y, y)
    xy_kernel = ExpSineSquared()(x, y)
    # compute mmd
    mmd_value = mmd(x_kernel, y_kernel, xy_kernel)
    return mmd_value


def kernel_inception_distance(real, fake):
    # from KID score on TorchMetrics
    mmd_value = poly_mmd(real, fake)
    return mmd_value
