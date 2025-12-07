import numpy as np
from scipy.linalg import sqrtm
from scipy import linalg

# A lot of this is directly from the Google Research FAD repo
def _stable_trace_sqrt_product(sigma_test, sigma_train, eps=1e-7):
    """Avoids some problems when computing the srqt of product of sigmas.
  Based on Dougal J. Sutherland's contribution here:
  https://github.com/bioinf-jku/TTUR/blob/master/fid.py
  Args:
    sigma_test: Test covariance matrix.
    sigma_train: Train covariance matirx.
    eps: Small number; used to avoid singular product.
  Returns:
    The Trace of the square root of the product of the passed convariance
    matrices.
  Raises:
    ValueError: If the sqrt of the product of the sigmas contains complex
        numbers with large imaginary parts.
  """
    # product might be almost singular
    sqrt_product, _ = sqrtm(sigma_test.dot(sigma_train), disp=False)
    if not np.isfinite(sqrt_product).all():
        # add eps to the diagonal to avoid a singular product.
        offset = np.eye(sigma_test.shape[0]) * eps
        sqrt_product = sqrtm((sigma_test + offset).dot(sigma_train + offset))

    # Might have a slight imaginary component.
    if not np.allclose(np.diagonal(sqrt_product).imag, 0, atol=1e-2):
        raise ValueError('sqrt_product contains large complex numbers.')
    sqrt_product = sqrt_product.real

    return np.trace(sqrt_product)


def compute_fad(mu_train, sigma_train, mu_test, sigma_test, eps=1e-6):
    if len(mu_train.shape) != 1:
        raise ValueError('mu_train must be 1 dimensional.')
    if len(sigma_train.shape) != 2:
        raise ValueError('sigma_train must be 2 dimensional.')
    
    if mu_test.shape != mu_train.shape:
        raise ValueError('mu_test should have the same shape as mu_train')
    if sigma_test.shape != sigma_train.shape:
        raise ValueError('sigma_test should have the same shape as sigma_train')
    
    mu_diff = mu_test - mu_train
    trace_sqrt_product = _stable_trace_sqrt_product(sigma_test, sigma_train)

    return mu_diff.dot(mu_diff) + np.trace(sigma_test) + np.trace(
        sigma_train) - 2 * trace_sqrt_product

def compute_embedding_stats(features):
    # get the mean and cov
    mu = np.mean(features, axis=0)
    
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_audio_distance(ref_feats, recon_feats):
    mu_bg, sig_bg = compute_embedding_stats(ref_feats)
    mu_fg, sig_fg = compute_embedding_stats(recon_feats)
    return compute_fad(mu_bg, sig_bg, mu_fg, sig_fg)
