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
    if not np.allclose(np.diagonal(sqrt_product).imag, 0, atol=1e-3):
        raise ValueError('sqrt_product contains large complex numbers.')
    sqrt_product = sqrt_product.real

    return np.trace(sqrt_product)


def compute_fad(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def compute_embedding_stats(features):
    # get the mean and cov
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_audio_distance(ref_feats, recon_feats):
    mu_bg, sig_bg = compute_embedding_stats(ref_feats)
    mu_fg, sig_fg = compute_embedding_stats(recon_feats)
    return compute_fad(mu_bg, sig_bg, mu_fg, sig_fg)
