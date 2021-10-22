import numpy as np


def _gaussian_fit(img):
    assert img.ndim == 2, '_gaussian_fit: Wrong image dimentions'
    assert (img >= 0).all(), '_gaussian_fit: negative image content'
    assert (img > 0).any(), '_gaussian_fit: blank image'
    img_n = img / img.sum()

    mu = np.fromfunction(lambda i, j: (img_n[np.newaxis, ...] * np.stack([i, j])).sum(axis=(1, 2)), shape=img.shape)
    cov = np.fromfunction(
        lambda i, j: ((img_n[np.newaxis, ...] * np.stack([i * i, j * i, i * j, j * j])).sum(axis=(1, 2)))
        - np.stack([mu[0] ** 2, mu[0] * mu[1], mu[0] * mu[1], mu[1] ** 2]),
        shape=img.shape,
    ).reshape(2, 2)
    return mu, cov


def _get_val_metric_single(img):
    """Returns a vector of gaussian fit results to the image.
    The components are: [mu0, mu1, sigma0^2, sigma1^2, covariance, integral]
    """
    assert img.ndim == 2, '_get_val_metric_single: Wrong image dimentions'

    img = np.where(img < 0, 0, img)

    mu, cov = _gaussian_fit(img)

    return np.array((*mu, *cov.diagonal(), cov[0, 1], img.sum()))


_METRIC_NAMES = ['Mean0', 'Mean1', 'Sigma0^2', 'Sigma1^2', 'Cov01', 'Sum']


get_val_metric = np.vectorize(_get_val_metric_single, signature='(m,n)->(k)')


def get_val_metric_v(imgs):
    """Returns a vector of gaussian fit results to the image.
    The components are: [mu0, mu1, sigma0^2, sigma1^2, covariance, integral]
    """
    assert imgs.ndim == 3, 'get_val_metric_v: Wrong images dimentions'
    assert (imgs >= 0).all(), 'get_val_metric_v: Negative image content'
    assert (imgs > 0).any(axis=(1, 2)).all(), 'get_val_metric_v: some images are empty'
    imgs_n = imgs / imgs.sum(axis=(1, 2), keepdims=True)
    mu = np.fromfunction(
        lambda i, j: (imgs_n[:, np.newaxis, ...] * np.stack([i, j])[np.newaxis, ...]).sum(axis=(2, 3)),
        shape=imgs.shape[1:],
    )

    cov = np.fromfunction(
        lambda i, j: ((imgs_n[:, np.newaxis, ...] * np.stack([i * i, j * j, i * j])[np.newaxis, ...]).sum(axis=(2, 3)))
        - np.stack([mu[:, 0] ** 2, mu[:, 1] ** 2, mu[:, 0] * mu[:, 1]]).T,
        shape=imgs.shape[1:],
    )

    return np.concatenate([mu, cov, imgs.sum(axis=(1, 2))[:, np.newaxis]], axis=1)
