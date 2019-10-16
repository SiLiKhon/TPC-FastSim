import numpy as np

def _gaussian_fit(img):
    assert img.ndim == 2
    assert (img >= 0).all()
    assert (img > 0).any()
    img_n = img / img.sum()

    mu = np.fromfunction(
        lambda i, j: (img_n[np.newaxis,...] * np.stack([i, j])).sum(axis=(1, 2)),
        shape=img.shape
    )
    cov = np.fromfunction(
        lambda i, j: (
            (img_n[np.newaxis,...] * np.stack([i * i, j * i, i * j, j * j])).sum(axis=(1, 2))
        ) - np.stack([mu[0]**2, mu[0]*mu[1], mu[0]*mu[1], mu[1]**2]),
        shape=img.shape
    ).reshape(2, 2)
    return mu, cov

def _get_val_metric_single(img):
    """Returns a vector of gaussian fit results to the image.
    The components are: [mu0, mu1, sigma0^2, sigma1^2, covariance, integral]
    """
    assert img.ndim == 2

    img = np.where(img < 0, 0, img)

    mu, cov = _gaussian_fit(img)

    return np.array((*mu, *cov.diagonal(), cov[0, 1], img.sum()))

get_val_metric = np.vectorize(_get_val_metric_single, signature='(m,n)->(k)')