import io

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import PIL

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

_METRIC_NAMES = ['Mean0', 'Mean1', 'Sigma0^2', 'Sigma1^2', 'Cov01', 'Sum']

get_val_metric = np.vectorize(_get_val_metric_single, signature='(m,n)->(k)')

def get_val_metric_v(imgs):
    """Returns a vector of gaussian fit results to the image.
    The components are: [mu0, mu1, sigma0^2, sigma1^2, covariance, integral]
    """
    assert imgs.ndim == 3
    assert (imgs >= 0).all()
    assert (imgs > 0).any(axis=(1, 2)).all()
    imgs_n = imgs / imgs.sum(axis=(1, 2), keepdims=True)
    mu = np.fromfunction(
        lambda i, j: (imgs_n[:,np.newaxis,...] * np.stack([i, j])[np.newaxis,...]).sum(axis=(2, 3)),
        shape=imgs.shape[1:]
    )

    cov = np.fromfunction(
        lambda i, j: (
            (imgs_n[:,np.newaxis,...] * np.stack([i * i, j * j, i * j])[np.newaxis,...]).sum(axis=(2, 3))
        ) - np.stack([mu[:,0]**2, mu[:,1]**2, mu[:,0] * mu[:,1]]).T,
        shape=imgs.shape[1:]
    )

    return np.concatenate([mu, cov, imgs.sum(axis=(1, 2))[:,np.newaxis]], axis=1)


def make_histograms(data_real, data_gen, title, figsize=(8, 8), n_bins=100, logy=False):
    l = min(data_real.min(), data_gen.min())
    r = max(data_real.max(), data_gen.max())
    bins = np.linspace(l, r, n_bins + 1)
    
    fig = plt.figure(figsize=figsize)
    plt.hist(data_real, bins=bins, label='real')
    plt.hist(data_gen , bins=bins, label='generated', alpha=0.7)
    if logy:
        plt.yscale('log')
    plt.legend()
    plt.title(title)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    img = PIL.Image.open(buf)
    return np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[0], img.size[1], -1)

def make_metric_plots(images_real, images_gen):
    plots = {}
    try:
        metric_real = get_val_metric_v(images_real)
        metric_gen  = get_val_metric_v(images_gen )
    
        plots.update({name : make_histograms(real, gen, name)
                      for name, real, gen in zip(_METRIC_NAMES, metric_real.T, metric_gen.T)})
    except AssertionError:
        pass

    return plots
    
    
