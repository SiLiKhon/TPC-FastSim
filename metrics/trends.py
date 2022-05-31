import io

import numpy as np
import matplotlib.pyplot as plt
import PIL

from .plotting import _bootstrap_error


def calc_trend(x, y, do_plot=True, bins=100, window_size=20, **kwargs):
    assert x.ndim == 1, 'calc_trend: wrong x dim'
    assert y.ndim == 1, 'calc_trend: wrong y dim'

    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.7

    if isinstance(bins, int):
        bins = np.linspace(np.min(x), np.max(x), bins + 1)
    sel = x >= bins[0]
    x, y = x[sel], y[sel]
    cats = (x[:, np.newaxis] < bins[np.newaxis, 1:]).argmax(axis=1)

    def stats(arr):
        return (arr.mean(), arr.std() / (len(arr) - 1) ** 0.5, arr.std(), _bootstrap_error(arr, np.std))

    mean, mean_err, std, std_err, bin_centers = np.array(
        [
            stats(y[(cats >= left) & (cats < right)]) + ((bins[left] + bins[right]) / 2,)
            for left, right in zip(range(len(bins) - window_size), range(window_size, len(bins)))
        ]
    ).T

    if do_plot:
        mean_p_std_err = (mean_err**2 + std_err**2) ** 0.5
        plt.fill_between(bin_centers, mean - mean_err, mean + mean_err, **kwargs)
        kwargs['alpha'] *= 0.5
        kwargs = {k: v for k, v in kwargs.items() if k != 'label'}
        plt.fill_between(bin_centers, mean - std - mean_p_std_err, mean - std + mean_p_std_err, **kwargs)
        plt.fill_between(bin_centers, mean + std - mean_p_std_err, mean + std + mean_p_std_err, **kwargs)
        kwargs['alpha'] *= 0.25
        plt.fill_between(bin_centers, mean - std + mean_p_std_err, mean + std - mean_p_std_err, **kwargs)

    return (mean, std), (mean_err, std_err)


def make_trend_plot(
    feature_real,
    real,
    feature_gen,
    gen,
    name,
    calc_chi2=False,
    figsize=(8, 8),
    pdffile=None,
    label_real='real',
    label_gen='generated',
):
    feature_real = feature_real.squeeze()
    feature_gen = feature_gen.squeeze()
    real = real.squeeze()
    gen = gen.squeeze()

    bins = np.linspace(min(feature_real.min(), feature_gen.min()), max(feature_real.max(), feature_gen.max()), 100)

    fig = plt.figure(figsize=figsize)
    calc_trend(feature_real, real, bins=bins, label=label_real, color='blue')
    calc_trend(feature_gen, gen, bins=bins, label=label_gen, color='red')
    plt.legend()
    plt.title(name)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    if pdffile is not None:
        fig.savefig(pdffile, format='pdf')
    plt.close(fig)
    buf.seek(0)

    img = PIL.Image.open(buf)
    img_data = np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[1], img.size[0], -1)

    if calc_chi2:
        bins = np.linspace(min(feature_real.min(), feature_gen.min()), max(feature_real.max(), feature_gen.max()), 20)
        ((real_mean, real_std), (real_mean_err, real_std_err)) = calc_trend(
            feature_real, real, do_plot=False, bins=bins, window_size=1
        )
        ((gen_mean, gen_std), (gen_mean_err, gen_std_err)) = calc_trend(
            feature_gen, gen, do_plot=False, bins=bins, window_size=1
        )

        gen_upper = gen_mean + gen_std
        gen_lower = gen_mean - gen_std
        gen_err2 = gen_mean_err**2 + gen_std_err**2

        real_upper = real_mean + real_std
        real_lower = real_mean - real_std
        real_err2 = real_mean_err**2 + real_std_err**2

        chi2 = ((gen_upper - real_upper) ** 2 / (gen_err2 + real_err2)).sum() + (
            (gen_lower - real_lower) ** 2 / (gen_err2 + real_err2)
        ).sum()

        return img_data, chi2

    return img_data
