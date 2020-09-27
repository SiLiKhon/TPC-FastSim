import io

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import PIL


from .plotting import _bootstrap_error
from .gaussian_metrics import get_val_metric_v, _METRIC_NAMES
from .trends import make_trend_plot


def make_histograms(data_real, data_gen, title, figsize=(8, 8), n_bins=100, logy=False):
    l = min(data_real.min(), data_gen.min())
    r = max(data_real.max(), data_gen.max())
    bins = np.linspace(l, r, n_bins + 1)
    
    fig = plt.figure(figsize=figsize)
    plt.hist(data_real, bins=bins, density=True, label='real')
    plt.hist(data_gen , bins=bins, density=True, label='generated', alpha=0.7)
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


def make_metric_plots(images_real, images_gen, features=None, calc_chi2=False):
    plots = {}
    if calc_chi2:
        chi2 = 0

    try:
        metric_real = get_val_metric_v(images_real)
        metric_gen  = get_val_metric_v(images_gen )
    
        plots.update({name : make_histograms(real, gen, name)
                      for name, real, gen in zip(_METRIC_NAMES, metric_real.T, metric_gen.T)})

        if features is not None:
            for feature_name, (feature_real, feature_gen) in features.items():
                for metric_name, real, gen in zip(_METRIC_NAMES, metric_real.T, metric_gen.T):
                    name = f'{metric_name} vs {feature_name}'
                    if calc_chi2 and (metric_name != "Sum"):
                        plots[name], chi2_i = make_trend_plot(feature_real, real,
                                                              feature_gen, gen,
                                                              name, calc_chi2=True)
                        chi2 += chi2_i
                    else:
                        plots[name] = make_trend_plot(feature_real, real,
                                                      feature_gen, gen, name)

    except AssertionError as e:
        print(f"WARNING! Assertion error ({e})")

    if calc_chi2:
        return plots, chi2

    return plots


def make_images_for_model(model,
                          sample,
                          return_raw_data=False,
                          calc_chi2=False,
                          gen_more=None,
                          batch_size=128):
    X, Y = sample
    assert X.ndim == 2
    assert X.shape[1] == 4

    if gen_more is None:
        gen_features = X
    else:
        gen_features = np.tile(
            X,
            [gen_more] + [1] * (X.ndim - 1)
        )
    gen_scaled = np.concatenate([
        model.make_fake(gen_features[i:i+batch_size]).numpy()
        for i in range(0, len(gen_features), batch_size)
    ], axis=0)
    real = model.scaler.unscale(Y)
    gen = model.scaler.unscale(gen_scaled)
    gen[gen < 0] = 0
    gen1 = np.where(gen < 1., 0, gen)

    features = {
        'crossing_angle' : (X[:, 0], gen_features[:,0]),
        'dip_angle'      : (X[:, 1], gen_features[:,1]),
        'drift_length'   : (X[:, 2], gen_features[:,2]),
        'time_bin_fraction' : (X[:, 2] % 1, gen_features[:,2] % 1),
        'pad_coord_fraction' : (X[:, 3] % 1, gen_features[:,3] % 1)
    }

    images = make_metric_plots(real, gen, features=features, calc_chi2=calc_chi2)
    if calc_chi2:
        images, chi2 = images

    images1 = make_metric_plots(real, gen1, features=features)

    img_amplitude = make_histograms(Y.flatten(), gen_scaled.flatten(), 'log10(amplitude + 1)', logy=True)

    result = [images, images1, img_amplitude]

    if return_raw_data:
        result += [(gen_features, gen)]

    if calc_chi2:
        result += [chi2]

    return result


def evaluate_model(model, path, sample, gen_sample_name=None):
    path.mkdir()
    (
        images, images1, img_amplitude,
        gen_dataset, chi2
    ) = make_images_for_model(model, sample=sample,
                              calc_chi2=True, return_raw_data=True, gen_more=10)

    array_to_img = lambda arr: PIL.Image.fromarray(arr.reshape(arr.shape[1:]))

    for k, img in images.items():
        array_to_img(img).save(str(path / f"{k}.png"))
    for k, img in images1.items():
        array_to_img(img).save(str(path / f"{k}_amp_gt_1.png"))
    array_to_img(img_amplitude).save(str(path / "log10_amp_p_1.png"))

    if gen_sample_name is not None:
        with open(str(path / gen_sample_name), 'w') as f:
            for event_X, event_Y in zip(*gen_dataset):
                f.write('params: {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(*event_X))
                for ipad, time_distr in enumerate(event_Y, model.pad_range[0] + event_X[3].astype(int)):
                    for itime, amp in enumerate(time_distr, model.time_range[0] + event_X[2].astype(int)):
                        if amp < 1:
                            continue
                        f.write(" {:2d} {:3d} {:8.3e} ".format(ipad, itime, amp))
                f.write('\n')

    with open(str(path / 'stats'), 'w') as f:
        f.write(f"{chi2:.2f}\n")