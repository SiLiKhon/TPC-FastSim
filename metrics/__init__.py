import io

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL

from .gaussian_metrics import get_val_metric_v, _METRIC_NAMES
from .trends import make_trend_plot

mpl.use("Agg")


def make_histograms(
    data_real,
    data_gen,
    title,
    figsize=(8, 8),
    n_bins=100,
    logy=False,
    pdffile=None,
    label_real='real',
    label_gen='generated',
):
    left = min(data_real.min(), data_gen.min())
    right = max(data_real.max(), data_gen.max())
    bins = np.linspace(left, right, n_bins + 1)

    fig = plt.figure(figsize=figsize)
    plt.hist(data_real, bins=bins, density=True, label=label_real)
    plt.hist(data_gen, bins=bins, density=True, label=label_gen, alpha=0.7)
    if logy:
        plt.yscale('log')
    plt.legend()
    plt.title(title)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    if pdffile is not None:
        fig.savefig(pdffile, format='pdf')
    plt.close(fig)
    buf.seek(0)

    img = PIL.Image.open(buf)
    return np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[1], img.size[0], -1)


def make_metric_plots(
    images_real, images_gen, features=None, calc_chi2=False, make_pdfs=False, label_real='real', label_gen='generated'
):
    plots = {}
    if make_pdfs:
        pdf_plots = {}
    if calc_chi2:
        chi2 = 0

    try:
        metric_real = get_val_metric_v(images_real)
        metric_gen = get_val_metric_v(images_gen)

        for name, real, gen in zip(_METRIC_NAMES, metric_real.T, metric_gen.T):
            pdffile = None
            if make_pdfs:
                pdffile = io.BytesIO()
                pdf_plots[name] = pdffile
            plots[name] = make_histograms(real, gen, name, pdffile=pdffile, label_real=label_real, label_gen=label_gen)

        if features is not None:
            for feature_name, (feature_real, feature_gen) in features.items():
                for metric_name, real, gen in zip(_METRIC_NAMES, metric_real.T, metric_gen.T):
                    name = f'{metric_name} vs {feature_name}'
                    pdffile = None
                    if make_pdfs:
                        pdffile = io.BytesIO()
                        pdf_plots[name] = pdffile
                    if calc_chi2 and (metric_name != "Sum"):
                        plots[name], chi2_i = make_trend_plot(
                            feature_real,
                            real,
                            feature_gen,
                            gen,
                            name,
                            calc_chi2=True,
                            pdffile=pdffile,
                            label_real=label_real,
                            label_gen=label_gen,
                        )
                        chi2 += chi2_i
                    else:
                        plots[name] = make_trend_plot(
                            feature_real,
                            real,
                            feature_gen,
                            gen,
                            name,
                            pdffile=pdffile,
                            label_real=label_real,
                            label_gen=label_gen,
                        )

    except AssertionError as e:
        print(f"WARNING! Assertion error ({e})")

    result = {'plots': plots}
    if calc_chi2:
        result['chi2'] = chi2
    if make_pdfs:
        result['pdf_plots'] = pdf_plots

    return result


def make_images_for_model(
    model, sample, return_raw_data=False, calc_chi2=False, gen_more=None, batch_size=128, pdf_outputs=None
):
    X, Y = sample
    assert X.ndim == 2
    if model.data_version == 'data_v4plus':
        assert X.shape[1] == 6
    else:
        assert X.shape[1] == 4
    make_pdfs = pdf_outputs is not None
    if make_pdfs:
        assert isinstance(pdf_outputs, list)
        assert len(pdf_outputs) == 0

    if gen_more is None:
        gen_features = X
    else:
        gen_features = np.tile(X, [gen_more] + [1] * (X.ndim - 1))
    gen_scaled = np.concatenate(
        [model.make_fake(gen_features[i : i + batch_size]).numpy() for i in range(0, len(gen_features), batch_size)],
        axis=0,
    )
    real = model.scaler.unscale(Y)
    gen = model.scaler.unscale(gen_scaled)
    gen[gen < 0] = 0
    gen1 = np.where(gen < 1.0, 0, gen)

    features = {
        'crossing_angle': (X[:, 0], gen_features[:, 0]),
        'dip_angle': (X[:, 1], gen_features[:, 1]),
        'drift_length': (X[:, 2], gen_features[:, 2]),
        'time_bin_fraction': (X[:, 2] % 1, gen_features[:, 2] % 1),
        'pad_coord_fraction': (X[:, 3] % 1, gen_features[:, 3] % 1),
    }
    if model.data_version == 'data_v4plus' and model.include_pT_for_evaluation:
        features['pT'] = (X[:, 5], gen_features[:, 5])

    metric_plot_results = make_metric_plots(real, gen, features=features, calc_chi2=calc_chi2, make_pdfs=make_pdfs)
    images = metric_plot_results['plots']
    if calc_chi2:
        chi2 = metric_plot_results['chi2']
    if make_pdfs:
        images_pdf = metric_plot_results['pdf_plots']
        pdf_outputs.append(images_pdf)

    metric_plot_results1 = make_metric_plots(real, gen1, features=features, make_pdfs=make_pdfs)
    images1 = metric_plot_results1['plots']
    if make_pdfs:
        pdf_outputs.append(metric_plot_results1['pdf_plots'])

    pdffile = None
    if make_pdfs:
        pdffile = io.BytesIO()
        pdf_outputs.append(pdffile)
    img_amplitude = make_histograms(
        Y.flatten(), gen_scaled.flatten(), 'log10(amplitude + 1)', logy=True, pdffile=pdffile
    )

    pdffile_examples = None
    pdffile_examples_mask = None
    if make_pdfs:
        pdffile_examples = io.BytesIO()
        pdffile_examples_mask = io.BytesIO()
        images_pdf['examples'] = pdffile_examples
        images_pdf['examples_mask'] = pdffile_examples_mask
    images['examples'] = plot_individual_images(Y, gen_scaled, pdffile=pdffile_examples)
    images['examples_mask'] = plot_images_mask(Y, gen_scaled, pdffile=pdffile_examples_mask)

    result = [images, images1, img_amplitude]

    if return_raw_data:
        result += [(gen_features, gen)]

    if calc_chi2:
        result += [chi2]

    return result


def evaluate_model(model, path, sample, gen_sample_name=None):
    def array_to_img(arr):
        return PIL.Image.fromarray(arr.reshape(arr.shape[1:]))

    path.mkdir()
    pdf_outputs = []
    (images, images1, img_amplitude, gen_dataset, chi2) = make_images_for_model(
        model, sample=sample, calc_chi2=True, return_raw_data=True, gen_more=10, pdf_outputs=pdf_outputs
    )
    images_pdf, images1_pdf, img_amplitude_pdf = pdf_outputs

    for k, img in images.items():
        array_to_img(img).save(str(path / f"{k}.png"))
    for k, img in images1.items():
        array_to_img(img).save(str(path / f"{k}_amp_gt_1.png"))
    array_to_img(img_amplitude).save(str(path / "log10_amp_p_1.png"))

    def buf_to_file(buf, filename):
        with open(filename, 'wb') as f:
            f.write(buf.getbuffer())

    for k, img in images_pdf.items():
        buf_to_file(img, str(path / f"{k}.pdf"))
    for k, img in images1_pdf.items():
        buf_to_file(img, str(path / f"{k}_amp_gt_1.pdf"))
    buf_to_file(img_amplitude_pdf, str(path / "log10_amp_p_1.pdf"))

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


def plot_individual_images(real, gen, n=10, pdffile=None, label_real='real', label_gen='generated'):
    assert real.ndim == 3 == gen.ndim
    assert real.shape[1:] == gen.shape[1:]
    N_max = min(len(real), len(gen))
    assert n * 2 <= N_max

    idx = np.sort(np.random.choice(N_max, n * 2, replace=False))
    real = real[idx]
    gen = gen[idx]

    size_x = 12
    size_y = size_x / real.shape[2] * real.shape[1] * n * 1.2 / 4

    fig, axx = plt.subplots(n, 4, figsize=(size_x, size_y))
    axx = [(ax[0], ax[1]) for ax in axx] + [(ax[2], ax[3]) for ax in axx]

    for ax, img_real, img_fake in zip(axx, real, gen):
        ax[0].imshow(img_real, aspect='auto')
        ax[0].set_title(label_real)
        ax[0].axis('off')
        ax[1].imshow(img_fake, aspect='auto')
        ax[1].set_title(label_gen)
        ax[1].axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    if pdffile is not None:
        fig.savefig(pdffile, format='pdf')
    plt.close(fig)
    buf.seek(0)

    img = PIL.Image.open(buf)
    return np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[1], img.size[0], -1)


def plot_images_mask(real, gen, pdffile=None, label_real='real', label_gen='generated'):
    assert real.ndim == 3 == gen.ndim
    assert real.shape[1:] == gen.shape[1:]

    size_x = 6
    size_y = size_x / real.shape[2] * real.shape[1] * 2.4

    fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(size_x, size_y))
    ax0.imshow((real >= 1.0).any(axis=0), aspect='auto')
    ax0.set_title(label_real)
    ax1.imshow((gen >= 1.0).any(axis=0), aspect='auto')
    ax1.set_title(label_gen)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    if pdffile is not None:
        fig.savefig(pdffile, format='pdf')
    plt.close(fig)
    buf.seek(0)

    img = PIL.Image.open(buf)
    return np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[1], img.size[0], -1)
