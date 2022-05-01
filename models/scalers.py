import numpy as np

from metrics.gaussian_metrics import get_val_metric_v as gaussian_fit


class Identity:
    def scale(self, x):
        return x

    def unscale(self, x):
        return x


class Logarithmic:
    def scale(self, x):
        return np.log10(1 + x)

    def unscale(self, x):
        return 10**x - 1


class Gaussian:
    def __init__(self, shape=(8, 16)):
        self.shape = shape

    def scale(self, x):
        result = gaussian_fit(x)
        result[:, -1] = np.log1p(result[:, -1])
        result[:, 4] /= result[:, 2] * result[:, 3]
        return result

    def unscale(self, x):
        m0, m1, D00, D11, D01, logA = x.T
        D00 = np.clip(D00, 0.05, None)
        D11 = np.clip(D11, 0.05, None)
        D01 = np.clip(D01, -1.0, 1.0)
        D01 *= D00 * D11

        A = np.expm1(logA)

        cov = np.stack([np.stack([D00, D01], axis=1), np.stack([D01, D11], axis=1)], axis=2)  # N x 2 x 2
        invcov = np.linalg.inv(cov)
        mu = np.stack([m0, m1], axis=1)

        xx0 = np.arange(self.shape[0])
        xx1 = np.arange(self.shape[1])
        xx0, xx1 = np.meshgrid(xx0, xx1, indexing='ij')
        xx = np.stack([xx0, xx1], axis=2)
        residuals = xx[None, ...] - mu[:, None, None, :]  # N x H x W x 2

        result = np.exp(-0.5 * np.einsum('ijkl,ilm,ijkm->ijk', residuals, invcov, residuals))

        result /= result.sum(axis=(1, 2), keepdims=True)
        result *= A[:, None, None]

        return result


def get_scaler(scaler_type):
    if scaler_type == 'identity':
        return Identity()
    elif scaler_type == 'logarithmic':
        return Logarithmic()
    elif scaler_type == 'gaussian':
        return Gaussian()
    else:
        raise NotImplementedError(scaler_type)
