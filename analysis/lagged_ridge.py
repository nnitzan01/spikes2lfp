"""
Causal lagged ridge regression (a Wiener filter on a lag basis): the linear baseline with the same
kind of input history as the LSTM (reviewer 3, comment 3).

Each input column z (a unit's spikes, a region's rate, a covariate) is expanded into n_basis
features, z filtered with raised-cosine lag kernels b_k over lags 0 .. n_lags - 1 bins:

    f_k(t) = sum_l b_k(l) z(t - l)          (only the present and the past)

The kernels are log-stretched (Pillow et al., 2008): narrow at short lags, wide at long ones.
Features are z-scored with training statistics and the targets centred; for each penalty alpha
the weights are

    W(alpha) = (F'F + alpha n I)^-1 F'Y      (n = training rows; alpha is per row)

solved for all penalties and targets at once through one eigendecomposition of F'F.
"""
import numpy as np
from scipy.signal import fftconvolve


def raised_cosine_basis(n_lags, n_basis, offset=None):
    """(n_lags x n_basis) log-stretched raised cosines tiling lags 0 .. n_lags - 1, each summing to 1."""
    if n_basis == 1:
        return np.ones((n_lags, 1)) / n_lags
    offset = offset or max(1.0, n_lags / 50)                 # sets how strongly short lags are favoured
    x = np.log(np.arange(n_lags) + offset)
    centres = np.linspace(x[0], x[-1], n_basis)
    spacing = centres[1] - centres[0]
    arg = np.clip((x[:, None] - centres[None, :]) * np.pi / (2 * spacing), -np.pi, np.pi)
    basis = 0.5 * (1 + np.cos(arg))
    return basis / basis.sum(axis=0)


def lagged_features(Z, basis, rows, columns_per_pass=64):
    """
    Lag-basis features (len(rows) x n_columns * n_basis, float32; column order: input-major,
    basis-minor) of Z (bins x n_columns) at the given rows. Bins before the start of Z count as 0.
    """
    n_lags, n_basis = basis.shape
    out = np.empty((len(rows), Z.shape[1] * n_basis), dtype=np.float32)
    for a in range(0, Z.shape[1], columns_per_pass):
        block = np.asarray(Z[:, a:a + columns_per_pass], dtype=np.float64)
        for k in range(n_basis):
            filtered = fftconvolve(block, basis[:, k:k + 1], axes=0)[:Z.shape[0]]
            out[:, (a + np.arange(block.shape[1])) * n_basis + k] = filtered[rows]
    return out


class RidgePath:
    """Ridge fits of Y (rows x targets) on F (rows x features) for several penalties (see module docstring)."""

    def __init__(self, F, Y, alphas, chunk=8192):
        self.alphas = np.asarray(alphas, dtype=float)
        n, p = F.shape
        self.mean = F.mean(axis=0, dtype=np.float64)
        sd = F.std(axis=0, dtype=np.float64)
        self.scale = np.where(sd > 0, sd, 1.0)
        self.y_mean = Y.mean(axis=0)
        gram, fty = np.zeros((p, p)), np.zeros((p, Y.shape[1]))
        for a in range(0, n, chunk):
            Fz = self.standardise(F[a:a + chunk])
            gram += Fz.T @ Fz
            fty += Fz.T @ (Y[a:a + chunk] - self.y_mean)
        evals, self.evecs = np.linalg.eigh(gram)
        self.evals = np.clip(evals, 0, None)
        self.proj = self.evecs.T @ fty                          # eigenbasis x targets
        self.n_rows = n

    def standardise(self, F):
        return (np.asarray(F, dtype=np.float64) - self.mean) / self.scale

    def weights(self, alpha):
        """Standardised-feature weights (features x targets) for one penalty."""
        return self.evecs @ (self.proj / (self.evals + alpha * self.n_rows)[:, None])

    def predict(self, F, W):
        return self.standardise(F) @ W + self.y_mean


def r2_columns(Y, P):
    """R2 of each column of the predictions P against Y (rows x targets)."""
    ss = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ss > 0, 1 - ((Y - P) ** 2).sum(axis=0) / ss, np.nan)


def fit_select(Z, Y, basis, train_rows, val_rows, alphas, columns_per_pass=64):
    """
    Fit the lagged ridge on train_rows and choose each target's penalty on val_rows. Z: inputs
    (bins x columns), Y: targets (bins x targets). Returns (RidgePath, standardised weights at the
    chosen penalties: features x targets, chosen alpha per target, validation R2 per alpha x target).
    """
    path = RidgePath(lagged_features(Z, basis, train_rows, columns_per_pass), Y[train_rows], alphas)
    F_val = lagged_features(Z, basis, val_rows, columns_per_pass)
    val_r2 = np.stack([r2_columns(Y[val_rows], path.predict(F_val, path.weights(a))) for a in alphas])
    best = np.nanargmax(np.where(np.isfinite(val_r2), val_r2, -np.inf), axis=0)
    W = np.stack([path.weights(a) for a in alphas])[best, :, np.arange(Y.shape[1])].T     # features x targets
    return path, W, np.asarray(alphas)[best], val_r2


def fit_select_predict(Z, Y, basis, train_rows, val_rows, test_rows, alphas, columns_per_pass=64):
    """
    fit_select, then predict test_rows, building the test features block by block. Returns (test
    predictions: len(test_rows) x targets, chosen alpha per target, validation R2 per alpha x target).
    """
    path, W, alpha, val_r2 = fit_select(Z, Y, basis, train_rows, val_rows, alphas, columns_per_pass)
    return predict_lagged(path, W, Z, basis, test_rows, columns_per_pass), alpha, val_r2


def predict_lagged(path, W, Z, basis, rows, columns_per_pass=64):
    """path.predict on the lag features of Z at rows, built one block of input columns at a time (bounded memory)."""
    n_basis = basis.shape[1]
    prediction = np.tile(path.y_mean, (len(rows), 1))
    for a in range(0, Z.shape[1], columns_per_pass):
        cols = np.arange(a * n_basis, min(Z.shape[1], a + columns_per_pass) * n_basis)
        F = lagged_features(Z[:, a:a + columns_per_pass], basis, rows)
        prediction += ((F - path.mean[cols]) / path.scale[cols]) @ W[cols]
    return prediction
