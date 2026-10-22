"""Fingerprinting localiser: learn delay-vector -> (x, y) with uncertainty.

Rather than solving the (very hard) physical inverse problem in an anisotropic,
heterogeneous, non-stationary medium, we calibrate: stimulate at known points,
record the delay vector at each, and fit a regressor. A Gaussian Process gives
both a position estimate and a calibrated uncertainty, and is cheap to recalibrate
as the living medium drifts.
"""
from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .medium import MycelialMedium


def build_fingerprint_dataset(
    medium: MycelialMedium,
    electrodes: np.ndarray,
    n_samples: int = 400,
    delay_noise_s: float = 0.0,
    ref: int = 0,
    rng: np.random.Generator | None = None,
    electrode_bias_s: np.ndarray | None = None,
):
    """Sample stimuli at random known positions; return (X, y).

    `electrode_bias_s` is a fixed timing offset per electrode (contact impedance,
    front-end filtering), shared by every stimulus. Unlike `delay_noise_s` it does
    not average out; calibration absorbs it only while the electrode stays put.

    X : array (n_samples, n_electrodes - 1)  -- delays relative to `ref`
    y : array (n_samples, 2)                 -- true (x, y) in cm
    """
    rng = rng or np.random.default_rng(1)
    margin = 1.0
    xs = rng.uniform(margin, medium.size_cm - margin, size=n_samples)
    ys = rng.uniform(margin, medium.size_cm - margin, size=n_samples)
    X, Y = [], []
    keep = [k for k in range(len(electrodes)) if k != ref]
    for x, y in zip(xs, ys):
        t = medium.arrival_times((x, y), electrodes)
        if electrode_bias_s is not None:
            t = t + electrode_bias_s
        delays = t - t[ref]
        delays = delays[keep]
        if delay_noise_s > 0:
            delays = delays + rng.normal(0, delay_noise_s, size=delays.shape)
        X.append(delays)
        Y.append([x, y])
    return np.asarray(X), np.asarray(Y)


class FingerprintLocalizer:
    """Gaussian-Process localiser mapping a delay vector to a 2D position."""

    def __init__(self, length_scale: float = 1.0, seed: int = 0):
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e3))
            * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        )
        # one GP per output coordinate keeps per-axis uncertainty interpretable
        self._gp_x = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)
        self._gp_y = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "FingerprintLocalizer":
        self._gp_x.fit(X, Y[:, 0])
        self._gp_y.fit(X, Y[:, 1])
        return self

    def predict(self, X: np.ndarray):
        """Return (positions (n,2), sigma (n,)) where sigma is positional std in cm."""
        mx, sx = self._gp_x.predict(X, return_std=True)
        my, sy = self._gp_y.predict(X, return_std=True)
        pos = np.column_stack([mx, my])
        sigma = np.hypot(sx, sy)
        return pos, sigma

    @staticmethod
    def error_cm(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pred - true, axis=1)
