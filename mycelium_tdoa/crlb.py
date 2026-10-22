"""Identifiability analysis for source localisation on a mycelial medium.

The headline tool is the Cramer-Rao lower bound (CRLB): the smallest position
variance any unbiased estimator can achieve, given the electrode geometry, the
timing noise, and the *actual* (warped, anisotropic) travel-time geometry of the
medium. Because it depends on the medium only through the gradients of the
travel-time fields, it turns "does this localise well?" into a rigorous,
falsifiable design question that needs no hardware.

Measurement model (TDOA with unknown emission time):
    t_e = t_emit + travel_e(p) + n_e ,   n_e ~ N(0, sigma_t^2),  i.i.d. over e.
Marginalising the nuisance t_emit (its derivative is 1 for every electrode) gives
the 2x2 Fisher information for the position p = (x, y):
    F(p) = (1 / sigma_t^2) * G^T P G ,
where G is the E x 2 matrix of per-electrode travel-time gradients and
P = I_E - (1/E) 1 1^T projects out the common-mode emission-time direction.
The positional CRLB (RMSE) is sqrt(trace(F^-1)). It scales linearly with sigma_t,
so the sigma_t = 1 map is purely *geometric* (a generalised GDOP) and can be
rescaled for any noise level for free.
"""
from __future__ import annotations

import numpy as np

from .medium import MycelialMedium
from .signals import estimate_delays_averaged


# --------------------------------------------------------------------- gradients
def travel_time_fields(medium: MycelialMedium, electrodes: np.ndarray) -> np.ndarray:
    """Stack of travel-time fields, one per electrode: shape (E, n, n)."""
    return np.stack([medium.travel_time_field(ex, ey) for ex, ey in electrodes])


def field_gradients(fields: np.ndarray, dx: float):
    """Per-electrode spatial gradients (gx, gy), each shape (E, n, n)."""
    gy, gx = [], []
    for f in fields:
        gyf, gxf = np.gradient(f, dx)  # axis 0 = y (rows), axis 1 = x (cols)
        gy.append(gyf)
        gx.append(gxf)
    return np.stack(gx), np.stack(gy)


# --------------------------------------------------------------------- CRLB maps
def geometric_crlb_map(medium: MycelialMedium, electrodes: np.ndarray,
                       cap: float | None = None):
    """sigma_t = 1 positional CRLB (cm) over the whole substrate (generalised GDOP).

    Multiply the returned map by sigma_t (s) to get the CRLB for that noise level.
    Large values mark ill-posed regions (poor geometry / outside the array).
    """
    fields = travel_time_fields(medium, electrodes)
    gx, gy = field_gradients(fields, medium.dx)
    E = len(electrodes)
    # F = G^T P G  with P = I - 11^T/E  =>  centred sums across electrodes
    sx = gx.sum(0)
    sy = gy.sum(0)
    Fxx = (gx * gx).sum(0) - sx * sx / E
    Fyy = (gy * gy).sum(0) - sy * sy / E
    Fxy = (gx * gy).sum(0) - sx * sy / E
    det = Fxx * Fyy - Fxy * Fxy
    det = np.where(det <= 1e-12, np.nan, det)
    trace_inv = (Fxx + Fyy) / det           # trace(F^-1) for sigma_t = 1
    crlb = np.sqrt(np.clip(trace_inv, 0, None))
    if cap is not None:
        crlb = np.minimum(crlb, cap)
    return crlb


def crlb_at(medium: MycelialMedium, electrodes: np.ndarray, point, sigma_t: float):
    """CRLB positional RMSE (cm) at a single point for a given timing noise."""
    gmap = geometric_crlb_map(medium, electrodes)
    i, j = divmod(medium.node_of(*point), medium.grid_n)
    return float(sigma_t * gmap[i, j])


def n_minus_1_crlb_map(medium: MycelialMedium, electrodes: np.ndarray):
    """sigma_t = 1 CRLB map that survives the loss of any single electrode.

    Worst case over the E maps computed with one electrode removed: a point is
    feasible on this map only if it stays feasible whichever electrode fails.
    """
    maps = [geometric_crlb_map(medium, np.delete(electrodes, k, axis=0))
            for k in range(len(electrodes))]
    stacked = np.stack([np.where(np.isfinite(m), m, np.inf) for m in maps])
    worst = stacked.max(0)
    return np.where(np.isfinite(worst), worst, np.nan)


def electrode_noise_cov(electrodes: np.ndarray, sigma_t: float, rho: float,
                        corr_len_cm: float) -> np.ndarray:
    """Timing-noise covariance with spatially correlated components.

    A fraction `rho` of each electrode's noise variance is shared with its
    neighbours, decaying as exp(-distance / corr_len_cm); the rest is independent.
    rho = 0 recovers the i.i.d. model of Eq. (1).
    """
    d = np.linalg.norm(electrodes[:, None, :] - electrodes[None, :, :], axis=-1)
    return sigma_t**2 * ((1.0 - rho) * np.eye(len(electrodes)) + rho * np.exp(-d / corr_len_cm))


def crlb_map_cov(medium: MycelialMedium, electrodes: np.ndarray, cov: np.ndarray):
    """Positional CRLB (cm) for a general timing-noise covariance `cov` (E x E).

    With W = cov^-1 and H = [G, 1], the Fisher information of (p, t_emit) is
    H^T W H; the Schur complement on t_emit gives the 2x2 position block
        F = G^T W G - (G^T W 1)(1^T W G) / (1^T W 1).
    For cov = sigma_t^2 I this equals Eq. (2). Noise that is common to every
    electrode is indistinguishable from t_emit and costs nothing.
    """
    fields = travel_time_fields(medium, electrodes)
    gx, gy = field_gradients(fields, medium.dx)
    W = np.linalg.inv(cov)
    w1 = W.sum(1)
    c = w1.sum()
    Wgx = np.tensordot(W, gx, axes=1)
    Wgy = np.tensordot(W, gy, axes=1)
    ux = np.tensordot(w1, gx, axes=1)
    uy = np.tensordot(w1, gy, axes=1)
    Fxx = (gx * Wgx).sum(0) - ux * ux / c
    Fyy = (gy * Wgy).sum(0) - uy * uy / c
    Fxy = (gx * Wgy).sum(0) - ux * uy / c
    det = Fxx * Fyy - Fxy * Fxy
    det = np.where(det <= 1e-12, np.nan, det)
    return np.sqrt(np.clip((Fxx + Fyy) / det, 0, None))


def feasible_fraction(geom_map: np.ndarray, sigma_t: float, threshold_cm: float,
                      mask: np.ndarray | None = None) -> float:
    """Fraction of substrate where CRLB (= sigma_t * geom_map) is below threshold."""
    crlb = sigma_t * geom_map
    valid = np.isfinite(crlb)
    if mask is not None:
        valid &= mask
    return float(np.mean(crlb[valid] < threshold_cm))


# --------------------------------------------------- empirical noise calibration
def calibrate_timing_noise(medium: MycelialMedium, electrodes: np.ndarray,
                           snr_db_values, n_reps: int = 6, n_trials: int = 40,
                           max_tau: float = 30.0, seed: int = 0,
                           pink_fraction: float = 0.0, margin_cm: float = 1.0):
    """Empirically measure per-electrode timing precision sigma_t (s) vs SNR.

    Runs the real delay-estimation pipeline (synthesise -> averaged GCC-PHAT) and
    converts the relative-delay error variance to an absolute per-electrode
    sigma_t (var of a TDOA ~ 2 sigma_t^2). This ties the Fisher analysis to the
    measured behaviour of the signal stage rather than an assumed number.

    Each trial draws a new stimulus position: the delay spread, and so the error,
    depends on where the stimulus is, and a single position is not representative.
    `pink_fraction` is passed to the trace synthesis (share of 1/f noise power).
    """
    rng = np.random.default_rng(seed)
    out = {}
    for snr in snr_db_values:
        errs = []
        for _ in range(n_trials):
            src = tuple(rng.uniform(margin_cm, medium.size_cm - margin_cm, size=2))
            true = medium.arrival_times(src, electrodes)
            true_rel = true - true[0]
            est = estimate_delays_averaged(true, n_repeats=n_reps, snr_db=snr,
                                           max_tau=max_tau, rng=rng,
                                           pink_fraction=pink_fraction)
            errs.append((est - true_rel)[1:])  # drop the reference (always 0)
        errs = np.asarray(errs)
        var_tdoa = np.mean(errs**2)
        out[float(snr)] = float(np.sqrt(var_tdoa / 2.0))
    return out


def fit_sigma_t_vs_snr(calib: dict):
    """Fit sigma_t = a / sqrt(SNR_linear) (time-delay-estimation scaling)."""
    snr_db = np.array(sorted(calib))
    sig = np.array([calib[s] for s in snr_db])
    snr_lin = 10 ** (snr_db / 10.0)
    a = np.mean(sig * np.sqrt(snr_lin))  # least-squares through origin in 1/sqrt(SNR)
    return float(a), (lambda snr_db_q: a / np.sqrt(10 ** (np.asarray(snr_db_q) / 10.0)))


# ------------------------------------------------------------- ML estimator (MC)
def ml_localize(fields: np.ndarray, observed_tdoa: np.ndarray, ref: int = 0):
    """Maximum-likelihood TDOA position on the grid (efficient, asymptotically unbiased).

    Returns the (x, y) grid point minimising the whitened TDOA residual. Used to
    *validate* the CRLB: an efficient estimator should reach it where geometry is good.
    """
    E, n, _ = fields.shape
    model = fields - fields[ref]                      # (E, n, n) modelled TDOAs
    keep = [e for e in range(E) if e != ref]
    resid = observed_tdoa[:, None, None] - model[keep]  # (E-1, n, n)
    # whitened cost with C^-1 = (1/sigma^2)(I - 11^T/E); sigma cancels in argmin
    s = resid.sum(0)
    cost = (resid**2).sum(0) - s * s / E
    idx = np.argmin(cost)
    i, j = divmod(idx, n)
    return j, i  # (col=x index, row=y index)


def mmse_localize(fields: np.ndarray, observed_tdoa: np.ndarray, sigma_t: float,
                  xx: np.ndarray, yy: np.ndarray, prior_mask: np.ndarray, ref: int = 0):
    """Posterior-mean position under a uniform prior on `prior_mask`.

    Same Gaussian TDOA likelihood as `ml_localize`, but averaged over the grid
    instead of maximised. Under that prior this is the minimum-mean-square-error
    estimator: no estimator, biased or not, has a lower expected squared error.
    """
    E = fields.shape[0]
    model = fields - fields[ref]
    keep = [e for e in range(E) if e != ref]
    resid = observed_tdoa[:, None, None] - model[keep]
    s = resid.sum(0)
    cost = (resid**2).sum(0) - s * s / E
    logp = np.where(prior_mask, -cost / (2.0 * sigma_t**2), -np.inf)
    w = np.exp(logp - logp[prior_mask].max())
    w /= w.sum()
    return float((w * xx).sum()), float((w * yy).sum())


def bayes_comparison(medium: MycelialMedium, electrodes: np.ndarray, sigma_t: float,
                     n_trials: int = 300, margin_cm: float = 1.0, seed: int = 0):
    """Monte-Carlo RMSE of the ML and MMSE estimators for uniformly drawn stimuli.

    The CRLB only bounds *unbiased* estimators. On a bounded substrate a learned
    localiser is pulled towards the interior and can sit below it where the bound
    is large; the MMSE RMSE is the right floor for such estimators.
    Returns (ml_rmse, mmse_rmse, crlb_rms) over the same stimuli; crlb_rms is the
    root-mean-square of the pointwise CRLB, for reference.
    """
    rng = np.random.default_rng(seed)
    fields = travel_time_fields(medium, electrodes)
    gmap = geometric_crlb_map(medium, electrodes)
    lo, hi = margin_cm, medium.size_cm - margin_cm
    prior = (medium.xx >= lo) & (medium.xx <= hi) & (medium.yy >= lo) & (medium.yy <= hi)
    E = len(electrodes)
    se_ml, se_mmse, crlb_sq = [], [], []
    for _ in range(n_trials):
        px, py = rng.uniform(lo, hi, size=2)
        i0, j0 = divmod(medium.node_of(px, py), medium.grid_n)
        tx, ty = j0 * medium.dx, i0 * medium.dx  # grid point the stimulus snaps to
        obs = fields[:, i0, j0] + rng.normal(0, sigma_t, size=E)
        obs_tdoa = (obs - obs[0])[1:]
        jx, iy = ml_localize(fields, obs_tdoa, ref=0)
        se_ml.append((jx * medium.dx - tx) ** 2 + (iy * medium.dx - ty) ** 2)
        mx, my = mmse_localize(fields, obs_tdoa, sigma_t, medium.xx, medium.yy, prior)
        se_mmse.append((mx - tx) ** 2 + (my - ty) ** 2)
        c = sigma_t * gmap[i0, j0]
        crlb_sq.append(c * c if np.isfinite(c) else np.nan)
    return (float(np.sqrt(np.mean(se_ml))), float(np.sqrt(np.mean(se_mmse))),
            float(np.sqrt(np.nanmean(crlb_sq))))


def validate_crlb(medium: MycelialMedium, electrodes: np.ndarray, points,
                  sigma_t: float, n_trials: int = 300, seed: int = 0):
    """Monte-Carlo: empirical ML localisation RMSE vs CRLB at each test point."""
    rng = np.random.default_rng(seed)
    fields = travel_time_fields(medium, electrodes)
    gmap = geometric_crlb_map(medium, electrodes)
    E = len(electrodes)
    res = []
    for (px, py) in points:
        node = medium.node_of(px, py)
        i0, j0 = divmod(node, medium.grid_n)
        true_t = fields[:, i0, j0]
        true_tdoa = (true_t - true_t[0])[1:]
        ex, ey = [], []
        for _ in range(n_trials):
            noise = rng.normal(0, sigma_t, size=E)
            obs = (true_t + noise)
            obs_tdoa = (obs - obs[0])[1:]
            jx, iy = ml_localize(fields, obs_tdoa, ref=0)
            ex.append(jx * medium.dx)
            ey.append(iy * medium.dx)
        ex = np.array(ex); ey = np.array(ey)
        emp_rmse = np.sqrt(np.mean((ex - px) ** 2 + (ey - py) ** 2))
        crlb = sigma_t * gmap[i0, j0]
        res.append((px, py, float(crlb), float(emp_rmse)))
    return res
