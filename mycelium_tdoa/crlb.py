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
                           max_tau: float = 30.0, seed: int = 0):
    """Empirically measure per-electrode timing precision sigma_t (s) vs SNR.

    Runs the real delay-estimation pipeline (synthesise -> averaged GCC-PHAT) and
    converts the relative-delay error variance to an absolute per-electrode
    sigma_t (var of a TDOA ~ 2 sigma_t^2). This ties the Fisher analysis to the
    measured behaviour of the signal stage rather than an assumed number.
    """
    rng = np.random.default_rng(seed)
    src = (medium.size_cm * 0.55, medium.size_cm * 0.4)
    true = medium.arrival_times(src, electrodes)
    true_rel = true - true[0]
    out = {}
    for snr in snr_db_values:
        errs = []
        for _ in range(n_trials):
            est = estimate_delays_averaged(true, n_repeats=n_reps, snr_db=snr,
                                           max_tau=max_tau, rng=rng)
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
