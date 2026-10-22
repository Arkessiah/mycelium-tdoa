"""Identifiability experiment: the part that produces model-independent value.

Outputs:
  id_01_crlb_map.png   -- where on the substrate localisation is feasible vs ill-posed,
                          for two electrode geometries (a generalised GDOP map).
  id_02_validation.png -- empirical ML localisation RMSE vs CRLB (efficiency check),
                          at unit noise and at the operating noise level.
  id_03_phase_diagram.png -- feasible-area fraction over (anisotropy x SNR): a design tool,
                          for white noise and for noise with half its power 1/f.
  id_04_robustness.png -- N-1 electrode loss, spatially correlated timing noise, and
                          the estimators (ML, MMSE, GP) against the bound.
  id_metrics.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from mycelium_tdoa import MycelialMedium, default_electrodes
from mycelium_tdoa.crlb import (
    geometric_crlb_map,
    n_minus_1_crlb_map,
    electrode_noise_cov,
    crlb_map_cov,
    feasible_fraction,
    calibrate_timing_noise,
    fit_sigma_t_vs_snr,
    validate_crlb,
    bayes_comparison,
)
from mycelium_tdoa.localizer import build_fingerprint_dataset, FingerprintLocalizer

SNR_VALUES = [4, 6, 8, 10, 12, 16]
PINK_FRACTION = 0.5  # share of noise power that is 1/f in the coloured-noise scenario

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def star_electrodes(size_cm=20.0, n=8, margin=2.5):
    c = size_cm / 2
    r = c - margin
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([c + r * np.cos(ang), c + r * np.sin(ang)])


def _hull_xy(electrodes):
    h = ConvexHull(electrodes)
    pts = electrodes[h.vertices]
    return np.vstack([pts, pts[0]])


def fig_crlb_maps(sigma_t_ref, snr_ref):
    medium = MycelialMedium(seed=3)
    layouts = [("6 electrodes (corners + mid-edges)", default_electrodes(medium.size_cm)),
               ("8 electrodes (ring)", star_electrodes(medium.size_cm))]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ext = [0, medium.size_cm, 0, medium.size_cm]
    for ax, (name, el) in zip(axes, layouts):
        gmap = geometric_crlb_map(medium, el, cap=None)
        crlb = sigma_t_ref * gmap
        im = ax.imshow(np.clip(crlb, 0, 6), origin="lower", extent=ext, cmap="viridis")
        cs = ax.contour(medium.xx, medium.yy, crlb, levels=[1, 2, 3], colors="w", linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f cm")
        hull = _hull_xy(el)
        ax.plot(hull[:, 0], hull[:, 1], "w--", lw=0.8, alpha=0.7)
        ax.scatter(el[:, 0], el[:, 1], c="red", edgecolor="k", s=55, zorder=5)
        frac = feasible_fraction(gmap, sigma_t_ref, threshold_cm=2.0)
        ax.set_title(f"{name}\nCRLB @ {snr_ref:.0f} dB | <2 cm over {frac*100:.0f}% of area")
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
        fig.colorbar(im, ax=ax, label="CRLB position RMSE (cm)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "id_01_crlb_map.png"), dpi=130)
    plt.close(fig)


def fig_validation(sigma_op, snr_op):
    """ML efficiency at unit noise and at the operating noise level.

    Grid-search ML is only asymptotically efficient; at realistic noise it can
    drift away from the bound, so checking sigma_t = 1 alone is not enough.
    """
    medium = MycelialMedium(seed=3)
    el = default_electrodes(medium.size_cm)
    pts = [(10, 10), (6, 6), (14, 5), (5, 15), (16, 16), (10, 3), (8, 12), (13, 13)]
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ratios = {}
    lim = 0.0
    for st, label, colour in [(1.0, "σt = 1 s", "purple"),
                              (sigma_op, f"σt = {sigma_op:.2f} s ({snr_op:.0f} dB)", "darkorange")]:
        res = validate_crlb(medium, el, pts, sigma_t=st, n_trials=400)
        crlb = np.array([r[2] for r in res])
        emp = np.array([r[3] for r in res])
        ratios[st] = float(np.median(emp / crlb))
        lim = max(lim, crlb.max(), emp.max())
        ax.scatter(crlb, emp, c=colour, s=45, zorder=5, label=f"{label}: median ratio {ratios[st]:.2f}")
    lim *= 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (efficient)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("CRLB position RMSE (cm)")
    ax.set_ylabel("empirical ML RMSE (cm)")
    ax.set_title("CRLB is achievable: ML estimator reaches the bound")
    ax.legend(fontsize=7); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "id_02_validation.png"), dpi=130)
    plt.close(fig)
    return ratios[1.0], ratios[sigma_op]


def phase_table(sigma_by_snr, aniso_vals, threshold_cm):
    """Feasible-area table using the sigma_t *measured* at each SNR.

    The 1/sqrt(SNR) fit holds for white noise but not for 1/f noise, whose timing
    error has a threshold below ~10 dB; the measured values are used for both so
    the two diagrams are built the same way.
    """
    el = default_electrodes(20.0)
    feas = np.zeros((len(SNR_VALUES), len(aniso_vals)))
    for ci, a in enumerate(aniso_vals):
        gmap = geometric_crlb_map(MycelialMedium(seed=3, aniso_ratio=a), el)
        for ri, snr in enumerate(SNR_VALUES):
            feas[ri, ci] = feasible_fraction(gmap, sigma_by_snr[float(snr)], threshold_cm) * 100.0
    return feas


def fig_phase_diagram(calib, calib_pink, threshold_cm=2.0):
    aniso_vals = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    tables = [("white noise", phase_table(calib, aniso_vals, threshold_cm)),
              (f"{PINK_FRACTION:.0%} of noise power 1/f", phase_table(calib_pink, aniso_vals, threshold_cm))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (name, feas) in zip(axes, tables):
        im = ax.imshow(feas, origin="lower", cmap="magma", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(aniso_vals))); ax.set_xticklabels(aniso_vals)
        ax.set_yticks(range(len(SNR_VALUES))); ax.set_yticklabels(SNR_VALUES)
        ax.set_xlabel("anisotropy ratio"); ax.set_ylabel("SNR (dB)")
        ax.set_title(f"% of substrate with CRLB < {threshold_cm:.0f} cm — {name}")
        for ri in range(len(SNR_VALUES)):
            for ci in range(len(aniso_vals)):
                ax.text(ci, ri, f"{feas[ri,ci]:.0f}", ha="center", va="center",
                        color="white" if feas[ri, ci] < 55 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="% feasible area")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "id_03_phase_diagram.png"), dpi=130)
    plt.close(fig)
    return tables[0][1].tolist(), tables[1][1].tolist(), aniso_vals, SNR_VALUES


def fig_robustness(sigma_op, snr_op, threshold_cm=2.0):
    """Three ways the idealised bound can mislead, at the operating noise level."""
    medium = MycelialMedium(seed=3)
    el = default_electrodes(medium.size_cm)
    ext = [0, medium.size_cm, 0, medium.size_cm]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) losing any single electrode
    gmap = geometric_crlb_map(medium, el)
    n1 = n_minus_1_crlb_map(medium, el)
    frac_all = feasible_fraction(gmap, sigma_op, threshold_cm)
    frac_n1 = feasible_fraction(n1, sigma_op, threshold_cm)
    crlb_n1 = sigma_op * n1
    im = axes[0].imshow(np.clip(crlb_n1, 0, 6), origin="lower", extent=ext, cmap="viridis")
    cs = axes[0].contour(medium.xx, medium.yy, crlb_n1, levels=[1, 2, 3], colors="w", linewidths=0.8)
    axes[0].clabel(cs, inline=True, fontsize=7, fmt="%.0f cm")
    axes[0].scatter(el[:, 0], el[:, 1], c="red", edgecolor="k", s=55, zorder=5)
    axes[0].set_title(f"(a) Worst case over any one dead electrode\n<{threshold_cm:.0f} cm over "
                      f"{frac_n1*100:.0f}% of area (all electrodes: {frac_all*100:.0f}%)")
    axes[0].set_xlabel("x (cm)"); axes[0].set_ylabel("y (cm)")
    fig.colorbar(im, ax=axes[0], label="CRLB position RMSE (cm)")

    # (b) spatially correlated timing noise, same per-electrode variance
    rhos = [0.0, 0.25, 0.5, 0.75, 0.9]
    corr = {}
    for ell in [5.0, 20.0]:
        corr[ell] = [feasible_fraction(crlb_map_cov(medium, el, electrode_noise_cov(el, sigma_op, r, ell)),
                                       1.0, threshold_cm) * 100.0 for r in rhos]
        axes[1].plot(rhos, corr[ell], "o-", label=f"correlation length {ell:.0f} cm")
    axes[1].set_xlabel("correlated share of timing-noise variance ρ")
    axes[1].set_ylabel(f"% of substrate with CRLB < {threshold_cm:.0f} cm")
    axes[1].set_title(f"(b) Correlated noise at {snr_op:.0f} dB")
    axes[1].legend(fontsize=8)

    # (c) estimators against the bound, all with the per-electrode noise of Eq. (1)
    ml_rmse, mmse_rmse, crlb_rms = bayes_comparison(medium, el, sigma_op, n_trials=400)
    Xtr, Ytr = build_fingerprint_dataset(medium, el, n_samples=400, timing_noise_s=sigma_op,
                                         rng=np.random.default_rng(1))
    Xte, Yte = build_fingerprint_dataset(medium, el, n_samples=400, timing_noise_s=sigma_op,
                                         rng=np.random.default_rng(99))
    gp_err = FingerprintLocalizer.error_cm(FingerprintLocalizer(length_scale=2.0).fit(Xtr, Ytr).predict(Xte)[0], Yte)
    gp_rmse = float(np.sqrt(np.mean(gp_err**2)))
    names = ["RMS of pointwise\nCRLB (unbiased)", "ML\n(grid search)", "GP\nfingerprinting",
             "MMSE\n(Bayesian optimum)"]
    vals = [crlb_rms, ml_rmse, gp_rmse, mmse_rmse]
    axes[2].bar(range(4), vals, color=["grey", "purple", "darkorange", "teal"])
    for k, v in enumerate(vals):
        axes[2].text(k, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels(names, fontsize=8)
    axes[2].set_ylabel("position RMSE over the substrate (cm)")
    axes[2].set_title(f"(c) Estimators vs bound at {snr_op:.0f} dB (σt = {sigma_op:.2f} s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "id_04_robustness.png"), dpi=130)
    plt.close(fig)
    return {
        "feasible_pct_all_electrodes": round(frac_all * 100.0, 2),
        "feasible_pct_any_one_dead": round(frac_n1 * 100.0, 2),
        "correlated_noise_rho": rhos,
        "correlated_noise_feasible_pct": {f"{k:.0f}cm": [round(v, 2) for v in vs] for k, vs in corr.items()},
        "rmse_crlb_rms_cm": round(crlb_rms, 3),
        "rmse_ml_cm": round(ml_rmse, 3),
        "rmse_gp_cm": round(gp_rmse, 3),
        "rmse_mmse_cm": round(mmse_rmse, 3),
    }


def main():
    medium = MycelialMedium(seed=3)
    el = default_electrodes(medium.size_cm)

    calib = calibrate_timing_noise(medium, el, SNR_VALUES, n_trials=100)
    a_fit, sigma_fn = fit_sigma_t_vs_snr(calib)
    calib_pink = calibrate_timing_noise(medium, el, SNR_VALUES, n_trials=100,
                                        pink_fraction=PINK_FRACTION, seed=1)
    a_fit_pink, _ = fit_sigma_t_vs_snr(calib_pink)
    snr_ref = 10.0
    sigma_ref = float(sigma_fn(snr_ref))

    fig_crlb_maps(sigma_ref, snr_ref)
    eff_unit, eff_op = fig_validation(sigma_ref, snr_ref)
    feas, feas_pink, aniso_vals, snr_vals = fig_phase_diagram(calib, calib_pink)
    robustness = fig_robustness(sigma_ref, snr_ref)

    metrics = {
        "timing_noise_sigma_t_s": {str(k): round(v, 3) for k, v in calib.items()},
        "sigma_t_fit_a": round(a_fit, 3),
        "sigma_t_at_10dB_s": round(sigma_ref, 3),
        "timing_noise_sigma_t_half_pink_s": {str(k): round(v, 3) for k, v in calib_pink.items()},
        "sigma_t_fit_a_half_pink": round(a_fit_pink, 3),
        "median_empirical_over_crlb_ratio": round(eff_unit, 3),
        "median_empirical_over_crlb_ratio_at_10dB": round(eff_op, 3),
        "phase_diagram_feasible_pct": feas,
        "phase_diagram_feasible_pct_half_pink": feas_pink,
        "phase_diagram_anisotropy": aniso_vals,
        "phase_diagram_snr_db": snr_vals,
        "feasibility_threshold_cm": 2.0,
        "robustness_at_10dB": robustness,
    }
    with open(os.path.join(OUT, "id_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
