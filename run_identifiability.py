"""Identifiability experiment: the part that produces model-independent value.

Outputs:
  id_01_crlb_map.png   -- where on the substrate localisation is feasible vs ill-posed,
                          for two electrode geometries (a generalised GDOP map).
  id_02_validation.png -- empirical ML localisation RMSE vs CRLB (efficiency check).
  id_03_phase_diagram.png -- feasible-area fraction over (anisotropy x SNR): a design tool.
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
    feasible_fraction,
    calibrate_timing_noise,
    fit_sigma_t_vs_snr,
    validate_crlb,
)

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


def fig_validation():
    medium = MycelialMedium(seed=3)
    el = default_electrodes(medium.size_cm)
    pts = [(10, 10), (6, 6), (14, 5), (5, 15), (16, 16), (10, 3), (8, 12), (13, 13)]
    res = validate_crlb(medium, el, pts, sigma_t=1.0, n_trials=400)
    crlb = np.array([r[2] for r in res])
    emp = np.array([r[3] for r in res])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = max(crlb.max(), emp.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (efficient)")
    ax.scatter(crlb, emp, c="purple", s=45, zorder=5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("CRLB position RMSE (cm)")
    ax.set_ylabel("empirical ML RMSE (cm)")
    ax.set_title("CRLB is achievable: ML estimator reaches the bound")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "id_02_validation.png"), dpi=130)
    plt.close(fig)
    return float(np.median(emp / crlb))


def fig_phase_diagram(sigma_fn, threshold_cm=2.0):
    aniso_vals = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    snr_vals = [4, 6, 8, 10, 12, 16]
    el = default_electrodes(20.0)
    feas = np.zeros((len(snr_vals), len(aniso_vals)))
    for ci, a in enumerate(aniso_vals):
        medium = MycelialMedium(seed=3, aniso_ratio=a)
        gmap = geometric_crlb_map(medium, el)
        for ri, snr in enumerate(snr_vals):
            st = float(sigma_fn(snr))
            feas[ri, ci] = feasible_fraction(gmap, st, threshold_cm) * 100.0
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    im = ax.imshow(feas, origin="lower", cmap="magma", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(aniso_vals))); ax.set_xticklabels(aniso_vals)
    ax.set_yticks(range(len(snr_vals))); ax.set_yticklabels(snr_vals)
    ax.set_xlabel("anisotropy ratio"); ax.set_ylabel("SNR (dB)")
    ax.set_title(f"Feasible area: % of substrate with CRLB < {threshold_cm:.0f} cm")
    for ri in range(len(snr_vals)):
        for ci in range(len(aniso_vals)):
            ax.text(ci, ri, f"{feas[ri,ci]:.0f}", ha="center", va="center",
                    color="white" if feas[ri, ci] < 55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="% feasible area")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "id_03_phase_diagram.png"), dpi=130)
    plt.close(fig)
    return feas.tolist(), aniso_vals, snr_vals


def main():
    medium = MycelialMedium(seed=3)
    el = default_electrodes(medium.size_cm)

    calib = calibrate_timing_noise(medium, el, [4, 6, 8, 10, 12, 16], n_trials=30)
    a_fit, sigma_fn = fit_sigma_t_vs_snr(calib)
    snr_ref = 10.0
    sigma_ref = float(sigma_fn(snr_ref))

    fig_crlb_maps(sigma_ref, snr_ref)
    eff = fig_validation()
    feas, aniso_vals, snr_vals = fig_phase_diagram(sigma_fn)

    metrics = {
        "timing_noise_sigma_t_s": {str(k): round(v, 3) for k, v in calib.items()},
        "sigma_t_fit_a": round(a_fit, 3),
        "sigma_t_at_10dB_s": round(sigma_ref, 3),
        "median_empirical_over_crlb_ratio": round(eff, 3),
        "phase_diagram_feasible_pct": feas,
        "phase_diagram_anisotropy": aniso_vals,
        "phase_diagram_snr_db": snr_vals,
        "feasibility_threshold_cm": 2.0,
    }
    with open(os.path.join(OUT, "id_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
