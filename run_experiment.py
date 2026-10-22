"""End-to-end experiment for the mycelium-tdoa testbed.

Runs four stages and writes figures + a metrics summary:
  1. Medium + warped wavefront visualisation.
  2. GCC-PHAT delay recovery on synthetic noisy traces; its measured RMSE over
     many stimulus positions sets the timing noise used by stages 3 and 4.
  3. Fingerprint localisation accuracy (GP, with uncertainty).
  4. Non-stationarity: drift degrades a fixed model; recalibration restores it.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mycelium_tdoa import (
    MycelialMedium,
    default_electrodes,
    synthesize_traces,
    build_fingerprint_dataset,
    FingerprintLocalizer,
)
from mycelium_tdoa.signals import estimate_delays_averaged

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
FS = 1.0  # samples per second (abstract; only relative delays matter)


def stage1_medium(medium, electrodes):
    src = (6.0, 14.0)
    tt = medium.travel_time_field(*src)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    im0 = ax[0].imshow(medium.slowness, origin="lower",
                       extent=[0, medium.size_cm] * 2, cmap="magma")
    ax[0].set_title("Heterogeneous slowness field")
    ax[0].set_xlabel("x (cm)"); ax[0].set_ylabel("y (cm)")
    fig.colorbar(im0, ax=ax[0], label="slowness (s/cm)")
    ax[0].scatter(electrodes[:, 0], electrodes[:, 1], c="cyan", edgecolor="k", s=60, label="electrodes")
    ax[0].legend(loc="upper right", fontsize=8)

    im1 = ax[1].imshow(tt, origin="lower", extent=[0, medium.size_cm] * 2, cmap="viridis")
    cs = ax[1].contour(medium.xx, medium.yy, tt, levels=12, colors="w", linewidths=0.6)
    ax[1].clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    ax[1].scatter(*src, c="red", s=80, marker="*", label="stimulus")
    ax[1].scatter(electrodes[:, 0], electrodes[:, 1], c="cyan", edgecolor="k", s=60)
    ax[1].set_title("Travel time — note the warped, non-circular front")
    ax[1].set_xlabel("x (cm)"); ax[1].set_ylabel("y (cm)")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(im1, ax=ax[1], label="arrival time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_medium.png"), dpi=130)
    plt.close(fig)


def stage2_gccphat(medium, electrodes, rng):
    src = (15.0, 5.0)
    true = medium.arrival_times(src, electrodes)
    true_rel = true - true[0]
    max_tau = 30.0
    traces = synthesize_traces(true, fs=FS, duration_s=400, snr_db=8.0, rng=rng)
    est_rel = estimate_delays_averaged(true, n_repeats=6, fs=FS, duration_s=400,
                                       snr_db=8.0, ref=0, max_tau=max_tau, rng=rng)
    err = est_rel - true_rel

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    off = 3.0
    for k in range(traces.shape[0]):
        ax[0].plot(traces[k] + k * off, lw=0.7)
    ax[0].set_title("Synthetic electrode traces (SNR 8 dB)")
    ax[0].set_xlabel("sample"); ax[0].set_yticks([])
    idx = np.arange(len(true_rel))
    ax[1].bar(idx - 0.2, true_rel, 0.4, label="true Δt")
    ax[1].bar(idx + 0.2, est_rel, 0.4, label="GCC-PHAT (6 reps)")
    ax[1].set_title("Pairwise delay recovery")
    ax[1].set_xlabel("electrode (rel. to #0)"); ax[1].set_ylabel("delay (s)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_gccphat.png"), dpi=130)
    plt.close(fig)
    return float(np.sqrt(np.mean(err**2)))


def measure_delay_noise(medium, electrodes, rng, n_sources=20, snr_db=8.0, n_repeats=6):
    """RMSE of GCC-PHAT relative delays over random stimulus positions.

    This is the timing noise the localisation stages are run at. A single source
    is not representative (the delay spread depends on where the stimulus is), and
    a hand-picked, smaller value makes the end-to-end accuracy optimistic.
    """
    margin = 1.0
    errs = []
    for _ in range(n_sources):
        src = tuple(rng.uniform(margin, medium.size_cm - margin, size=2))
        true = medium.arrival_times(src, electrodes)
        est = estimate_delays_averaged(true, n_repeats=n_repeats, fs=FS, duration_s=400,
                                       snr_db=snr_db, ref=0, max_tau=30.0, rng=rng)
        errs.append((est - (true - true[0]))[1:])
    return float(np.sqrt(np.mean(np.square(errs))))


def stage3_localise(medium, electrodes, rng, noise):
    Xtr, Ytr = build_fingerprint_dataset(medium, electrodes, n_samples=400,
                                         delay_noise_s=noise, rng=rng)
    Xte, Yte = build_fingerprint_dataset(medium, electrodes, n_samples=120,
                                         delay_noise_s=noise, rng=np.random.default_rng(99))
    loc = FingerprintLocalizer(length_scale=2.0).fit(Xtr, Ytr)
    pred, sigma = loc.predict(Xte)
    err = FingerprintLocalizer.error_cm(pred, Yte)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(electrodes[:, 0], electrodes[:, 1], c="k", marker="s", s=60, label="electrodes")
    for p, t in zip(pred, Yte):
        ax[0].plot([t[0], p[0]], [t[1], p[1]], "r-", lw=0.4, alpha=0.6)
    ax[0].scatter(Yte[:, 0], Yte[:, 1], c="green", s=18, label="true")
    ax[0].scatter(pred[:, 0], pred[:, 1], c="orange", s=18, label="predicted")
    ax[0].set_title("Localisation: true vs predicted")
    ax[0].set_xlabel("x (cm)"); ax[0].set_ylabel("y (cm)")
    ax[0].legend(fontsize=8); ax[0].set_aspect("equal")

    ax[1].scatter(sigma, err, s=18, c="purple", alpha=0.7)
    ax[1].set_title("Per-prediction GP uncertainty vs error")
    ax[1].set_xlabel("GP positional σ (cm)"); ax[1].set_ylabel("actual error (cm)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_localisation.png"), dpi=130)
    plt.close(fig)

    # Reference points that give the error a scale. No-skill: always predict the
    # mean training position. Ceiling: the same localiser with noise-free delays,
    # i.e. what is left once timing noise is removed (map learnability + grid).
    err_noskill = FingerprintLocalizer.error_cm(np.tile(Ytr.mean(axis=0), (len(Yte), 1)), Yte)
    Xtr0, Ytr0 = build_fingerprint_dataset(medium, electrodes, n_samples=400,
                                           delay_noise_s=0.0, rng=np.random.default_rng(1))
    Xte0, Yte0 = build_fingerprint_dataset(medium, electrodes, n_samples=120,
                                           delay_noise_s=0.0, rng=np.random.default_rng(99))
    loc0 = FingerprintLocalizer(length_scale=2.0).fit(Xtr0, Ytr0)
    err_ceiling = FingerprintLocalizer.error_cm(loc0.predict(Xte0)[0], Yte0)
    # fraction of test stimuli whose true position falls within 2 sigma of the GP
    coverage_2sigma = float(np.mean(err <= 2.0 * sigma))

    return {
        "localisation_median_error_cm": float(np.median(err)),
        "localisation_mean_error_cm": float(np.mean(err)),
        "localisation_p90_error_cm": float(np.percentile(err, 90)),
        "localisation_noskill_median_error_cm": float(np.median(err_noskill)),
        "localisation_noisefree_median_error_cm": float(np.median(err_ceiling)),
        "localisation_coverage_2sigma": coverage_2sigma,
    }


def stage4_drift(medium, electrodes, rng, noise):
    Xtr, Ytr = build_fingerprint_dataset(medium, electrodes, n_samples=400,
                                         delay_noise_s=noise, rng=rng)
    loc = FingerprintLocalizer(length_scale=2.0).fit(Xtr, Ytr)

    # a "later day": the SAME substrate after biological drift
    drifted = medium.drifted(drift_strength=0.12, angle_shift_deg=8.0, seed=7)
    Xd, Yd = build_fingerprint_dataset(drifted, electrodes, n_samples=120,
                                       delay_noise_s=noise, rng=np.random.default_rng(5))
    err_stale = FingerprintLocalizer.error_cm(loc.predict(Xd)[0], Yd)

    Xr, Yr = build_fingerprint_dataset(drifted, electrodes, n_samples=400,
                                       delay_noise_s=noise, rng=np.random.default_rng(6))
    loc_recal = FingerprintLocalizer(length_scale=2.0).fit(Xr, Yr)
    err_recal = FingerprintLocalizer.error_cm(loc_recal.predict(Xd)[0], Yd)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.boxplot([err_stale, err_recal], tick_labels=["stale model\n(after drift)", "recalibrated"])
    ax.set_ylabel("localisation error (cm)")
    ax.set_title("Non-stationarity: drift degrades, recalibration restores")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_drift.png"), dpi=130)
    plt.close(fig)
    return float(np.median(err_stale)), float(np.median(err_recal))


def main():
    rng = np.random.default_rng(42)
    medium = MycelialMedium(seed=3)
    electrodes = default_electrodes(medium.size_cm)

    stage1_medium(medium, electrodes)
    delay_rmse_example = stage2_gccphat(medium, electrodes, rng)
    delay_rmse = measure_delay_noise(medium, electrodes, rng)
    loc_metrics = stage3_localise(medium, electrodes, rng, delay_rmse)
    drift_stale, drift_recal = stage4_drift(medium, electrodes, rng, delay_rmse)

    metrics = {
        "delay_recovery_rmse_example_s": round(delay_rmse_example, 4),
        "delay_recovery_rmse_s": round(delay_rmse, 4),
        **{k: round(v, 3) for k, v in loc_metrics.items()},
        "substrate_side_cm": medium.size_cm,
        "drift_median_error_stale_cm": round(drift_stale, 3),
        "drift_median_error_recalibrated_cm": round(drift_recal, 3),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
