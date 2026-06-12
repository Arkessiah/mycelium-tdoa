"""Time-delay estimation (GCC-PHAT) and synthetic spike-train generation.

GCC-PHAT (generalised cross-correlation with phase transform) is the robust,
standard way to estimate the delay between two noisy copies of a signal. We use
it to recover the pairwise arrival-time differences between electrodes from raw
extracellular voltage traces, instead of fragile threshold/onset picking.
"""
from __future__ import annotations

import numpy as np


def gcc_phat(sig: np.ndarray, ref: np.ndarray, fs: float = 1.0, interp: int = 16,
             beta: float = 0.5, max_tau: float | None = None):
    """Estimate the delay (seconds) of `sig` relative to `ref` via regularised GCC-PHAT.

    Positive tau => `sig` is delayed with respect to `ref`.

    `beta` in [0, 1] sets the phase-transform strength. beta=1 is classic PHAT
    (fragile at low SNR because it whitens noise-dominated bins); beta<1 keeps some
    magnitude weighting (PHAT-beta), which is markedly more robust on noisy traces.
    `max_tau` (seconds) restricts the search to physically plausible delays, which
    removes gross outliers when the SNR is low.
    """
    n = sig.shape[0] + ref.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)
    R = SIG * np.conj(REF)
    denom = np.abs(R) ** beta + 1e-3 * np.max(np.abs(R))
    R /= denom
    cc = np.fft.irfft(R, n=interp * n)
    max_shift = interp * n // 2
    if max_tau is not None:
        max_shift = min(max_shift, int(interp * fs * max_tau))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / float(interp * fs), cc


def spike_kernel(width: int = 120) -> np.ndarray:
    """A slow, smooth biphasic 'spike' resembling fungal extracellular events."""
    t = np.linspace(-3, 3, width)
    k = (1 - t**2) * np.exp(-(t**2) / 2.0)  # Ricker / Mexican-hat
    return k / np.max(np.abs(k))


def synthesize_traces(
    delays_s: np.ndarray,
    fs: float = 1.0,
    duration_s: float = 400.0,
    snr_db: float = 8.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build one noisy trace per electrode, each a delayed copy of a shared spike.

    `delays_s` are shifted so the earliest arrival is 0. Returns array
    (n_electrodes, n_samples).
    """
    rng = rng or np.random.default_rng(0)
    delays_s = np.asarray(delays_s) - np.min(delays_s)
    n_samples = int(duration_s * fs)
    kernel = spike_kernel()
    onset = n_samples // 3
    traces = np.zeros((len(delays_s), n_samples))
    sig_amp = 1.0
    noise_amp = sig_amp / (10 ** (snr_db / 20.0))
    for k, d in enumerate(delays_s):
        start = onset + int(round(d * fs))
        end = min(start + len(kernel), n_samples)
        traces[k, start:end] += sig_amp * kernel[: end - start]
        traces[k] += noise_amp * rng.standard_normal(n_samples)
    return traces


def estimate_pairwise_delays(traces: np.ndarray, fs: float = 1.0, ref: int = 0,
                             max_tau: float | None = None) -> np.ndarray:
    """Estimate each electrode's delay relative to electrode `ref` (seconds)."""
    out = np.zeros(traces.shape[0])
    for k in range(traces.shape[0]):
        if k == ref:
            continue
        tau, _ = gcc_phat(traces[k], traces[ref], fs=fs, max_tau=max_tau)
        out[k] = tau
    return out


def estimate_delays_averaged(
    delays_s: np.ndarray,
    n_repeats: int = 6,
    fs: float = 1.0,
    duration_s: float = 400.0,
    snr_db: float = 8.0,
    ref: int = 0,
    max_tau: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Repeat the stimulus `n_repeats` times and average the cross-spectra.

    Averaging independent repetitions is the standard, physically realistic way to
    beat low SNR: the coherent spike adds up while incoherent noise averages out.
    """
    rng = rng or np.random.default_rng(0)
    n_e = len(delays_s)
    n = 2 * int(duration_s * fs)
    interp = 16
    acc = np.zeros((n_e, n // 2 + 1), dtype=complex)
    for _ in range(n_repeats):
        traces = synthesize_traces(delays_s, fs=fs, duration_s=duration_s,
                                   snr_db=snr_db, rng=rng)
        REF = np.fft.rfft(traces[ref], n=n)
        for k in range(n_e):
            SIG = np.fft.rfft(traces[k], n=n)
            acc[k] += SIG * np.conj(REF)
    out = np.zeros(n_e)
    max_shift = interp * n // 2
    if max_tau is not None:
        max_shift = min(max_shift, int(interp * fs * max_tau))
    for k in range(n_e):
        if k == ref:
            continue
        R = acc[k]
        R = R / (np.abs(R) ** 0.5 + 1e-3 * np.max(np.abs(R)))
        cc = np.fft.irfft(R, n=interp * n)
        cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
        out[k] = (np.argmax(np.abs(cc)) - max_shift) / float(interp * fs)
    return out
