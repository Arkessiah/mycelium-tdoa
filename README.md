# mycelium-tdoa

**An open testbed for acoustic-style source localisation on living mycelium.**

Most work on fungal bioelectronics stops at *characterisation*: it shows that
mycelium propagates electrical spikes, that propagation is directional, and that
the medium is dispersive. What nobody has closed is the **inverse problem** — given
the delays measured across an electrode array, *where* did the stimulus happen, and
with what uncertainty?

This repository is a fully reproducible, dependency-light simulation testbed for
exactly that. It lets you develop and stress-test the full localisation pipeline
**before** touching a single electrode, and it is designed to become the open
reference implementation that real recordings can be plugged into.

> ⚠️ This is a *simulator + method* testbed. The medium model is a deliberate,
> honest abstraction (a discrete eikonal solve on an anisotropic, heterogeneous
> slowness field), not a biophysical ground truth. Its job is to exercise the
> signal-processing and learning stack and to quantify how each non-ideality
> (anisotropy, heterogeneity, low SNR, drift) degrades localisation.

## Why a simulator first

Real mycelium violates every assumption of textbook TDOA:

1. **No constant velocity.** Propagation is slow *and* dispersive — `c` is a field,
   not a number.
2. **Anisotropy + heterogeneity.** Speed depends on direction and location, so the
   wavefront is warped, not circular.
3. **Non-stationarity.** The organism grows; any calibration goes stale over days.

The simulator reproduces all three so you can measure their impact and design
around them.

## The pipeline

```
stimulus ──▶ mycelial medium ──▶ electrode traces ──▶ GCC-PHAT ──▶ delay vector ──▶ GP localiser ──▶ (x, y) ± σ
            (anisotropic,         (noisy, slow         (regularised,   (relative to    (fingerprinting,
             heterogeneous)        spikes)              averaged)       a reference)    with uncertainty)
```

- **Medium** (`medium.py`): heterogeneous, anisotropic slowness field; travel times
  via Dijkstra on an 8-connected lattice (a discrete eikonal solve).
- **Signals** (`signals.py`): slow biphasic spikes, additive white and/or 1/f noise, and
  **regularised GCC-PHAT** with stimulus-repeat averaging for low-SNR robustness.
- **Localiser** (`localizer.py`): a **fingerprinting** approach — stimulate at known
  points, learn `delay-vector → (x, y)` with a Gaussian Process that returns
  calibrated per-prediction uncertainty and is cheap to recalibrate as the medium
  drifts.

Fingerprinting is the recommended MVP route: it sidesteps the (very hard) physical
inverse problem and turns localisation into a supervised-learning problem.

## Quickstart

```bash
pip install -r requirements.txt
python run_experiment.py        # writes figures/ and prints metrics.json
```

## What the experiment shows (reference run)

All localisation stages run at the delay noise the GCC-PHAT stage actually
measures, not at a hand-picked value. Every error is shown next to its scale: a
no-skill baseline (always predict the mean training position) and a noise-free
ceiling.

| Stage | Result |
|---|---|
| Warped wavefront | Travel-time contours are clearly non-circular (see `figures/01_medium.png`) |
| Delay recovery (GCC-PHAT, 6 reps, 8 dB, 20 stimulus positions) | RMSE ≈ 3.5 s on delays spanning up to ~30 s; errors are close to Gaussian |
| Same, half of the noise power 1/f (same SNR) | RMSE ≈ 7.2 s: coloured noise doubles the timing error |
| Localisation (fingerprinting) | **median error ≈ 1.4 cm** (p90 ≈ 2.6 cm) on a 20 cm substrate |
| — no-skill baseline / noise-free ceiling | ≈ 6.8 cm / ≈ 0.1 cm |
| — GP uncertainty | 97.5 % of true positions fall within 2σ |
| Drift (stale model → recalibrated) | ≈ 2.0 cm → ≈ 1.5 cm |
| Per-electrode timing offsets (fixed) | absorbed by calibration (≈ 1.4 cm); one reseated electrode (+1 sd) → ≈ 1.4 cm |
| One dead electrode (refit on the other five) | worst case ≈ 1.6 cm |

Findings worth carrying into a write-up:

- **Timing noise dominates.** With noise-free delays the localiser reaches ≈ 0.1 cm,
  so almost all of the ≈ 1.4 cm comes from delay estimation. Earlier versions of this
  README quoted ≈ 0.2 cm, which came from running localisation at 0.3 s of delay
  noise, ten times less than the delay stage measures.
- **At low SNR you must average over repeated stimuli**, and the noise spectrum
  matters as much as its power: 1/f noise, typical of extracellular recordings,
  doubles the delay error at the same SNR.
- **Drift is real but second-order at this SNR.** Recalibration still helps
  (≈ 2.0 → ≈ 1.5 cm), but the gain is far smaller than the gap between the measured
  and the noise-free error.
- **The array degrades gracefully.** Fixed per-electrode offsets are absorbed by
  fingerprinting, and losing any single electrode costs at most ≈ 0.3 cm, provided
  the localiser is refitted without it.

## The sim-to-real flywheel

The simulator generates labelled training data for free. Real electrode recordings,
when they arrive, are used to **calibrate the slowness field** so the simulator
matches the physical substrate — which makes the next batch of synthetic training
data more realistic, which improves the localiser. Open-sourcing both the simulator
and the trained models lets the community contribute real recordings that improve
the shared sim. That loop is the point.

## Roadmap

- [ ] Replace the synthetic medium with field-calibrated slowness from real arrays.
- [ ] Active learning: stimulate where the localiser is most uncertain.
- [ ] Online recalibration to track drift automatically.
- [ ] Closed-loop demo: localise → trigger an actuator.
- [ ] Provenance: pin datasets + model artefacts (DVC, optional permanent anchoring).

## License

MIT.
