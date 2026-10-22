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

## Two layers: an engineering demo and a scientific result

This repo deliberately separates what is a *pipeline demo* from what is a
*model-independent result*, because they carry very different weight.

- `run_experiment.py` — the **engineering demo**. End-to-end pipeline (medium →
  GCC-PHAT → fingerprint localiser), useful to see the parts work. Its headline
  accuracy is **not** a scientific claim: it measures how learnable the simulator's
  own static map is.
- `run_identifiability.py` — the **scientific contribution**. A Cramer-Rao /
  Fisher analysis that asks *when source localisation on this medium is even
  possible*, validated against an efficient estimator. Its outputs (the CRLB map
  and the feasibility phase diagram) are design tools that hold for **any** unbiased
  estimator, and are framed explicitly as conditional on the propagation model — so
  they survive as a sensitivity analysis even if the model is imperfect.

## Quickstart

```bash
pip install -r requirements.txt
python run_experiment.py         # engineering demo  -> figures/0*.png
python run_identifiability.py    # CRLB + phase diagram -> figures/id_*.png
```

## What the experiment shows (reference run)

All localisation stages run at the delay noise the GCC-PHAT stage actually
measures, not at a hand-picked value. Every error is shown next to its scale: a
no-skill baseline (always predict the mean training position) and a noise-free
ceiling.

| Stage | Result |
|---|---|
| Warped wavefront | Travel-time contours are clearly non-circular (see `figures/01_medium.png`) |
| Delay recovery (GCC-PHAT, 6 reps, 8 dB, 20 stimulus positions) | RMSE ≈ 3.3 s on delays spanning up to ~30 s; errors are close to Gaussian |
| Same, half of the noise power 1/f (same SNR) | RMSE ≈ 6.6 s: at 8 dB coloured noise doubles the timing error |
| Localisation (fingerprinting) | **median error ≈ 1.8 cm** (p90 ≈ 3.6 cm) on a 20 cm substrate |
| — no-skill baseline / noise-free ceiling | ≈ 6.8 cm / ≈ 0.1 cm |
| — GP uncertainty | 99 % of true positions fall within 2σ |
| Drift (stale model → recalibrated) | ≈ 2.5 cm → ≈ 2.1 cm |
| Per-electrode timing offsets (fixed) | absorbed by calibration (≈ 1.8 cm); one reseated electrode (+1 sd) → ≈ 1.9 cm |
| One dead electrode (refit on the other five) | worst case ≈ 2.1 cm |

Findings worth carrying into a write-up:

- **Timing noise dominates.** With noise-free delays the localiser reaches ≈ 0.1 cm,
  so almost all of the ≈ 1.8 cm comes from delay estimation. Versions up to 0.2.0
  quoted ≈ 0.2–0.26 cm, which came from running localisation at 0.3 s of delay
  noise, about ten times less than the delay stage measures.
- **At low SNR you must average over repeated stimuli**, and the noise spectrum
  matters as much as its power: 1/f noise, typical of extracellular recordings,
  doubles the delay error at 8 dB and has a threshold below ~10 dB (see the
  identifiability section).
- **Drift is real but second-order at this SNR.** Recalibration still helps
  (≈ 2.5 → ≈ 2.1 cm), but the gain is far smaller than the gap between the measured
  and the noise-free error.
- **Fixed per-electrode offsets are absorbed by fingerprinting**, and the learned
  localiser degrades gracefully when it can be refitted after losing an electrode.
  The bound tells a harsher story about that loss (below).

## Identifiability: the model-independent result

`run_identifiability.py` answers the question that actually matters: *under what
conditions is the inverse problem solvable at all?* It computes the Cramer-Rao lower
bound on position error using the gradients of the real (warped) travel-time fields,
so it is medium-aware rather than an idealised isotropic calculation.

| Output | What it tells you |
|---|---|
| CRLB map (`id_01`) | Where on the substrate localisation is feasible vs ill-posed. Inside the electrode hull, ~1–2 cm; outside it, the bound explodes. A medium-aware generalised GDOP. |
| Validation (`id_02`) | An efficient ML estimator reaches the bound, at unit noise (empirical/CRLB ≈ 1.05) and at the operating noise of 10 dB (≈ 1.08), so the bound is real and achievable, not just theoretical. |
| Phase diagram (`id_03`) | Feasible substrate area over (anisotropy × SNR), for white noise and for noise with half its power 1/f. **SNR dominates**: below ~8 dB localisation is infeasible over most of the substrate; anisotropy steadily shrinks the feasible region. With 1/f noise the timing error has a threshold and **nothing below 10 dB is feasible**. |
| Robustness (`id_04`) | (a) If any one of the six electrodes may fail, the area with CRLB < 2 cm at 10 dB falls from ≈ 30 % to ≈ 1 %: the layout has no redundancy. (b) Spatially correlated timing noise does not hurt and long-range correlation helps (≈ 30 → ≈ 49 %), because the common part is indistinguishable from the unknown emission time. (c) Over the whole substrate the GP (RMSE ≈ 2.2 cm) is within ≈ 12 % of the Bayesian optimum (MMSE ≈ 1.9 cm). |

Timing precision is not assumed — it is **measured** by running the real delay
pipeline across SNRs (`calibrate_timing_noise`, a new random stimulus position per
trial) and fitting the `sigma_t ∝ 1/√SNR` law, then propagated through the Fisher
analysis. The phase diagrams use the value measured at each SNR, because the law
does not hold for 1/f noise. This ties the signal stage to the localisation bound.

**The CRLB bounds unbiased estimators only.** Near the edges of a bounded substrate
the pointwise bound grows large (its RMS over the substrate is ≈ 4.9 cm at 10 dB),
and learned or grid-limited estimators sit below it because they are pulled
towards the interior. `bayes_comparison` computes the minimum-mean-square-error
estimator under a uniform prior, which is the correct floor for any estimator.

**Design takeaway** (for this 6-electrode layout, 20 cm substrate, 2 cm tolerance):
you need roughly SNR ≥ 12 dB to localise over a majority of the area, stronger
anisotropy raises that requirement, 1/f noise makes 10 dB a hard floor, and
tolerating one dead electrode needs a denser or redundant array. That is a hardware
spec produced with no hardware.

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
