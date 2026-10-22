"""mycelium-tdoa: an open testbed for TDOA source localisation on living mycelium."""
from .medium import MycelialMedium, default_electrodes
from .signals import (
    gcc_phat,
    spike_kernel,
    synthesize_traces,
    estimate_pairwise_delays,
)
from .localizer import build_fingerprint_dataset, FingerprintLocalizer
from .crlb import (
    geometric_crlb_map,
    crlb_at,
    feasible_fraction,
    calibrate_timing_noise,
    fit_sigma_t_vs_snr,
    validate_crlb,
    ml_localize,
)

__version__ = "0.2.0"
__all__ = [
    "MycelialMedium",
    "default_electrodes",
    "gcc_phat",
    "spike_kernel",
    "synthesize_traces",
    "estimate_pairwise_delays",
    "build_fingerprint_dataset",
    "FingerprintLocalizer",
    "geometric_crlb_map",
    "crlb_at",
    "feasible_fraction",
    "calibrate_timing_noise",
    "fit_sigma_t_vs_snr",
    "validate_crlb",
    "ml_localize",
]
