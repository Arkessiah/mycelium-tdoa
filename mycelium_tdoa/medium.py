"""Anisotropic, heterogeneous mycelial medium simulator.

Travel times are computed on a regular grid by solving the discrete shortest-path
problem (Dijkstra on an 8-connected lattice with direction-dependent edge weights).
This is a discrete eikonal solve: it naturally produces warped, non-circular
wavefronts driven by both spatial heterogeneity and anisotropy of the medium,
which is exactly what real mycelium exhibits.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class MycelialMedium:
    """A synthetic 2D mycelial substrate with a heterogeneous, anisotropic slowness field.

    Parameters
    ----------
    size_cm : float
        Side length of the (square) substrate in centimetres.
    grid_n : int
        Grid resolution per side (grid_n x grid_n cells).
    base_slowness : float
        Baseline slowness in seconds per centimetre (1 / propagation speed).
        Mycelial signalling is slow; the absolute value only rescales time,
        TDOA depends on *relative* differences.
    hetero_strength : float
        Fractional strength of spatial heterogeneity (e.g. 0.5 = +/-50%).
    hetero_scale_cm : float
        Correlation length of the heterogeneity in cm (patch size).
    aniso_ratio : float
        Anisotropy ratio (>=1). 1.0 = isotropic. Perpendicular-to-grain
        slowness is aniso_ratio times the along-grain slowness.
    aniso_angle_deg : float
        Orientation of the fast axis ("grain") in degrees.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        size_cm: float = 20.0,
        grid_n: int = 80,
        base_slowness: float = 1.0,
        hetero_strength: float = 0.5,
        hetero_scale_cm: float = 4.0,
        aniso_ratio: float = 2.0,
        aniso_angle_deg: float = 30.0,
        seed: int = 0,
    ) -> None:
        self.size_cm = float(size_cm)
        self.grid_n = int(grid_n)
        self.base_slowness = float(base_slowness)
        self.aniso_ratio = float(aniso_ratio)
        self.aniso_angle = np.deg2rad(aniso_angle_deg)
        self.rng = np.random.default_rng(seed)

        self.dx = self.size_cm / (self.grid_n - 1)  # cm per cell
        # cell centre coordinates
        coords = np.linspace(0.0, self.size_cm, self.grid_n)
        self.xx, self.yy = np.meshgrid(coords, coords)  # shape (n, n)

        self.slowness = self._make_slowness_field(hetero_strength, hetero_scale_cm)
        self._graph = self._build_graph()
        self._tt_cache: dict[tuple[float, float], np.ndarray] = {}

    # ------------------------------------------------------------------ fields
    def _make_slowness_field(self, strength: float, scale_cm: float) -> np.ndarray:
        """Smooth random heterogeneity around the baseline slowness."""
        noise = self.rng.standard_normal((self.grid_n, self.grid_n))
        sigma_cells = max(scale_cm / self.dx, 0.5)
        smooth = gaussian_filter(noise, sigma=sigma_cells, mode="reflect")
        smooth /= (smooth.std() + 1e-12)
        field = self.base_slowness * (1.0 + strength * smooth)
        return np.clip(field, 1e-3, None)

    # ------------------------------------------------------------------ graph
    def _aniso_multiplier(self, dy: float, dx: float) -> float:
        """Direction-dependent slowness multiplier for an edge of orientation (dy, dx).

        Fast along the grain (~1/mean), slow perpendicular (~aniso_ratio/mean).
        Normalised so the *directional mean* is 1: this isolates the effect of
        anisotropy (shape) from the overall slowness level (scale), so a sweep over
        aniso_ratio does not secretly also change the mean propagation speed.
        """
        phi = np.arctan2(dy, dx)
        raw = 1.0 + (self.aniso_ratio - 1.0) * np.sin(phi - self.aniso_angle) ** 2
        mean = 0.5 * (1.0 + self.aniso_ratio)  # directional mean of raw
        return raw / mean

    def _build_graph(self) -> csr_matrix:
        n = self.grid_n
        slow = self.slowness
        rows, cols, data = [], [], []
        # 8-connectivity offsets (only "forward" half; matrix symmetrised after)
        offsets = [(-1, 0), (0, -1), (-1, -1), (-1, 1)]
        for di, dj in offsets:
            dist = np.hypot(di, dj) * self.dx
            mult = self._aniso_multiplier(di, dj)
            i0 = max(0, -di)
            i1 = min(n, n - di)
            j0 = max(0, -dj)
            j1 = min(n, n - dj)
            for i in range(i0, i1):
                for j in range(j0, j1):
                    a = i * n + j
                    b = (i + di) * n + (j + dj)
                    w = dist * mult * 0.5 * (slow[i, j] + slow[i + di, j + dj])
                    rows.append(a)
                    cols.append(b)
                    data.append(w)
        m = csr_matrix((data, (rows, cols)), shape=(n * n, n * n))
        return m + m.T  # symmetric => undirected

    # ------------------------------------------------------------------ utils
    def node_of(self, x_cm: float, y_cm: float) -> int:
        j = int(round(np.clip(x_cm, 0, self.size_cm) / self.dx))
        i = int(round(np.clip(y_cm, 0, self.size_cm) / self.dx))
        return i * self.grid_n + j

    def xy_of_node(self, node: int) -> tuple[float, float]:
        i, j = divmod(node, self.grid_n)
        return j * self.dx, i * self.dx

    def travel_time_field(self, x_cm: float, y_cm: float) -> np.ndarray:
        """Travel time from a source point to every grid cell (shape grid_n x grid_n)."""
        key = (round(x_cm, 6), round(y_cm, 6))
        if key not in self._tt_cache:
            src = self.node_of(x_cm, y_cm)
            d = dijkstra(self._graph, indices=src, directed=False)
            self._tt_cache[key] = d.reshape(self.grid_n, self.grid_n)
        return self._tt_cache[key]

    def arrival_times(self, src_xy: tuple[float, float], electrodes: np.ndarray) -> np.ndarray:
        """Arrival times of a stimulus at `src_xy` to each electrode (seconds)."""
        field = self.travel_time_field(*src_xy)
        out = np.empty(len(electrodes))
        for k, (ex, ey) in enumerate(electrodes):
            i, j = divmod(self.node_of(ex, ey), self.grid_n)
            out[k] = field[i, j]
        return out

    def drifted(self, drift_strength: float = 0.12, angle_shift_deg: float = 8.0,
                seed: int = 123) -> "MycelialMedium":
        """Return a copy representing the *same* substrate after biological drift.

        The living medium grows and remodels: the slowness field is perturbed by a
        small smooth field and the anisotropy axis rotates slightly. This is the
        non-stationarity that makes a fixed calibration go stale over days.
        """
        rng = np.random.default_rng(seed)
        new = MycelialMedium.__new__(MycelialMedium)
        new.size_cm = self.size_cm
        new.grid_n = self.grid_n
        new.base_slowness = self.base_slowness
        new.aniso_ratio = self.aniso_ratio
        new.aniso_angle = self.aniso_angle + np.deg2rad(angle_shift_deg)
        new.rng = rng
        new.dx = self.dx
        new.xx, new.yy = self.xx, self.yy
        noise = rng.standard_normal((self.grid_n, self.grid_n))
        sigma_cells = max(4.0 / self.dx, 0.5)
        smooth = gaussian_filter(noise, sigma=sigma_cells, mode="reflect")
        smooth /= (smooth.std() + 1e-12)
        new.slowness = np.clip(self.slowness * (1.0 + drift_strength * smooth), 1e-3, None)
        new._graph = new._build_graph()
        new._tt_cache = {}
        return new


def default_electrodes(size_cm: float = 20.0, margin: float = 2.0) -> np.ndarray:
    """A reasonable 6-electrode layout (corners + two mid-edges)."""
    s, m = size_cm, margin
    return np.array(
        [
            [m, m],
            [s - m, m],
            [m, s - m],
            [s - m, s - m],
            [s / 2, m],
            [s / 2, s - m],
        ]
    )
