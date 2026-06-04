import numpy as np


def make_circular_probe(
    shape: tuple[int, int] = (64, 64),
    radius: float = 24.0,
    soft_edge_width: float = 4.0,
    phase_curvature: float = 0.0,
) -> np.ndarray:
    """Create a smooth circular probe in real space."""
    ny, nx = shape
    yy, xx = np.indices(shape)

    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    edge_start = radius - soft_edge_width / 2
    edge_end = radius + soft_edge_width / 2

    amp = np.ones(shape, dtype=np.float32)
    amp[rr >= edge_end] = 0.0

    edge = (rr > edge_start) & (rr < edge_end)
    t = (rr[edge] - edge_start) / max(soft_edge_width, 1e-12)
    amp[edge] = 0.5 * (1.0 + np.cos(np.pi * t))

    if phase_curvature == 0.0:
        phase = np.zeros(shape, dtype=np.float32)
    else:
        yy0 = (yy - cy) / ny
        xx0 = (xx - cx) / nx
        phase = phase_curvature * (xx0**2 + yy0**2)

    probe = amp * np.exp(1j * phase)
    probe = probe / max(np.max(np.abs(probe)), 1e-12)

    return probe.astype(np.complex64)