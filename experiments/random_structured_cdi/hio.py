import numpy as np
from typing import Optional, Callable

from .forward_model import apply_linear_operator, apply_adjoint_operator


def _get_rng(seed: Optional[int]):
    return np.random.default_rng(seed)


def _random_phase_like(amplitude, rng):
    """
    Generate a complex field with the given amplitude and random phase.
    """
    phase = rng.uniform(0.0, 2.0 * np.pi, size=amplitude.shape)
    return amplitude * np.exp(1j * phase)


def _replace_amplitude(z, amplitude, eps: float = 1e-12):
    """
    Replace the amplitude of z while preserving its phase.
    """
    abs_z = np.abs(z)
    return amplitude * z / np.maximum(abs_z, eps)


def hio(
    amplitude,
    phase_masks,
    support,
    *,
    beta: float = 0.9,
    n_iter: int = 1000,
    seed: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
    callback: Optional[Callable] = None,
):
    """
    Reconstruct an object from random structured CDI amplitude data using HIO.

    Parameters
    ----------
    amplitude:
        Measured diffraction amplitude, shape (H, W).
    phase_masks:
        Random phase masks defining the structured CDI operator,
        shape (n_layers, H, W).
    support:
        Boolean support mask in the object domain, shape (H, W).
    beta:
        HIO feedback parameter.
    n_iter:
        Number of HIO iterations.
    seed:
        Random seed used when x0 is not provided.
    x0:
        Optional initial object-domain estimate, shape (H, W).
    callback:
        Optional callback called as callback(x, t, info).

    Returns
    -------
    x:
        Reconstructed complex object, shape (H, W).
    """
    amplitude = np.asarray(amplitude)
    phase_masks = np.asarray(phase_masks)
    support = np.asarray(support, dtype=bool)

    if amplitude.ndim != 2:
        raise ValueError("amplitude must be 2D with shape (H, W).")

    if phase_masks.ndim != 3:
        raise ValueError("phase_masks must have shape (n_layers, H, W).")

    if phase_masks.shape[1:] != amplitude.shape:
        raise ValueError(
            f"phase_masks spatial shape mismatch: expected {amplitude.shape}, "
            f"got {phase_masks.shape[1:]}"
        )

    if support.shape != amplitude.shape:
        raise ValueError(
            f"support shape mismatch: expected {amplitude.shape}, got {support.shape}"
        )

    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must satisfy 0 < beta <= 1.")

    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    rng = _get_rng(seed)

    if x0 is None:
        z0 = _random_phase_like(amplitude, rng)
        x = apply_adjoint_operator(z0, phase_masks)
    else:
        x = np.asarray(x0, dtype=phase_masks.dtype)
        if x.shape != amplitude.shape:
            raise ValueError(
                f"x0 shape mismatch: expected {amplitude.shape}, got {x.shape}"
            )

    for t in range(n_iter):
        x_prev = x

        # Forward projection: enforce measured amplitude.
        z = apply_linear_operator(x, phase_masks)
        z_proj = _replace_amplitude(z, amplitude)

        # Backprojection to the object domain.
        x_back = apply_adjoint_operator(z_proj, phase_masks)

        # HIO real-space update.
        x_next = x_prev.copy()
        x_next[support] = x_back[support]
        x_next[~support] = x_prev[~support] - beta * x_back[~support]

        x = x_next

        if callback is not None:
            z_check = apply_linear_operator(x, phase_masks)

            relative_amplitude_error = np.mean(
                (np.abs(z_check) - amplitude)**2
            ) / np.prod(z_check.shape)

            support_violation = np.linalg.norm(x[~support]) / max(
                np.linalg.norm(x),
                1e-12,
            )

            info = {
                "relative_amplitude_error": float(relative_amplitude_error),
                "support_violation": float(support_violation),
            }

            callback(x, t, info)

    return np.where(support, x, 0)