import math
from typing import Optional, Literal

import numpy as np


# ------------------------------------------------------------
# RNG helper
# ------------------------------------------------------------
def get_rng(seed: Optional[int]):
    return np.random.default_rng(seed)


# ------------------------------------------------------------
# Centered FFT / IFFT
# ------------------------------------------------------------
def fft2c(x):
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(x, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def ifft2c(x):
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(x, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


# ------------------------------------------------------------
# Random phase masks
# ------------------------------------------------------------
def random_phase_masks(n_layers: int, shape, rng, *, dtype=np.complex64):
    """
    Generate random phase masks.

    Parameters
    ----------
    n_layers:
        Number of random phase modulation layers.
    shape:
        Spatial shape (H, W).
    rng:
        NumPy random generator.
    dtype:
        Complex dtype of the returned masks.

    Returns
    -------
    phase_masks:
        Array with shape (n_layers, H, W).
    """
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1.")

    theta = rng.uniform(0.0, 2.0 * math.pi, size=(n_layers,) + tuple(shape))
    return np.exp(1j * theta).astype(dtype)


# ------------------------------------------------------------
# Support utilities
# ------------------------------------------------------------
def apply_support(x, support):
    """
    Apply a support mask to the object.

    Parameters
    ----------
    x:
        Complex object, shape (H, W).
    support:
        Boolean support mask, shape (H, W).

    Returns
    -------
    supported_x:
        Object with values outside the support set to zero.
    """
    if support is None:
        return x

    if support.shape != x.shape:
        raise ValueError(f"support shape mismatch: expected {x.shape}, got {support.shape}")

    return np.where(support, x, 0)


# ------------------------------------------------------------
# Forward and adjoint linear operators
# ------------------------------------------------------------
def apply_linear_operator(x, phase_masks):
    """
    Apply the structured random CDI linear operator.

    The operator is

        A x = F D_L ... F D_2 F D_1 x,

    where each D_i is a random phase mask and F is a centered unitary FFT.

    Parameters
    ----------
    x:
        Complex object, shape (H, W).
    phase_masks:
        Random phase masks, shape (n_layers, H, W).

    Returns
    -------
    z:
        Complex wavefield after the final FFT, shape (H, W).
    """
    if x.ndim != 2:
        raise ValueError("x must be 2D with shape (H, W).")

    if phase_masks.ndim != 3:
        raise ValueError("phase_masks must have shape (n_layers, H, W).")

    if phase_masks.shape[1:] != x.shape:
        raise ValueError(
            f"phase_masks spatial shape mismatch: expected {x.shape}, got {phase_masks.shape[1:]}"
        )

    z = x
    for mask in phase_masks:
        z = fft2c(mask * z)

    return z


def apply_adjoint_operator(z, phase_masks):
    """
    Apply the adjoint of the structured random CDI linear operator.

    If

        A x = F D_L ... F D_1 x,

    then

        A^H z = D_1^H F^H ... D_L^H F^H z.

    Parameters
    ----------
    z:
        Complex wavefield, shape (H, W).
    phase_masks:
        Random phase masks, shape (n_layers, H, W).

    Returns
    -------
    x:
        Back-propagated complex field, shape (H, W).
    """
    if z.ndim != 2:
        raise ValueError("z must be 2D with shape (H, W).")

    if phase_masks.ndim != 3:
        raise ValueError("phase_masks must have shape (n_layers, H, W).")

    if phase_masks.shape[1:] != z.shape:
        raise ValueError(
            f"phase_masks spatial shape mismatch: expected {z.shape}, got {phase_masks.shape[1:]}"
        )

    x = z
    for mask in phase_masks[::-1]:
        x = np.conj(mask) * ifft2c(x)

    return x


# ------------------------------------------------------------
# Forward model: amplitude-only measurement
# ------------------------------------------------------------
def forward_random_structured_cdi(x, phase_masks, support=None):
    """
    Compute noiseless amplitude measurements for random structured CDI.

    Parameters
    ----------
    x:
        Complex object, shape (H, W).
    phase_masks:
        Random phase masks, shape (n_layers, H, W).
    support:
        Optional boolean support mask, shape (H, W).

    Returns
    -------
    amplitude:
        Noiseless amplitude measurement, shape (H, W).
    """
    if x.ndim != 2:
        raise ValueError("x must be 2D with shape (H, W).")

    x_supported = apply_support(x, support)
    z = apply_linear_operator(x_supported, phase_masks)
    return np.abs(z)


# ------------------------------------------------------------
# Anscombe transform
# ------------------------------------------------------------
def anscombe_transform(y):
    return np.sqrt(y + 3.0 / 8.0)


# ------------------------------------------------------------
# Main API
# ------------------------------------------------------------
def make_random_structured_cdi(
    x,
    n_layers: int,
    *,
    support=None,
    seed: Optional[int] = None,
    noise_model: Literal["gaussian", "poisson"] = "gaussian",
    noise_param: float = 1e-3,
    dtype=np.complex64,
):
    """
    Generate a random structured CDI instance.

    Parameters
    ----------
    x:
        Ground-truth complex object, shape (H, W).
    n_layers:
        Number of random phase modulation layers.
    support:
        Optional boolean support mask, shape (H, W).
        If provided, the object is masked before propagation.
    seed:
        Random seed.
    noise_model:
        Either "gaussian" or "poisson".
    noise_param:
        For Gaussian noise, this is the amplitude noise variance.
        For Poisson noise, this is the photon scaling factor.
    dtype:
        Complex dtype used for phase masks and the object.

    Returns
    -------
    If noise_model == "gaussian":
        phase_masks, y, noise_var

    If noise_model == "poisson":
        phase_masks, y, noise_var, photon_counts
    """
    rng = get_rng(seed)

    if x.ndim != 2:
        raise ValueError("x must be 2D with shape (H, W).")

    if support is not None:
        support = np.asarray(support, dtype=bool)
        if support.shape != x.shape:
            raise ValueError(f"support shape mismatch: expected {x.shape}, got {support.shape}")

    x = np.asarray(x, dtype=dtype)

    # ------------------------------------------------------------
    # Phase masks
    # ------------------------------------------------------------
    phase_masks = random_phase_masks(
        n_layers=n_layers,
        shape=x.shape,
        rng=rng,
        dtype=dtype,
    )

    # ------------------------------------------------------------
    # Noiseless forward model
    # ------------------------------------------------------------
    amplitude = forward_random_structured_cdi(
        x,
        phase_masks,
        support=support,
    )

    # ------------------------------------------------------------
    # Noise model
    # ------------------------------------------------------------
    if noise_model == "gaussian":
        var = float(noise_param)

        noise = rng.normal(size=amplitude.shape) * math.sqrt(var)
        y = amplitude + noise
        noise_var = var

        return phase_masks, y, noise_var

    if noise_model == "poisson":
        scale = float(noise_param)

        intensity = amplitude**2
        scaled = scale * intensity

        y_poisson = rng.poisson(scaled)
        photon_counts = int(np.max(y_poisson))

        # Variance-stabilizing transform
        y = anscombe_transform(y_poisson)

        # Normalize back to amplitude scale
        y = y / math.sqrt(scale)

        # Effective Gaussian variance after Anscombe transform
        noise_var = 1.0 / (4.0 * scale)

        return phase_masks, y, noise_var, photon_counts

    raise ValueError(f"Unknown noise_model: {noise_model}")