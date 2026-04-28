import numpy as np
from typing import Optional, Literal
import math


# ------------------------------------------------------------
# RNG helper
# ------------------------------------------------------------
def get_rng(seed: Optional[int]):
    return np.random.default_rng(seed)


# ------------------------------------------------------------
# Centered FFT
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


# ------------------------------------------------------------
# Random phase mask
# ------------------------------------------------------------
def random_phase_mask(shape, rng):
    theta = rng.uniform(0.0, 2.0 * math.pi, size=shape)
    return np.exp(1j * theta)


# ------------------------------------------------------------
# Forward model (noise-free)
# ------------------------------------------------------------
def forward_cdp(x, masks):
    """
    x: (H, W)
    masks: (M, H, W)

    returns: (M, H, W)
    """
    if x.ndim != 2:
        raise ValueError("x must be 2D (H, W)")

    z = masks * x[None, :, :]   # (M, H, W)
    Z = fft2c(z)

    return np.abs(Z)


# ------------------------------------------------------------
# Anscombe transform
# ------------------------------------------------------------
def anscombe_transform(y):
    return np.sqrt(y + 3.0 / 8.0)


# ------------------------------------------------------------
# Main API
# ------------------------------------------------------------
def make_cdp(
    x,
    num_masks: int,
    *,
    seed: Optional[int] = None,
    noise_model: Literal["gaussian", "poisson"] = "gaussian",
    noise_param: float = 1e-3,
):
    rng = get_rng(seed)
    if x.ndim != 2:
        raise ValueError("x must be (H, W)")

    H, W = x.shape

    # ------------------------------------------------------------
    # masks
    # ------------------------------------------------------------
    masks = random_phase_mask((num_masks, H, W), rng)

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    amplitude = forward_cdp(x, masks)

    # ------------------------------------------------------------
    # noise
    # ------------------------------------------------------------
    if noise_model == "gaussian":
        var = noise_param

        noise = rng.normal(size=amplitude.shape) * math.sqrt(var)
        y = amplitude + noise

        noise_var = var

    elif noise_model == "poisson":
        scale = noise_param

        intensity = amplitude ** 2
        scaled = scale * intensity

        y_poisson = rng.poisson(scaled)

        # Anscombe
        y = anscombe_transform(y_poisson)

        # normalize by sqrt(scale)
        y = y / math.sqrt(scale)

        # effective variance
        noise_var = 1.0 / (4.0 * scale)

    else:
        raise ValueError(f"Unknown noise_model: {noise_model}")

    return masks, y, noise_var