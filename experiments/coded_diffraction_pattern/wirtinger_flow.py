import numpy as np


# ------------------------------------------------------------
# FFT (centered, unitary)
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
# CDP linear operator
# ------------------------------------------------------------
def cdp_forward_linear(x, masks):
    """
    Apply A x for CDP.

    Args:
        x: (H, W)
        masks: (M, H, W)

    Returns:
        (M, H, W)
    """
    return fft2c(masks * x[None, :, :])


def cdp_adjoint_linear(z, masks):
    """
    Apply A^H z for CDP.

    Args:
        z: (M, H, W)
        masks: (M, H, W)

    Returns:
        (H, W)
    """
    return np.sum(np.conj(masks) * ifft2c(z), axis=0)


# ------------------------------------------------------------
# Step size schedule
# ------------------------------------------------------------
def wf_stepsize(t, mu_max=0.4, tau=300.0):
    """
    Wirtinger Flow step size schedule.
    """
    return min(1.0 - np.exp(-t / tau), mu_max)


# ------------------------------------------------------------
# Single WF step
# ------------------------------------------------------------
def amplitude_wf_step(x, y, masks, step_size, eps=1e-12):
    """
    Perform one WF step for amplitude loss.

    Args:
        x: (H, W)
        y: (M, H, W)
        masks: (M, H, W)
        step_size: float

    Returns:
        updated x
    """
    # Forward
    w = cdp_forward_linear(x, masks)

    # Projection onto amplitude constraint
    projected = y * w / (np.abs(w) + eps)

    # Residual (negative gradient direction)
    residual = projected - w

    # Backprojection
    direction = cdp_adjoint_linear(residual, masks)

    # Normalize by number of measurements (important!)
    direction = direction / masks.shape[0]

    # Gradient step
    x = x + step_size * direction

    return x


# ------------------------------------------------------------
# Main solver
# ------------------------------------------------------------
def run_wirtinger_flow_cdp(
    y,
    masks,
    *,
    x0=None,
    n_iter=50,
    mu_max=0.4,
    tau=300.0,
    callback=None,
    seed=0,
):
    """
    Run amplitude-based Wirtinger Flow for CDP.

    Args:
        y: (M, H, W) amplitude measurements
        masks: (M, H, W)
        x0: initial estimate (H, W). If None, random init.
        n_iter: number of iterations
        mu_max, tau: step size parameters
        normalize: whether to normalize x each iteration
        callback: optional function(x, t)
        seed: random seed (if x0 is None)

    Returns:
        x: reconstructed signal (H, W)
    """
    M, H, W = y.shape

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    if x0 is None:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(H, W)) + 1j * rng.normal(size=(H, W))
        x = x.astype(np.complex64)
    else:
        x = x0.astype(np.complex64)

    # Normalize initial estimate
    x = x / (np.linalg.norm(x) + 1e-12)

    # --------------------------------------------------------
    # Iterations
    # --------------------------------------------------------
    for t in range(n_iter):
        step = wf_stepsize(t, mu_max=mu_max, tau=tau)

        x = amplitude_wf_step(x, y, masks, step)

        if callback is not None:
            callback(x, t)

    # --------------------------------------------------------
    # Final scaling (match measurement energy)
    # --------------------------------------------------------
    scale = np.sqrt(np.mean(y**2))
    x = scale * x

    return x.astype(np.complex64)