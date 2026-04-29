import numpy as np


def fft2c(x):
    """Apply centered orthonormal 2D FFT."""
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(x, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def ifft2c(x):
    """Apply centered orthonormal 2D inverse FFT."""
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(x, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def spectral_preprocess(y, eps=1e-12, clip_min=-1.0):
    """
    Apply the standard spectral preprocessing function.

    The input y is an amplitude observation. We normalize by the empirical
    second moment so that the preprocessing is scale-stable.
    """
    intensity = np.abs(y) ** 2
    intensity_norm = intensity / (np.mean(intensity) + eps)
    return np.maximum(1.0 - 1.0 / (intensity_norm + eps), clip_min)


def cdp_apply_A(x, masks):
    """Apply CDP forward linear operator before amplitude."""
    return fft2c(masks * x[None, :, :])


def cdp_apply_AH(z, masks):
    """Apply adjoint CDP operator."""
    return np.sum(np.conj(masks) * ifft2c(z), axis=0)


def spectral_initialization_cdp(
    y,
    masks,
    *,
    n_iter=400,
    seed=0,
    lamb=10.0,
    eps=1e-12,
):
    rng = np.random.default_rng(seed)

    M, H, W = y.shape

    # ------------------------------------------------------------
    # normalize measurements (deepinverse style)
    # ------------------------------------------------------------
    y_norm = y / (np.mean(y) + eps)

    weights = np.maximum(1.0 - 1.0 / (y_norm + eps), -5.0)

    # ------------------------------------------------------------
    # init
    # ------------------------------------------------------------
    x = rng.normal(size=(H, W)) + 1j * rng.normal(size=(H, W))
    x = x.astype(np.complex64)
    x /= np.linalg.norm(x) + eps

    # ------------------------------------------------------------
    # power iteration with damping
    # ------------------------------------------------------------
    for _ in range(n_iter):
        Ax = fft2c(masks * x[None, :, :])
        Wx = np.sum(np.conj(masks) * ifft2c(weights * Ax), axis=0)

        x = Wx + lamb * x  

        x = x / (np.linalg.norm(x) + eps)

    # scale
    x = np.sqrt(np.mean(y**2)) * x

    return x.astype(np.complex64)