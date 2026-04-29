import numpy as np
from scipy.special import ive

from experiments.coded_diffraction_pattern.wirtinger_flow import (
    cdp_forward_linear,
    cdp_adjoint_linear,
)


def bessel_ratio_i1_i0(kappa, eps=1e-12):
    """Compute I1(kappa) / I0(kappa) stably using exponentially scaled Bessel functions."""
    return ive(1, kappa) / (ive(0, kappa) + eps)


def random_complex_normal(shape, seed=0):
    """Generate a normalized complex Gaussian random array."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    x = x.astype(np.complex64)
    x /= np.linalg.norm(x) + 1e-12
    return x


def prvbem_step(
    m,
    y,
    masks,
    sigma_n2,
    *,
    sigma_x2=1.0,
    eps=1e-12,
):
    """
    Perform one prVBEM update for the CDP model.

    Args:
        m: Current posterior mean, shape (H, W).
        y: Amplitude observations, shape (M, H, W).
        masks: CDP masks, shape (M, H, W).
        sigma_n2: Current noise variance.
        sigma_x2: Prior variance of x.

    Returns:
        m_new: Updated posterior mean.
        Sigma: Scalar posterior variance.
        y_bar: Phase-completed observation.
        z: Current linear prediction D m.
    """
    n_masks = masks.shape[0]

    z = cdp_forward_linear(m, masks)

    kappa = 2.0 * y * np.abs(z) / (sigma_n2 + eps)
    ratio = bessel_ratio_i1_i0(kappa, eps=eps)

    phase = np.exp(1j * np.angle(z))
    y_bar = y * phase * ratio

    residual = y_bar - z
    backproj = cdp_adjoint_linear(residual, masks)

    denominator = n_masks + sigma_n2 / sigma_x2
    Sigma = sigma_n2 * sigma_x2 / (sigma_n2 + n_masks * sigma_x2)

    m_new = (backproj + n_masks * m) / denominator

    return m_new.astype(np.complex64), float(Sigma), y_bar, z


def estimate_noise_variance(
    y,
    y_bar,
    z,
    Sigma,
    masks,
    *,
    min_noise=1e-12,
):
    """
    Estimate noise variance using the VB-EM M-step.

    This uses the CDP simplification D^H D = n_masks * I.
    """
    n_masks = masks.shape[0]
    n_pixels = y.shape[-2] * y.shape[-1]
    n_obs = y.size

    data_energy = np.sum(np.abs(y) ** 2)
    cross_term = -2.0 * np.real(np.vdot(y_bar, z))
    mean_energy = np.sum(np.abs(z) ** 2)

    variance_trace = Sigma * n_masks * n_pixels

    sigma_n2 = (data_energy + cross_term + mean_energy + variance_trace) / n_obs
    sigma_n2 = max(float(np.real(sigma_n2)), min_noise)

    return sigma_n2


def run_prvbem_cdp(
    y,
    masks,
    *,
    x0=None,
    n_iter=300,
    sigma_x2=1.0,
    sigma_n2_init=None,
    min_noise=1e-12,
    seed=0,
    callback=None,
):
    """
    Run prVBEM for coded diffraction pattern phase retrieval.

    Args:
        y: Amplitude observations, shape (M, H, W).
        masks: CDP masks, shape (M, H, W).
        x0: Optional initial mean, shape (H, W).
        n_iter: Number of VB-EM iterations.
        sigma_x2: Prior variance of x.
        sigma_n2_init: Initial noise variance. If None, initialized from measurement energy.
        min_noise: Lower bound for estimated noise variance.
        seed: Random seed used when x0 is None.
        callback: Optional callback(m, t, info).

    Returns:
        m: Posterior mean estimate.
        info: Dictionary containing noise and variance histories.
    """
    if y.ndim != 3:
        raise ValueError(f"y must have shape (M, H, W), got {y.shape}.")
    if masks.shape != y.shape:
        raise ValueError(f"masks and y must have the same shape, got {masks.shape} and {y.shape}.")

    _, H, W = y.shape

    if x0 is None:
        m = random_complex_normal((H, W), seed=seed)
        m *= np.sqrt(np.mean(y ** 2))
    else:
        m = x0.astype(np.complex64)

    if sigma_n2_init is None:
        sigma_n2 = float(0.1 * np.mean(y ** 2))
    else:
        sigma_n2 = float(sigma_n2_init)

    sigma_n2 = max(sigma_n2, min_noise)

    noise_history = []
    Sigma_history = []

    for t in range(n_iter):
        m, Sigma, y_bar, z = prvbem_step(
            m,
            y,
            masks,
            sigma_n2,
            sigma_x2=sigma_x2,
        )

        sigma_n2 = estimate_noise_variance(
            y,
            y_bar,
            z,
            Sigma,
            masks,
            min_noise=min_noise,
        )

        noise_history.append(sigma_n2)
        Sigma_history.append(Sigma)

        if callback is not None:
            info_t = {
                "sigma_n2": sigma_n2,
                "Sigma": Sigma,
            }
            callback(m, t, info_t)

    info = {
        "noise_history": np.asarray(noise_history),
        "Sigma_history": np.asarray(Sigma_history),
    }

    return m.astype(np.complex64), info