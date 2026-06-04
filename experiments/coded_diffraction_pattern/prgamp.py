import numpy as np


# ------------------------------------------------------------
# FFT (reuse WF implementation)
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
# CDP operators
# ------------------------------------------------------------
def A_forward(x, masks):
    return fft2c(masks * x[None, :, :])


def A_adjoint(z, masks):
    return np.sum(np.conj(masks) * ifft2c(z), axis=0)


# ------------------------------------------------------------
# g_out (phaseless channel)
# ------------------------------------------------------------
def g_out_phaseless(p, pvar, y, wvar, eps=1e-12):
    """
    Borrowed from https://github.com/ricedsp/D-AMP_Toolbox/blob/master/Algorithms/DprGAMP.m
    """
    abs_p = np.abs(p)
    B = 2.0 * y * abs_p / (wvar + pvar + eps)

    # Stable approximation of I1/I0 (Amos bounds)
    I1overI0 = np.minimum(
        B / (np.sqrt(B**2 + 4.0)),
        B / (0.5 + np.sqrt(B**2 + 0.25)),
    )

    y_sca = y / (1.0 + wvar / (pvar + eps))
    p_sca = abs_p / (1.0 + pvar / (wvar + eps))

    zhat = (p_sca + y_sca * I1overI0) * np.exp(1j * np.angle(p))

    sigma2_z = (
        y_sca**2
        + p_sca**2
        + (1.0 + B * I1overI0) / (1.0 / (wvar + eps) + 1.0 / (pvar + eps))
        - np.abs(zhat) ** 2
    )

    g = (zhat - p) / (pvar + eps)

    # Uniform variance approximation
    dg = (np.mean(sigma2_z) / (pvar + eps) - 1.0) / (pvar + eps)

    return g, dg


# ------------------------------------------------------------
# g_in (Gaussian prior CN(0,1))
# ------------------------------------------------------------
def g_in_gaussian(r, sigma2_r):
    """
    Denoiser for CN(0,1) prior.
    """
    x = r / (1.0 + sigma2_r)
    var = sigma2_r / (1.0 + sigma2_r)

    return x, var


# ------------------------------------------------------------
# Main solver
# ------------------------------------------------------------
def run_prgamp_cdp(
    y,
    masks,
    *,
    x0=None,
    n_iter=100,
    beta=0.25,
    wvar=1e-6,
    seed=0,
    callback=None,
):
    """
    prGAMP for CDP (uniform variance version).

    Args:
        y: (M,H,W)
        masks: (M,H,W)
        x0: (H,W)
        beta: damping
        wvar: noise variance
    """
    M, H, W = y.shape
    n = H * W
    m = y.size  # M * H * W

    # ||A||_F^2 = m
    A_norm2 = float(m)

    # --------------------------------------------------------
    # init
    # --------------------------------------------------------
    if x0 is None:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(H, W)) + 1j * rng.normal(size=(H, W))
    else:
        x = x0.copy()

    x = x.astype(np.complex64)

    sigma2_x = np.var(x)
    s = np.zeros_like(y, dtype=np.complex64)

    sigma2_p = 1.
    sigma2_s = 1.

    x_bar = x.copy()

    # --------------------------------------------------------
    # iteration
    # --------------------------------------------------------
    for t in range(n_iter):

        # ---- variance update ----
        sigma2_p = beta * (A_norm2 / m) * sigma2_x + (1 - beta) * sigma2_p
        sigma2_p = max(sigma2_p, 1e-12)

        # ---- forward ----
        p = A_forward(x, masks) - sigma2_p * s

        # ---- output nonlinearity ----
        g, dg = g_out_phaseless(p, sigma2_p, y, wvar)

        s = beta * g + (1 - beta) * s
        sigma2_s = -beta * dg + (1 - beta) * sigma2_s

        # ---- input variance ----
        sigma2_r = 1.0 / ((A_norm2 / n) * sigma2_s + 1e-12)

        # ---- damping ----
        x_bar = beta * x + (1 - beta) * x_bar

        # ---- backprojection ----
        r = x_bar + sigma2_r * A_adjoint(s, masks)

        # ---- denoiser ----
        x, sigma2_x = g_in_gaussian(r, sigma2_r)

        if callback is not None:
            info = {
                "sigma2_x": sigma2_x,
                "sigma2_r": sigma2_r,
                "sigma2_p": sigma2_p,
            }
            callback(x, t, info)

    return x.astype(np.complex64)