from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PtychographyData:
    """Container for synthetic ptychography data."""
    obj: np.ndarray
    probe: np.ndarray
    positions: np.ndarray
    exit_waves: np.ndarray
    farplane_waves: np.ndarray
    intensity: np.ndarray
    counts: np.ndarray
    amplitude: np.ndarray
    anscombe: np.ndarray
    photon_scale: float


def extract_patches(
    obj: np.ndarray,
    positions: np.ndarray,
    patch_shape: tuple[int, int],
) -> np.ndarray:
    """Extract object patches using top-left pixel coordinates."""
    ph, pw = patch_shape
    patches = np.empty((len(positions), ph, pw), dtype=obj.dtype)

    for i, (y, x) in enumerate(positions):
        patches[i] = obj[y:y + ph, x:x + pw]

    return patches


def fft2_centered(x: np.ndarray) -> np.ndarray:
    """Apply centered orthonormal 2D FFT over the last two axes."""
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(x, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def forward_ptychography(
    obj: np.ndarray,
    probe: np.ndarray,
    positions: np.ndarray,
    photon_scale: float = 1e4,
    rng: np.random.Generator | None = None,
) -> PtychographyData:
    """Generate noisy ptychographic diffraction data."""
    rng = np.random.default_rng() if rng is None else rng

    patches = extract_patches(obj, positions, probe.shape)
    exit_waves = patches * probe[None, :, :]

    farplane_waves = fft2_centered(exit_waves)
    intensity = np.abs(farplane_waves) ** 2

    lam = photon_scale * intensity
    counts = rng.poisson(lam).astype(np.float32)

    amplitude = np.sqrt(counts / photon_scale).astype(np.float32)

    # Anscombe transform for Poisson counts.
    anscombe = (2.0 * np.sqrt((counts + 3.0 / 8.0)/photon_scale)).astype(np.float32)

    return PtychographyData(
        obj=obj,
        probe=probe,
        positions=positions,
        exit_waves=exit_waves.astype(np.complex64),
        farplane_waves=farplane_waves.astype(np.complex64),
        intensity=intensity.astype(np.float32),
        counts=counts,
        amplitude=amplitude,
        anscombe=anscombe,
        photon_scale=float(photon_scale),
    )


#Used for reproduction of Ptycho-EP data
@dataclass(frozen=True)
class GaussianPtychographyData:
    """Container for Gaussian amplitude-noise ptychography data."""

    obj: np.ndarray
    probe: np.ndarray
    positions: np.ndarray
    exit_waves: np.ndarray
    farplane_waves: np.ndarray
    intensity: np.ndarray
    noiseless_amplitude: np.ndarray
    observed_amplitude: np.ndarray
    noise_var: float
    snr_db: float


def forward_ptychography_gaussian(
    obj: np.ndarray,
    probe: np.ndarray,
    positions: np.ndarray,
    *,
    snr_db: float = 30.0,
    rng: np.random.Generator | None = None,
) -> GaussianPtychographyData:
    """Generate ptychography data with additive Gaussian amplitude noise."""
    rng = np.random.default_rng() if rng is None else rng

    patches = extract_patches(
        obj,
        positions,
        probe.shape,
    )
    exit_waves = patches * probe[None, :, :]

    farplane_waves = fft2_centered(exit_waves)
    intensity = np.abs(farplane_waves) ** 2
    noiseless_amplitude = np.abs(farplane_waves)

    signal_power = float(np.mean(intensity))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = signal_power / snr_linear

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_var),
        size=noiseless_amplitude.shape,
    )

    observed_amplitude = (
        noiseless_amplitude + noise
    ).astype(np.float32)

    return GaussianPtychographyData(
        obj=obj,
        probe=probe,
        positions=positions,
        exit_waves=exit_waves.astype(np.complex64),
        farplane_waves=farplane_waves.astype(np.complex64),
        intensity=intensity.astype(np.float32),
        noiseless_amplitude=noiseless_amplitude.astype(np.float32),
        observed_amplitude=observed_amplitude,
        noise_var=float(noise_var),
        snr_db=float(snr_db),
    )