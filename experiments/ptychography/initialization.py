from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.ptychography.forward_model import (
    extract_patches,
    fft2_centered,
)


@dataclass(frozen=True)
class BlindPtychographyInitialization:
    """Initial states for blind ptychography."""

    obj: np.ndarray
    probe: np.ndarray
    detector: np.ndarray


def _normalize_rms(
    x: np.ndarray,
    *,
    target_rms: float = 1.0,
) -> np.ndarray:
    """Normalize an array to a target root-mean-square amplitude."""
    power = np.mean(
        np.abs(x) ** 2,
        dtype=np.float32,
    )

    rms = np.sqrt(
        np.maximum(
            power,
            np.float32(1e-12),
        )
    )

    return (
        x
        * np.float32(target_rms)
        / rms
    ).astype(
        np.complex64,
        copy=False,
    )


def make_initial_wave(
    shape: tuple[int, int],
    *,
    mode: str,
    rng: np.random.Generator,
    perturbation_scale: float = 0.3,
    target_rms: float = 1.0,
) -> np.ndarray:
    """Create a complex-valued object initialization."""
    if mode == "ones":
        wave = np.ones(
            shape,
            dtype=np.complex64,
        )

    elif mode == "complex_normal":
        real = rng.standard_normal(
            shape,
            dtype=np.float32,
        )
        imag = rng.standard_normal(
            shape,
            dtype=np.float32,
        )

        wave = (
            (real + 1j * imag)
            * np.float32(
                1.0 / np.sqrt(2.0)
            )
        ).astype(
            np.complex64,
            copy=False,
        )

    elif mode == "random_phase":
        phase = rng.uniform(
            -np.pi,
            np.pi,
            size=shape,
        ).astype(
            np.float32,
            copy=False,
        )

        wave = np.exp(
            1j * phase
        ).astype(
            np.complex64,
            copy=False,
        )

    elif mode == "perturbed_ones":
        real = rng.standard_normal(
            shape,
            dtype=np.float32,
        )
        imag = rng.standard_normal(
            shape,
            dtype=np.float32,
        )

        noise = (
            (real + 1j * imag)
            * np.float32(
                1.0 / np.sqrt(2.0)
            )
        )

        wave = (
            np.complex64(1.0)
            + np.float32(
                perturbation_scale
            )
            * noise
        ).astype(
            np.complex64,
            copy=False,
        )

    else:
        raise ValueError(
            f"Unknown initialization mode: {mode}"
        )

    return _normalize_rms(
        wave,
        target_rms=target_rms,
    )


def make_detector_initialization(
    *,
    obj_init: np.ndarray,
    probe_init: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Create a forward-consistent detector-wave initialization."""
    patches = extract_patches(
        obj_init,
        positions,
        probe_init.shape,
    )

    patches = patches.astype(
        np.complex64,
        copy=False,
    )

    exit_waves = (
        patches
        * probe_init[None, :, :]
    ).astype(
        np.complex64,
        copy=False,
    )

    detector = fft2_centered(
        exit_waves
    ).astype(
        np.complex64,
        copy=False,
    )

    return detector


def make_blind_ptychography_initialization(
    *,
    object_shape: tuple[int, int],
    probe_init: np.ndarray,
    positions: np.ndarray,
    object_mode: str,
    seed: int,
    perturbation_scale: float = 0.3,
    target_object_rms: float = 1.0,
) -> BlindPtychographyInitialization:
    """Create forward-consistent blind ptychography initialization."""
    rng = np.random.default_rng(
        seed
    )

    obj_init = make_initial_wave(
        object_shape,
        mode=object_mode,
        rng=rng,
        perturbation_scale=perturbation_scale,
        target_rms=target_object_rms,
    )

    probe_init = np.asarray(
        probe_init,
        dtype=np.complex64,
    ).copy()

    detector_init = make_detector_initialization(
        obj_init=obj_init,
        probe_init=probe_init,
        positions=positions,
    )

    return BlindPtychographyInitialization(
        obj=obj_init,
        probe=probe_init,
        detector=detector_init,
    )