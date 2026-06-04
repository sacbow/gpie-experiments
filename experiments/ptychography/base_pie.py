from __future__ import annotations

from typing import Callable

import numpy as np

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

def ifft2_centered(x: np.ndarray) -> np.ndarray:
    """Apply centered orthonormal 2D FFT over the last two axes."""
    return np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.fftshift(x, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def fourier_projector(
    exit_wave: np.ndarray,
    measured_modulus: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project the exit wave onto the measured Fourier-modulus constraint."""
    farplane_wave = fft2_centered(exit_wave)
    current_modulus = np.abs(farplane_wave)

    eps = 1e-12
    projected_farplane = measured_modulus * farplane_wave / np.maximum(
        current_modulus,
        eps,
    )

    projected_exit_wave = ifft2_centered(projected_farplane)
    error = np.mean((current_modulus - measured_modulus) ** 2)

    return projected_exit_wave.astype(exit_wave.dtype, copy=False), float(error)


class BasePIE:
    """Base class for fixed-probe PIE-type reconstruction."""

    def __init__(
        self,
        *,
        obj_shape: tuple[int, int],
        probe: np.ndarray,
        positions: np.ndarray,
        measured_modulus: np.ndarray,
        beta_object: float = 0.1,
        obj_init: np.ndarray | None = None,
        dtype: np.dtype = np.complex64,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
        seed: int | None = None,
        shuffle: bool = True,
    ) -> None:
        self.obj_shape = tuple(obj_shape)
        self.probe = np.asarray(probe, dtype=dtype)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.measured_modulus = np.asarray(measured_modulus, dtype=np.float32)

        self.beta_object = float(beta_object)
        self.dtype = dtype
        self.callback = callback
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)

        self._validate_inputs()

        if obj_init is None:
            self.obj = self._init_object()
        else:
            self.obj = self._set_object_init(obj_init)

    def _validate_inputs(self) -> None:
        """Validate input shapes."""
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (n_scan, 2).")

        if self.measured_modulus.shape[0] != self.positions.shape[0]:
            raise ValueError(
                "measured_modulus and positions must have the same length."
            )

        if self.measured_modulus.shape[1:] != self.probe.shape:
            raise ValueError(
                "Each measured pattern must have the same shape as the probe."
            )

        if len(self.obj_shape) != 2:
            raise ValueError("obj_shape must be a tuple of length 2.")

    def _init_object(self) -> np.ndarray:
        """Initialize the object with unit amplitude and random phase."""
        phase = self.rng.uniform(-np.pi, np.pi, size=self.obj_shape)
        return np.exp(1j * phase).astype(self.dtype)

    def _set_object_init(self, obj_init: np.ndarray) -> np.ndarray:
        """Set the object initialization from an external array."""
        obj = np.asarray(obj_init, dtype=self.dtype)

        if obj.shape != self.obj_shape:
            raise ValueError(
                f"obj_init has shape {obj.shape}, expected {self.obj_shape}."
            )

        return obj.copy()

    def set_object(self, obj: np.ndarray) -> None:
        """Replace the current object estimate."""
        self.obj = self._set_object_init(obj)

    def run(self, n_iter: int = 100) -> np.ndarray:
        """Run PIE-type reconstruction."""
        n_scan = len(self.positions)
        order = np.arange(n_scan)

        for it in range(n_iter):
            if self.shuffle:
                self.rng.shuffle(order)

            err = 0.0

            for idx in order:
                y, x = self.positions[idx]
                measured = self.measured_modulus[idx]

                ph, pw = self.probe.shape
                obj_patch = self.obj[y:y + ph, x:x + pw]
                exit_wave = self.probe * obj_patch

                projected_exit_wave, error_val = fourier_projector(
                    exit_wave,
                    measured,
                )
                err += error_val

                self._update_object(
                    projected_exit_wave=projected_exit_wave,
                    exit_wave=exit_wave,
                    position=(y, x),
                )

            avg_err = err / n_scan

            if self.callback is not None:
                self.callback(it, avg_err, self.obj)

        return self.obj

    def _update_object(
        self,
        *,
        projected_exit_wave: np.ndarray,
        exit_wave: np.ndarray,
        position: tuple[int, int],
    ) -> None:
        """Update the object estimate for one scan position."""
        raise NotImplementedError