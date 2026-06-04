import numpy as np

from .base_pie import BasePIE


class PIE(BasePIE):
    """Classic PIE with a fixed probe."""

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)

        self.probe_conj = np.conj(self.probe)
        self.probe_abs = np.abs(self.probe)
        self.probe_abs2 = self.probe_abs**2
        self.probe_max = float(np.max(self.probe_abs))

    def _update_object(
        self,
        *,
        projected_exit_wave: np.ndarray,
        exit_wave: np.ndarray,
        position: tuple[int, int],
    ) -> None:
        y, x = position

        diff = projected_exit_wave - exit_wave

        denom = self.probe_max * (
            self.probe_abs2 + self.alpha * self.probe_max**2
        )

        update = self.beta_object * self.probe_abs * self.probe_conj * diff / denom

        self.obj[y:y + self.probe.shape[0], x:x + self.probe.shape[1]] += update