import numpy as np

from datasets.loader import get_image


def make_complex_object(
    name: str | None = None,
    shape: tuple[int, int] = (256, 256),
    phase_scale: float = np.pi,
    amplitude_min: float = 0.1,
    *,
    amplitude_name: str | None = None,
    phase_name: str = "coins",
    normalize: str = "max",
) -> np.ndarray:
    """Create a synthetic complex-valued object from two natural images."""
    if shape[0] != shape[1]:
        raise ValueError(
            "Only square object shapes are supported for now."
        )

    size = shape[0]

    if amplitude_name is None:
        amplitude_name = name if name is not None else "eagle"

    amplitude_image = get_image(
        name=amplitude_name,
        size=size,
        normalize=normalize,
    )
    phase_image = get_image(
        name=phase_name,
        size=size,
        normalize=normalize,
    )

    amplitude = (
        amplitude_min
        + (1.0 - amplitude_min) * amplitude_image
    )
    phase = phase_scale * phase_image

    obj = amplitude * np.exp(1j * phase)

    return obj.astype(np.complex64)