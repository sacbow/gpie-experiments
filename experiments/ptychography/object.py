import numpy as np

from datasets.loader import get_image


def make_complex_object(
    name: str = "coins",
    shape: tuple[int, int] = (256, 256),
    phase_scale: float = np.pi,
    amplitude_min: float = 0.1,
) -> np.ndarray:
    """Create a synthetic complex-valued object."""
    size = shape[0]
    if shape[0] != shape[1]:
        raise ValueError("Only square object shapes are supported for now.")

    img = get_image(name="eagle", size=size, normalize="max")
    phase_img = get_image(name="coins", size=size, normalize="max")

    amplitude = amplitude_min + (1.0 - amplitude_min) * img
    phase = phase_scale * phase_img

    obj = amplitude * np.exp(1j * phase)
    return obj.astype(np.complex64)