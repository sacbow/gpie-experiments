import numpy as np
from skimage import data, color, transform


# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------
def _to_float32(x):
    """Convert uint8 image to float32 in [0, 1]."""
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    return x.astype(np.float32)


def _center_crop_to_square(x):
    """Crop image to centered square."""
    H, W = x.shape
    if H == W:
        return x

    if H > W:
        start = (H - W) // 2
        return x[start:start + W, :]
    else:
        start = (W - H) // 2
        return x[:, start:start + H]


def _resize(x, size):
    """Resize image to (size, size)."""
    return transform.resize(
        x,
        (size, size),
        anti_aliasing=True,
        mode="reflect",
    ).astype(np.float32)


def _normalize(x, mode="max"):
    """
    Normalize image.

    mode:
        - "max": divide by max
        - "l2": unit L2 norm
    """
    eps = 1e-12

    if mode == "max":
        return x / (np.max(x) + eps)

    elif mode == "l2":
        return x / (np.linalg.norm(x) + eps)

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


# ------------------------------------------------------------
# Dataset loaders
# ------------------------------------------------------------
def load_coins(size=128, normalize="max"):
    """
    Load skimage 'coins' dataset.

    Original: (303, 384), uint8
    """
    x = data.coins()  # grayscale already

    x = _to_float32(x)
    x = _center_crop_to_square(x)
    x = _resize(x, size)
    x = _normalize(x, normalize)

    return x


def load_eagle(size=128, normalize="max"):
    """
    Load skimage 'eagle' dataset.

    Original: (2019, 1826), uint8 or RGB depending on version.
    """
    x = data.eagle()

    # grayscale if needed
    if x.ndim == 3:
        x = color.rgb2gray(x)

    x = _to_float32(x)
    x = _center_crop_to_square(x)
    x = _resize(x, size)
    x = _normalize(x, normalize)

    return x


# ------------------------------------------------------------
# Unified interface
# ------------------------------------------------------------
def get_image(name="coins", size=128, normalize="max"):
    """
    Unified dataset interface.

    Args:
        name: "coins" or "eagle"
        size: output resolution
        normalize: "max" or "l2"

    Returns:
        x: (size, size) float32
    """
    if name == "coins":
        return load_coins(size=size, normalize=normalize)

    elif name == "eagle":
        return load_eagle(size=size, normalize=normalize)

    else:
        raise ValueError(f"Unknown dataset: {name}")