import numpy as np
from skimage import color, data, transform


# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------
def _to_float32(x):
    """Convert uint8 image to float32 in [0, 1]."""
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0

    return x.astype(np.float32)


def _center_crop_to_square(x):
    """Crop image to a centered square."""
    height, width = x.shape

    if height == width:
        return x

    if height > width:
        start = (height - width) // 2
        return x[start:start + width, :]

    start = (width - height) // 2
    return x[:, start:start + height]


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
    Normalize an image.

    Modes:
        - "max": divide by the maximum value
        - "l2": normalize to unit L2 norm
    """
    eps = 1e-12

    if mode == "max":
        return x / (np.max(x) + eps)

    if mode == "l2":
        return x / (np.linalg.norm(x) + eps)

    raise ValueError(
        f"Unknown normalization mode: {mode}"
    )


# ------------------------------------------------------------
# Dataset loaders
# ------------------------------------------------------------
def load_coins(size=128, normalize="max"):
    """
    Load the skimage 'coins' dataset.

    Original shape: (303, 384), uint8.
    """
    x = data.coins()

    x = _to_float32(x)
    x = _center_crop_to_square(x)
    x = _resize(x, size)
    x = _normalize(x, normalize)

    return x


def load_eagle(size=128, normalize="max"):
    """
    Load the skimage 'eagle' dataset.

    The original image may be grayscale or RGB depending on
    the installed scikit-image version.
    """
    x = data.eagle()

    if x.ndim == 3:
        x = color.rgb2gray(x)

    x = _to_float32(x)
    x = _center_crop_to_square(x)
    x = _resize(x, size)
    x = _normalize(x, normalize)

    return x


def load_camera(size=128, normalize="max"):
    """
    Load the skimage 'camera' dataset.

    Original shape: (512, 512), uint8.
    """
    x = data.camera()

    x = _to_float32(x)
    x = _center_crop_to_square(x)
    x = _resize(x, size)
    x = _normalize(x, normalize)

    return x


def load_moon(size=128, normalize="max"):
    """
    Load the skimage 'moon' dataset.

    Original image is grayscale.
    """
    x = data.moon()

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
    Load and preprocess a sample image.

    Args:
        name:
            Dataset name. Supported values are
            "coins", "eagle", and "camera".
        size:
            Output spatial resolution.
        normalize:
            Normalization mode, either "max" or "l2".

    Returns:
        Array with shape (size, size) and dtype float32.
    """
    if name == "coins":
        return load_coins(
            size=size,
            normalize=normalize,
        )

    if name == "eagle":
        return load_eagle(
            size=size,
            normalize=normalize,
        )

    if name == "camera":
        return load_camera(
            size=size,
            normalize=normalize,
        )

    if name == "moon":
        return load_moon(
            size=size,
            normalize=normalize,
        )

    raise ValueError(
        f"Unknown dataset: {name}"
    )