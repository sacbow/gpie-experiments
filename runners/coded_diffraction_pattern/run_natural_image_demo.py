from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gpie import (
    AmplitudeMeasurement,
    GaussianPrior,
    fft2,
    model,
    pmse,
    replicate,
)

from datasets.loader import get_image
from experiments.coded_diffraction_pattern.forward_model import make_cdp
from utils.config import ConfigError, ExperimentConfig


SUPPORTED_OBJECT_MODES = {
    "real",
    "complex",
}

SUPPORTED_NOISE_MODELS = {
    "gaussian",
    "poisson",
}


# ------------------------------------------------------------
# gPIE model
# ------------------------------------------------------------
@model
def coded_diffraction_pattern(
    shape,
    n_measurements,
    phase_masks,
    noise,
    object_dtype,
    damping,
):
    """Build a coded diffraction pattern model for gPIE."""
    x = ~GaussianPrior(
        event_shape=shape,
        label="object",
        dtype=object_dtype,
    )

    x_batch = replicate(
        x,
        batch_size=n_measurements,
    )
    y = fft2(phase_masks * x_batch)

    AmplitudeMeasurement(
        var=noise,
        label="measurement",
        damping=damping,
    ) << y


# ------------------------------------------------------------
# Config validation
# ------------------------------------------------------------
def _require_mapping(
    mapping: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> dict[str, Any]:
    """Return a required nested mapping."""
    value = mapping.get(key)

    if not isinstance(value, dict):
        raise ConfigError(
            f"'{section}.{key}' must be a JSON object."
        )

    return value


def _require_nonempty_string(
    mapping: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> str:
    """Return a required non-empty string."""
    value = mapping.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"'{section}.{key}' must be a non-empty string."
        )

    return value.strip()


def _require_positive_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> int:
    """Return a required positive integer."""
    value = mapping.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ConfigError(
            f"'{section}.{key}' must be a positive integer."
        )

    return value


def _require_nonnegative_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> int:
    """Return a required non-negative integer."""
    value = mapping.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ConfigError(
            f"'{section}.{key}' must be a non-negative integer."
        )

    return value


def _require_number(
    mapping: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> float:
    """Return a required finite real number."""
    value = mapping.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ConfigError(
            f"'{section}.{key}' must be a finite number."
        )

    return float(value)


def _require_positive_number(
    mapping: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> float:
    """Return a required positive finite number."""
    value = _require_number(
        mapping,
        key,
        section=section,
    )

    if value <= 0:
        raise ConfigError(
            f"'{section}.{key}' must be positive."
        )

    return value


def validate_demo_config(
    config: ExperimentConfig,
) -> None:
    """Validate settings specific to the natural-image demo."""
    if config.problem != "coded_diffraction_pattern":
        raise ConfigError(
            "This runner requires 'experiment.problem' to be "
            "'coded_diffraction_pattern'."
        )

    data_config = config.section("data")
    forward_config = config.section("forward_model")
    algorithm_config = config.section("algorithm")

    _require_positive_int(
        data_config,
        "size",
        section="data",
    )
    _require_nonnegative_int(
        data_config,
        "seed",
        section="data",
    )

    object_config = _require_mapping(
        data_config,
        "object",
        section="data",
    )

    mode = _require_nonempty_string(
        object_config,
        "mode",
        section="data.object",
    )
    if mode not in SUPPORTED_OBJECT_MODES:
        supported = ", ".join(
            sorted(SUPPORTED_OBJECT_MODES)
        )
        raise ConfigError(
            f"'data.object.mode' must be one of: {supported}"
        )

    _require_nonempty_string(
        object_config,
        "amplitude_image",
        section="data.object",
    )
    _require_nonempty_string(
        object_config,
        "normalize",
        section="data.object",
    )

    if mode == "complex":
        _require_nonempty_string(
            object_config,
            "phase_image",
            section="data.object",
        )
        _require_number(
            object_config,
            "phase_scale",
            section="data.object",
        )

    _require_positive_int(
        forward_config,
        "n_masks",
        section="forward_model",
    )

    noise_model = _require_nonempty_string(
        forward_config,
        "noise_model",
        section="forward_model",
    )
    if noise_model not in SUPPORTED_NOISE_MODELS:
        supported = ", ".join(
            sorted(SUPPORTED_NOISE_MODELS)
        )
        raise ConfigError(
            "'forward_model.noise_model' must be one of: "
            f"{supported}"
        )

    _require_positive_number(
        forward_config,
        "noise_param",
        section="forward_model",
    )

    gpie_config = _require_mapping(
        algorithm_config,
        "gpie",
        section="algorithm",
    )

    _require_positive_int(
        gpie_config,
        "n_iter",
        section="algorithm.gpie",
    )
    _require_positive_int(
        gpie_config,
        "block_size",
        section="algorithm.gpie",
    )

    damping = _require_number(
        gpie_config,
        "damping",
        section="algorithm.gpie",
    )
    if not 0.0 <= damping <= 1.0:
        raise ConfigError(
            "'algorithm.gpie.damping' must be in [0, 1]."
        )

    schedule = _require_nonempty_string(
        gpie_config,
        "schedule",
        section="algorithm.gpie",
    )
    if schedule not in {"parallel", "sequential"}:
        raise ConfigError(
            "'algorithm.gpie.schedule' must be "
            "'parallel' or 'sequential'."
        )


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------
def make_real_image(
    *,
    size: int,
    image_name: str,
    normalize: str,
) -> np.ndarray:
    """Create a real-valued natural-image object."""
    image = get_image(
        image_name,
        size=size,
        normalize=normalize,
    )

    return np.asarray(
        image,
        dtype=np.float32,
    )


def make_complex_image(
    *,
    size: int,
    amplitude_image: str,
    phase_image: str,
    normalize: str,
    phase_scale: float,
) -> np.ndarray:
    """Create a complex object from amplitude and phase images."""
    amplitude = get_image(
        amplitude_image,
        size=size,
        normalize=normalize,
    )
    phase_base = get_image(
        phase_image,
        size=size,
        normalize=normalize,
    )

    phase = phase_scale * phase_base
    image = amplitude * np.exp(1j * phase)

    return np.asarray(
        image,
        dtype=np.complex64,
    )


def make_ground_truth(
    *,
    size: int,
    object_config: Mapping[str, Any],
) -> np.ndarray:
    """Create the configured real or complex ground truth."""
    mode = object_config["mode"]
    normalize = object_config["normalize"]

    if mode == "real":
        return make_real_image(
            size=size,
            image_name=object_config["amplitude_image"],
            normalize=normalize,
        )

    if mode == "complex":
        return make_complex_image(
            size=size,
            amplitude_image=object_config["amplitude_image"],
            phase_image=object_config["phase_image"],
            normalize=normalize,
            phase_scale=object_config["phase_scale"],
        )

    raise ValueError(
        f"Unknown object mode: {mode}"
    )


# ------------------------------------------------------------
# Forward simulation compatibility
# ------------------------------------------------------------
def generate_cdp_observation(
    *,
    x_true: np.ndarray,
    n_masks: int,
    seed: int,
    noise_model: str,
    noise_param: float,
) -> tuple[np.ndarray, np.ndarray, float, Any | None]:
    """
    Generate CDP observations while accepting either supported return format.

    Some experiment code expects make_cdp() to return three values:

        masks, y_obs, noise_var

    while other code expects four:

        masks, y_obs, noise_var, photon_counts
    """
    result = make_cdp(
        x_true,
        num_masks=n_masks,
        seed=seed,
        noise_model=noise_model,
        noise_param=noise_param,
    )

    if not isinstance(result, tuple):
        raise TypeError(
            "make_cdp() must return a tuple."
        )

    if len(result) == 3:
        masks, y_obs, noise_var = result
        photon_counts = None
    elif len(result) == 4:
        masks, y_obs, noise_var, photon_counts = result
    else:
        raise ValueError(
            "make_cdp() must return either three or four values, "
            f"but returned {len(result)}."
        )

    return (
        np.asarray(masks, dtype=np.complex64),
        np.asarray(y_obs, dtype=np.float32),
        float(noise_var),
        photon_counts,
    )


# ------------------------------------------------------------
# Output utilities
# ------------------------------------------------------------
def _to_builtin_scalar(
    value: Any,
) -> int | float | None:
    """Convert an optional NumPy scalar to a JSON-compatible value."""
    if value is None:
        return None

    array = np.asarray(value)

    if array.ndim != 0:
        raise ValueError(
            "Expected photon-count metadata to be scalar."
        )

    scalar = array.item()

    if isinstance(scalar, (int, np.integer)):
        return int(scalar)

    return float(scalar)


def save_image(
    output_dir: Path,
    name: str,
    image: np.ndarray,
    *,
    cmap: str,
) -> None:
    """Save a single image."""
    plt.imsave(
        output_dir / name,
        image,
        cmap=cmap,
    )


def save_demo_figures(
    *,
    output_dir: Path,
    mode: str,
    x_true: np.ndarray,
    x_est: np.ndarray,
) -> None:
    """Save reconstruction figures."""
    if mode == "real":
        save_image(
            output_dir,
            "true_real.png",
            x_true,
            cmap="gray",
        )
        save_image(
            output_dir,
            "est_real.png",
            x_est,
            cmap="gray",
        )
        return

    save_image(
        output_dir,
        "true_amp_cmp.png",
        np.abs(x_true),
        cmap="gray",
    )
    save_image(
        output_dir,
        "true_phase_cmp.png",
        np.angle(x_true),
        cmap="twilight",
    )
    save_image(
        output_dir,
        "est_amp_cmp.png",
        np.abs(x_est),
        cmap="gray",
    )
    save_image(
        output_dir,
        "est_phase_cmp.png",
        np.angle(x_est),
        cmap="twilight",
    )


def save_metrics(
    *,
    output_dir: Path,
    error: float,
    noise_var: float,
    photon_counts: Any | None,
) -> None:
    """Save scalar experiment metrics."""
    metrics = {
        "pmse": float(error),
        "effective_noise_variance": float(noise_var),
        "maximum_photon_counts": _to_builtin_scalar(
            photon_counts
        ),
    }

    path = output_dir / "metrics.json"
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------
def run_demo(
    *,
    size: int,
    seed: int,
    object_config: Mapping[str, Any],
    n_masks: int,
    noise_model: str,
    noise_param: float,
    gpie_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    output_config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Run the natural-image coded diffraction pattern demo."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode = object_config["mode"]

    print("==== Natural Image Demo ====")
    print(f"mode        : {mode}")
    print(f"size        : {size}")
    print(f"masks       : {n_masks}")
    print(f"noise_model : {noise_model}")
    print(f"noise_param : {noise_param}")
    print()

    # --------------------------------------------------------
    # Data generation
    # --------------------------------------------------------
    x_true = make_ground_truth(
        size=size,
        object_config=object_config,
    )

    masks, y_obs, noise_var, photon_counts = (
        generate_cdp_observation(
            x_true=x_true,
            n_masks=n_masks,
            seed=seed,
            noise_model=noise_model,
            noise_param=noise_param,
        )
    )

    print(
        f"Effective noise variance: {noise_var:.3e}"
    )
    if photon_counts is not None:
        print(
            f"Maximum photon counts: {photon_counts}"
        )

    object_dtype = (
        np.float32
        if mode == "real"
        else np.complex64
    )

    graph = coded_diffraction_pattern(
        shape=(size, size),
        n_measurements=n_masks,
        phase_masks=masks,
        noise=noise_var,
        object_dtype=object_dtype,
        damping=gpie_config["damping"],
    )

    graph.set_observation(
        y_obs,
        precision=1.0 / noise_var,
        label="measurement",
    )

    graph.set_init_rng(
        np.random.default_rng(seed)
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------
    print("Running gPIE...")

    graph.run(
        n_iter=gpie_config["n_iter"],
        schedule=gpie_config["schedule"],
        block_size=gpie_config["block_size"],
        device=runtime_config["device"],
        fft_engine=runtime_config.get("fft_engine"),
        verbose=runtime_config.get("verbose", False),
    )

    x_est = np.asarray(
        graph["object"]["mean"][0]
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    error = pmse(
        x_est,
        x_true,
    )
    print(
        f"Final PMSE = {error:.6e}"
    )

    save_metrics(
        output_dir=output_dir,
        error=error,
        noise_var=noise_var,
        photon_counts=photon_counts,
    )

    if output_config.get(
        "save_reconstruction",
        True,
    ):
        np.savez_compressed(
            output_dir / "result.npz",
            x_true=x_true,
            x_est=x_est,
            masks=masks,
            y_obs=y_obs,
            noise_var=np.asarray(noise_var),
        )

    if output_config.get(
        "save_figures",
        True,
    ):
        save_demo_figures(
            output_dir=output_dir,
            mode=mode,
            x_true=x_true,
            x_est=x_est,
        )

    print(
        f"Results saved to: {output_dir}"
    )


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def run(
    *,
    config: ExperimentConfig,
    run_dir: Path,
) -> None:
    """Run the natural-image demo from a resolved config."""
    validate_demo_config(config)

    data_config = config.section("data")
    forward_config = config.section(
        "forward_model"
    )
    algorithm_config = config.section(
        "algorithm"
    )

    run_demo(
        size=data_config["size"],
        seed=data_config["seed"],
        object_config=data_config["object"],
        n_masks=forward_config["n_masks"],
        noise_model=forward_config["noise_model"],
        noise_param=forward_config["noise_param"],
        gpie_config=algorithm_config["gpie"],
        runtime_config=config.runtime,
        output_config=config.output,
        output_dir=run_dir,
    )