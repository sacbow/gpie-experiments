from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gpie import (
    AmplitudeMeasurement,
    SupportPrior,
    fft2,
    model,
    pmse,
)

from datasets.loader import get_image
from experiments.random_structured_cdi.forward_model import (
    make_random_structured_cdi,
)
from utils.config import ConfigError, ExperimentConfig


SUPPORTED_NOISE_MODELS = {
    "gaussian",
    "poisson",
}


# ------------------------------------------------------------
# gPIE model
# ------------------------------------------------------------
@model
def random_structured_cdi_model(
    support,
    phase_masks,
    noise_var,
    damping,
):
    """Build a random structured CDI reconstruction model."""
    x = ~SupportPrior(
        support=support,
        label="object",
        dtype=np.complex64,
    )

    for phase_mask in phase_masks:
        x = fft2(phase_mask * x)

    AmplitudeMeasurement(
        var=noise_var,
        label="measurement",
        damping=damping,
    ) << x


# ------------------------------------------------------------
# Config validation helpers
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
    """Validate the random structured CDI demo config."""
    if config.problem != "random_structured_cdi":
        raise ConfigError(
            "This runner requires 'experiment.problem' to be "
            "'random_structured_cdi'."
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

    _require_nonempty_string(
        object_config,
        "amplitude_image",
        section="data.object",
    )
    _require_nonempty_string(
        object_config,
        "phase_image",
        section="data.object",
    )
    _require_nonempty_string(
        object_config,
        "normalize",
        section="data.object",
    )
    _require_number(
        object_config,
        "phase_scale",
        section="data.object",
    )

    _require_positive_int(
        forward_config,
        "n_layers",
        section="forward_model",
    )

    support_config = _require_mapping(
        forward_config,
        "support",
        section="forward_model",
    )

    support_type = _require_nonempty_string(
        support_config,
        "type",
        section="forward_model.support",
    )
    if support_type != "circular":
        raise ConfigError(
            "'forward_model.support.type' must be 'circular'."
        )

    radius_fraction = _require_positive_number(
        support_config,
        "radius_fraction",
        section="forward_model.support",
    )
    if radius_fraction >= 0.5:
        raise ConfigError(
            "'forward_model.support.radius_fraction' must be "
            "smaller than 0.5."
        )

    noise_model = _require_nonempty_string(
        forward_config,
        "noise_model",
        section="forward_model",
    )
    if noise_model not in SUPPORTED_NOISE_MODELS:
        raise ConfigError(
            "'forward_model.noise_model' must be "
            "'gaussian' or 'poisson'."
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
    _require_nonnegative_int(
        gpie_config,
        "init_seed",
        section="algorithm.gpie",
    )
    _require_positive_int(
        gpie_config,
        "block_size",
        section="algorithm.gpie",
    )

    damping = gpie_config.get("damping")

    if damping != "auto":
        damping = _require_number(
            gpie_config,
            "damping",
            section="algorithm.gpie",
        )

        if not 0.0 <= damping <= 1.0:
            raise ConfigError(
                "'algorithm.gpie.damping' must be in [0, 1] "
                "or equal to 'auto'."
            )

    schedule = _require_nonempty_string(
        gpie_config,
        "schedule",
        section="algorithm.gpie",
    )
    if schedule not in {
        "parallel",
        "sequential",
    }:
        raise ConfigError(
            "'algorithm.gpie.schedule' must be "
            "'parallel' or 'sequential'."
        )


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------
def make_circular_support(
    shape: tuple[int, int],
    radius_fraction: float,
) -> np.ndarray:
    """Create a centered circular support mask."""
    height, width = shape

    center_y = height // 2
    center_x = width // 2
    radius = radius_fraction * min(height, width)

    yy, xx = np.ogrid[:height, :width]
    distance_squared = (
        (yy - center_y) ** 2
        + (xx - center_x) ** 2
    )

    return distance_squared <= radius**2


def make_complex_image(
    *,
    size: int,
    object_config: Mapping[str, Any],
    support: np.ndarray,
) -> np.ndarray:
    """Create a supported complex natural image."""
    amplitude = get_image(
        object_config["amplitude_image"],
        size=size,
        normalize=object_config["normalize"],
    )

    phase_base = get_image(
        object_config["phase_image"],
        size=size,
        normalize=object_config["normalize"],
    )

    phase = (
        object_config["phase_scale"]
        * phase_base
    )

    image = amplitude * np.exp(1j * phase)

    return np.asarray(
        np.where(
            support,
            image,
            0.0,
        ),
        dtype=np.complex64,
    )


# ------------------------------------------------------------
# Observation generation
# ------------------------------------------------------------
def generate_observation(
    *,
    x_true: np.ndarray,
    n_layers: int,
    support: np.ndarray,
    seed: int,
    noise_model: str,
    noise_param: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    int | None,
]:
    """Generate observations using the independent forward model."""
    result = make_random_structured_cdi(
        x_true,
        n_layers=n_layers,
        support=support,
        seed=seed,
        noise_model=noise_model,
        noise_param=noise_param,
        dtype=np.complex64,
    )

    if noise_model == "gaussian":
        phase_masks, y_obs, noise_var = result
        photon_counts = None
    elif noise_model == "poisson":
        (
            phase_masks,
            y_obs,
            noise_var,
            photon_counts,
        ) = result
    else:
        raise ValueError(
            f"Unsupported noise model: {noise_model}"
        )

    return (
        np.asarray(
            phase_masks,
            dtype=np.complex64,
        ),
        np.asarray(
            y_obs,
            dtype=np.float32,
        ),
        float(noise_var),
        photon_counts,
    )


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
def align_global_phase(
    estimate: np.ndarray,
    reference: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Align the estimate under the global phase ambiguity."""
    correlation = np.vdot(
        estimate[support],
        reference[support],
    )

    if np.abs(correlation) <= 1e-12:
        return estimate.copy()

    return (
        estimate
        * np.exp(1j * np.angle(correlation))
    ).astype(np.complex64)


def save_figures(
    *,
    output_dir: Path,
    x_true: np.ndarray,
    x_est: np.ndarray,
    support: np.ndarray,
    y_obs: np.ndarray,
) -> None:
    """Save observation and reconstruction figures."""
    true_phase = np.angle(x_true) * support
    estimated_phase = np.angle(x_est) * support

    plt.imsave(
        output_dir / "support.png",
        support.astype(np.float32),
        cmap="gray",
    )
    plt.imsave(
        output_dir / "observation.png",
        y_obs,
        cmap="gray",
    )
    plt.imsave(
        output_dir / "true_amplitude.png",
        np.abs(x_true),
        cmap="gray",
    )
    plt.imsave(
        output_dir / "true_phase.png",
        true_phase,
        cmap="twilight",
    )
    plt.imsave(
        output_dir / "estimated_amplitude.png",
        np.abs(x_est),
        cmap="gray",
    )
    plt.imsave(
        output_dir / "estimated_phase.png",
        estimated_phase,
        cmap="twilight",
    )


def save_metrics(
    *,
    output_dir: Path,
    error: float,
    noise_var: float,
    photon_counts: int | None,
    support: np.ndarray,
) -> None:
    """Save scalar experiment results."""
    metrics = {
        "pmse": float(error),
        "effective_noise_variance": float(noise_var),
        "maximum_photon_counts": photon_counts,
        "support_fraction": float(
            np.mean(support)
        ),
    }

    with (
        output_dir / "metrics.json"
    ).open(
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
    data_seed: int,
    object_config: Mapping[str, Any],
    n_layers: int,
    support_config: Mapping[str, Any],
    noise_model: str,
    noise_param: float,
    gpie_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    output_config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Run the random structured CDI natural-image demo."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    support = make_circular_support(
        shape=(size, size),
        radius_fraction=support_config[
            "radius_fraction"
        ],
    )

    x_true = make_complex_image(
        size=size,
        object_config=object_config,
        support=support,
    )

    (
        phase_masks,
        y_obs,
        noise_var,
        photon_counts,
    ) = generate_observation(
        x_true=x_true,
        n_layers=n_layers,
        support=support,
        seed=data_seed,
        noise_model=noise_model,
        noise_param=noise_param,
    )

    graph = random_structured_cdi_model(
        support=support,
        phase_masks=phase_masks,
        noise_var=noise_var,
        damping=gpie_config["damping"],
    )

    graph.set_observation(
        y_obs,
        precision=1.0 / noise_var,
        label="measurement",
    )

    graph.set_init_rng(
        np.random.default_rng(
            gpie_config["init_seed"]
        )
    )

    graph.run(
        n_iter=gpie_config["n_iter"],
        schedule=gpie_config["schedule"],
        block_size=gpie_config["block_size"],
        device=runtime_config["device"],
        fft_engine=runtime_config.get(
            "fft_engine"
        ),
        verbose=runtime_config.get(
            "verbose",
            False,
        ),
    )

    x_est = np.asarray(
        graph["object"]["mean"][0],
        dtype=np.complex64,
    )

    x_est_aligned = align_global_phase(
        x_est,
        x_true,
        support,
    )

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
        support=support,
    )

    if output_config.get(
        "save_reconstruction",
        True,
    ):
        np.savez_compressed(
            output_dir / "result.npz",
            x_true=x_true,
            x_est=x_est,
            x_est_aligned=x_est_aligned,
            support=support,
            phase_masks=phase_masks,
            y_obs=y_obs,
            noise_var=np.asarray(
                noise_var,
                dtype=np.float32,
            ),
        )

    if output_config.get(
        "save_figures",
        True,
    ):
        save_figures(
            output_dir=output_dir,
            x_true=x_true,
            x_est=x_est_aligned,
            support=support,
            y_obs=y_obs,
        )


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def run(
    *,
    config: ExperimentConfig,
    run_dir: Path,
) -> None:
    """Run the demo from a resolved experiment config."""
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
        data_seed=data_config["seed"],
        object_config=data_config["object"],
        n_layers=forward_config["n_layers"],
        support_config=forward_config["support"],
        noise_model=forward_config["noise_model"],
        noise_param=forward_config["noise_param"],
        gpie_config=algorithm_config["gpie"],
        runtime_config=config.runtime,
        output_config=config.output,
        output_dir=run_dir,
    )