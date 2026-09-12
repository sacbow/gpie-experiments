from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gpie import (
    AmplitudeMeasurement,
    GaussianPrior,
    fft2,
    model,
    pmse,
)

from experiments.ptychography.forward_model import (
    forward_ptychography_gaussian,
)
from experiments.ptychography.object import (
    make_complex_object,
)
from experiments.ptychography.pie import PIE
from experiments.ptychography.probe import (
    make_circular_probe,
)
from experiments.ptychography.scan import (
    make_fermat_scan_positions,
)
from utils.config import ConfigError, ExperimentConfig


SUPPORTED_METHODS = {
    "pie",
    "gpie",
}


# ------------------------------------------------------------
# gPIE model
# ------------------------------------------------------------
@model
def ptychography_graph_known_probe(
    obj_shape: tuple[int, int],
    probe: np.ndarray,
    indices: Sequence[tuple[slice, slice]],
    noise_var: float,
    damping: float | str,
) -> None:
    """Build a known-probe ptychography graph."""
    obj = ~GaussianPrior(
        event_shape=obj_shape,
        label="object",
        dtype=np.complex64,
    )

    patches = obj.extract_patches(indices)
    exit_waves = probe * patches
    detector_waves = fft2(exit_waves)

    AmplitudeMeasurement(
        var=noise_var,
        label="measurement",
        damping=damping,
    ) << detector_waves


# ------------------------------------------------------------
# Configuration
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


def validate_benchmark_config(
    config: ExperimentConfig,
) -> None:
    """Validate the known-probe benchmark configuration."""
    if config.problem != "ptychography":
        raise ConfigError(
            "This runner requires "
            "'experiment.problem' to be 'ptychography'."
        )

    data_config = config.section("data")
    forward_config = config.section("forward_model")
    algorithm_config = config.section("algorithm")

    if int(data_config["object_size"]) <= 0:
        raise ConfigError(
            "'data.object_size' must be positive."
        )

    if int(data_config["eval_crop_size"]) <= 0:
        raise ConfigError(
            "'data.eval_crop_size' must be positive."
        )

    if int(data_config["n_trials"]) <= 0:
        raise ConfigError(
            "'data.n_trials' must be positive."
        )

    probe_config = _require_mapping(
        forward_config,
        "probe",
        section="forward_model",
    )
    scan_config = _require_mapping(
        forward_config,
        "scan",
        section="forward_model",
    )

    if probe_config["soft_edge_width"] != 0:
        raise ConfigError(
            "The information-limit benchmark requires "
            "a hard circular probe with soft_edge_width=0."
        )

    if scan_config["type"] != "fermat":
        raise ConfigError(
            "'forward_model.scan.type' must be 'fermat'."
        )

    ratios = scan_config["target_sampling_ratios"]

    if (
        not isinstance(ratios, list)
        or not ratios
        or any(float(value) <= 0 for value in ratios)
    ):
        raise ConfigError(
            "'forward_model.scan.target_sampling_ratios' "
            "must be a non-empty list of positive numbers."
        )

    methods = algorithm_config["methods"]

    unknown = set(methods) - SUPPORTED_METHODS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ConfigError(
            f"Unsupported method(s): {names}"
        )


# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------
def positions_to_indices(
    positions: np.ndarray,
    probe_shape: tuple[int, int],
) -> list[tuple[slice, slice]]:
    """Convert top-left scan positions to object slices."""
    probe_h, probe_w = probe_shape

    return [
        (
            slice(int(y), int(y) + probe_h),
            slice(int(x), int(x) + probe_w),
        )
        for y, x in positions
    ]


def sort_positions_center_out(
    positions: np.ndarray,
    *,
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
) -> np.ndarray:
    """Sort scan positions by probe-center distance from the object center."""
    object_center = np.asarray(
        [
            (object_shape[0] - 1) / 2.0,
            (object_shape[1] - 1) / 2.0,
        ]
    )

    probe_half = np.asarray(
        [
            (probe_shape[0] - 1) / 2.0,
            (probe_shape[1] - 1) / 2.0,
        ]
    )

    centers = positions.astype(np.float64) + probe_half
    distance_squared = np.sum(
        (centers - object_center) ** 2,
        axis=1,
    )

    order = np.argsort(distance_squared)

    return positions[order]


def compute_coverage_count(
    *,
    object_shape: tuple[int, int],
    probe: np.ndarray,
    positions: np.ndarray,
    min_count: int,
) -> int:
    """Count object pixels illuminated at least min_count times."""
    if min_count <= 0:
        raise ValueError(
            "min_count must be positive."
        )

    probe_support = np.abs(probe) > 1e-8
    coverage = np.zeros(
        object_shape,
        dtype=np.int32,
    )

    probe_h, probe_w = probe.shape

    for y, x in positions:
        y = int(y)
        x = int(x)

        coverage[
            y:y + probe_h,
            x:x + probe_w,
        ] += probe_support

    return int(
        np.count_nonzero(
            coverage >= min_count
        )
    )


def compute_sampling_ratio(
    *,
    object_shape: tuple[int, int],
    probe: np.ndarray,
    positions: np.ndarray,
    min_coverage_count: int,
) -> tuple[float, int]:
    """Compute measurement-to-illuminated-pixel sampling ratio."""
    coverage_pixels = compute_coverage_count(
        object_shape=object_shape,
        probe=probe,
        positions=positions,
        min_count=min_coverage_count,
    )

    if coverage_pixels == 0:
        raise RuntimeError(
            "No object pixels satisfy the coverage criterion."
        )

    n_measurements = (
        len(positions)
        * probe.shape[0]
        * probe.shape[1]
    )

    sampling_ratio = (
        n_measurements / coverage_pixels
    )

    return float(sampling_ratio), coverage_pixels


def find_scan_geometry(
    *,
    target_ratio: float,
    object_shape: tuple[int, int],
    probe: np.ndarray,
    scan_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Find a Fermat step whose sampling ratio is closest to the target."""
    search_config = scan_config["step_search"]

    step_min = float(search_config["min"])
    step_max = float(search_config["max"])
    resolution = float(
        search_config["resolution"]
    )

    if resolution <= 0:
        raise ValueError(
            "Step-search resolution must be positive."
        )

    steps = np.arange(
        step_min,
        step_max + 0.5 * resolution,
        resolution,
    )

    best: dict[str, Any] | None = None

    for step in steps:
        positions = make_fermat_scan_positions(
            object_shape=object_shape,
            probe_shape=probe.shape,
            step=float(step),
            margin=int(scan_config["margin"]),
            jitter=int(scan_config["jitter"]),
            rng=np.random.default_rng(0),
        )

        positions = sort_positions_center_out(
            positions,
            object_shape=object_shape,
            probe_shape=probe.shape,
        )

        ratio, coverage_pixels = (
            compute_sampling_ratio(
                object_shape=object_shape,
                probe=probe,
                positions=positions,
                min_coverage_count=int(
                    scan_config[
                        "coverage_min_count"
                    ]
                ),
            )
        )

        error = abs(
            ratio - target_ratio
        )

        candidate = {
            "target_sampling_ratio": float(
                target_ratio
            ),
            "sampling_ratio": float(ratio),
            "scan_step": float(step),
            "positions": positions,
            "n_scans": int(len(positions)),
            "coverage_pixels": int(
                coverage_pixels
            ),
            "ratio_error": float(error),
        }

        if (
            best is None
            or candidate["ratio_error"]
            < best["ratio_error"]
        ):
            best = candidate

    if best is None:
        raise RuntimeError(
            "Could not construct a scan geometry."
        )

    return best


# ------------------------------------------------------------
# Object and initialization
# ------------------------------------------------------------
def make_initial_object(
    shape: tuple[int, int],
    seed: int,
) -> np.ndarray:
    """Draw an initial object from a standard complex normal distribution."""
    rng = np.random.default_rng(seed)

    real = rng.normal(
        size=shape
    )
    imag = rng.normal(
        size=shape
    )

    obj = (
        real + 1j * imag
    ) / np.sqrt(2.0)

    return obj.astype(
        np.complex64
    )


def center_crop(
    image: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    """Extract a centered square crop."""
    height, width = image.shape[-2:]

    if crop_size > min(height, width):
        raise ValueError(
            "crop_size exceeds image dimensions."
        )

    y0 = (
        height - crop_size
    ) // 2
    x0 = (
        width - crop_size
    ) // 2

    return image[
        ...,
        y0:y0 + crop_size,
        x0:x0 + crop_size,
    ]


def compute_pmse(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    crop_size: int,
) -> float:
    """Compute phase-invariant PMSE on the central crop."""
    estimate_crop = center_crop(
        estimate,
        crop_size,
    )
    reference_crop = center_crop(
        reference,
        crop_size,
    )

    return float(
        pmse(
            estimate_crop,
            reference_crop,
            normalize=False,
        )
    )


def pmse_to_db(
    value: float,
) -> float:
    """Convert PMSE to the positive dB convention used in the paper."""
    value = max(
        float(value),
        1e-30,
    )

    return float(
        -10.0 * np.log10(value)
    )


# ------------------------------------------------------------
# Algorithm runners
# ------------------------------------------------------------
def run_gpie_once(
    *,
    obj_true: np.ndarray,
    obj_init: np.ndarray,
    probe: np.ndarray,
    positions: np.ndarray,
    observed: np.ndarray,
    noise_var: float,
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    crop_size: int,
    log_every: int,
) -> tuple[
    np.ndarray,
    dict[str, list[float]],
]:
    """Run one known-probe gPIE reconstruction."""
    indices = positions_to_indices(
        positions,
        probe.shape,
    )

    graph = ptychography_graph_known_probe(
        obj_shape=obj_true.shape,
        probe=probe,
        indices=indices,
        noise_var=noise_var,
        damping=config["damping"],
    )

    graph.set_sample(
        "object",
        obj_true,
    )

    graph.set_observation(
        observed.astype(np.float32),
        precision=1.0 / noise_var,
        label="measurement",
    )

    graph.set_init_strategy(
        "object",
        "manual",
        data=obj_init,
        verbose=False,
    )

    history = {
        "iteration": [],
        "pmse": [],
        "pmse_db": [],
    }

    def monitor(
        current_graph,
        iteration: int,
    ) -> None:
        """Record the gPIE reconstruction error."""
        if (
            iteration % log_every != 0
            and iteration
            != config["n_iter"] - 1
        ):
            return

        estimate = current_graph[
            "object"
        ]["mean"]

        reference = (
            current_graph
            .get_wave("object")
            .get_sample()
        )

        error = compute_pmse(
            estimate,
            reference,
            crop_size=crop_size,
        )

        history["iteration"].append(
            int(iteration)
        )
        history["pmse"].append(
            float(error)
        )
        history["pmse_db"].append(
            pmse_to_db(error)
        )

    graph.run(
        n_iter=config["n_iter"],
        schedule=config["schedule"],
        block_size=config["block_size"],
        device=runtime_config["device"],
        fft_engine=runtime_config.get(
            "fft_engine"
        ),
        callback=monitor,
        verbose=runtime_config.get(
            "verbose",
            False,
        ),
    )

    estimate = np.asarray(
        graph["object"]["mean"]
    )

    if (
        estimate.ndim == 3
        and estimate.shape[0] == 1
    ):
        estimate = estimate[0]

    return (
        estimate.astype(
            np.complex64
        ),
        history,
    )


def run_pie_once(
    *,
    obj_true: np.ndarray,
    obj_init: np.ndarray,
    probe: np.ndarray,
    positions: np.ndarray,
    observed: np.ndarray,
    config: Mapping[str, Any],
    crop_size: int,
    log_every: int,
    seed: int,
) -> tuple[
    np.ndarray,
    dict[str, list[float]],
]:
    """Run one fixed-probe PIE reconstruction."""
    history = {
        "iteration": [],
        "pmse": [],
        "pmse_db": [],
    }

    def callback(
        iteration: int,
        _fourier_error: float,
        estimate: np.ndarray,
    ) -> None:
        """Record the PIE reconstruction error."""
        if (
            iteration % log_every != 0
            and iteration
            != config["n_iter"] - 1
        ):
            return

        error = compute_pmse(
            estimate,
            obj_true,
            crop_size=crop_size,
        )

        history["iteration"].append(
            int(iteration)
        )
        history["pmse"].append(
            float(error)
        )
        history["pmse_db"].append(
            pmse_to_db(error)
        )

    algorithm = PIE(
        obj_shape=obj_true.shape,
        probe=probe,
        positions=positions,
        measured_modulus=observed,
        alpha=config["alpha"],
        beta_object=config["beta_object"],
        obj_init=obj_init,
        callback=callback,
        seed=seed,
        shuffle=config["shuffle"],
    )

    estimate = algorithm.run(
        n_iter=config["n_iter"]
    )

    return (
        estimate.astype(
            np.complex64
        ),
        history,
    )


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
def summarize(
    values: list[float],
) -> dict[str, float]:
    """Compute summary statistics."""
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    q1, median, q3 = np.quantile(
        array,
        [0.25, 0.50, 0.75],
    )

    return {
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "mean": float(
            np.mean(array)
        ),
        "std": float(
            np.std(array)
        ),
    }


def save_rows(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Save rows as CSV."""
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(
    path: Path,
    data: Any,
) -> None:
    """Save JSON data."""
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )
        file.write("\n")


def save_reconstruction(
    *,
    output_dir: Path,
    method: str,
    target_ratio: float,
    obj_true: np.ndarray,
    obj_est: np.ndarray,
    crop_size: int,
) -> None:
    """Save the central crop of a representative reconstruction."""
    ratio_name = (
        f"{target_ratio:.2f}"
        .replace(".", "p")
    )

    obj_true_crop = center_crop(
        obj_true,
        crop_size,
    )
    obj_est_crop = center_crop(
        obj_est,
        crop_size,
    )

    # Align the global phase using the displayed crop.
    inner = np.vdot(
        obj_true_crop,
        obj_est_crop,
    )

    if np.abs(inner) > 1e-12:
        phase_offset = np.angle(inner)
        obj_est_crop = (
            obj_est_crop
            * np.exp(-1j * phase_offset)
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 8),
    )

    panels = [
        (
            np.abs(obj_true_crop),
            "True amplitude",
            "gray",
        ),
        (
            np.abs(obj_est_crop),
            f"{method}: amplitude",
            "gray",
        ),
        (
            np.angle(obj_true_crop),
            "True phase",
            "twilight",
        ),
        (
            np.angle(obj_est_crop),
            f"{method}: phase",
            "twilight",
        ),
    ]

    for axis, (image, title, cmap) in zip(
        axes.ravel(),
        panels,
    ):
        if cmap == "twilight":
            axis.imshow(
                image,
                cmap=cmap,
                vmin=-np.pi,
                vmax=np.pi,
            )
        else:
            axis.imshow(
                image,
                cmap=cmap,
            )

        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(
        f"Central {crop_size} x {crop_size} crop",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"reconstruction_{method}_alpha_{ratio_name}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_summary_plot(
    *,
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    methods: list[str],
) -> None:
    """Plot median PMSE in dB against actual sampling ratio."""
    fig, axis = plt.subplots(
        figsize=(7, 5),
    )

    for method in methods:
        rows = [
            row
            for row in summary_rows
            if row["method"] == method
        ]

        rows.sort(
            key=lambda row: row[
                "sampling_ratio"
            ]
        )

        x = np.asarray(
            [
                row["sampling_ratio"]
                for row in rows
            ]
        )
        median = np.asarray(
            [
                row["median_pmse_db"]
                for row in rows
            ]
        )
        q1 = np.asarray(
            [
                row["q1_pmse_db"]
                for row in rows
            ]
        )
        q3 = np.asarray(
            [
                row["q3_pmse_db"]
                for row in rows
            ]
        )

        axis.plot(
            x,
            median,
            marker="o",
            linewidth=2,
            label=method,
        )

        axis.fill_between(
            x,
            q1,
            q3,
            alpha=0.2,
        )

    axis.set_xlabel(
        "Sampling ratio"
    )
    axis.set_ylabel(
        "PMSE [dB]"
    )
    axis.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )
    axis.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir
        / "sampling_ratio_vs_pmse.png",
        dpi=200,
    )
    plt.close(fig)


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------
def benchmark(
    *,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    """Run the PIE versus gPIE known-probe benchmark."""
    data_config = config.section(
        "data"
    )
    forward_config = config.section(
        "forward_model"
    )
    algorithm_config = config.section(
        "algorithm"
    )

    object_size = int(
        data_config["object_size"]
    )
    object_shape = (
        object_size,
        object_size,
    )

    crop_size = int(
        data_config["eval_crop_size"]
    )

    object_config = data_config[
        "object"
    ]
    probe_config = forward_config[
        "probe"
    ]
    scan_config = forward_config[
        "scan"
    ]

    obj_true = make_complex_object(
        shape=object_shape,
        amplitude_name=object_config[
            "amplitude_image"
        ],
        phase_name=object_config[
            "phase_image"
        ],
        normalize=object_config[
            "normalize"
        ],
        phase_scale=float(
            object_config[
                "phase_scale"
            ]
        ),
        amplitude_min=float(
            object_config[
                "amplitude_min"
            ]
        ),
    )

    probe_size = int(
        probe_config["size"]
    )

    probe = make_circular_probe(
        shape=(
            probe_size,
            probe_size,
        ),
        radius=float(
            probe_config["radius"]
        ),
        soft_edge_width=float(
            probe_config[
                "soft_edge_width"
            ]
        ),
        phase_curvature=float(
            probe_config[
                "phase_curvature"
            ]
        ),
    )

    target_ratios = [
        float(value)
        for value in scan_config[
            "target_sampling_ratios"
        ]
    ]

    geometries = [
        find_scan_geometry(
            target_ratio=ratio,
            object_shape=object_shape,
            probe=probe,
            scan_config=scan_config,
        )
        for ratio in target_ratios
    ]

    geometry_rows = []

    for geometry in geometries:
        geometry_rows.append(
            {
                key: value
                for key, value
                in geometry.items()
                if key != "positions"
            }
        )

        print(
            "target alpha="
            f"{geometry['target_sampling_ratio']:.3f}, "
            "actual alpha="
            f"{geometry['sampling_ratio']:.3f}, "
            "step="
            f"{geometry['scan_step']:.2f}, "
            "scans="
            f"{geometry['n_scans']}"
        )

    save_rows(
        output_dir
        / "scan_geometries.csv",
        geometry_rows,
    )
    save_json(
        output_dir
        / "scan_geometries.json",
        geometry_rows,
    )

    raw_rows: list[
        dict[str, Any]
    ] = []

    histories: dict[
        str,
        dict[str, Any],
    ] = {}

    methods = list(
        algorithm_config["methods"]
    )
    n_trials = int(
        data_config["n_trials"]
    )
    log_every = int(
        algorithm_config[
            "log_every"
        ]
    )

    for geometry_index, geometry in enumerate(
        geometries
    ):
        target_ratio = geometry[
            "target_sampling_ratio"
        ]
        actual_ratio = geometry[
            "sampling_ratio"
        ]
        positions = geometry[
            "positions"
        ]

        for trial in range(
            n_trials
        ):
            data_seed = (
                int(
                    data_config[
                        "data_seed_base"
                    ]
                )
                + 1000 * geometry_index
                + trial
            )
            init_seed = (
                int(
                    data_config[
                        "init_seed_base"
                    ]
                )
                + 1000 * geometry_index
                + trial
            )

            data_rng = (
                np.random.default_rng(
                    data_seed
                )
            )

            data = (
                forward_ptychography_gaussian(
                    obj=obj_true,
                    probe=probe,
                    positions=positions,
                    snr_db=float(
                        forward_config[
                            "snr_db"
                        ]
                    ),
                    rng=data_rng,
                )
            )

            observed = (
                data.observed_amplitude
            )

            obj_init = (
                make_initial_object(
                    object_shape,
                    init_seed,
                )
            )

            for method in methods:
                print(
                    f"[alpha={actual_ratio:.3f}, "
                    f"trial={trial:02d}] "
                    f"running {method}..."
                )

                try:
                    if method == "gpie":
                        estimate, history = (
                            run_gpie_once(
                                obj_true=obj_true,
                                obj_init=obj_init,
                                probe=probe,
                                positions=positions,
                                observed=observed,
                                noise_var=data.noise_var,
                                config=algorithm_config[
                                    "gpie"
                                ],
                                runtime_config=config.runtime,
                                crop_size=crop_size,
                                log_every=log_every,
                            )
                        )

                    elif method == "pie":
                        estimate, history = (
                            run_pie_once(
                                obj_true=obj_true,
                                obj_init=obj_init,
                                probe=probe,
                                positions=positions,
                                observed=observed,
                                config=algorithm_config[
                                    "pie"
                                ],
                                crop_size=crop_size,
                                log_every=log_every,
                                seed=init_seed,
                            )
                        )

                    else:
                        raise ValueError(
                            f"Unknown method: {method}"
                        )

                    error = compute_pmse(
                        estimate,
                        obj_true,
                        crop_size=crop_size,
                    )
                    error_db = pmse_to_db(
                        error
                    )

                except Exception as exc:
                    print(
                        f"  failed: {exc}"
                    )

                    estimate = None
                    history = {
                        "iteration": [],
                        "pmse": [],
                        "pmse_db": [],
                    }
                    error = float(
                        "nan"
                    )
                    error_db = float(
                        "nan"
                    )

                print(
                    "  PMSE="
                    f"{error:.6e}, "
                    "PMSE[dB]="
                    f"{error_db:.3f}"
                )

                raw_rows.append(
                    {
                        "method": method,
                        "target_sampling_ratio": float(
                            target_ratio
                        ),
                        "sampling_ratio": float(
                            actual_ratio
                        ),
                        "scan_step": float(
                            geometry[
                                "scan_step"
                            ]
                        ),
                        "n_scans": int(
                            geometry[
                                "n_scans"
                            ]
                        ),
                        "coverage_pixels": int(
                            geometry[
                                "coverage_pixels"
                            ]
                        ),
                        "trial": int(
                            trial
                        ),
                        "data_seed": int(
                            data_seed
                        ),
                        "init_seed": int(
                            init_seed
                        ),
                        "pmse": float(
                            error
                        ),
                        "pmse_db": float(
                            error_db
                        ),
                    }
                )

                save_rows(
                    output_dir
                    / "raw_results.csv",
                    raw_rows,
                )
                save_json(
                    output_dir
                    / "raw_results.json",
                    raw_rows,
                )

                history_key = (
                    f"{method}"
                    f"_alpha_{target_ratio:.2f}"
                    f"_trial_{trial:02d}"
                )
                histories[
                    history_key
                ] = history

                if (
                    trial == 0
                    and estimate is not None
                    and config.output.get(
                        "save_figures",
                        True,
                    )
                ):
                    save_reconstruction(
                        output_dir=output_dir,
                        method=method,
                        target_ratio=target_ratio,
                        obj_true=obj_true,
                        obj_est=estimate,
                        crop_size=crop_size,
                    )

    summary_rows = []

    for geometry in geometries:
        for method in methods:
            rows = [
                row
                for row in raw_rows
                if (
                    row["method"]
                    == method
                    and row[
                        "target_sampling_ratio"
                    ]
                    == geometry[
                        "target_sampling_ratio"
                    ]
                )
                and np.isfinite(
                    row["pmse_db"]
                )
            ]

            values = [
                row["pmse_db"]
                for row in rows
            ]

            if not values:
                stats = {
                    "q1": float("nan"),
                    "median": float("nan"),
                    "q3": float("nan"),
                    "mean": float("nan"),
                    "std": float("nan"),
                }
            else:
                stats = summarize(
                    values
                )

            summary_rows.append(
                {
                    "method": method,
                    "target_sampling_ratio": float(
                        geometry[
                            "target_sampling_ratio"
                        ]
                    ),
                    "sampling_ratio": float(
                        geometry[
                            "sampling_ratio"
                        ]
                    ),
                    "scan_step": float(
                        geometry[
                            "scan_step"
                        ]
                    ),
                    "n_scans": int(
                        geometry[
                            "n_scans"
                        ]
                    ),
                    "q1_pmse_db": stats[
                        "q1"
                    ],
                    "median_pmse_db": stats[
                        "median"
                    ],
                    "q3_pmse_db": stats[
                        "q3"
                    ],
                    "mean_pmse_db": stats[
                        "mean"
                    ],
                    "std_pmse_db": stats[
                        "std"
                    ],
                }
            )

    save_rows(
        output_dir
        / "summary.csv",
        summary_rows,
    )
    save_json(
        output_dir
        / "summary.json",
        summary_rows,
    )

    if config.output.get(
        "save_history",
        True,
    ):
        save_json(
            output_dir
            / "histories.json",
            histories,
        )

    if config.output.get(
        "save_figures",
        True,
    ):
        save_summary_plot(
            output_dir=output_dir,
            summary_rows=summary_rows,
            methods=methods,
        )


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def run(
    *,
    config: ExperimentConfig,
    run_dir: Path,
) -> None:
    """Run the known-probe PIE versus gPIE benchmark."""
    validate_benchmark_config(
        config
    )

    benchmark(
        config=config,
        output_dir=run_dir,
    )