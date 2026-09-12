from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gpie import (
    AmplitudeMeasurement,
    GaussianPrior,
    UncertainArray,
    fft2,
    model,
)

from experiments.ptychography.forward_model import (
    extract_patches,
    fft2_centered,
)
from experiments.ptychography.initialization import (
    make_blind_ptychography_initialization,
)
from experiments.ptychography.object import (
    make_complex_object,
)
from experiments.ptychography.probe import (
    make_aperture_probe,
    make_image_probe,
)
from utils.config import ExperimentConfig


@model
def blind_ptychography_model(
    obj_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    indices: Sequence[tuple[slice, slice]],
    noise_var: float,
    damping: float | str,
) -> None:
    """Build a blind ptychography model."""
    obj = ~GaussianPrior(
        event_shape=obj_shape,
        label="object",
        dtype=np.complex64,
    )

    probe = ~GaussianPrior(
        event_shape=probe_shape,
        label="probe",
        dtype=np.complex64,
    )

    patches = obj.extract_patches(indices)
    exit_waves = probe * patches
    detector_waves = fft2(exit_waves)

    AmplitudeMeasurement(
        var=noise_var,
        damping=damping,
        label="measurement",
    ) << detector_waves


def positions_to_indices(
    positions: np.ndarray,
    probe_shape: tuple[int, int],
) -> list[tuple[slice, slice]]:
    """Convert top-left scan positions to slice indices."""
    probe_h, probe_w = probe_shape

    return [
        (
            slice(int(y), int(y) + probe_h),
            slice(int(x), int(x) + probe_w),
        )
        for y, x in positions
    ]


def make_centered_raster_positions(
    *,
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    grid_shape: tuple[int, int],
    step: int,
    jitter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a centered raster grid sorted from the center outward."""
    if step <= 0:
        raise ValueError(
            "step must be positive."
        )

    if jitter < 0:
        raise ValueError(
            "jitter must be non-negative."
        )

    obj_h, obj_w = object_shape
    probe_h, probe_w = probe_shape
    grid_h, grid_w = grid_shape

    center_y = (
        obj_h - probe_h
    ) / 2.0
    center_x = (
        obj_w - probe_w
    ) / 2.0

    offsets_y = (
        np.arange(
            grid_h,
            dtype=np.float32,
        )
        - (grid_h - 1) / 2.0
    ) * step

    offsets_x = (
        np.arange(
            grid_w,
            dtype=np.float32,
        )
        - (grid_w - 1) / 2.0
    ) * step

    positions = []

    for dy in offsets_y:
        for dx in offsets_x:
            y = int(
                round(
                    center_y
                    + float(dy)
                )
            )
            x = int(
                round(
                    center_x
                    + float(dx)
                )
            )

            if jitter > 0:
                y += int(
                    rng.integers(
                        -jitter,
                        jitter + 1,
                    )
                )
                x += int(
                    rng.integers(
                        -jitter,
                        jitter + 1,
                    )
                )

            if (
                y < 0
                or x < 0
                or y + probe_h > obj_h
                or x + probe_w > obj_w
            ):
                raise ValueError(
                    f"Invalid scan position {(y, x)} "
                    f"for object shape {object_shape} "
                    f"and probe shape {probe_shape}."
                )

            positions.append(
                (y, x)
            )

    positions = np.asarray(
        positions,
        dtype=np.int64,
    )

    probe_center_offset = np.asarray(
        [
            (probe_h - 1) / 2.0,
            (probe_w - 1) / 2.0,
        ],
        dtype=np.float32,
    )

    object_center = np.asarray(
        [
            (obj_h - 1) / 2.0,
            (obj_w - 1) / 2.0,
        ],
        dtype=np.float32,
    )

    scan_centers = (
        positions.astype(
            np.float32
        )
        + probe_center_offset
    )

    distance_squared = np.sum(
        (
            scan_centers
            - object_center
        ) ** 2,
        axis=1,
    )

    order = np.argsort(
        distance_squared
    )

    return positions[order]


def make_gaussian_amplitude_data(
    *,
    obj: np.ndarray,
    probe: np.ndarray,
    positions: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """Generate Gaussian-noisy amplitude measurements."""
    patches = extract_patches(
        obj,
        positions,
        probe.shape,
    )

    exit_waves = (
        patches
        * probe[None, :, :]
    ).astype(
        np.complex64,
        copy=False,
    )

    farplane_waves = fft2_centered(
        exit_waves
    ).astype(
        np.complex64,
        copy=False,
    )

    noiseless_amplitude = np.abs(
        farplane_waves
    ).astype(
        np.float32,
        copy=False,
    )

    signal_power = np.mean(
        noiseless_amplitude**2,
        dtype=np.float32,
    )

    snr_linear = np.float32(
        10.0 ** (
            float(snr_db) / 10.0
        )
    )

    noise_var = np.float32(
        signal_power
        / snr_linear
    )

    noise_std = np.sqrt(
        noise_var
    ).astype(
        np.float32
    )

    noise = rng.standard_normal(
        size=noiseless_amplitude.shape,
        dtype=np.float32,
    )

    observed = (
        noiseless_amplitude
        + noise_std * noise
    ).astype(
        np.float32,
        copy=False,
    )

    return (
        observed,
        noiseless_amplitude,
        float(noise_var),
    )


def seed_measurement_initial_message(
    *,
    graph,
    detector_init: np.ndarray,
    precision: float,
) -> None:
    """Seed the initial backward message from the amplitude measurement."""
    measurement = graph.get_factor(
        "measurement"
    )
    detector_wave = measurement.input

    initial_message = UncertainArray(
        detector_init,
        dtype=np.complex64,
        precision=float(
            precision
        ),
        batched=True,
    )

    if detector_wave.precision_mode == "scalar":
        initial_message = (
            initial_message.as_scalar_precision()
        )
    else:
        initial_message = (
            initial_message.as_array_precision()
        )

    detector_wave.receive_message(
        measurement,
        initial_message,
    )

    measurement.last_backward_messages[
        detector_wave
    ] = initial_message


def center_crop(
    image: np.ndarray,
    size: int,
) -> np.ndarray:
    """Extract a centered square crop."""
    height, width = image.shape[-2:]

    if size > min(
        height,
        width,
    ):
        raise ValueError(
            "Crop size exceeds image size."
        )

    y0 = (
        height - size
    ) // 2
    x0 = (
        width - size
    ) // 2

    return image[
        ...,
        y0:y0 + size,
        x0:x0 + size,
    ]


def save_history(
    path: Path,
    history: list[dict[str, float]],
) -> None:
    """Save fitness history to CSV."""
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "iteration",
                "fitness",
            ],
        )
        writer.writeheader()
        writer.writerows(
            history
        )


def save_fitness_plot(
    path: Path,
    history: list[dict[str, float]],
) -> None:
    """Save the measurement-fitness curve."""
    iterations = [
        row["iteration"]
        for row in history
    ]
    fitness = [
        row["fitness"]
        for row in history
    ]

    fig, axis = plt.subplots(
        figsize=(6, 4)
    )

    axis.semilogy(
        iterations,
        fitness,
        marker="o",
        markersize=3,
    )

    axis.set_xlabel(
        "Iteration"
    )
    axis.set_ylabel(
        "Measurement fitness"
    )
    axis.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)


def save_reconstruction_figure(
    *,
    path: Path,
    obj_true: np.ndarray,
    obj_est: np.ndarray,
    probe_true: np.ndarray,
    probe_init: np.ndarray,
    probe_est: np.ndarray,
    object_crop_size: int,
) -> None:
    """Save qualitative object and probe reconstruction results."""
    obj_true_crop = center_crop(
        obj_true,
        object_crop_size,
    )
    obj_est_crop = center_crop(
        obj_est,
        object_crop_size,
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7),
    )

    axes[0, 0].imshow(
        np.abs(obj_true_crop),
        cmap="gray",
    )
    axes[0, 0].set_title(
        "True object amplitude"
    )

    axes[0, 1].imshow(
        np.abs(obj_est_crop),
        cmap="gray",
    )
    axes[0, 1].set_title(
        "Estimated object amplitude"
    )

    axes[0, 2].imshow(
        np.angle(obj_est_crop),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[0, 2].set_title(
        "Estimated object phase"
    )

    axes[0, 3].axis(
        "off"
    )

    axes[1, 0].imshow(
        np.abs(probe_true),
        cmap="gray",
    )
    axes[1, 0].set_title(
        "True probe"
    )

    axes[1, 1].imshow(
        np.abs(probe_init),
        cmap="gray",
    )
    axes[1, 1].set_title(
        "Initial probe"
    )

    axes[1, 2].imshow(
        np.abs(probe_est),
        cmap="gray",
    )
    axes[1, 2].set_title(
        "Estimated probe amplitude"
    )

    axes[1, 3].imshow(
        np.angle(probe_est),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1, 3].set_title(
        "Estimated probe phase"
    )

    for axis in axes.ravel():
        axis.axis(
            "off"
        )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def run(
    *,
    config: ExperimentConfig,
    run_dir: Path,
) -> None:
    """Run a blind ptychography smoke test."""
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
        data_config[
            "object_size"
        ]
    )
    probe_size = int(
        data_config[
            "probe_size"
        ]
    )

    obj_shape = (
        object_size,
        object_size,
    )
    probe_shape = (
        probe_size,
        probe_size,
    )

    data_seed = int(
        data_config[
            "data_seed"
        ]
    )
    init_seed = int(
        data_config[
            "init_seed"
        ]
    )

    data_rng = (
        np.random.default_rng(
            data_seed
        )
    )

    obj_true = make_complex_object(
        name=data_config[
            "object_name"
        ],
        shape=obj_shape,
        phase_scale=float(
            data_config[
                "phase_scale"
            ]
        ),
        amplitude_min=float(
            data_config[
                "amplitude_min"
            ]
        ),
    ).astype(
        np.complex64,
        copy=False,
    )

    probe_true = make_image_probe(
        image_name=data_config[
            "probe_image"
        ],
        shape=probe_shape,
        radius=float(
            data_config[
                "probe_radius"
            ]
        ),
    ).astype(
        np.complex64,
        copy=False,
    )

    probe_init = make_aperture_probe(
        shape=probe_shape,
        radius=float(
            data_config[
                "probe_radius"
            ]
        ),
    ).astype(
        np.complex64,
        copy=False,
    )

    scan_config = forward_config[
        "scan"
    ]

    positions = make_centered_raster_positions(
        object_shape=obj_shape,
        probe_shape=probe_shape,
        grid_shape=tuple(
            scan_config[
                "grid_shape"
            ]
        ),
        step=int(
            scan_config[
                "step"
            ]
        ),
        jitter=int(
            scan_config[
                "jitter"
            ]
        ),
        rng=data_rng,
    )

    (
        observed,
        _noiseless_amplitude,
        noise_var,
    ) = make_gaussian_amplitude_data(
        obj=obj_true,
        probe=probe_true,
        positions=positions,
        snr_db=float(
            forward_config[
                "snr_db"
            ]
        ),
        rng=data_rng,
    )

    indices = positions_to_indices(
        positions,
        probe_shape,
    )

    graph = blind_ptychography_model(
        obj_shape=obj_shape,
        probe_shape=probe_shape,
        indices=indices,
        noise_var=noise_var,
        damping=algorithm_config[
            "damping"
        ],
    )

    graph.set_sample(
        "object",
        obj_true,
    )
    graph.set_sample(
        "probe",
        probe_true,
    )

    graph.set_observation(
        observed,
        precision=1.0 / noise_var,
        label="measurement",
    )

    initialization = (
        make_blind_ptychography_initialization(
            object_shape=obj_shape,
            probe_init=probe_init,
            positions=positions,
            object_mode=algorithm_config[
                "object_init"
            ],
            seed=init_seed,
            perturbation_scale=float(
                algorithm_config.get(
                    "object_init_perturbation",
                    0.3,
                )
            ),
        )
    )

    graph.set_all_init_strategies(
        {
            "object": (
                "manual",
                initialization.obj,
            ),
            "probe": (
                "manual",
                initialization.probe,
            ),
        },
        verbose=False,
    )

    use_measurement_warm_start = bool(
        algorithm_config.get(
            "measurement_warm_start",
            True,
        )
    )

    if use_measurement_warm_start:
        seed_measurement_initial_message(
            graph=graph,
            detector_init=initialization.detector,
            precision=float(
                algorithm_config.get(
                    "measurement_init_precision",
                    1.0,
                )
            ),
        )

    measurement = graph.get_factor(
        "measurement"
    )

    n_iter = int(
        algorithm_config[
            "n_iter"
        ]
    )
    log_every = int(
        algorithm_config[
            "log_every"
        ]
    )

    history: list[
        dict[str, float]
    ] = []

    def monitor(
        current_graph,
        iteration: int,
    ) -> None:
        """Record measurement fitness."""
        if (
            iteration % log_every != 0
            and iteration
            != n_iter - 1
        ):
            return

        fitness = (
            measurement.compute_fitness()
        )

        history.append(
            {
                "iteration": int(
                    iteration
                ),
                "fitness": float(
                    fitness
                ),
            }
        )

        print(
            f"[iter={iteration:04d}] "
            f"fitness={fitness:.6e}"
        )

    print(
        f"object shape: {obj_shape}"
    )
    print(
        f"probe shape: {probe_shape}"
    )
    print(
        f"scans: {len(positions)}"
    )
    print(
        f"scan step: "
        f"{scan_config['step']}"
    )
    print(
        f"noise variance: "
        f"{noise_var:.6e}"
    )
    print(
        f"object init: "
        f"{algorithm_config['object_init']}"
    )
    print(
        "measurement warm start: "
        f"{use_measurement_warm_start}"
    )

    if use_measurement_warm_start:
        print(
            "measurement init precision: "
            f"{algorithm_config.get('measurement_init_precision', 1.0)}"
        )

    graph.run(
        n_iter=n_iter,
        schedule=algorithm_config[
            "schedule"
        ],
        block_size=int(
            algorithm_config[
                "block_size"
            ]
        ),
        device=config.runtime[
            "device"
        ],
        fft_engine=config.runtime.get(
            "fft_engine"
        ),
        callback=monitor,
        verbose=config.runtime.get(
            "verbose",
            False,
        ),
    )

    obj_est = np.asarray(
        graph["object"]["mean"]
    )
    probe_est = np.asarray(
        graph["probe"]["mean"]
    )

    if (
        obj_est.ndim == 3
        and obj_est.shape[0] == 1
    ):
        obj_est = obj_est[0]

    if (
        probe_est.ndim == 3
        and probe_est.shape[0] == 1
    ):
        probe_est = probe_est[0]

    obj_est = obj_est.astype(
        np.complex64,
        copy=False,
    )
    probe_est = probe_est.astype(
        np.complex64,
        copy=False,
    )

    save_history(
        run_dir
        / "fitness.csv",
        history,
    )

    save_fitness_plot(
        run_dir
        / "fitness.png",
        history,
    )

    save_reconstruction_figure(
        path=run_dir
        / "reconstruction.png",
        obj_true=obj_true,
        obj_est=obj_est,
        probe_true=probe_true,
        probe_init=probe_init,
        probe_est=probe_est,
        object_crop_size=int(
            data_config[
                "object_crop_size"
            ]
        ),
    )

    np.save(
        run_dir
        / "object_estimate.npy",
        obj_est,
    )
    np.save(
        run_dir
        / "probe_estimate.npy",
        probe_est,
    )
    np.save(
        run_dir
        / "scan_positions.npy",
        positions,
    )
    np.save(
        run_dir
        / "detector_initialization.npy",
        initialization.detector,
    )

    if history:
        print(
            "final fitness: "
            f"{history[-1]['fitness']:.6e}"
        )