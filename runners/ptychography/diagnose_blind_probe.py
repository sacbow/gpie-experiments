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
    GaussianMeasurement,
    GaussianPrior,
    fft2,
    model,
)

from experiments.ptychography.forward_model import (
    extract_patches,
    fft2_centered,
)
from experiments.ptychography.object import make_complex_object
from experiments.ptychography.probe import (
    make_aperture_probe,
    make_image_probe,
)
from utils.config import ExperimentConfig


# ------------------------------------------------------------
# gPIE model
# ------------------------------------------------------------

@model
def blind_ptychography_model(
    obj_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    indices: Sequence[tuple[slice, slice]],
    noise_var: float,
    measurement_model: str,
    amplitude_damping: float | str,
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

    if measurement_model == "gaussian":
        GaussianMeasurement(
            var=noise_var,
            label="measurement",
        ) << detector_waves

    elif measurement_model == "amplitude":
        AmplitudeMeasurement(
            var=noise_var,
            damping=amplitude_damping,
            label="measurement",
        ) << detector_waves

    else:
        raise ValueError(
            f"Unknown measurement model: {measurement_model}"
        )


# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------

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
) -> np.ndarray:
    """Create a centered raster grid sorted from center outward."""
    obj_h, obj_w = object_shape
    probe_h, probe_w = probe_shape
    grid_h, grid_w = grid_shape

    center_y = (obj_h - probe_h) / 2.0
    center_x = (obj_w - probe_w) / 2.0

    offsets_y = (
        np.arange(grid_h, dtype=np.float32)
        - (grid_h - 1) / 2.0
    ) * step

    offsets_x = (
        np.arange(grid_w, dtype=np.float32)
        - (grid_w - 1) / 2.0
    ) * step

    positions = []

    for dy in offsets_y:
        for dx in offsets_x:
            y = int(round(center_y + float(dy)))
            x = int(round(center_x + float(dx)))

            if (
                y < 0
                or x < 0
                or y + probe_h > obj_h
                or x + probe_w > obj_w
            ):
                raise ValueError(
                    f"Invalid scan position {(y, x)}."
                )

            positions.append((y, x))

    positions = np.asarray(
        positions,
        dtype=np.int64,
    )

    object_center = np.asarray(
        [
            (obj_h - 1) / 2.0,
            (obj_w - 1) / 2.0,
        ],
        dtype=np.float32,
    )

    probe_offset = np.asarray(
        [
            (probe_h - 1) / 2.0,
            (probe_w - 1) / 2.0,
        ],
        dtype=np.float32,
    )

    scan_centers = (
        positions.astype(np.float32)
        + probe_offset
    )

    distance2 = np.sum(
        (scan_centers - object_center) ** 2,
        axis=1,
    )

    order = np.argsort(distance2)

    return positions[order]


# ------------------------------------------------------------
# Synthetic data
# ------------------------------------------------------------

def compute_farplane_waves(
    *,
    obj: np.ndarray,
    probe: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Compute noiseless complex diffraction waves."""
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

    return fft2_centered(
        exit_waves
    ).astype(
        np.complex64,
        copy=False,
    )


def compute_noise_variance(
    farplane_waves: np.ndarray,
    *,
    snr_db: float,
) -> float:
    """Compute noise variance from the diffraction signal power."""
    signal_power = np.mean(
        np.abs(farplane_waves) ** 2,
        dtype=np.float32,
    )

    snr_linear = np.float32(
        10.0 ** (
            float(snr_db) / 10.0
        )
    )

    noise_var = np.float32(
        signal_power / snr_linear
    )

    return float(noise_var)


def make_amplitude_observation(
    farplane_waves: np.ndarray,
    *,
    noise_var: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Gaussian-noisy amplitude observations."""
    amplitude = np.abs(
        farplane_waves
    ).astype(
        np.float32,
        copy=False,
    )

    noise = rng.standard_normal(
        amplitude.shape,
        dtype=np.float32,
    )

    noise *= np.float32(
        np.sqrt(noise_var)
    )

    return (
        amplitude + noise
    ).astype(
        np.float32,
        copy=False,
    )


def make_gaussian_observation(
    farplane_waves: np.ndarray,
    *,
    noise_var: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate circular complex Gaussian observations."""
    real_noise = rng.standard_normal(
        farplane_waves.shape,
        dtype=np.float32,
    )

    imag_noise = rng.standard_normal(
        farplane_waves.shape,
        dtype=np.float32,
    )

    scale = np.float32(
        np.sqrt(
            noise_var / 2.0
        )
    )

    noise = scale * (
        real_noise
        + 1j * imag_noise
    )

    return (
        farplane_waves + noise
    ).astype(
        np.complex64,
        copy=False,
    )


# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------

def make_object_initialization(
    *,
    shape: tuple[int, int],
    mode: str,
    seed: int,
) -> np.ndarray:
    """Create a deterministic object initialization."""
    if mode == "ones":
        return np.ones(
            shape,
            dtype=np.complex64,
        )

    if mode == "random":
        rng = np.random.default_rng(seed)

        real = rng.standard_normal(
            shape,
            dtype=np.float32,
        )
        imag = rng.standard_normal(
            shape,
            dtype=np.float32,
        )

        return (
            (real + 1j * imag)
            / np.float32(np.sqrt(2.0))
        ).astype(
            np.complex64,
            copy=False,
        )

    raise ValueError(
        f"Unknown object initialization: {mode}"
    )


def select_probe_initialization(
    *,
    mode: str,
    probe_true: np.ndarray,
    probe_aperture: np.ndarray,
) -> np.ndarray:
    """Select the probe initialization."""
    if mode == "truth":
        return probe_true.copy()

    if mode == "aperture":
        return probe_aperture.copy()

    raise ValueError(
        f"Unknown probe initialization: {mode}"
    )


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def _to_cpu_array(x) -> np.ndarray:
    """Convert NumPy or CuPy arrays to NumPy."""
    if hasattr(x, "get"):
        x = x.get()

    return np.asarray(x)


def _get_image_mean(
    graph,
    label: str,
) -> np.ndarray:
    """Extract a batch-size-one posterior mean as a 2-D NumPy array."""
    x = _to_cpu_array(
        graph[label]["mean"]
    )

    if (
        x.ndim == 3
        and x.shape[0] == 1
    ):
        x = x[0]

    return x


def normalized_measurement_fitness(
    *,
    obj: np.ndarray,
    probe: np.ndarray,
    positions: np.ndarray,
    observation: np.ndarray,
    measurement_model: str,
) -> float:
    """Compute normalized residual in the observation domain."""
    predicted = compute_farplane_waves(
        obj=obj,
        probe=probe,
        positions=positions,
    )

    if measurement_model == "gaussian":
        residual = (
            predicted - observation
        )

    elif measurement_model == "amplitude":
        residual = (
            np.abs(predicted)
            - observation
        )

    else:
        raise ValueError(
            f"Unknown measurement model: {measurement_model}"
        )

    numerator = np.sum(
        np.abs(residual) ** 2,
        dtype=np.float64,
    )

    denominator = np.sum(
        np.abs(observation) ** 2,
        dtype=np.float64,
    )

    return float(
        numerator
        / max(
            denominator,
            1e-12,
        )
    )


def center_crop(
    image: np.ndarray,
    size: int,
) -> np.ndarray:
    """Extract a centered square crop."""
    height, width = image.shape

    y0 = (
        height - size
    ) // 2
    x0 = (
        width - size
    ) // 2

    return image[
        y0:y0 + size,
        x0:x0 + size,
    ]


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

def save_history(
    path: Path,
    rows: list[dict],
) -> None:
    """Save convergence history."""
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "condition",
                "measurement",
                "probe_init",
                "iteration",
                "fitness",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def save_fitness_plot(
    path: Path,
    histories: dict[str, list[dict]],
) -> None:
    """Save fitness curves for all diagnostic conditions."""
    fig, axis = plt.subplots(
        figsize=(7, 5)
    )

    for condition_name, history in histories.items():
        iterations = [
            row["iteration"]
            for row in history
        ]

        fitness = [
            row["fitness"]
            for row in history
        ]

        axis.semilogy(
            iterations,
            fitness,
            label=condition_name,
        )

    axis.set_xlabel(
        "Iteration"
    )
    axis.set_ylabel(
        "Normalized measurement fitness"
    )
    axis.grid(True)
    axis.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_reconstruction(
    *,
    path: Path,
    obj_true: np.ndarray,
    obj_est: np.ndarray,
    probe_true: np.ndarray,
    probe_init: np.ndarray,
    probe_est: np.ndarray,
    crop_size: int,
    title: str,
) -> None:
    """Save qualitative object and probe reconstruction results."""
    obj_true_crop = center_crop(
        obj_true,
        crop_size,
    )

    obj_est_crop = center_crop(
        obj_est,
        crop_size,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11, 7),
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
        "Estimated probe"
    )

    for axis in axes.ravel():
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# ------------------------------------------------------------
# Single diagnostic condition
# ------------------------------------------------------------

def run_condition(
    *,
    condition: dict,
    obj_true: np.ndarray,
    obj_init: np.ndarray,
    probe_true: np.ndarray,
    probe_aperture: np.ndarray,
    positions: np.ndarray,
    amplitude_observation: np.ndarray,
    gaussian_observation: np.ndarray,
    noise_var: float,
    algorithm_config: dict,
    runtime_config: dict,
    run_dir: Path,
    crop_size: int,
) -> list[dict]:
    """Run one blind ptychography diagnostic condition."""
    condition_name = str(
        condition["name"]
    )
    measurement_model = str(
        condition["measurement"]
    )
    probe_init_mode = str(
        condition["probe_init"]
    )

    probe_init = select_probe_initialization(
        mode=probe_init_mode,
        probe_true=probe_true,
        probe_aperture=probe_aperture,
    )

    if measurement_model == "gaussian":
        observation = gaussian_observation
    elif measurement_model == "amplitude":
        observation = amplitude_observation
    else:
        raise ValueError(
            f"Unknown measurement model: {measurement_model}"
        )

    indices = positions_to_indices(
        positions,
        probe_true.shape,
    )

    graph = blind_ptychography_model(
        obj_shape=obj_true.shape,
        probe_shape=probe_true.shape,
        indices=indices,
        noise_var=noise_var,
        measurement_model=measurement_model,
        amplitude_damping=algorithm_config[
            "amplitude_damping"
        ],
    )

    graph.set_observation(
        observation,
        label="measurement",
    )

    graph.set_all_init_strategies(
        {
            "object": (
                "manual",
                obj_init,
            ),
            "probe": (
                "manual",
                probe_init,
            ),
        },
        verbose=False,
    )

    n_iter = int(
        algorithm_config["n_iter"]
    )
    log_every = int(
        algorithm_config["log_every"]
    )

    history: list[dict] = []

    initial_fitness = (
        normalized_measurement_fitness(
            obj=obj_init,
            probe=probe_init,
            positions=positions,
            observation=observation,
            measurement_model=measurement_model,
        )
    )

    history.append(
        {
            "condition": condition_name,
            "measurement": measurement_model,
            "probe_init": probe_init_mode,
            "iteration": -1,
            "fitness": initial_fitness,
        }
    )

    print(
        f"\n[{condition_name}]"
    )
    print(
        f"  measurement={measurement_model}"
    )
    print(
        f"  probe_init={probe_init_mode}"
    )
    print(
        f"  initial fitness={initial_fitness:.6e}"
    )

    def monitor(
        current_graph,
        iteration: int,
    ) -> None:
        """Record measurement-domain fitness."""
        if (
            iteration % log_every != 0
            and iteration != n_iter - 1
        ):
            return

        obj_est = _get_image_mean(
            current_graph,
            "object",
        )

        probe_est = _get_image_mean(
            current_graph,
            "probe",
        )

        fitness = (
            normalized_measurement_fitness(
                obj=obj_est,
                probe=probe_est,
                positions=positions,
                observation=observation,
                measurement_model=measurement_model,
            )
        )

        history.append(
            {
                "condition": condition_name,
                "measurement": measurement_model,
                "probe_init": probe_init_mode,
                "iteration": int(iteration),
                "fitness": fitness,
            }
        )

        print(
            f"  iter={iteration:04d}, "
            f"fitness={fitness:.6e}"
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
        device=runtime_config[
            "device"
        ],
        fft_engine=runtime_config.get(
            "fft_engine"
        ),
        callback=monitor,
        verbose=runtime_config.get(
            "verbose",
            False,
        ),
    )

    obj_est = _get_image_mean(
        graph,
        "object",
    ).astype(
        np.complex64,
        copy=False,
    )

    probe_est = _get_image_mean(
        graph,
        "probe",
    ).astype(
        np.complex64,
        copy=False,
    )

    condition_dir = (
        run_dir / condition_name
    )
    condition_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        condition_dir
        / "object_estimate.npy",
        obj_est,
    )

    np.save(
        condition_dir
        / "probe_estimate.npy",
        probe_est,
    )

    save_reconstruction(
        path=condition_dir
        / "reconstruction.png",
        obj_true=obj_true,
        obj_est=obj_est,
        probe_true=probe_true,
        probe_init=probe_init,
        probe_est=probe_est,
        crop_size=crop_size,
        title=condition_name,
    )

    print(
        f"  final fitness="
        f"{history[-1]['fitness']:.6e}"
    )

    return history


# ------------------------------------------------------------
# Experiment entry point
# ------------------------------------------------------------

def run(
    *,
    config: ExperimentConfig,
    run_dir: Path,
) -> None:
    """Run the blind ptychography diagnostic experiment."""
    data_config = config.section(
        "data"
    )
    forward_config = config.section(
        "forward_model"
    )
    algorithm_config = config.section(
        "algorithm"
    )

    runtime_config = config.runtime

    object_size = int(
        data_config["object_size"]
    )
    probe_size = int(
        data_config["probe_size"]
    )

    obj_shape = (
        object_size,
        object_size,
    )
    probe_shape = (
        probe_size,
        probe_size,
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

    probe_aperture = make_aperture_probe(
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

    obj_init = make_object_initialization(
        shape=obj_shape,
        mode=algorithm_config[
            "object_init"
        ],
        seed=int(
            data_config[
                "init_seed"
            ]
        ),
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
    )

    farplane_true = compute_farplane_waves(
        obj=obj_true,
        probe=probe_true,
        positions=positions,
    )

    noise_var = compute_noise_variance(
        farplane_true,
        snr_db=float(
            forward_config[
                "snr_db"
            ]
        ),
    )

    data_seed = int(
        data_config[
            "data_seed"
        ]
    )

    amplitude_rng = np.random.default_rng(
        data_seed
    )
    gaussian_rng = np.random.default_rng(
        data_seed + 1
    )

    amplitude_observation = (
        make_amplitude_observation(
            farplane_true,
            noise_var=noise_var,
            rng=amplitude_rng,
        )
    )

    gaussian_observation = (
        make_gaussian_observation(
            farplane_true,
            noise_var=noise_var,
            rng=gaussian_rng,
        )
    )

    np.save(
        run_dir
        / "scan_positions.npy",
        positions,
    )

    np.save(
        run_dir
        / "probe_true.npy",
        probe_true,
    )

    np.save(
        run_dir
        / "probe_aperture.npy",
        probe_aperture,
    )

    print(
        f"object shape: {obj_shape}"
    )
    print(
        f"probe shape: {probe_shape}"
    )
    print(
        f"n scans: {len(positions)}"
    )
    print(
        f"scan step: {scan_config['step']}"
    )
    print(
        f"SNR: {forward_config['snr_db']} dB"
    )
    print(
        f"noise variance: {noise_var:.6e}"
    )
    print(
        f"object init: "
        f"{algorithm_config['object_init']}"
    )

    histories = {}
    all_rows = []

    for condition in algorithm_config[
        "conditions"
    ]:
        history = run_condition(
            condition=condition,
            obj_true=obj_true,
            obj_init=obj_init,
            probe_true=probe_true,
            probe_aperture=probe_aperture,
            positions=positions,
            amplitude_observation=amplitude_observation,
            gaussian_observation=gaussian_observation,
            noise_var=noise_var,
            algorithm_config=algorithm_config,
            runtime_config=runtime_config,
            run_dir=run_dir,
            crop_size=int(
                data_config[
                    "object_crop_size"
                ]
            ),
        )

        condition_name = str(
            condition["name"]
        )

        histories[
            condition_name
        ] = history

        all_rows.extend(
            history
        )

    save_history(
        run_dir
        / "fitness.csv",
        all_rows,
    )

    save_fitness_plot(
        run_dir
        / "fitness.png",
        histories,
    )