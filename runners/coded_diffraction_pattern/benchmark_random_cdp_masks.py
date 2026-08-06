from __future__ import annotations

import csv
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

from experiments.coded_diffraction_pattern.forward_model import make_cdp
from experiments.coded_diffraction_pattern.prgamp import run_prgamp_cdp
from experiments.coded_diffraction_pattern.prvbem import run_prvbem_cdp
from experiments.coded_diffraction_pattern.spectral_initialization import (
    spectral_initialization_cdp,
)
from experiments.coded_diffraction_pattern.wirtinger_flow import (
    run_wirtinger_flow_cdp,
)
from utils.config import ConfigError, ExperimentConfig


SUPPORTED_METHODS = {
    "gpie",
    "wf",
    "prvbem",
    "prgamp",
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
):
    """Build a batched coded diffraction pattern model."""
    x = ~GaussianPrior(
        event_shape=shape,
        label="object",
        dtype=np.complex64,
    )

    x_batch = replicate(
        x,
        batch_size=n_measurements,
    )
    y = fft2(phase_masks * x_batch)

    AmplitudeMeasurement(
        var=noise,
        label="measurement",
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


def _require_positive_number(
    mapping: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> float:
    """Return a required positive real number."""
    value = mapping.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise ConfigError(
            f"'{section}.{key}' must be a positive number."
        )

    return float(value)


def _require_number(
    mapping: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> float:
    """Return a required real number."""
    value = mapping.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ConfigError(
            f"'{section}.{key}' must be a number."
        )

    return float(value)


def validate_benchmark_config(
    config: ExperimentConfig,
) -> None:
    """Validate problem-specific benchmark settings."""
    if config.problem != "coded_diffraction_pattern":
        raise ConfigError(
            "This runner requires "
            "'experiment.problem' to be "
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
    _require_positive_int(
        data_config,
        "n_trials",
        section="data",
    )
    _require_nonnegative_int(
        data_config,
        "data_seed_base",
        section="data",
    )
    _require_nonnegative_int(
        data_config,
        "init_seed_base",
        section="data",
    )

    mask_counts = data_config.get("mask_counts")
    if (
        not isinstance(mask_counts, list)
        or not mask_counts
    ):
        raise ConfigError(
            "'data.mask_counts' must be a non-empty list."
        )

    for index, value in enumerate(mask_counts):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ConfigError(
                "'data.mask_counts' must contain only "
                f"positive integers; invalid value at index {index}."
            )

    noise_model = forward_config.get("noise_model")
    if noise_model not in {"gaussian", "poisson"}:
        raise ConfigError(
            "'forward_model.noise_model' must be "
            "'gaussian' or 'poisson'."
        )

    _require_positive_number(
        forward_config,
        "noise_param",
        section="forward_model",
    )

    methods = algorithm_config.get("methods")
    if (
        not isinstance(methods, list)
        or not methods
        or not all(isinstance(item, str) for item in methods)
    ):
        raise ConfigError(
            "'algorithm.methods' must be a non-empty "
            "list of strings."
        )

    if len(methods) != len(set(methods)):
        raise ConfigError(
            "'algorithm.methods' must not contain duplicates."
        )

    unknown_methods = set(methods) - SUPPORTED_METHODS
    if unknown_methods:
        names = ", ".join(sorted(unknown_methods))
        raise ConfigError(
            f"Unsupported benchmark method(s): {names}"
        )

    if "gpie" in methods:
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

        schedule = gpie_config.get("schedule")
        if schedule not in {"parallel", "sequential"}:
            raise ConfigError(
                "'algorithm.gpie.schedule' must be "
                "'parallel' or 'sequential'."
            )

    if "wf" in methods:
        wf_config = _require_mapping(
            algorithm_config,
            "wf",
            section="algorithm",
        )
        _require_positive_int(
            wf_config,
            "n_iter",
            section="algorithm.wf",
        )
        _require_positive_number(
            wf_config,
            "mu_max",
            section="algorithm.wf",
        )
        _require_positive_number(
            wf_config,
            "tau",
            section="algorithm.wf",
        )

        si_config = _require_mapping(
            wf_config,
            "spectral_initialization",
            section="algorithm.wf",
        )
        _require_positive_int(
            si_config,
            "n_iter",
            section="algorithm.wf.spectral_initialization",
        )
        _require_positive_number(
            si_config,
            "lamb",
            section="algorithm.wf.spectral_initialization",
        )

    if "prvbem" in methods:
        prvbem_config = _require_mapping(
            algorithm_config,
            "prvbem",
            section="algorithm",
        )
        _require_positive_int(
            prvbem_config,
            "n_iter",
            section="algorithm.prvbem",
        )

    if "prgamp" in methods:
        prgamp_config = _require_mapping(
            algorithm_config,
            "prgamp",
            section="algorithm",
        )
        _require_positive_int(
            prgamp_config,
            "n_iter",
            section="algorithm.prgamp",
        )

        beta = _require_number(
            prgamp_config,
            "beta",
            section="algorithm.prgamp",
        )
        if not 0.0 <= beta <= 1.0:
            raise ConfigError(
                "'algorithm.prgamp.beta' must be in [0, 1]."
            )


# ------------------------------------------------------------
# Data utilities
# ------------------------------------------------------------
def make_random_complex_ground_truth(
    shape: tuple[int, int],
    seed: int,
) -> np.ndarray:
    """Generate a complex Gaussian ground truth with unit average power."""
    rng = np.random.default_rng(seed)

    x = (
        rng.normal(size=shape)
        + 1j * rng.normal(size=shape)
    )
    x = x / np.sqrt(2.0)

    power = np.mean(np.abs(x) ** 2)
    x = x / np.sqrt(power + 1e-12)

    return x.astype(np.complex64)


def make_random_initialization(
    shape: tuple[int, int],
    seed: int,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate a normalized random complex initialization."""
    rng = np.random.default_rng(seed)

    x = (
        rng.normal(size=shape)
        + 1j * rng.normal(size=shape)
    ).astype(np.complex64)

    x = x / (
        np.linalg.norm(x)
        + 1e-12
    )
    x = scale * x

    return x.astype(np.complex64)


# ------------------------------------------------------------
# Result utilities
# ------------------------------------------------------------
def summarize(
    values: list[float] | np.ndarray,
) -> dict[str, float]:
    """Compute quartiles, mean, and standard deviation."""
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    q1, q2, q3 = np.quantile(
        array,
        [0.25, 0.50, 0.75],
    )

    return {
        "q1": float(q1),
        "q2": float(q2),
        "q3": float(q3),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def save_raw_results(
    output_dir: Path,
    raw_results: list[dict[str, Any]],
) -> None:
    """Save trial-level results as CSV and JSON."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "method",
        "n_masks",
        "trial",
        "data_seed",
        "init_seed",
        "pmse",
    ]

    csv_path = output_dir / "raw_results.csv"
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(raw_results)

    json_path = output_dir / "raw_results.json"
    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            raw_results,
            file,
            indent=2,
        )
        file.write("\n")


def save_summary(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    """Save summary statistics as CSV and JSON."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "method",
        "n_masks",
        "q1",
        "q2",
        "q3",
        "mean",
        "std",
    ]

    csv_path = output_dir / "summary.csv"
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = output_dir / "summary.json"
    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_rows,
            file,
            indent=2,
        )
        file.write("\n")


def plot_mask_vs_pmse(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    methods: list[str],
    mask_counts: list[int],
) -> None:
    """Plot mask count versus PMSE with interquartile bands."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(7, 5),
    )

    styles = [
        {
            "linestyle": "-",
            "marker": "o",
        },
        {
            "linestyle": "--",
            "marker": "s",
        },
        {
            "linestyle": "-.",
            "marker": "d",
        },
        {
            "linestyle": ":",
            "marker": "^",
        },
    ]

    for index, method in enumerate(methods):
        rows = [
            row
            for row in summary_rows
            if row["method"] == method
        ]
        rows.sort(
            key=lambda row: row["n_masks"]
        )

        xs = np.asarray(
            [row["n_masks"] for row in rows],
            dtype=float,
        )
        q1 = np.asarray(
            [row["q1"] for row in rows],
            dtype=float,
        )
        q2 = np.asarray(
            [row["q2"] for row in rows],
            dtype=float,
        )
        q3 = np.asarray(
            [row["q3"] for row in rows],
            dtype=float,
        )

        style = styles[index % len(styles)]

        axis.plot(
            xs,
            q2,
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2,
            markersize=10,
            label=method,
        )

        axis.fill_between(
            xs,
            q1,
            q3,
            alpha=0.2,
        )

    axis.set_xlabel(
        "Number of masks",
        fontsize=20,
    )
    axis.set_ylabel(
        "Final PMSE",
        fontsize=20,
    )

    axis.set_xticks(mask_counts)
    axis.tick_params(
        axis="both",
        labelsize=15,
    )

    axis.set_yscale("log")
    axis.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.6,
    )
    axis.legend(fontsize=11)

    figure.tight_layout()
    figure.savefig(
        output_dir / "mask_vs_pmse.png",
        dpi=200,
    )
    plt.close(figure)


# ------------------------------------------------------------
# Method runners
# ------------------------------------------------------------
def run_gpie_once(
    x_true: np.ndarray,
    masks: np.ndarray,
    y_obs: np.ndarray,
    noise_var: float,
    *,
    n_iter: int,
    init_seed: int,
    device: str,
    fft_engine: str | None,
    schedule: str,
    block_size: int,
    verbose: bool,
) -> np.ndarray:
    """Run one gPIE reconstruction."""
    shape = x_true.shape
    n_masks = masks.shape[0]

    graph = coded_diffraction_pattern(
        shape=shape,
        n_measurements=n_masks,
        phase_masks=masks.astype(np.complex64),
        noise=float(noise_var),
    )

    graph.set_observation(
        y_obs.astype(np.float32),
        precision=1.0 / noise_var,
        label="measurement",
    )

    graph.set_init_rng(
        np.random.default_rng(init_seed)
    )

    graph.run(
        n_iter=n_iter,
        device=device,
        fft_engine=fft_engine,
        schedule=schedule,
        block_size=block_size,
        verbose=verbose,
    )

    return graph["object"]["mean"][0].astype(
        np.complex64
    )


def run_wf_once(
    masks: np.ndarray,
    y_obs: np.ndarray,
    *,
    n_iter: int,
    init_seed: int,
    si_iter: int,
    si_lamb: float,
    mu_max: float,
    tau: float,
) -> np.ndarray:
    """Run one Wirtinger Flow reconstruction."""
    x0 = spectral_initialization_cdp(
        y_obs.astype(np.float32),
        masks.astype(np.complex64),
        n_iter=si_iter,
        seed=init_seed,
        lamb=si_lamb,
    )

    return run_wirtinger_flow_cdp(
        y_obs.astype(np.float32),
        masks.astype(np.complex64),
        x0=x0,
        n_iter=n_iter,
        mu_max=mu_max,
        tau=tau,
        callback=None,
        seed=init_seed,
    ).astype(np.complex64)


def run_prvbem_once(
    x_true: np.ndarray,
    masks: np.ndarray,
    y_obs: np.ndarray,
    *,
    n_iter: int,
    init_seed: int,
) -> np.ndarray:
    """Run one prVBEM reconstruction."""
    scale = np.sqrt(
        np.mean(y_obs**2)
    )
    x0 = make_random_initialization(
        x_true.shape,
        init_seed,
        scale=scale,
    )

    x_est, _ = run_prvbem_cdp(
        y_obs.astype(np.float32),
        masks.astype(np.complex64),
        x0=x0,
        n_iter=n_iter,
        sigma_x2=1.0,
        sigma_n2_init=None,
        seed=init_seed,
        callback=None,
    )

    return x_est.astype(np.complex64)


def run_prgamp_once(
    x_true: np.ndarray,
    masks: np.ndarray,
    y_obs: np.ndarray,
    noise_var: float,
    *,
    n_iter: int,
    init_seed: int,
    beta: float,
) -> np.ndarray:
    """Run one prGAMP reconstruction."""
    scale = np.sqrt(
        np.mean(y_obs**2)
    )
    x0 = make_random_initialization(
        x_true.shape,
        init_seed,
        scale=scale,
    )

    return run_prgamp_cdp(
        y_obs.astype(np.float32),
        masks.astype(np.complex64),
        x0=x0,
        n_iter=n_iter,
        beta=beta,
        wvar=noise_var,
        seed=init_seed,
        callback=None,
    ).astype(np.complex64)


def run_method_once(
    method: str,
    *,
    x_true: np.ndarray,
    masks: np.ndarray,
    y_obs: np.ndarray,
    noise_var: float,
    init_seed: int,
    algorithm_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> np.ndarray:
    """Dispatch one reconstruction method."""
    if method == "gpie":
        method_config = algorithm_config["gpie"]

        return run_gpie_once(
            x_true,
            masks,
            y_obs,
            noise_var,
            n_iter=method_config["n_iter"],
            init_seed=init_seed,
            device=runtime_config["device"],
            fft_engine=runtime_config.get("fft_engine"),
            schedule=method_config["schedule"],
            block_size=method_config["block_size"],
            verbose=runtime_config.get(
                "verbose",
                False,
            ),
        )

    if method == "wf":
        method_config = algorithm_config["wf"]
        si_config = method_config[
            "spectral_initialization"
        ]

        return run_wf_once(
            masks,
            y_obs,
            n_iter=method_config["n_iter"],
            init_seed=init_seed,
            si_iter=si_config["n_iter"],
            si_lamb=si_config["lamb"],
            mu_max=method_config["mu_max"],
            tau=method_config["tau"],
        )

    if method == "prvbem":
        method_config = algorithm_config["prvbem"]

        return run_prvbem_once(
            x_true,
            masks,
            y_obs,
            n_iter=method_config["n_iter"],
            init_seed=init_seed,
        )

    if method == "prgamp":
        method_config = algorithm_config["prgamp"]

        return run_prgamp_once(
            x_true,
            masks,
            y_obs,
            noise_var,
            n_iter=method_config["n_iter"],
            init_seed=init_seed,
            beta=method_config["beta"],
        )

    raise ValueError(
        f"Unknown reconstruction method: {method}"
    )


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------
def benchmark_random_cdp_masks(
    *,
    size: int,
    mask_counts: list[int],
    n_trials: int,
    noise_model: str,
    noise_param: float,
    data_seed_base: int,
    init_seed_base: int,
    output_dir: Path,
    methods: list[str],
    algorithm_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> None:
    """Run a mask-count sweep on random complex Gaussian objects."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_results: list[dict[str, Any]] = []

    for n_masks in mask_counts:
        print(
            f"\n===== masks = {n_masks} ====="
        )

        for trial in range(n_trials):
            data_seed = (
                data_seed_base
                + 1000 * n_masks
                + trial
            )
            init_seed = (
                init_seed_base
                + 1000 * n_masks
                + trial
            )

            x_true = make_random_complex_ground_truth(
                shape=(size, size),
                seed=data_seed,
            )

            masks, y_obs, noise_var = make_cdp(
                x_true,
                num_masks=n_masks,
                seed=data_seed,
                noise_model=noise_model,
                noise_param=noise_param,
            )

            masks = masks.astype(np.complex64)
            y_obs = y_obs.astype(np.float32)

            for method in methods:
                print(
                    f"[masks={n_masks:02d}, "
                    f"trial={trial:02d}] "
                    f"running {method}..."
                )

                try:
                    x_est = run_method_once(
                        method,
                        x_true=x_true,
                        masks=masks,
                        y_obs=y_obs,
                        noise_var=noise_var,
                        init_seed=init_seed,
                        algorithm_config=algorithm_config,
                        runtime_config=runtime_config,
                    )
                    error = pmse(
                        x_est,
                        x_true,
                    )
                except Exception as exc:
                    print(
                        f"  failed: {exc}"
                    )
                    error = float("nan")

                print(
                    f"  PMSE = {error:.6e}"
                )

                raw_results.append(
                    {
                        "method": method,
                        "n_masks": int(n_masks),
                        "trial": int(trial),
                        "data_seed": int(data_seed),
                        "init_seed": int(init_seed),
                        "pmse": float(error),
                    }
                )

                save_raw_results(
                    output_dir,
                    raw_results,
                )

    summary_rows: list[dict[str, Any]] = []

    for method in methods:
        for n_masks in mask_counts:
            values = [
                row["pmse"]
                for row in raw_results
                if (
                    row["method"] == method
                    and row["n_masks"] == n_masks
                )
            ]

            finite_values = np.asarray(
                values,
                dtype=np.float64,
            )
            finite_values = finite_values[
                np.isfinite(finite_values)
            ]

            if finite_values.size == 0:
                statistics = {
                    "q1": float("nan"),
                    "q2": float("nan"),
                    "q3": float("nan"),
                    "mean": float("nan"),
                    "std": float("nan"),
                }
            else:
                statistics = summarize(
                    finite_values
                )

            summary_rows.append(
                {
                    "method": method,
                    "n_masks": int(n_masks),
                    **statistics,
                }
            )

    save_summary(
        output_dir,
        summary_rows,
    )
    plot_mask_vs_pmse(
        output_dir,
        summary_rows,
        methods,
        mask_counts,
    )

    print(
        f"\nSaved benchmark results to: {output_dir}"
    )


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def run(
    *,
    config: ExperimentConfig,
    run_dir: Path,
) -> None:
    """Run the benchmark from a resolved experiment config."""
    validate_benchmark_config(config)

    data_config = config.section("data")
    forward_config = config.section(
        "forward_model"
    )
    algorithm_config = config.section(
        "algorithm"
    )
    runtime_config = config.runtime

    benchmark_random_cdp_masks(
        size=data_config["size"],
        mask_counts=list(
            data_config["mask_counts"]
        ),
        n_trials=data_config["n_trials"],
        noise_model=forward_config[
            "noise_model"
        ],
        noise_param=forward_config[
            "noise_param"
        ],
        data_seed_base=data_config[
            "data_seed_base"
        ],
        init_seed_base=data_config[
            "init_seed_base"
        ],
        output_dir=run_dir,
        methods=list(
            algorithm_config["methods"]
        ),
        algorithm_config=algorithm_config,
        runtime_config=runtime_config,
    )