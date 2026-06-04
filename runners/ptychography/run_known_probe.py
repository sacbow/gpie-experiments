from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse as gpie_pmse

from experiments.ptychography.forward_model import forward_ptychography
from experiments.ptychography.object import make_complex_object
from experiments.ptychography.pie import PIE
from experiments.ptychography.probe import make_circular_probe
from experiments.ptychography.scan import (
    make_fermat_scan_positions,
    make_raster_scan_positions,
)


@model
def ptychography_graph_known_probe(
    obj_shape: tuple[int, int],
    prb: np.ndarray,
    indices: Sequence[tuple[slice, slice]],
    noise: float,
    *,
    dtype: Any = np.complex64,
    meas_label: str = "measurement",
    obj_label: str = "object",
    damping: float | str = "auto",
) -> None:
    """Build a fixed-probe ptychography graph."""
    obj = ~GaussianPrior(
        event_shape=obj_shape,
        label=obj_label,
        dtype=dtype,
    )

    patches = obj.extract_patches(indices)
    exit_waves = prb * patches

    AmplitudeMeasurement(
        var=noise,
        label=meas_label,
        damping=damping,
    ) << fft2(exit_waves)


def positions_to_indices(
    positions: np.ndarray,
    probe_shape: tuple[int, int],
) -> list[tuple[slice, slice]]:
    """Convert top-left pixel positions to slice indices."""
    ph, pw = probe_shape
    indices = []

    for y, x in positions:
        y0 = int(y)
        x0 = int(x)
        indices.append((slice(y0, y0 + ph), slice(x0, x0 + pw)))

    return indices


def parse_damping(value: str) -> float | str:
    """Parse a damping argument."""
    if value == "auto":
        return "auto"

    return float(value)


def phase_align_cpu(x_est: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    """Align the global phase of x_est to x_true on CPU."""
    inner = np.vdot(x_true, x_est)
    phase = np.angle(inner)
    return x_est * np.exp(-1j * phase)


def l2_normalize_cpu(x: np.ndarray) -> np.ndarray:
    """L2-normalize an array over all elements."""
    norm = np.linalg.norm(x)
    return x / max(norm, 1e-12)


def pmse_cpu(
    x_est: np.ndarray,
    x_true: np.ndarray,
    *,
    normalize: bool = False,
) -> float:
    """Compute phase-aligned MSE using the gPIE convention on CPU."""
    x_est = np.asarray(x_est)
    x_true = np.asarray(x_true)

    if normalize:
        x_est = l2_normalize_cpu(x_est)
        x_true = l2_normalize_cpu(x_true)

    x_est = phase_align_cpu(x_est, x_true)

    return float(np.sum(np.abs(x_est - x_true) ** 2))


def crop_for_metric(
    x,
    margin: int,
):
    """Crop the last two spatial axes without changing the array backend."""
    if margin <= 0:
        return x

    if 2 * margin >= min(x.shape[-2:]):
        raise ValueError(
            f"eval_crop_margin={margin} is too large for shape {x.shape}."
        )

    return x[..., margin:-margin, margin:-margin]


def cropped_pmse_cpu(
    x_est: np.ndarray,
    x_true: np.ndarray,
    *,
    margin: int,
    normalize: bool = True,
) -> float:
    """Compute CPU PMSE after center cropping."""
    return pmse_cpu(
        crop_for_metric(x_est, margin),
        crop_for_metric(x_true, margin),
        normalize=normalize,
    )


def backend_cropped_pmse(
    x_est,
    x_true,
    *,
    margin: int,
    normalize: bool = True,
) -> float:
    """Compute backend-aware cropped PMSE."""
    return gpie_pmse(
        crop_for_metric(x_est, margin),
        crop_for_metric(x_true, margin),
        normalize=normalize,
    )


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    """Normalize a complex image for visualization."""
    scale = max(np.max(np.abs(x)), 1e-12)
    return x / scale


def get_object_mean_cpu(graph, obj_label: str = "object") -> np.ndarray:
    """Extract the object posterior mean after graph.run has returned to CPU."""
    mean = np.asarray(graph[obj_label]["mean"])

    if mean.ndim == 3 and mean.shape[0] == 1:
        return mean[0]

    return mean


def get_object_sample_backend(graph, obj_label: str = "object"):
    """Extract the object sample without forcing CPU conversion."""
    return graph.get_wave(obj_label).get_sample()


def set_observation(
    graph,
    observed: np.ndarray,
    noise_var: float,
    meas_label: str = "measurement",
) -> None:
    """Set observed measurement data on a gPIE graph."""
    precision = 1.0 / float(noise_var)

    if hasattr(graph, "set_observation"):
        graph.set_observation(
            observed.astype(np.float32),
            precision=precision,
        )
        return

    measurement = graph.get_factor(meas_label)
    measurement.set_observed(
        observed.astype(np.float32),
        precision=precision,
    )


def set_gpie_object_initialization(
    graph,
    obj_shape: tuple[int, int],
    init_mode: str,
    obj_true: np.ndarray,
    obj_label: str = "object",
) -> None:
    """Set gPIE object initialization strategy."""
    if init_mode == "ones":
        obj_init = np.ones(obj_shape, dtype=np.complex64)
        graph.set_init_strategy(obj_label, "manual", data=obj_init)
    elif init_mode == "true":
        graph.set_init_strategy(
            obj_label,
            "manual",
            data=obj_true.astype(np.complex64),
        )
    elif init_mode == "random":
        graph.set_init_strategy(obj_label, "sample")
    elif init_mode == "default":
        return
    else:
        raise ValueError(f"Unknown gPIE init mode: {init_mode}")


def make_pie_object_initialization(
    obj_shape: tuple[int, int],
    init_mode: str,
    obj_true: np.ndarray,
) -> np.ndarray | None:
    """Create PIE object initialization."""
    if init_mode == "ones":
        return np.ones(obj_shape, dtype=np.complex64)

    if init_mode == "true":
        return obj_true.astype(np.complex64).copy()

    if init_mode in ("random", "default"):
        return None

    raise ValueError(f"Unknown PIE init mode: {init_mode}")


def make_scan_positions(
    *,
    scan_type: str,
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    step: float,
    margin: int,
    jitter: int,
    n_scans: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate deterministic center-out scan positions."""
    if scan_type == "raster":
        return make_raster_scan_positions(
            object_shape=object_shape,
            probe_shape=probe_shape,
            step=int(round(step)),
            margin=margin,
            jitter=jitter,
            rng=rng,
            n_scans=n_scans,
        )

    if scan_type == "fermat":
        return make_fermat_scan_positions(
            object_shape=object_shape,
            probe_shape=probe_shape,
            step=float(step),
            margin=margin,
            jitter=jitter,
            rng=rng,
            n_scans=n_scans,
        )

    raise ValueError(f"Unknown scan type: {scan_type}")


def compute_observation_rate(
    *,
    object_shape: tuple[int, int],
    probe: np.ndarray,
    positions: np.ndarray,
    threshold_ratio: float = 0.1,
) -> dict[str, float | int]:
    """Compute coverage and diffraction-pixel observation rate."""
    obj_h, obj_w = object_shape
    ph, pw = probe.shape

    intensity = np.abs(probe) ** 2
    threshold = threshold_ratio * float(np.max(intensity))
    active_probe = intensity >= threshold

    covered = np.zeros(object_shape, dtype=bool)

    for y, x in positions:
        y0 = int(y)
        x0 = int(x)
        covered[y0:y0 + ph, x0:x0 + pw] |= active_probe

    covered_pixels = int(np.count_nonzero(covered))
    object_pixels = int(obj_h * obj_w)
    diffraction_pixels = int(len(positions) * ph * pw)

    if covered_pixels == 0:
        observation_rate = float("inf")
    else:
        observation_rate = float(diffraction_pixels / covered_pixels)

    covered_fraction = float(covered_pixels / object_pixels)

    return {
        "covered_pixels": covered_pixels,
        "object_pixels": object_pixels,
        "diffraction_pixels": diffraction_pixels,
        "observation_rate": observation_rate,
        "covered_fraction": covered_fraction,
        "probe_threshold_ratio": float(threshold_ratio),
        "probe_active_pixels": int(np.count_nonzero(active_probe)),
    }


def make_pie_callback(
    *,
    obj_true: np.ndarray,
    n_iter: int,
    log_every: int,
    eval_crop_margin: int,
) -> tuple[dict[str, list[float]], Any]:
    """Create a PIE callback and its history dictionary."""
    history = {
        "iteration": [],
        "crop_pmse": [],
    }

    def callback(it: int, fourier_error: float, obj_est: np.ndarray) -> None:
        del fourier_error

        if it % log_every != 0 and it != n_iter - 1:
            return

        crop_err = cropped_pmse_cpu(
            obj_est,
            obj_true,
            margin=eval_crop_margin,
            normalize=True,
        )

        history["iteration"].append(int(it))
        history["crop_pmse"].append(float(crop_err))

        print(
            f"[PIE  iter={it:04d}] "
            f"crop_PMSE={crop_err:.6e}"
        )

    return history, callback


def save_history_csv(
    *,
    pie_history: dict[str, list[float]],
    gpie_history: dict[str, list[float]],
    path: Path,
) -> None:
    """Save PIE and gPIE convergence histories to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "iteration", "crop_pmse"])

        for it, crop_err in zip(
            pie_history["iteration"],
            pie_history["crop_pmse"],
        ):
            writer.writerow(["pie", it, crop_err])

        for it, crop_err in zip(
            gpie_history["iteration"],
            gpie_history["crop_pmse"],
        ):
            writer.writerow(["gpie", it, crop_err])


def save_complex_image_figure(
    *,
    image: np.ndarray,
    path: Path,
    title: str,
    crop_margin: int,
    reference: np.ndarray | None = None,
) -> None:
    """Save one complex image as amplitude and phase panels."""
    image_disp = normalize_for_display(image)

    if reference is not None:
        ref_disp = normalize_for_display(reference)
        image_disp = phase_align_cpu(image_disp, ref_disp)

    image_crop = crop_for_metric(image_disp, crop_margin)

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    panels = [
        (np.abs(image_disp), f"{title} amplitude"),
        (np.angle(image_disp), f"{title} phase"),
        (np.abs(image_crop), f"{title} amplitude crop"),
        (np.angle(image_crop), f"{title} phase crop"),
    ]

    for ax, (panel, panel_title) in zip(axes.ravel(), panels):
        if "phase" in panel_title.lower():
            im = ax.imshow(
                panel,
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
            )
        else:
            im = ax.imshow(panel, cmap="gray")

        ax.set_title(panel_title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_probe_figure(probe: np.ndarray, path: Path) -> None:
    """Save probe amplitude and phase."""
    probe_disp = normalize_for_display(probe)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(
        np.abs(probe_disp),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Probe amplitude")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        np.angle(probe_disp),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1].set_title("Probe phase")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_scan_positions_figure(
    *,
    positions: np.ndarray,
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    path: Path,
) -> None:
    """Save scan positions."""
    fig, ax = plt.subplots(figsize=(6, 6))

    y = positions[:, 0]
    x = positions[:, 1]

    ax.scatter(x, y, s=8)
    ax.set_xlim(0, object_shape[1] - probe_shape[1])
    ax.set_ylim(object_shape[0] - probe_shape[0], 0)
    ax.set_aspect("equal")
    ax.set_title("Scan positions")
    ax.set_xlabel("x (top-left pixel)")
    ax.set_ylabel("y (top-left pixel)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_convergence_figure(
    *,
    pie_history: dict[str, list[float]],
    gpie_history: dict[str, list[float]],
    path: Path,
) -> None:
    """Save cropped PMSE convergence curves for PIE and gPIE."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        np.asarray(pie_history["iteration"]),
        np.asarray(pie_history["crop_pmse"]),
        marker="o",
        label="PIE",
    )
    ax.plot(
        np.asarray(gpie_history["iteration"]),
        np.asarray(gpie_history["crop_pmse"]),
        marker="o",
        label="gPIE",
    )

    ax.set_title("Cropped PMSE")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PMSE")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_observed_figure(
    *,
    observed: np.ndarray,
    path: Path,
    max_patterns: int = 16,
) -> None:
    """Save a grid of observed diffraction patterns."""
    n = min(len(observed), max_patterns)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
    )
    axes = np.asarray(axes).reshape(-1)

    for i in range(n):
        axes[i].imshow(observed[i], cmap="gray")
        axes[i].set_title(f"Pattern {i}")
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def to_serializable_metadata(metadata: dict) -> dict:
    """Convert metadata values to JSON-serializable types."""
    converted = {}

    for key, value in metadata.items():
        if isinstance(value, tuple):
            converted[key] = list(value)
        elif isinstance(value, np.generic):
            converted[key] = value.item()
        else:
            converted[key] = value

    return converted


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run common-data known-probe ptychography with PIE and gPIE."
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results/ptychography/known_probe",
    )

    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--pie-seed", type=int, default=0)
    parser.add_argument("--gpie-init-seed", type=int, default=111)

    parser.add_argument("--object-name", type=str, default="coins")
    parser.add_argument("--object-size", type=int, default=256)
    parser.add_argument("--phase-scale", type=float, default=np.pi / 2)
    parser.add_argument("--amplitude-min", type=float, default=0.2)

    parser.add_argument("--probe-size", type=int, default=64)
    parser.add_argument("--probe-radius", type=float, default=30.0)
    parser.add_argument("--soft-edge-width", type=float, default=2.0)
    parser.add_argument("--phase-curvature", type=float, default=4.0)

    parser.add_argument(
        "--scan-type",
        type=str,
        default="fermat",
        choices=["fermat", "raster"],
    )
    parser.add_argument("--scan-step", type=float, default=20.0)
    parser.add_argument("--scan-margin", type=int, default=0)
    parser.add_argument("--scan-jitter", type=int, default=2)
    parser.add_argument(
        "--n-scans",
        type=int,
        default=None,
        help="Number of center-out scan positions.",
    )

    parser.add_argument("--photon-scale", type=float, default=1e3)
    parser.add_argument(
        "--probe-threshold-ratio",
        type=float,
        default=0.1,
        help="Probe intensity threshold ratio used for coverage computation.",
    )

    parser.add_argument("--pie-n-iter", type=int, default=2000)
    parser.add_argument("--gpie-n-iter", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--pie-beta-object", type=float, default=0.1)
    parser.add_argument("--pie-alpha", type=float, default=0.1)
    parser.add_argument(
        "--pie-init",
        type=str,
        default="ones",
        choices=["ones", "true", "random", "default"],
    )

    parser.add_argument(
        "--gpie-init",
        type=str,
        default="default",
        choices=["ones", "true", "random", "default"],
    )
    parser.add_argument("--gpie-damping", type=str, default=0.5)
    parser.add_argument(
        "--gpie-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--gpie-schedule",
        type=str,
        default="sequential",
        choices=["parallel", "sequential"],
    )
    parser.add_argument("--gpie-block-size", type=int, default=1)
    parser.add_argument("--save-graph", action="store_true")

    parser.add_argument(
        "--eval-crop-margin",
        type=int,
        default=-1,
        help="Margin removed from all sides for cropped PMSE. If negative, use probe_size // 2.",
    )

    return parser


def main() -> None:
    """Run common-data known-probe PIE and gPIE experiments."""
    args = build_parser().parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.data_seed)

    obj_shape = (args.object_size, args.object_size)
    probe_shape = (args.probe_size, args.probe_size)

    eval_crop_margin = args.eval_crop_margin
    if eval_crop_margin < 0:
        eval_crop_margin = args.probe_size // 2

    obj_true = make_complex_object(
        name=args.object_name,
        shape=obj_shape,
        phase_scale=args.phase_scale,
        amplitude_min=args.amplitude_min,
    ).astype(np.complex64)

    probe = make_circular_probe(
        shape=probe_shape,
        radius=args.probe_radius,
        soft_edge_width=args.soft_edge_width,
        phase_curvature=args.phase_curvature,
    ).astype(np.complex64)

    positions = make_scan_positions(
        scan_type=args.scan_type,
        object_shape=obj_shape,
        probe_shape=probe_shape,
        step=args.scan_step,
        margin=args.scan_margin,
        jitter=args.scan_jitter,
        n_scans=args.n_scans,
        rng=rng,
    )

    data = forward_ptychography(
        obj=obj_true,
        probe=probe,
        positions=positions,
        photon_scale=args.photon_scale,
        rng=rng,
    )

    observed = data.anscombe.astype(np.float32)
    noise_var = 1.0 / float(args.photon_scale)

    observation_stats = compute_observation_rate(
        object_shape=obj_shape,
        probe=probe,
        positions=positions,
        threshold_ratio=args.probe_threshold_ratio,
    )

    print("Observation statistics:")
    print(f"  covered pixels:      {observation_stats['covered_pixels']}")
    print(f"  object pixels:       {observation_stats['object_pixels']}")
    print(f"  diffraction pixels:  {observation_stats['diffraction_pixels']}")
    print(f"  covered fraction:    {observation_stats['covered_fraction']:.6e}")
    print(f"  observation rate:    {observation_stats['observation_rate']:.6e}")

    # ------------------------------------------------------------
    # Run PIE
    # ------------------------------------------------------------
    pie_history, pie_callback = make_pie_callback(
        obj_true=obj_true,
        n_iter=args.pie_n_iter,
        log_every=args.log_every,
        eval_crop_margin=eval_crop_margin,
    )

    pie_obj_init = make_pie_object_initialization(
        obj_shape=obj_shape,
        init_mode=args.pie_init,
        obj_true=obj_true,
    )

    pie_solver = PIE(
        obj_shape=obj_shape,
        probe=probe,
        obj_init=pie_obj_init,
        positions=positions,
        measured_modulus=observed,
        beta_object=args.pie_beta_object,
        alpha=args.pie_alpha,
        callback=pie_callback,
        seed=args.pie_seed,
    )

    obj_pie = pie_solver.run(n_iter=args.pie_n_iter)

    pie_final_crop_pmse = cropped_pmse_cpu(
        obj_pie,
        obj_true,
        margin=eval_crop_margin,
        normalize=True,
    )

    # ------------------------------------------------------------
    # Run gPIE
    # ------------------------------------------------------------
    indices = positions_to_indices(
        positions=positions,
        probe_shape=probe_shape,
    )

    graph = ptychography_graph_known_probe(
        obj_shape=obj_shape,
        prb=probe,
        indices=indices,
        noise=noise_var,
        damping=parse_damping(args.gpie_damping),
    )

    graph.set_sample("object", obj_true)

    set_observation(
        graph=graph,
        observed=observed,
        noise_var=noise_var,
        meas_label="measurement",
    )

    set_gpie_object_initialization(
        graph=graph,
        obj_shape=obj_shape,
        init_mode=args.gpie_init,
        obj_true=obj_true,
        obj_label="object",
    )

    graph.set_init_rng(np.random.default_rng(args.gpie_init_seed))

    gpie_history = {
        "iteration": [],
        "crop_pmse": [],
    }

    def gpie_monitor(g, t: int) -> None:
        """Monitor gPIE without forcing CPU conversion."""
        if t % args.log_every != 0 and t != args.gpie_n_iter - 1:
            return

        obj_est_current = g["object"]["mean"]
        obj_true_current = get_object_sample_backend(g, obj_label="object")

        crop_err = backend_cropped_pmse(
            obj_est_current,
            obj_true_current,
            margin=eval_crop_margin,
            normalize=True,
        )

        gpie_history["iteration"].append(int(t))
        gpie_history["crop_pmse"].append(float(crop_err))

        print(
            f"[gPIE iter={t:04d}] "
            f"crop_PMSE={crop_err:.6e}"
        )

    graph.run(
        n_iter=args.gpie_n_iter,
        device=args.gpie_device,
        schedule=args.gpie_schedule,
        block_size=args.gpie_block_size,
        callback=gpie_monitor,
    )

    obj_gpie = get_object_mean_cpu(graph, obj_label="object")

    gpie_final_crop_pmse = cropped_pmse_cpu(
        obj_gpie,
        obj_true,
        margin=eval_crop_margin,
        normalize=True,
    )

    metadata = {
        "data_seed": args.data_seed,
        "pie_seed": args.pie_seed,
        "gpie_init_seed": args.gpie_init_seed,
        "object_name": args.object_name,
        "object_shape": obj_shape,
        "phase_scale": args.phase_scale,
        "amplitude_min": args.amplitude_min,
        "probe_shape": probe_shape,
        "probe_radius": args.probe_radius,
        "soft_edge_width": args.soft_edge_width,
        "phase_curvature": args.phase_curvature,
        "scan_type": args.scan_type,
        "scan_step": args.scan_step,
        "scan_margin": args.scan_margin,
        "scan_jitter": args.scan_jitter,
        "n_scans": int(len(positions)),
        "photon_scale": args.photon_scale,
        "measurement": "scaled_anscombe",
        "noise_var": noise_var,
        "probe_threshold_ratio": args.probe_threshold_ratio,
        "eval_crop_margin": eval_crop_margin,
        "log_every": args.log_every,
        "pie_n_iter": args.pie_n_iter,
        "pie_beta_object": args.pie_beta_object,
        "pie_alpha": args.pie_alpha,
        "pie_init": args.pie_init,
        "pie_final_crop_pmse": pie_final_crop_pmse,
        "gpie_n_iter": args.gpie_n_iter,
        "gpie_init": args.gpie_init,
        "gpie_damping": args.gpie_damping,
        "gpie_device": args.gpie_device,
        "gpie_schedule": args.gpie_schedule,
        "gpie_block_size": args.gpie_block_size,
        "gpie_final_crop_pmse": gpie_final_crop_pmse,
        **observation_stats,
    }

    np.savez_compressed(
        outdir / "known_probe_result.npz",
        obj_true=obj_true,
        obj_pie=obj_pie,
        obj_gpie=obj_gpie,
        probe=probe,
        positions=positions,
        observed=observed,
        amplitude=data.amplitude,
        anscombe=data.anscombe,
        counts=data.counts,
        intensity=data.intensity,
        pie_history_iteration=np.asarray(pie_history["iteration"]),
        pie_history_crop_pmse=np.asarray(pie_history["crop_pmse"]),
        gpie_history_iteration=np.asarray(gpie_history["iteration"]),
        gpie_history_crop_pmse=np.asarray(gpie_history["crop_pmse"]),
    )

    with open(outdir / "metadata.json", "w") as f:
        json.dump(to_serializable_metadata(metadata), f, indent=2)

    save_history_csv(
        pie_history=pie_history,
        gpie_history=gpie_history,
        path=outdir / "history.csv",
    )

    save_complex_image_figure(
        image=obj_true,
        path=outdir / "ground_truth.png",
        title="Ground truth",
        crop_margin=eval_crop_margin,
        reference=None,
    )

    save_complex_image_figure(
        image=obj_pie,
        path=outdir / "pie_reconstruction.png",
        title="PIE",
        crop_margin=eval_crop_margin,
        reference=obj_true,
    )

    save_complex_image_figure(
        image=obj_gpie,
        path=outdir / "gpie_reconstruction.png",
        title="gPIE",
        crop_margin=eval_crop_margin,
        reference=obj_true,
    )

    save_probe_figure(
        probe=probe,
        path=outdir / "probe.png",
    )

    save_scan_positions_figure(
        positions=positions,
        object_shape=obj_shape,
        probe_shape=probe_shape,
        path=outdir / "scan_positions.png",
    )

    save_convergence_figure(
        pie_history=pie_history,
        gpie_history=gpie_history,
        path=outdir / "convergence.png",
    )

    save_observed_figure(
        observed=observed,
        path=outdir / "observed_patterns.png",
    )

    if args.save_graph:
        graph_path = os.path.join(outdir, "graph.html")
        graph.visualize(output_path=graph_path)

    print("Final results:")
    print(f"  PIE  crop PMSE: {pie_final_crop_pmse:.6e}")
    print(f"  gPIE crop PMSE: {gpie_final_crop_pmse:.6e}")
    print(f"  observation rate: {observation_stats['observation_rate']:.6e}")
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()