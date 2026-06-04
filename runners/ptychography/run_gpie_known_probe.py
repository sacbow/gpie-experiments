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

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse

from experiments.ptychography.forward_model import forward_ptychography
from experiments.ptychography.object import make_complex_object
from experiments.ptychography.probe import make_circular_probe
from experiments.ptychography.scan import make_raster_scan_positions


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


def backend_pmse(
    x_est,
    x_true,
    *,
    margin: int = 0,
    normalize: bool = False,
) -> float:
    """Compute backend-aware PMSE, optionally on a central crop."""
    x_est_eval = crop_for_metric(x_est, margin)
    x_true_eval = crop_for_metric(x_true, margin)

    return pmse(
        x_est_eval,
        x_true_eval,
        normalize=normalize,
    )


def center_crop_cpu(
    x: np.ndarray,
    margin: int,
) -> np.ndarray:
    """Crop a NumPy image by removing a margin from all sides."""
    if margin <= 0:
        return x

    if x.ndim < 2:
        raise ValueError("Input array must have at least two dimensions.")

    if 2 * margin >= min(x.shape[-2:]):
        raise ValueError(
            f"eval_crop_margin={margin} is too large for shape {x.shape}."
        )

    return x[..., margin:-margin, margin:-margin]


def phase_align_cpu(x_est: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    """Align the global phase of x_est to x_true on CPU."""
    inner = np.vdot(x_true, x_est)
    phase = np.angle(inner)
    return x_est * np.exp(-1j * phase)


def get_object_mean_cpu(graph, obj_label: str = "object") -> np.ndarray:
    """Extract the object posterior mean after graph.run has returned to CPU."""
    mean = np.asarray(graph[obj_label]["mean"])

    if mean.ndim == 3 and mean.shape[0] == 1:
        return mean[0]

    return mean


def get_object_sample_backend(graph, obj_label: str = "object"):
    """Extract the object sample without forcing CPU conversion."""
    return graph.get_wave(obj_label).get_sample()


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    """Normalize a complex image for visualization."""
    scale = max(np.max(np.abs(x)), 1e-12)
    return x / scale


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


def set_object_initialization(
    graph,
    obj_shape: tuple[int, int],
    init_mode: str,
    obj_true: np.ndarray,
    obj_label: str = "object",
) -> None:
    """Set the object initialization strategy."""
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
        raise ValueError(f"Unknown init mode: {init_mode}")


def save_history_csv(history: dict[str, list[float]], path: Path) -> None:
    """Save convergence history to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "full_pmse", "crop_pmse"])

        for it, full_err, crop_err in zip(
            history["iteration"],
            history["full_pmse"],
            history["crop_pmse"],
        ):
            writer.writerow([it, full_err, crop_err])


def save_reconstruction_figure(
    obj_true: np.ndarray,
    obj_est: np.ndarray,
    path: Path,
    *,
    crop_margin: int,
) -> None:
    """Save true and reconstructed object amplitude and phase."""
    obj_true_disp = normalize_for_display(obj_true)
    obj_est_disp = normalize_for_display(obj_est)
    obj_est_disp = phase_align_cpu(obj_est_disp, obj_true_disp)

    obj_true_crop = center_crop_cpu(obj_true_disp, crop_margin)
    obj_est_crop = center_crop_cpu(obj_est_disp, crop_margin)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    panels = [
        (np.abs(obj_true_disp), "True amplitude"),
        (np.angle(obj_true_disp), "True phase"),
        (np.abs(obj_est_disp), "Reconstructed amplitude"),
        (np.angle(obj_est_disp), "Reconstructed phase"),
        (np.abs(obj_true_crop), "True amplitude crop"),
        (np.angle(obj_true_crop), "True phase crop"),
        (np.abs(obj_est_crop), "Reconstructed amplitude crop"),
        (np.angle(obj_est_crop), "Reconstructed phase crop"),
    ]

    for ax, (image, title) in zip(axes.ravel(), panels):
        if "phase" in title.lower():
            im = ax.imshow(
                image,
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
            )
        else:
            im = ax.imshow(
                image,
                cmap="gray",
            )

        ax.set_title(title)
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
    positions: np.ndarray,
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    path: Path,
) -> None:
    """Save scan position plot."""
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
    history: dict[str, list[float]],
    path: Path,
) -> None:
    """Save convergence curves."""
    iteration = np.asarray(history["iteration"])
    full_pmse_values = np.asarray(history["full_pmse"])
    crop_pmse_values = np.asarray(history["crop_pmse"])

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(iteration, full_pmse_values, marker="o", label="Full image")
    ax.plot(iteration, crop_pmse_values, marker="o", label="Central crop")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PMSE")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    ax.set_title("gPIE known-probe ptychography convergence")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_observed_figure(
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


def parse_damping(value: str) -> float | str:
    """Parse damping argument."""
    if value == "auto":
        return "auto"

    return float(value)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run gPIE known-probe ptychography reconstruction."
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results/ptychography/gpie_known_probe",
    )

    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--init-seed", type=int, default=111)

    parser.add_argument("--object-name", type=str, default="coins")
    parser.add_argument("--object-size", type=int, default=256)
    parser.add_argument("--phase-scale", type=float, default=np.pi / 2)
    parser.add_argument("--amplitude-min", type=float, default=0.2)

    parser.add_argument("--probe-size", type=int, default=64)
    parser.add_argument("--probe-radius", type=float, default=24.0)
    parser.add_argument("--soft-edge-width", type=float, default=0.0)
    parser.add_argument("--phase-curvature", type=float, default=4.0)

    parser.add_argument("--scan-step", type=int, default=32)
    parser.add_argument("--scan-jitter", type=int, default=1)

    parser.add_argument("--photon-scale", type=float, default=1e3)
    parser.add_argument("--damping", type=str, default="auto")

    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument(
        "--eval-crop-margin",
        type=int,
        default=-1,
        help="Margin removed from all sides for cropped PMSE. If negative, use probe_size // 2.",
    )

    parser.add_argument(
        "--init",
        type=str,
        default="default",
        choices=["ones", "true", "random", "default"],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="sequential",
        choices=["parallel", "sequential"],
    )
    parser.add_argument("--block-size", type=int, default=1)

    parser.add_argument("--save-graph", action="store_true")

    return parser


def main() -> None:
    """Run gPIE known-probe ptychography experiment."""
    args = build_parser().parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_rng = np.random.default_rng(args.data_seed)

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

    positions = make_raster_scan_positions(
        object_shape=obj_shape,
        probe_shape=probe_shape,
        step=args.scan_step,
        jitter=args.scan_jitter,
        rng=data_rng,
    )

    data = forward_ptychography(
        obj=obj_true,
        probe=probe,
        positions=positions,
        photon_scale=args.photon_scale,
        rng=data_rng,
    )

    observed = data.anscombe.astype(np.float32)
    noise_var = 1.0 / float(args.photon_scale)

    indices = positions_to_indices(
        positions=positions,
        probe_shape=probe_shape,
    )

    graph = ptychography_graph_known_probe(
        obj_shape=obj_shape,
        prb=probe,
        indices=indices,
        noise=noise_var,
        damping=parse_damping(args.damping),
    )


    graph.set_sample("object", obj_true)

    set_observation(
        graph=graph,
        observed=observed,
        noise_var=noise_var,
        meas_label="measurement",
    )

    set_object_initialization(
        graph=graph,
        obj_shape=obj_shape,
        init_mode=args.init,
        obj_true=obj_true,
        obj_label="object",
    )

    graph.set_init_rng(np.random.default_rng(args.init_seed))

    history = {
        "iteration": [],
        "full_pmse": [],
        "crop_pmse": [],
    }

    def monitor(g, t: int) -> None:
        """Monitor reconstruction error without forcing CPU conversion."""
        if t % args.log_every != 0 and t != args.n_iter - 1:
            return

        obj_est_current = g["object"]["mean"]
        obj_true_current = get_object_sample_backend(g, obj_label="object")

        full_err = backend_pmse(
            obj_est_current,
            obj_true_current,
            margin=0,
        )
        crop_err = backend_pmse(
            obj_est_current,
            obj_true_current,
            margin=eval_crop_margin,
        )

        history["iteration"].append(int(t))
        history["full_pmse"].append(float(full_err))
        history["crop_pmse"].append(float(crop_err))

        print(
            f"[iter={t:04d}] "
            f"full_PMSE={full_err:.6e} "
            f"crop_PMSE={crop_err:.6e}"
        )

    graph.run(
        n_iter=args.n_iter,
        device=args.device,
        schedule=args.schedule,
        block_size=args.block_size,
        callback=monitor,
    )

    obj_est = get_object_mean_cpu(graph, obj_label="object")

    final_full_pmse = backend_pmse(
        obj_est,
        obj_true,
        margin=0,
    )
    final_crop_pmse = backend_pmse(
        obj_est,
        obj_true,
        margin=eval_crop_margin,
    )

    metadata = {
        "data_seed": args.data_seed,
        "init_seed": args.init_seed,
        "object_name": args.object_name,
        "object_shape": obj_shape,
        "phase_scale": args.phase_scale,
        "amplitude_min": args.amplitude_min,
        "probe_shape": probe_shape,
        "probe_radius": args.probe_radius,
        "soft_edge_width": args.soft_edge_width,
        "phase_curvature": args.phase_curvature,
        "scan_step": args.scan_step,
        "scan_jitter": args.scan_jitter,
        "n_scan": int(len(positions)),
        "photon_scale": args.photon_scale,
        "measurement": "scaled_anscombe",
        "noise_var": noise_var,
        "damping": args.damping,
        "n_iter": args.n_iter,
        "log_every": args.log_every,
        "eval_crop_margin": eval_crop_margin,
        "init": args.init,
        "device": args.device,
        "schedule": args.schedule,
        "block_size": args.block_size,
        "final_full_pmse": final_full_pmse,
        "final_crop_pmse": final_crop_pmse,
    }

    np.savez_compressed(
        outdir / "gpie_known_probe_result.npz",
        obj_true=obj_true,
        obj_est=obj_est,
        probe=probe,
        positions=positions,
        observed=observed,
        amplitude=data.amplitude,
        anscombe=data.anscombe,
        counts=data.counts,
        intensity=data.intensity,
        history_iteration=np.asarray(history["iteration"]),
        history_full_pmse=np.asarray(history["full_pmse"]),
        history_crop_pmse=np.asarray(history["crop_pmse"]),
    )

    with open(outdir / "metadata.json", "w") as f:
        json.dump(to_serializable_metadata(metadata), f, indent=2)

    save_history_csv(history, outdir / "history.csv")

    save_reconstruction_figure(
        obj_true=obj_true,
        obj_est=obj_est,
        path=outdir / "reconstruction.png",
        crop_margin=eval_crop_margin,
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
        history=history,
        path=outdir / "convergence.png",
    )

    save_observed_figure(
        observed=observed,
        path=outdir / "observed_patterns.png",
    )

    if args.save_graph:
        graph_path = os.path.join(outdir, "graph.html")
        graph.visualize(output_path=graph_path)

    print(f"Final full-image PMSE = {final_full_pmse:.6e}")
    print(f"Final cropped PMSE    = {final_crop_pmse:.6e}")
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()