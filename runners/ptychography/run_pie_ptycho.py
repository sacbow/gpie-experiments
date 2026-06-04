from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ptychography.forward_model import forward_ptychography
from experiments.ptychography.object import make_complex_object
from experiments.ptychography.pie import PIE
from experiments.ptychography.probe import make_circular_probe
from experiments.ptychography.scan import make_raster_scan_positions


def phase_align(x_est: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    """Align the global phase of x_est to x_true."""
    inner = np.vdot(x_true, x_est)
    phase = np.angle(inner)
    return x_est * np.exp(-1j * phase)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize an array over all elements."""
    norm = np.linalg.norm(x)
    return x / max(norm, 1e-12)


def pmse(
    x_est: np.ndarray,
    x_true: np.ndarray,
    *,
    normalize: bool = False,
) -> float:
    """Compute phase-aligned MSE using the gPIE convention."""
    x_est = np.asarray(x_est)
    x_true = np.asarray(x_true)

    if normalize:
        x_est = l2_normalize(x_est)
        x_true = l2_normalize(x_true)

    x_est = phase_align(x_est, x_true)

    return float(np.mean(np.abs(x_est - x_true) ** 2))


def center_crop(
    x: np.ndarray,
    margin: int,
) -> np.ndarray:
    """Crop image by removing a margin from all sides."""
    if margin <= 0:
        return x

    if x.ndim < 2:
        raise ValueError("Input array must have at least two dimensions.")

    if 2 * margin >= min(x.shape[-2:]):
        raise ValueError(
            f"eval_crop_margin={margin} is too large for shape {x.shape}."
        )

    return x[..., margin:-margin, margin:-margin]


def cropped_pmse(
    x_est: np.ndarray,
    x_true: np.ndarray,
    *,
    margin: int,
    normalize: bool = True,
) -> float:
    """Compute gPIE-style PMSE after center cropping."""
    return pmse(
        center_crop(x_est, margin),
        center_crop(x_true, margin),
        normalize=normalize,
    )


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    """Normalize a complex image for visualization."""
    scale = max(np.max(np.abs(x)), 1e-12)
    return x / scale


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, default="results/ptychography/pie")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--object-name", type=str, default="coins")
    parser.add_argument("--object-size", type=int, default=256)
    parser.add_argument("--phase-scale", type=float, default=np.pi/2)
    parser.add_argument("--amplitude-min", type=float, default=0.2)

    parser.add_argument("--probe-size", type=int, default=64)
    parser.add_argument("--probe-radius", type=float, default=24.0)
    parser.add_argument("--soft-edge-width", type=float, default=0.0)
    parser.add_argument("--phase-curvature", type=float, default=4.0)

    parser.add_argument("--scan-step", type=int, default=32)
    parser.add_argument("--scan-jitter", type=int, default=1)

    parser.add_argument("--photon-scale", type=float, default=1e3)
    parser.add_argument(
        "--measurement",
        type=str,
        default="anscombe",
        choices=["amplitude", "anscombe"],
    )

    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--beta-object", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument(
        "--eval-crop-margin",
        type=int,
        default=-1,
        help="Margin removed from all sides for cropped PMSE. If negative, use probe_size // 2.",
    )

    return parser


def make_callback(
    obj_true: np.ndarray,
    log_every: int,
    eval_crop_margin: int,
):
    """Create a callback that stores reconstruction history."""
    history = {
        "iteration": [],
        "fourier_error": [],
        "full_pmse": [],
        "crop_pmse": [],
    }

    def callback(it: int, fourier_error: float, obj_est: np.ndarray) -> None:
        full_err = pmse(
            obj_est,
            obj_true,
            normalize=False,
        )
        crop_err = cropped_pmse(
            obj_est,
            obj_true,
            margin=eval_crop_margin,
            normalize=False,
        )

        history["iteration"].append(int(it))
        history["fourier_error"].append(float(fourier_error))
        history["full_pmse"].append(float(full_err))
        history["crop_pmse"].append(float(crop_err))

        if (it % log_every) == 0:
            print(
                f"it={it:04d} "
                f"fourier_error={fourier_error:.6e} "
                f"full_PMSE={full_err:.6e} "
                f"crop_PMSE={crop_err:.6e}"
            )

    callback.history = history
    return callback


def save_history_csv(history: dict[str, list[float]], path: Path) -> None:
    """Save iteration history to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "fourier_error", "full_pmse", "crop_pmse"])

        for it, ferr, full_err, crop_err in zip(
            history["iteration"],
            history["fourier_error"],
            history["full_pmse"],
            history["crop_pmse"],
        ):
            writer.writerow([it, ferr, full_err, crop_err])


def save_probe_figure(probe: np.ndarray, path: Path) -> None:
    """Save a figure of the probe amplitude and phase."""
    probe_disp = normalize_for_display(probe)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(np.abs(probe_disp), cmap="gray", vmin=0.0, vmax=1.0)
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
    """Save a figure of scan positions."""
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


def save_reconstruction_figure(
    obj_true: np.ndarray,
    obj_est: np.ndarray,
    path: Path,
    *,
    crop_margin: int,
) -> None:
    """Save amplitude and phase images of the true and reconstructed objects."""
    obj_true_disp = normalize_for_display(obj_true)
    obj_est_disp = normalize_for_display(obj_est)
    obj_est_disp = phase_align(obj_est_disp, obj_true_disp)

    obj_true_crop = center_crop(obj_true_disp, crop_margin)
    obj_est_crop = center_crop(obj_est_disp, crop_margin)

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


def save_convergence_figure(history: dict[str, list[float]], path: Path) -> None:
    """Save convergence curves."""
    iteration = np.asarray(history["iteration"])
    fourier_error = np.asarray(history["fourier_error"])
    full_pmse_values = np.asarray(history["full_pmse"])
    crop_pmse_values = np.asarray(history["crop_pmse"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(iteration, fourier_error)
    axes[0].set_title("Fourier error")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Error")
    axes[0].set_yscale("log")

    axes[1].plot(iteration, full_pmse_values, label="Full image")
    axes[1].plot(iteration, crop_pmse_values, label="Central crop")
    axes[1].set_title("PMSE")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("PMSE")
    axes[1].set_yscale("log")
    axes[1].legend()

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


def main() -> None:
    """Run a fixed-probe PIE ptychography experiment."""
    args = build_parser().parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

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
        rng=rng,
    )

    data = forward_ptychography(
        obj=obj_true,
        probe=probe,
        positions=positions,
        photon_scale=args.photon_scale,
        rng=rng,
    )

    if args.measurement == "amplitude":
        measured = data.amplitude
    elif args.measurement == "anscombe":
        measured = data.anscombe
    else:
        raise ValueError(f"Unknown measurement mode: {args.measurement}")

    callback = make_callback(
        obj_true=obj_true,
        log_every=args.log_every,
        eval_crop_margin=eval_crop_margin,
    )

    obj_init = np.ones(obj_shape, dtype=np.complex64)

    solver = PIE(
        obj_shape=obj_shape,
        probe=probe,
        obj_init=obj_init,
        positions=positions,
        measured_modulus=measured,
        beta_object=args.beta_object,
        alpha=args.alpha,
        callback=callback,
        seed=args.seed,
    )

    obj_est = solver.run(n_iter=args.n_iter)

    final_full_pmse = pmse(
        obj_est,
        obj_true,
        normalize=False,
    )
    final_crop_pmse = cropped_pmse(
        obj_est,
        obj_true,
        margin=eval_crop_margin,
        normalize=False,
    )

    history = callback.history

    metadata = {
        "seed": args.seed,
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
        "measurement": args.measurement,
        "n_iter": args.n_iter,
        "beta_object": args.beta_object,
        "alpha": args.alpha,
        "eval_crop_margin": eval_crop_margin,
        "final_full_pmse": final_full_pmse,
        "final_crop_pmse": final_crop_pmse,
    }

    np.savez_compressed(
        outdir / "pie_ptycho_result.npz",
        obj_true=obj_true,
        obj_est=obj_est,
        probe=probe,
        positions=positions,
        measured=measured,
        amplitude=data.amplitude,
        anscombe=data.anscombe,
        counts=data.counts,
        intensity=data.intensity,
        history_iteration=np.asarray(history["iteration"]),
        history_fourier_error=np.asarray(history["fourier_error"]),
        history_full_pmse=np.asarray(history["full_pmse"]),
        history_crop_pmse=np.asarray(history["crop_pmse"]),
    )

    with open(outdir / "metadata.json", "w") as f:
        json.dump(to_serializable_metadata(metadata), f, indent=2)

    save_history_csv(history, outdir / "history.csv")
    save_probe_figure(probe, outdir / "probe.png")
    save_scan_positions_figure(
        positions=positions,
        object_shape=obj_shape,
        probe_shape=probe_shape,
        path=outdir / "scan_positions.png",
    )
    save_reconstruction_figure(
        obj_true=obj_true,
        obj_est=obj_est,
        path=outdir / "reconstruction.png",
        crop_margin=eval_crop_margin,
    )
    save_convergence_figure(
        history=history,
        path=outdir / "convergence.png",
    )

    print(f"Saved results to: {outdir}")
    print(f"Final full-image PMSE: {final_full_pmse:.6e}")
    print(f"Final cropped PMSE:    {final_crop_pmse:.6e}")


if __name__ == "__main__":
    main()