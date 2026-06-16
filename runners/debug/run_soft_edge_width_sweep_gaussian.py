from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from gpie import GaussianMeasurement, GaussianPrior, fft2, model, pmse

from experiments.ptychography.object import make_complex_object
from experiments.ptychography.probe import make_circular_probe
from experiments.ptychography.scan import make_raster_scan_positions


@model
def ptychography_graph_known_probe_gaussian(
    obj_shape: tuple[int, int],
    prb: np.ndarray,
    indices: Sequence[tuple[slice, slice]],
    noise: float,
    *,
    dtype: Any = np.complex64,
    meas_label: str = "measurement",
    obj_label: str = "object",
) -> None:
    """Build a fixed-probe ptychography graph with complex Gaussian measurements."""
    obj = ~GaussianPrior(
        event_shape=obj_shape,
        label=obj_label,
        dtype=dtype,
    )

    patches = obj.extract_patches(indices)
    exit_waves = prb * patches
    diffraction = fft2(exit_waves)

    GaussianMeasurement(
        var=noise,
        label=meas_label,
    ) << diffraction


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    values = []

    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))

    if not values:
        raise ValueError("At least one soft edge width must be provided.")

    return values


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


def crop_for_metric(x, margin: int):
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
    normalize: bool = True,
) -> float:
    """Compute backend-aware phase-aligned MSE."""
    x_est_eval = crop_for_metric(x_est, margin)
    x_true_eval = crop_for_metric(x_true, margin)

    return pmse(
        x_est_eval,
        x_true_eval,
        normalize=normalize,
    )


def centered_fft2_numpy(x: np.ndarray) -> np.ndarray:
    """Apply centered orthonormal 2D FFT over the last two axes using NumPy."""
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(x, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def make_complex_gaussian_noise(
    shape: tuple[int, ...],
    var: float,
    rng: np.random.Generator,
    *,
    dtype: Any = np.complex64,
) -> np.ndarray:
    """Generate circular complex Gaussian noise with E[|noise|^2] = var."""
    real = rng.normal(size=shape)
    imag = rng.normal(size=shape)
    noise = np.sqrt(var / 2.0) * (real + 1j * imag)
    return noise.astype(dtype)


def generate_gaussian_observation(
    obj: np.ndarray,
    probe: np.ndarray,
    indices: Sequence[tuple[slice, slice]],
    noise_var: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate complex Gaussian observations for fixed-probe ptychography."""
    patches = np.stack([obj[idx] for idx in indices], axis=0).astype(np.complex64)
    exit_waves = probe[None, :, :] * patches
    diffraction = centered_fft2_numpy(exit_waves).astype(np.complex64)

    noise = make_complex_gaussian_noise(
        diffraction.shape,
        noise_var,
        rng,
        dtype=np.complex64,
    )

    return (diffraction + noise).astype(np.complex64)


def get_object_sample_backend(graph, obj_label: str = "object"):
    """Extract the object sample without forcing CPU conversion."""
    return graph.get_wave(obj_label).get_sample()


def set_observation(
    graph,
    observed: np.ndarray,
    noise_var: float,
    meas_label: str = "measurement",
) -> None:
    """Set observed complex Gaussian measurement data on a gPIE graph."""
    precision = 1.0 / float(noise_var)

    if hasattr(graph, "set_observation"):
        graph.set_observation(
            observed.astype(np.complex64),
            precision=precision,
        )
        return

    measurement = graph.get_factor(meas_label)
    measurement.set_observed(
        observed.astype(np.complex64),
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


def summarize_probe(probe: np.ndarray) -> dict[str, float]:
    """Summarize the probe amplitude distribution."""
    amp = np.abs(probe).reshape(-1)

    return {
        "amp_min": float(np.min(amp)),
        "amp_p001": float(np.quantile(amp, 0.001)),
        "amp_p01": float(np.quantile(amp, 0.01)),
        "amp_p05": float(np.quantile(amp, 0.05)),
        "amp_median": float(np.median(amp)),
        "amp_p95": float(np.quantile(amp, 0.95)),
        "amp_max": float(np.max(amp)),
        "frac_lt_1e_4": float(np.mean(amp < 1e-4)),
        "frac_lt_1e_3": float(np.mean(amp < 1e-3)),
        "frac_lt_1e_2": float(np.mean(amp < 1e-2)),
    }


def format_probe_summary(summary: dict[str, float]) -> str:
    """Format probe diagnostics for one-line logging."""
    return (
        f"amp_min={summary['amp_min']:.3e} "
        f"amp_p001={summary['amp_p001']:.3e} "
        f"amp_p01={summary['amp_p01']:.3e} "
        f"amp_p05={summary['amp_p05']:.3e} "
        f"amp_median={summary['amp_median']:.3e} "
        f"amp_p95={summary['amp_p95']:.3e} "
        f"amp_max={summary['amp_max']:.3e} "
        f"frac_lt_1e-4={summary['frac_lt_1e_4']:.3e} "
        f"frac_lt_1e-3={summary['frac_lt_1e_3']:.3e} "
        f"frac_lt_1e-2={summary['frac_lt_1e_2']:.3e}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Sweep soft-edge widths for known-probe Gaussian ptychography."
    )

    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--obs-seed", type=int, default=1234)
    parser.add_argument("--init-seed", type=int, default=111)

    parser.add_argument("--object-name", type=str, default="coins")
    parser.add_argument("--object-size", type=int, default=256)
    parser.add_argument("--phase-scale", type=float, default=np.pi / 2)
    parser.add_argument("--amplitude-min", type=float, default=0.2)

    parser.add_argument("--probe-size", type=int, default=64)
    parser.add_argument("--probe-radius", type=float, default=24.0)
    parser.add_argument("--phase-curvature", type=float, default=4.0)

    parser.add_argument("--scan-step", type=int, default=32)
    parser.add_argument("--scan-jitter", type=int, default=1)

    parser.add_argument(
        "--noise-var",
        type=float,
        default=1e-3,
        help="Complex Gaussian observation noise variance.",
    )

    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument(
        "--soft-edge-widths",
        type=str,
        default="0.0,1.0,2.0,4.0,8.0",
        help="Comma-separated soft edge widths.",
    )

    parser.add_argument(
        "--eval-crop-margin",
        type=int,
        default=-1,
        help="If negative, use probe_size // 2.",
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

    return parser


def run_one_width(
    args: argparse.Namespace,
    *,
    soft_edge_width: float,
    obj_true: np.ndarray,
    positions: np.ndarray,
    indices: Sequence[tuple[slice, slice]],
    eval_crop_margin: int,
) -> tuple[float, float]:
    """Run one reconstruction for a given soft-edge width."""
    obj_shape = (args.object_size, args.object_size)
    probe_shape = (args.probe_size, args.probe_size)

    probe = make_circular_probe(
        shape=probe_shape,
        radius=args.probe_radius,
        soft_edge_width=soft_edge_width,
        phase_curvature=args.phase_curvature,
    ).astype(np.complex64)

    probe_summary = summarize_probe(probe)

    print("")
    print("=" * 100)
    print(f"[soft_edge_width={soft_edge_width:.6g}]")
    print(format_probe_summary(probe_summary))

    obs_rng = np.random.default_rng(args.obs_seed)

    observed = generate_gaussian_observation(
        obj=obj_true,
        probe=probe,
        indices=indices,
        noise_var=args.noise_var,
        rng=obs_rng,
    )

    graph = ptychography_graph_known_probe_gaussian(
        obj_shape=obj_shape,
        prb=probe,
        indices=indices,
        noise=args.noise_var,
    )

    graph.set_sample("object", obj_true)

    set_observation(
        graph=graph,
        observed=observed,
        noise_var=args.noise_var,
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

    last_full_pmse = float("nan")
    last_crop_pmse = float("nan")

    def monitor(g, t: int) -> None:
        """Monitor reconstruction error without forcing CPU conversion."""
        nonlocal last_full_pmse, last_crop_pmse

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

        last_full_pmse = float(full_err)
        last_crop_pmse = float(crop_err)

        print(
            f"[soft_edge_width={soft_edge_width:.6g}] "
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

    print(
        f"[soft_edge_width={soft_edge_width:.6g}] "
        f"final_full_PMSE={last_full_pmse:.6e} "
        f"final_crop_PMSE={last_crop_pmse:.6e}"
    )

    return last_full_pmse, last_crop_pmse


def main() -> None:
    """Run the soft-edge-width sweep."""
    args = build_parser().parse_args()

    soft_edge_widths = parse_float_list(args.soft_edge_widths)

    obj_shape = (args.object_size, args.object_size)
    probe_shape = (args.probe_size, args.probe_size)

    eval_crop_margin = args.eval_crop_margin
    if eval_crop_margin < 0:
        eval_crop_margin = args.probe_size // 2

    data_rng = np.random.default_rng(args.data_seed)

    obj_true = make_complex_object(
        name=args.object_name,
        shape=obj_shape,
        phase_scale=args.phase_scale,
        amplitude_min=args.amplitude_min,
    ).astype(np.complex64)

    positions = make_raster_scan_positions(
        object_shape=obj_shape,
        probe_shape=probe_shape,
        step=args.scan_step,
        jitter=args.scan_jitter,
        rng=data_rng,
    )

    indices = positions_to_indices(
        positions=positions,
        probe_shape=probe_shape,
    )

    print("Soft-edge-width sweep with complex Gaussian measurements")
    print(f"object_shape={obj_shape}")
    print(f"probe_shape={probe_shape}")
    print(f"n_scan={len(positions)}")
    print(f"soft_edge_widths={soft_edge_widths}")
    print(f"noise_var={args.noise_var:.3e}")
    print(f"n_iter={args.n_iter}")
    print(f"log_every={args.log_every}")
    print(f"device={args.device}")
    print(f"schedule={args.schedule}")
    print(f"block_size={args.block_size}")
    print(f"init={args.init}")
    print(f"eval_crop_margin={eval_crop_margin}")

    results = []

    for soft_edge_width in soft_edge_widths:
        final_full_pmse, final_crop_pmse = run_one_width(
            args,
            soft_edge_width=soft_edge_width,
            obj_true=obj_true,
            positions=positions,
            indices=indices,
            eval_crop_margin=eval_crop_margin,
        )

        results.append(
            {
                "soft_edge_width": soft_edge_width,
                "final_full_pmse": final_full_pmse,
                "final_crop_pmse": final_crop_pmse,
            }
        )

    print("")
    print("=" * 100)
    print("Summary")
    print("soft_edge_width,final_full_PMSE,final_crop_PMSE")

    for result in results:
        print(
            f"{result['soft_edge_width']:.6g},"
            f"{result['final_full_pmse']:.6e},"
            f"{result['final_crop_pmse']:.6e}"
        )


if __name__ == "__main__":
    main()