from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from gpie import AmplitudeMeasurement, GaussianPrior, fft2, model, pmse

from experiments.ptychography.forward_model import forward_ptychography
from experiments.ptychography.object import make_complex_object
from experiments.ptychography.probe import make_circular_probe
from experiments.ptychography.scan import make_raster_scan_positions


@model
def ptychography_graph_known_probe_extrinsic_policy(
    obj_shape: tuple[int, int],
    prb: np.ndarray,
    indices: Sequence[tuple[slice, slice]],
    noise: float,
    *,
    dtype: Any = np.complex64,
    meas_label: str = "measurement",
    obj_label: str = "object",
    damping: float | str = "auto",
    amplitude_eps: float = 1e-12,
    extrinsic_policy: str = "standard",
    weak_precision: float = 1.0,
    min_precision_diff_rel: float = 0.0,
) -> None:
    """Build a fixed-probe ptychography graph with an extrinsic-message policy."""
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
        eps=amplitude_eps,
        extrinsic_policy=extrinsic_policy,
        weak_precision=weak_precision,
        min_precision_diff_rel=min_precision_diff_rel,
    ) << fft2(exit_waves)


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    values = []

    for token in value.split(","):
        token = token.strip()
        if token:
            values.append(float(token))

    if not values:
        raise ValueError("At least one float value must be provided.")

    return values


def parse_str_list(value: str) -> list[str]:
    """Parse a comma-separated list of strings."""
    values = []

    for token in value.split(","):
        token = token.strip()
        if token:
            values.append(token)

    if not values:
        raise ValueError("At least one string value must be provided.")

    return values


def parse_damping(value: str) -> float | str:
    """Parse damping argument."""
    if value == "auto":
        return "auto"

    return float(value)


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
        description="Sweep AmplitudeMeasurement extrinsic-message policies."
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

    parser.add_argument("--soft-edge-width", type=float, default=4.0)

    parser.add_argument("--scan-step", type=int, default=24)
    parser.add_argument("--scan-jitter", type=int, default=1)

    parser.add_argument("--photon-scale", type=float, default=1e3)
    parser.add_argument("--damping", type=str, default="auto")

    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--amplitude-eps", type=float, default=1e-12)

    parser.add_argument(
        "--extrinsic-policies",
        type=str,
        default="standard,weak_on_negative_precision",
        help="Comma-separated extrinsic policies.",
    )

    parser.add_argument(
        "--weak-precisions",
        type=str,
        default="1.0,1e-2,1e-4",
        help="Comma-separated weak precisions used by weak policies.",
    )

    parser.add_argument(
        "--min-precision-diff-rel",
        type=float,
        default=0.0,
        help="Relative threshold for declaring EP residual precision invalid.",
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


def run_one_config(
    args: argparse.Namespace,
    *,
    extrinsic_policy: str,
    weak_precision: float,
    obj_true: np.ndarray,
    probe: np.ndarray,
    observed: np.ndarray,
    noise_var: float,
    indices: Sequence[tuple[slice, slice]],
    eval_crop_margin: int,
) -> tuple[float, float, float]:
    """Run one experiment for a given extrinsic policy configuration."""
    obj_shape = (args.object_size, args.object_size)

    print("")
    print("=" * 100)
    print(
        f"[soft_edge_width={args.soft_edge_width:.6g}] "
        f"[policy={extrinsic_policy}] "
        f"[weak_precision={weak_precision:.3e}] "
        f"[amplitude_eps={args.amplitude_eps:.3e}] "
        f"[min_precision_diff_rel={args.min_precision_diff_rel:.3e}]"
    )

    graph = ptychography_graph_known_probe_extrinsic_policy(
        obj_shape=obj_shape,
        prb=probe,
        indices=indices,
        noise=noise_var,
        damping=parse_damping(args.damping),
        amplitude_eps=args.amplitude_eps,
        extrinsic_policy=extrinsic_policy,
        weak_precision=weak_precision,
        min_precision_diff_rel=args.min_precision_diff_rel,
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

    last_full_pmse = float("nan")
    last_crop_pmse = float("nan")
    best_crop_pmse = float("inf")

    def monitor(g, t: int) -> None:
        """Monitor reconstruction error."""
        nonlocal last_full_pmse
        nonlocal last_crop_pmse
        nonlocal best_crop_pmse

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
        best_crop_pmse = min(best_crop_pmse, last_crop_pmse)

        print(
            f"[policy={extrinsic_policy}] "
            f"[weak_precision={weak_precision:.3e}] "
            f"[iter={t:04d}] "
            f"full_PMSE={full_err:.6e} "
            f"crop_PMSE={crop_err:.6e} "
            f"best_crop_PMSE={best_crop_pmse:.6e}"
        )

    graph.run(
        n_iter=args.n_iter,
        device=args.device,
        schedule=args.schedule,
        block_size=args.block_size,
        callback=monitor,
    )

    print(
        f"[policy={extrinsic_policy}] "
        f"[weak_precision={weak_precision:.3e}] "
        f"final_full_PMSE={last_full_pmse:.6e} "
        f"final_crop_PMSE={last_crop_pmse:.6e} "
        f"best_crop_PMSE={best_crop_pmse:.6e}"
    )

    return last_full_pmse, last_crop_pmse, best_crop_pmse


def main() -> None:
    """Run extrinsic-policy sweep experiments."""
    args = build_parser().parse_args()

    policies = parse_str_list(args.extrinsic_policies)
    weak_precisions = parse_float_list(args.weak_precisions)

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

    probe = make_circular_probe(
        shape=probe_shape,
        radius=args.probe_radius,
        soft_edge_width=args.soft_edge_width,
        phase_curvature=args.phase_curvature,
    ).astype(np.complex64)

    probe_summary = summarize_probe(probe)

    obs_rng = np.random.default_rng(args.obs_seed)

    data = forward_ptychography(
        obj=obj_true,
        probe=probe,
        positions=positions,
        photon_scale=args.photon_scale,
        rng=obs_rng,
    )

    observed = data.anscombe.astype(np.float32)
    noise_var = 1.0 / float(args.photon_scale)

    print("AmplitudeMeasurement extrinsic-policy sweep")
    print(f"object_shape={obj_shape}")
    print(f"probe_shape={probe_shape}")
    print(f"n_scan={len(positions)}")
    print(f"soft_edge_width={args.soft_edge_width:.6g}")
    print(format_probe_summary(probe_summary))
    print(f"policies={policies}")
    print(f"weak_precisions={weak_precisions}")
    print(f"amplitude_eps={args.amplitude_eps:.3e}")
    print(f"min_precision_diff_rel={args.min_precision_diff_rel:.3e}")
    print(f"noise_var={noise_var:.3e}")
    print(f"photon_scale={args.photon_scale:.3e}")
    print(f"n_iter={args.n_iter}")
    print(f"log_every={args.log_every}")
    print(f"device={args.device}")
    print(f"schedule={args.schedule}")
    print(f"block_size={args.block_size}")
    print(f"damping={args.damping}")
    print(f"init={args.init}")
    print(f"eval_crop_margin={eval_crop_margin}")

    results = []

    for policy in policies:
        if policy == "standard":
            config_weak_precisions = [1.0]
        else:
            config_weak_precisions = weak_precisions

        for weak_precision in config_weak_precisions:
            final_full_pmse, final_crop_pmse, best_crop_pmse = run_one_config(
                args,
                extrinsic_policy=policy,
                weak_precision=weak_precision,
                obj_true=obj_true,
                probe=probe,
                observed=observed,
                noise_var=noise_var,
                indices=indices,
                eval_crop_margin=eval_crop_margin,
            )

            results.append(
                {
                    "policy": policy,
                    "weak_precision": weak_precision,
                    "final_full_pmse": final_full_pmse,
                    "final_crop_pmse": final_crop_pmse,
                    "best_crop_pmse": best_crop_pmse,
                }
            )

    print("")
    print("=" * 100)
    print("Summary")
    print("policy,weak_precision,final_full_PMSE,final_crop_PMSE,best_crop_PMSE")

    for result in results:
        print(
            f"{result['policy']},"
            f"{result['weak_precision']:.3e},"
            f"{result['final_full_pmse']:.6e},"
            f"{result['final_crop_pmse']:.6e},"
            f"{result['best_crop_pmse']:.6e}"
        )


if __name__ == "__main__":
    main()