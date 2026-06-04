import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse, replicate

from experiments.coded_diffraction_pattern.forward_model import make_cdp
from experiments.coded_diffraction_pattern.spectral_initialization import (
    spectral_initialization_cdp,
)
from experiments.coded_diffraction_pattern.wirtinger_flow import (
    run_wirtinger_flow_cdp,
)
from experiments.coded_diffraction_pattern.prvbem import run_prvbem_cdp
from experiments.coded_diffraction_pattern.prgamp import run_prgamp_cdp


# ------------------------------------------------------------
# gPIE model
# ------------------------------------------------------------
@model
def coded_diffraction_pattern(shape, n_measurements, phase_masks, noise):
    """Build a batched CDP model for gPIE."""
    x = ~GaussianPrior(
        event_shape=shape,
        label="object",
        dtype=np.complex64,
    )

    x_batch = replicate(x, batch_size=n_measurements)
    y = fft2(phase_masks * x_batch)

    AmplitudeMeasurement(var=noise, label="measurement") << y


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def make_random_complex_ground_truth(shape, seed):
    """Generate a complex Gaussian ground truth with unit average power."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    x = x / np.sqrt(2.0)

    power = np.mean(np.abs(x) ** 2)
    x = x / np.sqrt(power + 1e-12)

    return x.astype(np.complex64)


def make_random_initialization(shape, seed, scale=1.0):
    """Generate a random complex initialization."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    x = x.astype(np.complex64)

    x = x / (np.linalg.norm(x) + 1e-12)
    x = scale * x

    return x.astype(np.complex64)


def summarize(values):
    """Compute Q1, median, and Q3."""
    values = np.asarray(values, dtype=np.float64)
    q1, q2, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    return {
        "q1": float(q1),
        "q2": float(q2),
        "q3": float(q3),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def save_raw_results(output_dir, raw_results):
    """Save raw trial-level results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "raw_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n_masks",
                "trial",
                "data_seed",
                "init_seed",
                "pmse",
            ],
        )
        writer.writeheader()
        for row in raw_results:
            writer.writerow(row)

    json_path = output_dir / "raw_results.json"
    with open(json_path, "w") as f:
        json.dump(raw_results, f, indent=2)


def save_summary(output_dir, summary_rows):
    """Save summary statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n_masks",
                "q1",
                "q2",
                "q3",
                "mean",
                "std",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)


def plot_mask_vs_pmse(output_dir, summary_rows, methods, mask_counts):
    """Plot mask count versus PMSE with interquartile bands."""

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    # --- styling presets for methods ---
    styles = [
        {"linestyle": "-",  "marker": "o"},
        {"linestyle": "--", "marker": "s"},
        {"linestyle": "-.", "marker": "d"},
        {"linestyle": ":",  "marker": "^"},
    ]



    for i, method in enumerate(methods):
        rows = [
            row for row in summary_rows
            if row["method"] == method
        ]
        rows = sorted(rows, key=lambda row: row["n_masks"])

        xs = np.asarray([row["n_masks"] for row in rows], dtype=float)
        q1 = np.asarray([row["q1"] for row in rows], dtype=float)
        q2 = np.asarray([row["q2"] for row in rows], dtype=float)
        q3 = np.asarray([row["q3"] for row in rows], dtype=float)


        style = styles[i % len(styles)]

        plt.plot(
            xs,
            q2,
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2,
            markersize=10,
            label=method,
        )

        plt.fill_between(
            xs,
            q1,
            q3,
            alpha=0.2,
        )

    # --- axis labels (bigger font) ---
    plt.xlabel("Number of masks", fontsize=20)
    plt.ylabel("Final PMSE", fontsize=20)

    # --- ticks ---
    plt.xticks(mask_counts, fontsize=15)
    plt.yticks(fontsize=15)

    # --- log scale ---
    plt.yscale("log")

    # --- grid & legend ---
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "mask_vs_pmse.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# Method runners
# ------------------------------------------------------------
def run_gpie_once(
    x_true,
    masks,
    y_obs,
    noise_var,
    *,
    n_iter,
    init_seed,
    device,
    schedule,
    block_size,
):
    """Run gPIE once."""
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

    graph.set_init_rng(np.random.default_rng(init_seed))

    graph.run(
        n_iter=n_iter,
        device=device,
        schedule=schedule,
        block_size=block_size,
    )

    return graph["object"]["mean"][0].astype(np.complex64)


def run_wf_once(
    x_true,
    masks,
    y_obs,
    *,
    n_iter,
    init_seed,
    si_iter,
    si_lamb,
    mu_max,
    tau,
):
    """Run WF once with spectral initialization."""
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
    )


def run_prvbem_once(
    x_true,
    masks,
    y_obs,
    *,
    n_iter,
    init_seed,
):
    """Run prVBEM once from random initialization."""
    scale = np.sqrt(np.mean(y_obs**2))
    x0 = make_random_initialization(x_true.shape, init_seed, scale=scale)

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
    x_true,
    masks,
    y_obs,
    noise_var,
    *,
    n_iter,
    init_seed,
    beta,
):
    """Run prGAMP once from random initialization."""
    scale = np.sqrt(np.mean(y_obs**2))
    x0 = make_random_initialization(x_true.shape, init_seed, scale=scale)

    return run_prgamp_cdp(
        y_obs.astype(np.float32),
        masks.astype(np.complex64),
        x0=x0,
        n_iter=n_iter,
        beta=beta,
        wvar=noise_var,
        seed=init_seed,
        callback=None,
    )


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------
def benchmark_random_cdp_masks(
    *,
    size,
    mask_counts,
    n_trials,
    noise_model,
    noise_param,
    data_seed_base,
    init_seed_base,
    output_dir,
    methods,
    gpie_iter,
    wf_iter,
    prvbem_iter,
    prgamp_iter,
    si_iter,
    si_lamb,
    wf_mu_max,
    wf_tau,
    prgamp_beta,
    device,
    schedule,
    block_size,
):
    """Run mask-count sweep for random complex Gaussian CDP."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_results = []

    for n_masks in mask_counts:
        print(f"\n===== masks = {n_masks} =====")

        for trial in range(n_trials):
            data_seed = data_seed_base + 1000 * n_masks + trial
            init_seed = init_seed_base + 1000 * n_masks + trial

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

            method_fns: dict[str, Callable[[], np.ndarray]] = {
                "gpie": lambda: run_gpie_once(
                    x_true,
                    masks,
                    y_obs,
                    noise_var,
                    n_iter=gpie_iter,
                    init_seed=init_seed,
                    device=device,
                    schedule=schedule,
                    block_size=block_size,
                ),
                "wf": lambda: run_wf_once(
                    x_true,
                    masks,
                    y_obs,
                    n_iter=wf_iter,
                    init_seed=init_seed,
                    si_iter=si_iter,
                    si_lamb=si_lamb,
                    mu_max=wf_mu_max,
                    tau=wf_tau,
                ),
                "prvbem": lambda: run_prvbem_once(
                    x_true,
                    masks,
                    y_obs,
                    n_iter=prvbem_iter,
                    init_seed=init_seed,
                ),
                "prgamp": lambda: run_prgamp_once(
                    x_true,
                    masks,
                    y_obs,
                    noise_var,
                    n_iter=prgamp_iter,
                    init_seed=init_seed,
                    beta=prgamp_beta,
                ),
            }

            for method in methods:
                print(f"[masks={n_masks:02d}, trial={trial:02d}] running {method}...")

                try:
                    x_est = method_fns[method]()
                    err = pmse(x_est, x_true)
                except Exception as exc:
                    print(f"  failed: {exc}")
                    err = float("nan")

                print(f"  PMSE = {err:.6e}")

                raw_results.append(
                    {
                        "method": method,
                        "n_masks": int(n_masks),
                        "trial": int(trial),
                        "data_seed": int(data_seed),
                        "init_seed": int(init_seed),
                        "pmse": float(err),
                    }
                )

                save_raw_results(output_dir, raw_results)

    summary_rows = []
    for method in methods:
        for n_masks in mask_counts:
            values = [
                row["pmse"]
                for row in raw_results
                if row["method"] == method and row["n_masks"] == n_masks
            ]
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]

            if values.size == 0:
                stats = {
                    "q1": float("nan"),
                    "q2": float("nan"),
                    "q3": float("nan"),
                    "mean": float("nan"),
                    "std": float("nan"),
                }
            else:
                stats = summarize(values)

            summary_rows.append(
                {
                    "method": method,
                    "n_masks": int(n_masks),
                    **stats,
                }
            )

    save_summary(output_dir, summary_rows)
    plot_mask_vs_pmse(output_dir, summary_rows, methods, mask_counts)

    print(f"\nSaved benchmark results to: {output_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Mask-count sweep for random complex Gaussian CDP."
    )

    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--mask-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["gpie", "wf", "prvbem", "prgamp"],
        choices=["gpie", "wf", "prvbem", "prgamp"],
    )

    parser.add_argument(
        "--noise-model",
        choices=["gaussian", "poisson"],
        default="gaussian",
    )
    parser.add_argument("--noise-param", type=float, default=1e-8)

    parser.add_argument("--data-seed-base", type=int, default=1000)
    parser.add_argument("--init-seed-base", type=int, default=2000)

    parser.add_argument("--gpie-iter", type=int, default=1000)
    parser.add_argument("--wf-iter", type=int, default=2000)
    parser.add_argument("--prvbem-iter", type=int, default=1000)
    parser.add_argument("--prgamp-iter", type=int, default=1000)

    parser.add_argument("--si-iter", type=int, default=50)
    parser.add_argument("--si-lamb", type=float, default=10.0)

    parser.add_argument("--wf-mu-max", type=float, default=0.4)
    parser.add_argument("--wf-tau", type=float, default=300.0)

    parser.add_argument("--prgamp-beta", type=float, default=0.25)

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--schedule",
        choices=["parallel", "sequential"],
        default="parallel",
    )
    parser.add_argument("--block-size", type=int, default=1)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/coded_diffraction_pattern/random_mask_sweep",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    benchmark_random_cdp_masks(
        size=args.size,
        mask_counts=args.mask_counts,
        n_trials=args.trials,
        noise_model=args.noise_model,
        noise_param=args.noise_param,
        data_seed_base=args.data_seed_base,
        init_seed_base=args.init_seed_base,
        output_dir=args.output_dir,
        methods=args.methods,
        gpie_iter=args.gpie_iter,
        wf_iter=args.wf_iter,
        prvbem_iter=args.prvbem_iter,
        prgamp_iter=args.prgamp_iter,
        si_iter=args.si_iter,
        si_lamb=args.si_lamb,
        wf_mu_max=args.wf_mu_max,
        wf_tau=args.wf_tau,
        prgamp_beta=args.prgamp_beta,
        device=args.device,
        schedule=args.schedule,
        block_size=args.block_size,
    )