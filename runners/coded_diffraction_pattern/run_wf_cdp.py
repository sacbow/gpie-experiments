import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from gpie import pmse

from datasets.loader import get_image
from experiments.coded_diffraction_pattern.forward_model import make_cdp
from experiments.coded_diffraction_pattern.spectral_initialization import (
    spectral_initialization_cdp,
)
from experiments.coded_diffraction_pattern.wirtinger_flow import (
    run_wirtinger_flow_cdp,
)


def make_complex_ground_truth(size: int) -> np.ndarray:
    """Create a complex-valued ground truth image."""
    amplitude = get_image("eagle", size=size, normalize="max")
    phase_base = get_image("coins", size=size, normalize="max")

    phase = 2.0 * np.pi * phase_base
    x = amplitude * np.exp(1j * phase)

    return x.astype(np.complex64)


def make_random_initialization(shape, seed: int) -> np.ndarray:
    """Create a random complex initialization."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    x0 = x0.astype(np.complex64)
    x0 /= np.linalg.norm(x0) + 1e-12
    return x0


def save_outputs(output_dir, x_true, x_est, pmse_history, log_interval):
    """Save reconstruction results and convergence plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.imsave(output_dir / "true_amplitude.png", np.abs(x_true), cmap="gray")
    plt.imsave(output_dir / "true_phase.png", np.angle(x_true), cmap="twilight")

    plt.imsave(output_dir / "reconstructed_amplitude.png", np.abs(x_est), cmap="gray")
    plt.imsave(output_dir / "reconstructed_phase.png", np.angle(x_est), cmap="twilight")

    if pmse_history:
        xs = np.arange(len(pmse_history)) * log_interval

        plt.figure()
        plt.plot(xs, pmse_history, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("PMSE")
        plt.yscale("log")
        plt.grid(True)
        plt.title("WF CDP convergence")
        plt.tight_layout()
        plt.savefig(output_dir / "pmse_history.png")
        plt.close()

    np.save(output_dir / "x_true.npy", x_true)
    np.save(output_dir / "x_est.npy", x_est)
    np.save(output_dir / "pmse_history.npy", np.asarray(pmse_history))


def run_wf_cdp(
    *,
    size: int,
    n_measurements: int,
    n_iter: int,
    noise_model: str,
    noise_param: float,
    data_seed: int,
    init_seed: int,
    init: str,
    si_iter: int,
    si_lamb: float,
    mu_max: float,
    tau: float,
    output_dir: str,
    log_interval: int,
):
    """Run amplitude-based Wirtinger Flow for CDP."""
    x_true = make_complex_ground_truth(size)

    masks, y_obs, noise_var = make_cdp(
        x_true,
        num_masks=n_measurements,
        seed=data_seed,
        noise_model=noise_model,
        noise_param=noise_param,
    )

    masks = masks.astype(np.complex64)
    y_obs = y_obs.astype(np.float32)

    if init == "random":
        x0 = make_random_initialization(x_true.shape, seed=init_seed)
    elif init == "spectral":
        x0 = spectral_initialization_cdp(
            y_obs,
            masks,
            n_iter=si_iter,
            seed=init_seed,
            lamb=si_lamb,
        )
    else:
        raise ValueError(f"Unknown initialization: {init}")

    pmse_history = []

    def monitor(x, t):
        if t % log_interval == 0 or t == n_iter - 1:
            err = pmse(x, x_true)
            pmse_history.append(err)
            print(f"[iter={t:04d}] PMSE = {err:.6e}")

    x_est = run_wirtinger_flow_cdp(
        y_obs,
        masks,
        x0=x0,
        n_iter=n_iter,
        mu_max=mu_max,
        tau=tau,
        callback=monitor,
        seed=init_seed,
    )

    final_err = pmse(x_est, x_true)

    save_outputs(
        output_dir=output_dir,
        x_true=x_true,
        x_est=x_est,
        pmse_history=pmse_history,
        log_interval=log_interval,
    )

    print(f"Noise variance used by amplitude model: {noise_var:.6e}")
    print(f"Final PMSE = {final_err:.6e}")
    print(f"Results saved to: {output_dir}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run amplitude-based Wirtinger Flow on CDP."
    )

    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--measurements", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=5000)

    parser.add_argument(
        "--noise-model",
        choices=["gaussian", "poisson"],
        default="gaussian",
    )
    parser.add_argument(
        "--noise-param",
        type=float,
        default=1e-4,
        help="Gaussian variance or Poisson scaling factor.",
    )

    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--init-seed", type=int, default=111)

    parser.add_argument(
        "--init",
        choices=["random", "spectral"],
        default="spectral",
    )
    parser.add_argument("--si-iter", type=int, default=50)
    parser.add_argument("--si-lamb", type=float, default=10.0)

    parser.add_argument("--mu-max", type=float, default=0.4)
    parser.add_argument("--tau", type=float, default=300)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/coded_diffraction_pattern/wf",
    )
    parser.add_argument("--log-interval", type=int, default=50)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_wf_cdp(
        size=args.size,
        n_measurements=args.measurements,
        n_iter=args.n_iter,
        noise_model=args.noise_model,
        noise_param=args.noise_param,
        data_seed=args.data_seed,
        init_seed=args.init_seed,
        init=args.init,
        si_iter=args.si_iter,
        si_lamb=args.si_lamb,
        mu_max=args.mu_max,
        tau=args.tau,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
    )