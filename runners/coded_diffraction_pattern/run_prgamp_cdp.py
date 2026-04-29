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
from experiments.coded_diffraction_pattern.prgamp import run_prgamp_cdp


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------
def make_complex_ground_truth(size):
    amp = get_image("eagle", size=size, normalize="max")
    phase_base = get_image("coins", size=size, normalize="max")

    phase = 2.0 * np.pi * phase_base
    x = amp * np.exp(1j * phase)

    return x.astype(np.complex64)


# ------------------------------------------------------------
# Random init
# ------------------------------------------------------------
def make_random_init(shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    x = x.astype(np.complex64)
    x /= np.linalg.norm(x) + 1e-12
    return x


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
def save_outputs(output_dir, x_true, x_est, pmse_history, log_interval):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.imsave(output_dir / "true_amp.png", np.abs(x_true), cmap="gray")
    plt.imsave(output_dir / "true_phase.png", np.angle(x_true), cmap="twilight")

    plt.imsave(output_dir / "est_amp.png", np.abs(x_est), cmap="gray")
    plt.imsave(output_dir / "est_phase.png", np.angle(x_est), cmap="twilight")

    if pmse_history:
        xs = np.arange(len(pmse_history)) * log_interval

        plt.figure()
        plt.plot(xs, pmse_history, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("PMSE")
        plt.yscale("log")
        plt.grid(True)
        plt.title("prGAMP convergence")
        plt.tight_layout()
        plt.savefig(output_dir / "pmse.png")
        plt.close()

    np.save(output_dir / "x_true.npy", x_true)
    np.save(output_dir / "x_est.npy", x_est)
    np.save(output_dir / "pmse_history.npy", np.asarray(pmse_history))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_prgamp(
    *,
    size,
    n_measurements,
    n_iter,
    noise_model,
    noise_param,
    data_seed,
    init_seed,
    init,
    si_iter,
    si_lamb,
    beta,
    wvar,
    output_dir,
    log_interval,
):
    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    if init == "random":
        x0 = make_random_init(x_true.shape, seed=init_seed)
    elif init == "spectral":
        x0 = spectral_initialization_cdp(
            y_obs,
            masks,
            n_iter=si_iter,
            seed=init_seed,
            lamb=si_lamb,
        )
    else:
        raise ValueError(f"Unknown init: {init}")

    # --------------------------------------------------------
    # Monitor
    # --------------------------------------------------------
    pmse_history = []

    def callback(x, t, info):
        if t % log_interval == 0 or t == n_iter - 1:
            err = pmse(x, x_true)
            pmse_history.append(err)
            print(
                f"[iter={t:04d}] PMSE={err:.6e}, "
                f"sigma2_x={info['sigma2_x']:.3e}"
            )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------
    x_est = run_prgamp_cdp(
        y_obs,
        masks,
        x0=x0,
        n_iter=n_iter,
        beta=beta,
        wvar=wvar if wvar is not None else noise_var,
        seed=init_seed,
        callback=callback,
    )

    final_err = pmse(x_est, x_true)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    save_outputs(
        output_dir,
        x_true,
        x_est,
        pmse_history,
        log_interval,
    )

    print(f"\nFinal PMSE = {final_err:.6e}")
    print(f"Noise variance used in GAMP = {wvar if wvar is not None else noise_var:.3e}")
    print(f"Results saved to: {output_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run prGAMP on coded diffraction pattern"
    )

    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--measurements", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=500)

    parser.add_argument(
        "--noise-model",
        choices=["gaussian", "poisson"],
        default="gaussian",
    )
    parser.add_argument("--noise-param", type=float, default=1e-6)

    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--init-seed", type=int, default=111)

    parser.add_argument(
        "--init",
        choices=["random", "spectral"],
        default="spectral",
    )
    parser.add_argument("--si-iter", type=int, default=50)
    parser.add_argument("--si-lamb", type=float, default=10.0)

    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument(
        "--wvar",
        type=float,
        default=None,
        help="Override noise variance (otherwise uses ground truth)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/coded_diffraction_pattern/prgamp",
    )
    parser.add_argument("--log-interval", type=int, default=50)

    return parser.parse_args()


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    run_prgamp(
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
        beta=args.beta,
        wvar=args.wvar,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
    )