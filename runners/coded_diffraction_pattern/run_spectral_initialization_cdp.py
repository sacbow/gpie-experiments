import argparse
import numpy as np
import matplotlib.pyplot as plt

from datasets.loader import get_image
from experiments.coded_diffraction_pattern.forward_model import make_cdp
from experiments.coded_diffraction_pattern.spectral_initialization import (
    spectral_initialization_cdp,
)


# ------------------------------------------------------------
# Utility: correlation
# ------------------------------------------------------------
def compute_correlation(x, y, eps=1e-12):
    num = np.abs(np.vdot(x, y))
    den = (np.linalg.norm(x) * np.linalg.norm(y) + eps)
    return float(num / den)


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
# Main experiment
# ------------------------------------------------------------
def run_spectral_init(
    size=128,
    n_measurements=4,
    noise_model="gaussian",
    noise_param=1e-4,
    si_iter=100,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------
    x_true = make_complex_ground_truth(size)

    # --------------------------------------------------------
    # Generate data
    # --------------------------------------------------------
    masks, y_obs, noise_var = make_cdp(
        x_true,
        num_masks=n_measurements,
        seed=seed,
        noise_model=noise_model,
        noise_param=noise_param,
    )

    # --------------------------------------------------------
    # Random initialization baseline
    # --------------------------------------------------------
    x_rand = rng.normal(size=x_true.shape) + 1j * rng.normal(size=x_true.shape)
    x_rand = x_rand.astype(np.complex64)

    corr_rand = compute_correlation(x_true, x_rand)

    print(f"Random init correlation: {corr_rand:.6e}")

    # --------------------------------------------------------
    # Spectral initialization (track trajectory)
    # --------------------------------------------------------
    correlations = []

    def callback(xk):
        corr = compute_correlation(x_true, xk)
        correlations.append(corr)

    x_si = spectral_initialization_cdp(
        y_obs,
        masks,
        n_iter=si_iter,
        seed=seed,
    )

    corr_final = compute_correlation(x_true, x_si)
    print(f"Spectral init final correlation: {corr_final:.6e}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Spectral initialization experiment for CDP"
    )

    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--measurements", type=int, default=4)
    parser.add_argument("--noise-model", choices=["gaussian", "poisson"], default="gaussian")
    parser.add_argument("--noise-param", type=float, default=1e-8)
    parser.add_argument("--si-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_spectral_init(
        size=args.size,
        n_measurements=args.measurements,
        noise_model=args.noise_model,
        noise_param=args.noise_param,
        si_iter=args.si_iter,
        seed=args.seed,
    )