import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.random_structured_cdi.forward_model import make_random_structured_cdi
from experiments.random_structured_cdi.hio import hio


# ------------------------------------------------------------
# Utility: correlation
# ------------------------------------------------------------
def compute_correlation(x, y, eps=1e-12):
    num = np.abs(np.vdot(x, y))
    den = np.linalg.norm(x) * np.linalg.norm(y) + eps
    return float(num / den)


# ------------------------------------------------------------
# Utility: phase alignment
# ------------------------------------------------------------
def phase_align(x_est, x_true, eps=1e-12):
    inner = np.vdot(x_true, x_est)
    phase = np.angle(inner + eps)
    return x_est * np.exp(-1j * phase)


# ------------------------------------------------------------
# Support from alpha
# ------------------------------------------------------------
def circular_support_from_alpha(shape, alpha):
    """
    Build a centered circular support from the target sampling rate alpha.

    alpha = (# diffraction pixels) / (# support pixels)

    For a square grid and a circular support, we approximately use:
        alpha ~= 1 / (pi * radius^2)
    where radius is normalized by min(H, W).

    Returns
    -------
    support:
        Boolean support mask.
    radius:
        Normalized support radius used to build the mask.
    realized_alpha:
        Actual discrete sampling rate after pixelization.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    h, w = shape
    radius = np.sqrt(1.0 / (np.pi * alpha))

    if radius >= 0.5:
        raise ValueError(
            f"Derived radius={radius:.4f} is too large. "
            "Please use a larger alpha."
        )

    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r_pix = radius * min(h, w)

    support = (yy - cy) ** 2 + (xx - cx) ** 2 <= r_pix**2
    support_pixels = int(np.sum(support))
    realized_alpha = (h * w) / max(support_pixels, 1)

    return support, float(radius), float(realized_alpha)


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------
def make_random_ground_truth(shape, support, rng, dtype=np.complex64):
    """
    Generate a random complex object:
        x_i ~ CN(0, 1) on the support,
        x_i = 0 outside the support.
    """
    real = rng.normal(size=shape)
    imag = rng.normal(size=shape)
    x = (real + 1j * imag) / np.sqrt(2.0)
    x = x.astype(dtype)
    x = np.where(support, x, 0)

    return x


# ------------------------------------------------------------
# Single trial
# ------------------------------------------------------------
def run_single_trial(
    *,
    size,
    alpha,
    n_layers,
    beta,
    n_iter,
    noise_var,
    seed,
):
    """
    Run one HIO trial for a given (alpha, noise_var).
    """
    rng = np.random.default_rng(seed)
    shape = (size, size)

    support, radius, realized_alpha = circular_support_from_alpha(shape, alpha)
    x_true = make_random_ground_truth(shape, support, rng)

    data_seed = int(rng.integers(0, 2**32 - 1))
    init_seed = int(rng.integers(0, 2**32 - 1))

    phase_masks, y_obs, _ = make_random_structured_cdi(
        x_true,
        n_layers=n_layers,
        support=support,
        seed=data_seed,
        noise_model="gaussian",
        noise_param=noise_var,
    )

    phase_masks = phase_masks.astype(np.complex64)
    y_obs = y_obs.astype(np.float32)

    x_est = hio(
        y_obs,
        phase_masks,
        support,
        beta=beta,
        n_iter=n_iter,
        seed=init_seed,
    )

    corr = compute_correlation(x_est, x_true)

    return {
        "x_true": x_true,
        "x_est": x_est,
        "support": support,
        "radius": radius,
        "realized_alpha": realized_alpha,
        "correlation": corr,
    }


# ------------------------------------------------------------
# Save correlation plot
# ------------------------------------------------------------
def save_correlation_plot(output_dir, alphas_realized, noise_vars, median_corr):
    """
    Save the median correlation vs alpha plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.5, 4.5))

    for i, noise_var in enumerate(noise_vars):
        plt.plot(
            alphas_realized,
            median_corr[i],
            marker="o",
            label=rf"$\sigma^2={noise_var:.0e}$",
        )

    plt.xlabel(r"Sampling rate $\alpha$")
    plt.ylabel("Median correlation")
    plt.ylim(0.0, 1.02)
    plt.grid(True)
    plt.legend()
    plt.title("HIO performance on random structured CDI")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_vs_alpha.png", dpi=200)
    plt.close()



# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
def run_hio(
    *,
    size,
    n_layers,
    n_iter,
    beta,
    alphas,
    noise_vars,
    n_trials,
    base_seed,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_noise = len(noise_vars)
    n_alpha = len(alphas)

    median_corr = np.zeros((n_noise, n_alpha), dtype=float)
    all_corr = np.zeros((n_noise, n_alpha, n_trials), dtype=float)

    # We compute realized alpha only once because it is determined by alpha and grid size.
    realized_alphas = []

    for alpha in alphas:
        support, radius, alpha_realized = circular_support_from_alpha((size, size), alpha)
        realized_alphas.append(alpha_realized)

    # Store representative reconstructions for each noise level.
    representatives_by_noise = []

    for i, noise_var in enumerate(noise_vars):
        representatives = []

        print(f"\n=== noise_var = {noise_var:.1e} ===")

        for j, alpha in enumerate(alphas):
            print(f"[alpha target = {alpha:.2f}, alpha realized = {realized_alphas[j]:.2f}]")

            case_results = []

            for trial in range(n_trials):
                seed = base_seed + 100000 * i + 1000 * j + trial

                result = run_single_trial(
                    size=size,
                    alpha=alpha,
                    n_layers=n_layers,
                    beta=beta,
                    n_iter=n_iter,
                    noise_var=noise_var,
                    seed=seed,
                )

                corr = result["correlation"]
                all_corr[i, j, trial] = corr
                case_results.append(result)

                print(
                    f"  trial={trial:02d} "
                    f"corr={corr:.6f}"
                )

            corr_values = np.asarray([r["correlation"] for r in case_results], dtype=float)
            med = float(np.median(corr_values))
            median_corr[i, j] = med

            rep_idx = int(np.argmin(np.abs(corr_values - med)))
            rep = case_results[rep_idx]
            representatives.append(rep)

            print(
                f"  median correlation = {med:.6f} "
                f"(representative trial = {rep_idx})"
            )

        representatives_by_noise.append(representatives)

    # Save median correlation plot.
    save_correlation_plot(
        output_dir=output_dir,
        alphas_realized=realized_alphas,
        noise_vars=noise_vars,
        median_corr=median_corr,
    )

    print(f"\nResults saved to: {output_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HIO on random structured CDI over alpha and noise levels"
    )

    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=10000)
    parser.add_argument("--beta", type=float, default=0.9)

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[2.20, 2.25, 2.30, 2.35, 2.40, 2.45, 2.50],
    )
    parser.add_argument(
        "--noise-vars",
        type=float,
        nargs="+",
        default=[1e-9, 1e-6, 1e-3],
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/random_structured_cdi/hio",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    run_hio(
        size=args.size,
        n_layers=args.layers,
        n_iter=args.n_iter,
        beta=args.beta,
        alphas=args.alphas,
        noise_vars=args.noise_vars,
        n_trials=args.trials,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
    )