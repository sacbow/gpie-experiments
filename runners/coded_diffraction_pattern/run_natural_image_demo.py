import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, replicate, pmse

from datasets.loader import get_image
from experiments.coded_diffraction_pattern.forward_model import make_cdp


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
@model
def cdp_complex(shape, n_measurements, phase_masks, noise):
    x = ~GaussianPrior(
        event_shape=shape,
        label="object",
        dtype=np.complex64,
    )

    x_batch = replicate(x, batch_size=n_measurements)
    y = fft2(phase_masks * x_batch)

    AmplitudeMeasurement(var=noise, label="measurement", damping = 0.25) << y


@model
def cdp_real(shape, n_measurements, phase_masks, noise):
    x = ~GaussianPrior(
        event_shape=shape,
        label="object",
        dtype=np.float32,
    )

    x_batch = replicate(x, batch_size=n_measurements)
    y = fft2(phase_masks * x_batch)

    AmplitudeMeasurement(
        var=noise,
        label="measurement",
        damping=0.25,
    ) << y


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------
def make_complex_image(size):
    amp = get_image("coins", size=size, normalize="max")
    phase_base = get_image("eagle", size=size, normalize="max")

    phase = 2.0 * np.pi * phase_base
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def make_real_image(size):
    x = get_image("coins", size=size, normalize="max")
    return x.astype(np.float32)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_demo(
    *,
    size,
    n_masks,
    noise_model,
    noise_param,
    n_iter,
    seed,
    output_dir,
    real,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==== Natural Image Demo ====")
    print(f"mode        : {'real' if real else 'complex'}")
    print(f"size        : {size}")
    print(f"masks       : {n_masks}")
    print(f"noise_model : {noise_model}")
    print(f"noise_param : {noise_param}")
    print()

    # --------------------------------------------------------
    # data
    # --------------------------------------------------------
    if real:
        x_true = make_real_image(size)
    else:
        x_true = make_complex_image(size)

    masks, y_obs, noise_var, photon_counts = make_cdp(
        x_true,
        num_masks=n_masks,
        seed=seed,
        noise_model=noise_model,
        noise_param=noise_param,
    )

    print(f"Effective noise variance: {noise_var:.3e}")
    print(f"Maximum photon counts: {photon_counts}")

    # --------------------------------------------------------
    # model selection
    # --------------------------------------------------------
    if real:
        model_fn = cdp_real
    else:
        model_fn = cdp_complex

    g = model_fn(
        shape=(size, size),
        n_measurements=n_masks,
        phase_masks=masks,
        noise=noise_var,
    )

    g.set_observation(
        y_obs,
        precision=1.0 / noise_var,
        label="measurement",
    )

    g.set_init_rng(np.random.default_rng(seed))

    # --------------------------------------------------------
    # run
    # --------------------------------------------------------
    print("Running gPIE...")
    g.run(n_iter=n_iter, schedule = "sequential")

    x_est = g["object"]["mean"][0]

    # --------------------------------------------------------
    # metrics
    # --------------------------------------------------------
    err = pmse(x_est, x_true)
    print(f"Final PMSE = {err:.6e}")

    # --------------------------------------------------------
    # save images
    # --------------------------------------------------------
    def save(name, img, cmap):
        plt.imsave(output_dir / name, img, cmap=cmap)

    if real:
        save("true_real.png", x_true, "gray")
        save("est_real.png", x_est, "gray")
    else:
        save("true_amp_cmp.png", np.abs(x_true), "gray")
        save("true_phase_cmp.png", np.angle(x_true), "twilight")
        save("est_amp_cmp.png", np.abs(x_est), "gray")
        save("est_phase_cmp.png", np.angle(x_est), "twilight")

    print(f"Results saved to: {output_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Natural image demo (gPIE, CDP, Poisson noise)"
    )

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--masks", type=int, default=3)

    parser.add_argument(
        "--noise-model",
        choices=["gaussian", "poisson"],
        default="poisson",
    )
    parser.add_argument("--noise-param", type=float, default=100)

    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=5)

    parser.add_argument("--real", action="store_true")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/coded_diffraction_pattern/natural_demo",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# entry
# ------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    run_demo(
        size=args.size,
        n_masks=args.masks,
        noise_model=args.noise_model,
        noise_param=args.noise_param,
        n_iter=args.n_iter,
        seed=args.seed,
        output_dir=args.output_dir,
        real=args.real,
    )