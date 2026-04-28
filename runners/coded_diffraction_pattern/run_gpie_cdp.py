import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse, replicate

from datasets.loader import get_image
from experiments.coded_diffraction_pattern.forward_model import make_cdp


@model
def coded_diffraction_pattern(shape, n_measurements, phase_masks, noise):
    """
    Build a batched coded diffraction pattern model.

    phase_masks has shape (M, H, W).
    """
    x = ~GaussianPrior(
        event_shape=shape,
        label="object",
        dtype=np.complex64,
    )

    x_batch = replicate(x, batch_size=n_measurements)
    y = fft2(phase_masks * x_batch)

    AmplitudeMeasurement(var=noise, label="measurement") << y


def make_complex_ground_truth(size: int) -> np.ndarray:
    """
    Create a complex-valued ground truth image.

    The amplitude is generated from eagle, and the phase is generated from coins.
    """
    amplitude = get_image("eagle", size=size, normalize="max")
    phase_base = get_image("coins", size=size, normalize="max")

    phase = 2.0 * np.pi * phase_base
    x = amplitude * np.exp(1j * phase)

    return x.astype(np.complex64)


def build_graph(shape, n_measurements, masks, noise_var):
    """
    Build a gPIE graph for CDP inference.
    """
    return coded_diffraction_pattern(
        shape=shape,
        n_measurements=n_measurements,
        phase_masks=masks.astype(np.complex64),
        noise=float(noise_var),
    )


def save_outputs(output_dir, x_true, x_est, pmse_history, log_interval):
    """
    Save reconstruction results and convergence plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_amp = np.abs(x_true)
    true_phase = np.angle(x_true)

    est_amp = np.abs(x_est)
    est_phase = np.angle(x_est)

    plt.imsave(output_dir / "true_amplitude.png", true_amp, cmap="gray")
    plt.imsave(output_dir / "true_phase.png", true_phase, cmap="twilight")

    plt.imsave(output_dir / "reconstructed_amplitude.png", est_amp, cmap="gray")
    plt.imsave(output_dir / "reconstructed_phase.png", est_phase, cmap="twilight")

    if pmse_history:
        xs = np.arange(len(pmse_history)) * log_interval

        plt.figure()
        plt.plot(xs, pmse_history, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("PMSE")
        plt.yscale("log")
        plt.grid(True)
        plt.title("gPIE CDP convergence")
        plt.tight_layout()
        plt.savefig(output_dir / "pmse_history.png")
        plt.close()

    np.save(output_dir / "x_true.npy", x_true)
    np.save(output_dir / "x_est.npy", x_est)
    np.save(output_dir / "pmse_history.npy", np.asarray(pmse_history))


def run_gpie_cdp(
    *,
    size: int,
    n_measurements: int,
    n_iter: int,
    noise_model: str,
    noise_param: float,
    data_seed: int,
    init_seed: int,
    device: str,
    schedule: str,
    block_size: int | None,
    output_dir: str,
    log_interval: int,
    save_graph: bool,
):
    """
    Run gPIE inference for the coded diffraction pattern problem.
    """
    shape = (size, size)

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

    g = build_graph(
        shape=shape,
        n_measurements=n_measurements,
        masks=masks,
        noise_var=noise_var,
    )

    g.set_observation(
        y_obs,
        precision=1.0 / noise_var,
    )

    pmse_history = []

    def monitor(graph, t):
        if t % log_interval == 0 or t == n_iter - 1:
            x_est = graph["object"]["mean"][0]
            err = pmse(x_est, x_true)
            pmse_history.append(err)
            print(f"[iter={t:04d}] PMSE = {err:.6e}")

    g.set_init_rng(np.random.default_rng(init_seed))

    g.run(
        n_iter=n_iter,
        device=device,
        schedule=schedule,
        block_size=block_size,
        callback=monitor,
    )

    x_est = g["object"]["mean"][0]

    save_outputs(
        output_dir=output_dir,
        x_true=x_true,
        x_est=x_est,
        pmse_history=pmse_history,
        log_interval=log_interval,
    )

    if save_graph:
        graph_path = os.path.join(output_dir, "graph.html")
        g.visualize(output_path=graph_path)

    final_err = pmse(x_est, x_true)
    print(f"Final PMSE = {final_err:.6e}")
    print(f"Results saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run gPIE on the coded diffraction pattern experiment."
    )

    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--measurements", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=400)

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
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--schedule",
        choices=["parallel", "sequential"],
        default="parallel",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/coded_diffraction_pattern/gpie",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-graph", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_gpie_cdp(
        size=args.size,
        n_measurements=args.measurements,
        n_iter=args.n_iter,
        noise_model=args.noise_model,
        noise_param=args.noise_param,
        data_seed=args.data_seed,
        init_seed=args.init_seed,
        device=args.device,
        schedule=args.schedule,
        block_size=args.block_size,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_graph=args.save_graph,
    )