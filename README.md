# gPIE Experiments

This repository contains numerical experiments and baseline implementations for evaluating gPIE on computational imaging problems.

The experiment-specific runners are located under `runners/`, while reusable forward models and baseline algorithms are located under `experiments/`.

## Project Structure

```text
gpie-experiments/
├── configs/        # JSON experiment configurations
├── datasets/       # Dataset loading utilities
├── experiments/    # Forward models and baseline algorithms
├── metrics/        # Evaluation metrics
├── notebooks/      # Analysis notebooks
├── results/        # Experiment outputs
├── runners/        # Experiment entrypoints
└── utils/          # Config, path, and CLI utilities
```

Configuration files are organized by problem:

```text
configs/
├── coded_diffraction_pattern/
├── ptychography/
└── random_structured_cdi/
```

## Configuration Files

Experiments are configured using JSON files. A minimal configuration has the following structure:

```json
{
  "experiment": {
    "problem": "ptychography",
    "name": "example",
    "method": "gpie",
    "entrypoint": "runners.ptychography.example:run"
  },
  "runtime": {
    "device": "cpu"
  }
}
```

The `entrypoint` field specifies the Python callable used to run the experiment. It must have the form:

```text
module.path:function_name
```

Experiment runners are expected to expose a callable with the following interface:

```python
def run(*, config, run_dir):
    ...
```

Here, `config` is the resolved experiment configuration and `run_dir` is the output directory created for the current run.

Default values are automatically added for omitted common sections such as `algorithm`, `data`, `forward_model`, `runtime`, and `output`.

## Command-Line Usage

Commands can be executed from the repository root using the `utils.cli` module.

### Validate a configuration

```bash
python -m utils.cli validate configs/ptychography/example.json
```

A path relative to the `configs/` directory can also be used:

```bash
python -m utils.cli validate ptychography/example.json
```

### Show the resolved configuration

This command prints the configuration after applying default values:

```bash
python -m utils.cli show ptychography/example.json
```

### Perform a dry run

A dry run validates and displays the resolved configuration without importing or executing the experiment runner:

```bash
python -m utils.cli run ptychography/example.json --dry-run
```

### Run an experiment

```bash
python -m utils.cli run ptychography/example.json
```

A custom run identifier can be specified with:

```bash
python -m utils.cli run ptychography/example.json --run-id test-run
```

## Experiment Outputs

By default, each run creates a new directory under:

```text
results/<problem>/<experiment-name>/<run-id>/
```

For example:

```text
results/
└── ptychography/
    └── example/
        └── 20260806-174500/
            ├── config.json
            └── metadata.json
```

The saved `config.json` contains the fully resolved configuration, including default values.

The `metadata.json` file records the execution status, timestamps, entrypoint, and elapsed time. If an experiment fails, basic error information is also recorded.

Experiment runners may save additional results, figures, histories, and reconstructed arrays in the same run directory.

## Optional Console Command

After adding the following entry to `pyproject.toml`:

```toml
[project.scripts]
gpie-exp = "utils.cli:main"
```

and installing the repository in editable mode:

```bash
pip install -e .
```

the CLI can be invoked as:

```bash
gpie-exp validate ptychography/example.json
gpie-exp show ptychography/example.json
gpie-exp run ptychography/example.json
```


## Coded Diffraction Pattern Experiments

The coded diffraction pattern experiments currently include:

* A random-mask benchmark comparing gPIE with baseline phase-retrieval methods.
* A natural-image reconstruction demo using gPIE.

### Random-Mask Benchmark

The benchmark evaluates reconstruction performance as the number of random phase masks changes.

The default configuration is:

```text
configs/coded_diffraction_pattern/random_mask_sweep.json
```

Validate the configuration with:

```bash
python -m utils.cli validate \
  coded_diffraction_pattern/random_mask_sweep.json
```

Display the fully resolved configuration with:

```bash
python -m utils.cli show \
  coded_diffraction_pattern/random_mask_sweep.json
```

Run the benchmark with:

```bash
python -m utils.cli run \
  coded_diffraction_pattern/random_mask_sweep.json
```

The benchmark writes trial-level results, summary statistics, and the final mask-count plot to the generated run directory:

```text
results/
└── coded_diffraction_pattern/
    └── random_mask_sweep/
        └── <run-id>/
            ├── config.json
            ├── metadata.json
            ├── raw_results.csv
            ├── raw_results.json
            ├── summary.csv
            ├── summary.json
            └── mask_vs_pmse.png
```

The reconstruction methods included in the benchmark are selected through the `algorithm.methods` field in the JSON configuration.

Currently supported methods are:

```json
{
  "algorithm": {
    "methods": [
      "gpie",
      "wf",
      "prvbem",
      "prgamp"
    ]
  }
}
```

### Natural-Image Demo

The natural-image demo reconstructs an image from coded diffraction pattern measurements using gPIE.

The default configuration is:

```text
configs/coded_diffraction_pattern/natural_image_demo.json
```

Validate the configuration with:

```bash
python -m utils.cli validate \
  coded_diffraction_pattern/natural_image_demo.json
```

Display the resolved configuration with:

```bash
python -m utils.cli show \
  coded_diffraction_pattern/natural_image_demo.json
```

Run the demo with:

```bash
python -m utils.cli run \
  coded_diffraction_pattern/natural_image_demo.json
```

The default experiment reconstructs a real-valued natural image. The object mode is controlled by:

```json
{
  "data": {
    "object": {
      "mode": "real"
    }
  }
}
```

A complex-valued object can be constructed from separate amplitude and phase images by setting:

```json
{
  "data": {
    "object": {
      "mode": "complex",
      "amplitude_image": "coins",
      "phase_image": "eagle",
      "normalize": "max",
      "phase_scale": 6.283185307179586
    }
  }
}
```

The noise model is selected in the forward-model section:

```json
{
  "forward_model": {
    "noise_model": "poisson",
    "noise_param": 100.0
  }
}
```

Supported noise models are:

```text
gaussian
poisson
```

The generated run directory contains the resolved configuration, execution metadata, scalar metrics, reconstructed arrays, and visualization images:

```text
results/
└── coded_diffraction_pattern/
    └── natural_image_demo/
        └── <run-id>/
            ├── config.json
            ├── metadata.json
            ├── metrics.json
            ├── result.npz
            ├── true_real.png
            └── est_real.png
```

For a complex-valued reconstruction, amplitude and phase images are saved instead:

```text
true_amp_cmp.png
true_phase_cmp.png
est_amp_cmp.png
est_phase_cmp.png
```

### Quick Smoke Tests

Before running the full experiments, reduce the problem size, iteration count, and number of trials in a copied configuration file.

For example, a short random-mask benchmark can use:

```json
{
  "data": {
    "size": 32,
    "mask_counts": [1],
    "n_trials": 1
  },
  "algorithm": {
    "methods": ["gpie"],
    "gpie": {
      "n_iter": 2,
      "schedule": "parallel",
      "block_size": 1
    }
  }
}
```

A short natural-image demo can use:

```json
{
  "data": {
    "size": 32
  },
  "forward_model": {
    "n_masks": 2
  },
  "algorithm": {
    "gpie": {
      "n_iter": 2,
      "damping": 0.25,
      "schedule": "sequential",
      "block_size": 1
    }
  }
}
```

Each smoke-test configuration must still contain the required `experiment` section and any other required fields not shown in these partial examples.
