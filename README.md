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
