from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .paths import resolve_config_path, resolve_project_path


SUPPORTED_PROBLEMS = {
    "coded_diffraction_pattern",
    "ptychography",
    "random_structured_cdi",
}


_ALLOWED_TOP_LEVEL_SECTIONS = {
    "experiment",
    "data",
    "forward_model",
    "algorithm",
    "runtime",
    "output",
}


_DEFAULTS: dict[str, Any] = {
    "data": {},
    "forward_model": {},
    "algorithm": {},
    "runtime": {
        "device": "cpu",
        "fft_engine": None,
        "verbose": False,
    },
    "output": {
        "root": "results",
        "save_history": True,
        "save_figures": True,
        "save_reconstruction": True,
    },
}


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge two mappings without mutating either input."""
    merged = deepcopy(dict(base))

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def _require_mapping(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    """Validate that a value is a JSON object."""
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a JSON object.")

    return value


def _require_nonempty_string(
    mapping: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> str:
    """Read a required non-empty string from a mapping."""
    value = mapping.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"'{section}.{key}' must be a non-empty string."
        )

    return value.strip()


def validate_config_data(data: Mapping[str, Any]) -> None:
    """Validate the common structure of an experiment configuration."""
    unknown_sections = set(data) - _ALLOWED_TOP_LEVEL_SECTIONS

    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ConfigError(
            f"Unknown top-level config section(s): {names}"
        )

    experiment = _require_mapping(
        data.get("experiment"),
        name="experiment",
    )

    problem = _require_nonempty_string(
        experiment,
        "problem",
        section="experiment",
    )
    _require_nonempty_string(
        experiment,
        "name",
        section="experiment",
    )
    _require_nonempty_string(
        experiment,
        "method",
        section="experiment",
    )
    entrypoint = _require_nonempty_string(
        experiment,
        "entrypoint",
        section="experiment",
    )

    if problem not in SUPPORTED_PROBLEMS:
        supported = ", ".join(sorted(SUPPORTED_PROBLEMS))
        raise ConfigError(
            f"Unsupported experiment problem '{problem}'. "
            f"Supported values: {supported}"
        )

    if ":" not in entrypoint:
        raise ConfigError(
            "'experiment.entrypoint' must use the form "
            "'module.path:function_name'."
        )

    for section_name in (
        _ALLOWED_TOP_LEVEL_SECTIONS - {"experiment"}
    ):
        _require_mapping(
            data.get(section_name),
            name=section_name,
        )

    runtime = data["runtime"]

    device = runtime.get("device")
    if device not in {"cpu", "cuda"}:
        raise ConfigError(
            "'runtime.device' must be either 'cpu' or 'cuda'."
        )

    fft_engine = runtime.get("fft_engine")
    if fft_engine not in {
        None,
        "numpy",
        "cupy",
        "fftw",
    }:
        raise ConfigError(
            "'runtime.fft_engine' must be null, "
            "'numpy', 'cupy', or 'fftw'."
        )

    if device == "cuda" and fft_engine not in {None, "cupy"}:
        raise ConfigError(
            "CUDA execution requires 'runtime.fft_engine' "
            "to be null or 'cupy'."
        )

    if device == "cpu" and fft_engine == "cupy":
        raise ConfigError(
            "The CuPy FFT engine requires "
            "'runtime.device' to be 'cuda'."
        )

    verbose = runtime.get("verbose")
    if not isinstance(verbose, bool):
        raise ConfigError(
            "'runtime.verbose' must be a boolean."
        )

    output = data["output"]

    output_root = output.get("root")
    if (
        not isinstance(output_root, str)
        or not output_root.strip()
    ):
        raise ConfigError(
            "'output.root' must be a non-empty string."
        )

    for key in (
        "save_history",
        "save_figures",
        "save_reconstruction",
    ):
        value = output.get(key)
        if not isinstance(value, bool):
            raise ConfigError(
                f"'output.{key}' must be a boolean."
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved experiment configuration loaded from JSON."""

    data: dict[str, Any]
    source_path: Path

    @property
    def experiment(self) -> dict[str, Any]:
        return self.data["experiment"]

    @property
    def problem(self) -> str:
        return self.experiment["problem"]

    @property
    def name(self) -> str:
        return self.experiment["name"]

    @property
    def method(self) -> str:
        return self.experiment["method"]

    @property
    def entrypoint(self) -> str:
        return self.experiment["entrypoint"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.data["runtime"]

    @property
    def output(self) -> dict[str, Any]:
        return self.data["output"]

    def section(self, name: str) -> dict[str, Any]:
        """Return one top-level configuration section."""
        try:
            value = self.data[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown config section: {name}"
            ) from exc

        if not isinstance(value, dict):
            raise TypeError(
                f"Config section '{name}' is not a mapping."
            )

        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the resolved configuration."""
        return deepcopy(self.data)


def load_config(path: str | Path) -> ExperimentConfig:
    """
    Load, resolve defaults, and validate an experiment JSON file.
    """
    config_path = resolve_config_path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw = json.load(file)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid JSON in {config_path} "
            f"at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            "The config root must be a JSON object."
        )

    resolved = _deep_merge(_DEFAULTS, raw)
    validate_config_data(resolved)

    return ExperimentConfig(
        data=resolved,
        source_path=config_path,
    )


def write_config(
    config: ExperimentConfig,
    path: str | Path,
) -> Path:
    """Write a resolved configuration as formatted JSON."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config.to_dict(),
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    return output_path