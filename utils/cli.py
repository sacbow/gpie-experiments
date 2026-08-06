from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import traceback
from typing import Any, Callable, Sequence

from .config import (
    ConfigError,
    ExperimentConfig,
    load_config,
    write_config,
)
from .paths import (
    create_run_directory,
    resolve_project_path,
)


def _load_entrypoint(
    spec: str,
) -> Callable[..., Any]:
    """
    Import a callable from a
    'module.path:function_name' string.
    """
    module_name, separator, attribute_name = spec.partition(":")

    if (
        not separator
        or not module_name
        or not attribute_name
    ):
        raise ConfigError(
            "Runner entrypoint must use the form "
            "'module.path:function_name'."
        )

    module = importlib.import_module(module_name)
    runner = getattr(
        module,
        attribute_name,
        None,
    )

    if runner is None:
        raise AttributeError(
            f"Entrypoint '{spec}' does not exist: "
            f"missing attribute '{attribute_name}'."
        )

    if not callable(runner):
        raise TypeError(
            f"Entrypoint '{spec}' is not callable."
        )

    return runner


def _print_resolved_config(
    config: ExperimentConfig,
) -> None:
    """Print a resolved configuration to standard output."""
    print(
        json.dumps(
            config.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )


def _write_metadata(
    path: Path,
    metadata: dict[str, Any],
) -> None:
    """Write run metadata as JSON."""
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
            default=str,
        )
        file.write("\n")


def _run_command(
    args: argparse.Namespace,
) -> int:
    config = load_config(args.config)

    if args.dry_run:
        _print_resolved_config(config)
        return 0

    output_root = resolve_project_path(
        config.output["root"]
    )

    run_dir = create_run_directory(
        problem=config.problem,
        experiment_name=config.name,
        output_root=output_root,
        run_id=args.run_id,
    )

    write_config(
        config,
        run_dir / "config.json",
    )

    started_at = datetime.now().astimezone()

    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "config_source": str(config.source_path),
        "entrypoint": config.entrypoint,
        "problem": config.problem,
        "experiment_name": config.name,
        "method": config.method,
    }

    metadata_path = run_dir / "metadata.json"
    _write_metadata(
        metadata_path,
        metadata,
    )

    try:
        runner = _load_entrypoint(
            config.entrypoint
        )

        runner(
            config=config,
            run_dir=run_dir,
        )

    except Exception as exc:
        finished_at = datetime.now().astimezone()

        metadata.update(
            {
                "status": "failed",
                "finished_at": finished_at.isoformat(),
                "elapsed_seconds": (
                    finished_at - started_at
                ).total_seconds(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )

        _write_metadata(
            metadata_path,
            metadata,
        )

        traceback.print_exc()
        print(f"Run directory: {run_dir}")
        return 1

    finished_at = datetime.now().astimezone()

    metadata.update(
        {
            "status": "completed",
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": (
                finished_at - started_at
            ).total_seconds(),
        }
    )

    _write_metadata(
        metadata_path,
        metadata,
    )

    print(f"Run completed: {run_dir}")
    return 0


def _validate_command(
    args: argparse.Namespace,
) -> int:
    config = load_config(args.config)
    print(f"Valid config: {config.source_path}")
    return 0


def _show_command(
    args: argparse.Namespace,
) -> int:
    config = load_config(args.config)
    _print_resolved_config(config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="gpie-exp",
        description=(
            "Run reproducible gPIE numerical "
            "experiments from JSON configs."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment.",
    )
    run_parser.add_argument(
        "config",
        help="Path to a JSON config file.",
    )
    run_parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional result directory name. "
            "A timestamp is used by default."
        ),
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and print the resolved config "
            "without running the experiment."
        ),
    )
    run_parser.set_defaults(
        handler=_run_command
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a config file.",
    )
    validate_parser.add_argument(
        "config",
        help="Path to a JSON config file.",
    )
    validate_parser.set_defaults(
        handler=_validate_command
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Print a resolved config.",
    )
    show_parser.add_argument(
        "config",
        help="Path to a JSON config file.",
    )
    show_parser.set_defaults(
        handler=_show_command
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Command-line entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.handler(args))
    except (
        ConfigError,
        FileNotFoundError,
        ImportError,
        AttributeError,
        TypeError,
    ) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())