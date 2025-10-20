"""Utility helpers for loading and working with YAML configuration files.

These helpers intentionally avoid third-party dependencies beyond PyYAML so
that configuration management can be used in training and evaluation scripts
without pulling in heavy frameworks.  Configurations are represented as nested
``dict`` objects by default, but ``ConfigNamespace`` provides attribute-style
access when desired.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration files or values are invalid."""


@dataclasses.dataclass
class ConfigNamespace:
    """A light-weight wrapper to allow attribute access for nested configs.

    The namespace is immutable once constructed to prevent accidental mutation
    of shared configuration objects.  To update values, create a copy via
    :meth:`clone` and apply changes to the copy.
    """

    _data: Mapping[str, Any]

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple proxy
        try:
            value = self._data[item]
        except KeyError as exc:  # pragma: no cover - simple proxy
            raise AttributeError(item) from exc
        if isinstance(value, Mapping) and not isinstance(value, ConfigNamespace):
            return ConfigNamespace(value)
        return value

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, ConfigNamespace) else v) for k, v in self._data.items()}

    def clone(self) -> "ConfigNamespace":
        return ConfigNamespace(copy.deepcopy(self._data))


def load_config(path: Path | str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a YAML configuration file and optionally apply CLI style overrides.

    Args:
        path: Path to a YAML configuration file.
        overrides: Optional iterable of ``key=value`` strings.  Nested keys can be
            referenced using dotted notation (e.g. ``training.lr=1e-4``).

    Returns:
        Parsed configuration as a Python dictionary.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config: Dict[str, Any] = yaml.safe_load(handle) or {}

    if overrides:
        for override in overrides:
            key, value = _split_override(override)
            _apply_override(config, key.split("."), value)
    return config


def save_config(config: Mapping[str, Any], path: Path | str) -> None:
    """Persist a configuration mapping to disk as YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)


def namespace(config: Mapping[str, Any]) -> ConfigNamespace:
    """Return a :class:`ConfigNamespace` view of the configuration mapping."""

    return ConfigNamespace(config)


def parse_cli_overrides(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse the common CLI arguments used by training/evaluation scripts."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML configuration file.")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="Optional key=value overrides applied after loading the YAML file.",
    )
    return parser.parse_known_intermixed_args(argv)[0]


def _split_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ConfigError(f"Invalid override syntax '{override}'. Expected format key=value.")
    key, raw_value = override.split("=", maxsplit=1)
    parsed_value: Any = yaml.safe_load(raw_value)
    return key, parsed_value


def _apply_override(config: MutableMapping[str, Any], keys: Iterable[str], value: Any) -> None:
    keys = list(keys)
    cursor: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], MutableMapping):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


__all__ = [
    "ConfigError",
    "ConfigNamespace",
    "load_config",
    "save_config",
    "namespace",
    "parse_cli_overrides",
]
