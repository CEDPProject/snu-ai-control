"""Helpers for single-version and segmented multi-version Kweather datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf


def _plain_list(value) -> list:
    if value is None:
        return []
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def configured_data_versions(
    data_version: str | None = None,
    data_versions=None,
) -> tuple[str, ...]:
    """Return the explicit dataset versions to load.

    `DATASET.data_version` is kept for backwards compatibility.  When
    `DATASET.data_versions` is provided, it becomes authoritative and the
    versions are interpreted as separate chronological segments.
    """
    versions = [str(value) for value in _plain_list(data_versions) if value]
    if versions:
        return tuple(versions)
    if not data_version:
        raise ValueError("DATASET.data_version or DATASET.data_versions is required")
    return (str(data_version),)


def dataset_version_label(
    data_version: str | None = None,
    data_versions=None,
) -> str:
    """Filesystem-safe label for result directories and metadata."""
    versions = configured_data_versions(data_version, data_versions)
    if len(versions) == 1:
        return versions[0]
    return "__".join(
        re.sub(r"[^0-9A-Za-z._-]+", "_", version).strip("._")
        for version in versions
    )


def dataset_label_from_cfg(cfg) -> str:
    return dataset_version_label(
        cfg.DATASET.get("data_version"),
        cfg.DATASET.get("data_versions"),
    )


def versions_from_cfg(cfg) -> tuple[str, ...]:
    return configured_data_versions(
        cfg.DATASET.get("data_version"),
        cfg.DATASET.get("data_versions"),
    )


def resolve_dataset_csv_path(
    datadir: str,
    data_version: str,
    environment_name: str,
) -> Path:
    """Resolve one Kweather CSV path from cwd, absolute path, or src-relative base."""
    configured_base = Path(datadir).expanduser()
    src_dir = Path(__file__).resolve().parents[1]
    base_candidates = [
        configured_base,
        Path.cwd() / configured_base,
        src_dir / configured_base,
    ]
    file_name = f"{environment_name}.csv"
    for candidate in base_candidates:
        csv_path = candidate.resolve() / data_version / file_name
        if csv_path.is_file():
            return csv_path
    searched = [
        str(candidate.resolve() / data_version / file_name)
        for candidate in base_candidates
    ]
    raise FileNotFoundError(f"Dataset CSV not found. Searched: {searched}")
