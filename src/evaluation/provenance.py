"""Preserve training membership even when a municipality has no holdout facts."""
from __future__ import annotations
import hashlib
from pathlib import Path


def reproducibility_metadata(root):
    """Hash source/configuration files and record installed direct dependencies."""
    from importlib.metadata import version, PackageNotFoundError
    root = Path(root)
    files = {}
    for directory in ["src", "scripts", "configs"]:
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".csv") and "__pycache__" not in path.parts:
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {}
    for package in ["numpy", "pandas", "pyarrow", "lightgbm", "catboost", "prophet", "cmdstanpy",
                    "ruptures", "scipy", "statsmodels", "torch", "chronos-forecasting"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"source_sha256": files, "package_versions": versions}


def training_territory_ids(metadata, predictions):
    ids = metadata.get("territory_ids")
    return list(ids) if ids is not None else sorted(predictions.territory_id.unique())


def verify_features(path, metadata):
    if metadata.get("status") not in (None, "complete"):
        raise ValueError("Forecast run is incomplete; finish or resume it before evaluating saved results")
    expected = metadata.get("features_sha256")
    if expected and hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError("Features differ from the baseline run; rerun forecasting before comparing models")
