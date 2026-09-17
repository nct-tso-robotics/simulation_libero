"""Resolve LIBERO-Pro evaluation data from its own path configuration."""

import os
from pathlib import Path

import yaml

libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")
)
config_file = os.path.join(libero_config_path, "config_libero_pro.yaml")


def get_default_path_dict(custom_location: str | None = None) -> dict[str, str]:
    """Return evaluation paths under robotics_assets/libero_pro or a supplied root.

    Args:
        custom_location: Directory containing assets, bddl_files and init_files.

    Returns:
        Absolute paths for evaluation files and demonstration datasets.

    Note:
        ROBOTICS_ASSETS_DIR defaults to ~/robotics_assets.
    """
    if custom_location is None:
        storage = Path(os.environ.get("ROBOTICS_ASSETS_DIR", "~/robotics_assets"))
        root = storage / "libero_pro"
    else:
        root = Path(custom_location)
    root = root.expanduser().resolve()
    return {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(root / "datasets"),
        "assets": str(root / "assets"),
        "custom_assets": str(root / "custom_assets"),
    }


def get_libero_path(query_key: str) -> str:
    """Read a configured path, using robotics_assets for fresh setups.

    Args:
        query_key: Benchmark path key, such as bddl_files or init_states.

    Returns:
        Absolute path. Relative configured paths resolve beside the config file.

    Raises:
        ValueError: If an existing configuration has an invalid or missing key.
    """
    path = Path(config_file).expanduser()
    if path.is_file():
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    else:
        config = get_default_path_dict()
    if not isinstance(config, dict) or not isinstance(config.get(query_key), str):
        raise ValueError(f"Set '{query_key}' to a directory path in {path}.")
    value = config[query_key]
    if not value.strip():
        raise ValueError(f"Set '{query_key}' to a directory path in {path}.")
    resolved = Path(os.path.expandvars(value)).expanduser()
    if not resolved.is_absolute():
        resolved = path.parent / resolved
    return str(resolved.resolve())


def set_libero_default_path(custom_location: str | None = None) -> None:
    """Save evaluation paths in config_libero_pro.yaml.

    Args:
        custom_location: Directory containing assets, bddl_files and init_files.
    """
    path = Path(config_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="w", encoding="utf-8") as file:
        yaml.safe_dump(get_default_path_dict(custom_location=custom_location), file)
