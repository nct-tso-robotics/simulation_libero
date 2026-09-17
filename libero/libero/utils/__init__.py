"""Shared LIBERO path helpers for dataset utilities."""

from libero.libero import (
    config_file as config_file,
    get_default_path_dict,
    get_libero_path as get_libero_path,
    libero_config_path as libero_config_path,
    set_libero_default_path,
)


def get_path_dict(root_location: str | None = None) -> dict[str, str]:
    """Return benchmark paths for the installation or a supplied root."""
    return get_default_path_dict(custom_location=root_location)


def set_libero_path(custom_location: str | None = None) -> None:
    """Save paths using the shared benchmark configuration."""
    set_libero_default_path(custom_location=custom_location)
