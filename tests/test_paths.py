"""Tests for LIBERO path configuration."""

import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

import libero.libero as libero
from libero.libero import utils

pytestmark = pytest.mark.integration


def test_fresh_import_uses_external_paths_and_ignores_other_configurations(
    tmp_path: Path,
) -> None:
    configuration = tmp_path / "fresh-config"
    configuration.mkdir()
    for name in ("config.yaml", "config_libero_plus.yaml"):
        (configuration / name).write_text("bddl_files: /unrelated/simulator\n")
    originals = {path.name: path.read_bytes() for path in configuration.iterdir()}
    environment = dict(
        os.environ,
        LIBERO_CONFIG_PATH=str(configuration),
        ROBOTICS_ASSETS_DIR=str(tmp_path / "robotics_assets"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from libero.libero import get_libero_path, set_libero_default_path; "
            "from libero.libero.utils import get_libero_path as utility_path; "
            "print(get_libero_path('bddl_files')); "
            "print(utility_path('bddl_files')); "
            "set_libero_default_path()",
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    expected = str(tmp_path / "robotics_assets/libero_pro/bddl_files")
    assert result.stdout.splitlines() == [expected, expected]
    own_config = configuration / "config_libero_pro.yaml"
    assert yaml.safe_load(own_config.read_text())["bddl_files"] == expected
    assert {
        path.name: path.read_bytes()
        for path in configuration.iterdir()
        if path != own_config
    } == originals


@pytest.mark.parametrize("key", ["bddl_files", "init_states", "datasets"])
@pytest.mark.parametrize("relative", [False, True])
def test_existing_paths_are_preserved_and_shared_by_utilities(
    path_config_factory: Callable[..., Path], key: str, relative: bool
) -> None:
    value = "../shared/data" if relative else "/shared/compatible-libero/data"
    config_path = path_config_factory(values={key: value})
    original = config_path.read_bytes()
    expected = str((config_path.parent / value).resolve())

    assert libero.get_libero_path(query_key=key) == expected
    assert utils.get_libero_path(query_key=key) == expected
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("value", [None, "", 3])
def test_invalid_config_path_reports_key_and_file(
    path_config_factory: Callable[..., Path], value: str | int | None
) -> None:
    path = path_config_factory(values={"bddl_files": value})
    with pytest.raises(
        ValueError,
        match=re.escape(f"Set 'bddl_files' to a directory path in {path}."),
    ):
        libero.get_libero_path(query_key="bddl_files")


def test_explicit_configuration_writes_the_selected_root(
    path_config_factory: Callable[..., Path], tmp_path: Path
) -> None:
    path = path_config_factory()
    root = tmp_path / "shared-benchmark"
    libero.set_libero_default_path(custom_location=str(root))
    assert path.is_file()
    for key, expected in libero.get_default_path_dict(
        custom_location=str(root)
    ).items():
        assert libero.get_libero_path(query_key=key) == expected
