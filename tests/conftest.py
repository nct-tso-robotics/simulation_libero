"""Fixtures for evaluation setup tests."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import libero.libero as libero
from libero.libero.benchmark import Task
from versatil_inference import check_assets


@pytest.fixture
def path_config_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., Path]:
    path = tmp_path / "config" / "config_libero_pro.yaml"
    monkeypatch.setattr(libero, "config_file", str(path))
    monkeypatch.setattr(check_assets, "config_file", str(path))

    def factory(values: dict | None = None) -> Path:
        if values is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(values), encoding="utf-8")
        return path

    return factory


@pytest.fixture
def evaluation_files_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_config_factory: Callable[..., Path],
) -> Callable[..., dict[str, Path]]:
    def factory(suites: tuple[str, ...] = ("libero_10",)) -> dict[str, Path]:
        root = tmp_path / "installation"
        paths = {
            "assets": root / "assets",
            "bddl_files": root / "bddl_files",
            "init_states": root / "init_files",
        }
        for pattern in check_assets.ASSET_PATTERNS:
            path = paths["assets"] / pattern.replace("*", "example")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"asset")
        benchmarks = {}
        for suite in suites:
            task = Task(
                name="pick_up_object",
                language="pick up object",
                problem="Libero",
                problem_folder=suite,
                bddl_file="pick_up_object.bddl",
                init_states_file="pick_up_object.pruned_init",
            )
            benchmarks[suite] = MagicMock(return_value=MagicMock(tasks=[task]))
            for key, name in (
                ("bddl_files", task.bddl_file),
                ("init_states", task.init_states_file),
            ):
                path = paths[key] / suite / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"task data")
        path_config_factory(values={key: str(value) for key, value in paths.items()})
        monkeypatch.setattr(
            check_assets.benchmark, "get_benchmark_dict", lambda: benchmarks
        )
        return paths

    return factory
