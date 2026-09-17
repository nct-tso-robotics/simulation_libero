"""Tests for rendering and stepping with evaluation data outside the package."""

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from libero.libero import benchmark, get_default_path_dict
from libero.libero.envs import OffScreenRenderEnv

pytestmark = pytest.mark.integration


@pytest.fixture
def environment_factory(
    path_config_factory: Callable[..., Path],
) -> Iterator[Callable[..., OffScreenRenderEnv]]:
    data_root = os.environ.get("LIBERO_TEST_DATA_ROOT")
    if data_root is None:
        pytest.skip("Set LIBERO_TEST_DATA_ROOT to run external-asset rendering tests.")
    path_config_factory(values=get_default_path_dict(custom_location=data_root))
    with ExitStack() as cleanup:

        def factory(task_index: int) -> OffScreenRenderEnv:
            suite = benchmark.get_benchmark_dict()["libero_10"]()
            environment = OffScreenRenderEnv(
                bddl_file_name=suite.get_task_bddl_file_path(task_index),
                camera_heights=64,
                camera_widths=64,
            )
            cleanup.callback(environment.close)
            environment.reset()
            environment.set_init_state(suite.get_task_init_states(task_index)[0])
            return environment

        yield factory


@pytest.mark.parametrize("task_index", range(10))
def test_libero_10_renders_and_steps_with_external_assets(
    environment_factory: Callable[..., OffScreenRenderEnv], task_index: int
) -> None:
    environment = environment_factory(task_index=task_index)
    observation, reward, done, info = environment.step(
        np.zeros(7, dtype=np.float32)  # (7,)
    )
    for camera in ("agentview_image", "robot0_eye_in_hand_image"):
        image = observation[camera]  # (64, 64, 3)
        assert image.shape == (64, 64, 3)
        assert np.isfinite(image).all()
        assert image.max() > image.min()
    assert Path(environment.env.custom_asset_dir).is_relative_to(
        Path(os.environ["LIBERO_TEST_DATA_ROOT"])
    )
