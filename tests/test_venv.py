"""Tests for libero.libero.envs.venv."""

import multiprocessing
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import gym
import numpy as np
import pytest

from libero.libero.envs.venv import SubprocEnvWorker

pytestmark = pytest.mark.integration


@pytest.fixture
def worker_factory() -> Iterator[Callable[..., SubprocEnvWorker]]:
    def create_environment() -> gym.Env:
        environment = gym.make("Pendulum-v1", disable_env_checker=True)
        environment.start_method = multiprocessing.get_start_method()
        return environment

    with ExitStack() as cleanup:

        def factory(share_memory: bool) -> SubprocEnvWorker:
            with ThreadPoolExecutor(max_workers=1) as executor:
                worker = executor.submit(
                    SubprocEnvWorker,
                    env_fn=create_environment,
                    share_memory=share_memory,
                ).result(timeout=30)
            cleanup.callback(worker.close)
            return worker

        yield factory


@pytest.mark.parametrize("share_memory", [False, True])
def test_worker_spawned_from_thread_resets_and_steps(
    worker_factory: Callable[..., SubprocEnvWorker], share_memory: bool
) -> None:
    worker = worker_factory(share_memory=share_memory)

    assert worker.get_env_attr(key="start_method") == "spawn"
    reset_result = worker.reset(seed=17)
    observation = (
        reset_result[0] if isinstance(reset_result, tuple) else reset_result
    ).copy()  # (3,)
    action = np.array([0.5], dtype=np.float32)  # (1,)
    transition = worker.step(action=action)
    next_observation = transition[0]  # (3,)
    np.testing.assert_allclose(np.linalg.norm(observation[:2]), 1, atol=1e-6)
    assert np.isfinite(next_observation).all()
    assert not np.array_equal(next_observation, observation)
    assert np.isfinite(transition[1])
    assert not transition[2]
