"""Tests for versatil_inference.server."""

import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import zmq
from tso_robotics_sockets import (
    CompressionType,
    InferenceRequestKey,
    InferenceResponseKey,
    ServerRoute,
    ServerStatus,
    TransportKey,
    decompress_array,
)
from versatil_constants.libero import LiberoCamera
from versatil_constants.shared import ActionComponent

from libero.libero.benchmark import Benchmark
from versatil_inference.environment import TASK_SUITE_MAX_STEPS
from versatil_inference.server import LiberoServer

pytestmark = pytest.mark.integration


@pytest.fixture
def server_factory(
    external_benchmark_factory: Callable[..., Benchmark],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[Callable[..., tuple[LiberoServer, zmq.Socket]]]:
    with ExitStack() as cleanup:

        def factory(task_names: tuple[str, ...]) -> tuple[LiberoServer, zmq.Socket]:
            suite = external_benchmark_factory(suite_name="libero_10")
            tasks = {task.name: task for task in suite.tasks}
            suite.tasks = [tasks[name] for name in task_names]
            suite.n_tasks = len(suite.tasks)
            monkeypatch.setitem(TASK_SUITE_MAX_STEPS, "libero_10", 20)
            with patch(
                "versatil_inference.environment.benchmark.get_benchmark_dict",
                return_value={"libero_10": lambda: suite},
            ):
                server = LiberoServer(
                    ip_address="127.0.0.1",
                    port=0,
                    task_suite_name="libero_10",
                    seed=7,
                    resolution=128,
                    num_steps_wait=5,
                    num_trials_per_task=1,
                    max_parallel_envs=2,
                    compression_type=CompressionType.RAW.value,
                    output_folder=str(tmp_path),
                    record_wrist_camera=True,
                )
            cleanup.callback(server.reply_socket.close, linger=0)
            cleanup.callback(server.executor.shutdown)
            cleanup.callback(server.shutdown)
            client = server.context.socket(zmq.REQ)
            cleanup.callback(client.close, linger=0)
            client.setsockopt(zmq.RCVTIMEO, 120_000)
            client.connect(server.reply_socket.getsockopt_string(zmq.LAST_ENDPOINT))
            return server, client

        yield factory


@pytest.fixture
def mock_policy_factory() -> Callable[..., Mock]:
    def factory(environment_count: int) -> Mock:
        action = {
            ActionComponent.POSITION.value: [0.0, 0.0, 0.0],
            ActionComponent.ORIENTATION.value: [0.0, 0.0, 0.0],
            ActionComponent.GRIPPER.value: [-1.0],
        }
        return Mock(return_value={index: action for index in range(environment_count)})

    return factory


def test_parallel_rollout_renders_cameras_with_mock_policy(
    server_factory: Callable[..., tuple[LiberoServer, zmq.Socket]],
    mock_policy_factory: Callable[..., Mock],
) -> None:
    server, client = server_factory(
        task_names=(
            "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
            "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        ),
    )
    policy = mock_policy_factory(environment_count=2)
    deadline = time.monotonic() + 120
    while (
        server.environment.current_status == ServerStatus.CREATING_ENV.value
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    assert server.environment.current_status == ServerStatus.WAITING_ACTION.value

    for timestep in range(5, 20):
        client.send_json(
            {
                TransportKey.ROUTE_NAME.value: ServerRoute.GET_OBSERVATION.value,
                InferenceRequestKey.REQUESTED_KEYS.value: [
                    LiberoCamera.AGENTVIEW.value,
                    LiberoCamera.EYE_IN_HAND.value,
                ],
                InferenceRequestKey.COMPRESSION_TYPE.value: CompressionType.RAW.value,
            }
        )
        server.handle_client_request()
        observation = client.recv_json()
        assert (
            observation[TransportKey.STATUS.value] == ServerStatus.WAITING_ACTION.value
        )
        assert observation[InferenceResponseKey.TIMESTEP.value] == {
            "0": timestep,
            "1": timestep,
        }
        for camera in (LiberoCamera.AGENTVIEW, LiberoCamera.EYE_IN_HAND):
            images = [
                decompress_array(data=data, method=CompressionType.RAW.value)
                for data in observation[camera.value].values()
            ]  # 2 × (128, 128, 3)
            assert len(images) == 2
            for image in images:
                assert image.shape == (128, 128, 3)
                assert image.dtype == np.uint8
                assert image.max() > image.min()
            assert not np.array_equal(images[0], images[1])
        client.send_json(
            {
                TransportKey.ROUTE_NAME.value: ServerRoute.SEND_ACTION.value,
                InferenceRequestKey.ACTIONS.value: policy(observation),
            }
        )
        server.handle_client_request()
        response = client.recv_json()

    assert response[TransportKey.STATUS.value] == ServerStatus.FINISHED.value
    assert server.environment.number_of_resets == [1, 1]
    assert policy.call_count == 15
    assert len(list(server.environment.rollout_directory.rglob("*.avi"))) == 4
