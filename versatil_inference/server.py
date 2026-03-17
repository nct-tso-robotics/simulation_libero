"""ZMQ server for running Libero simulation environments.

Accepts client connections and routes requests to vectorized Libero
environments for policy evaluation.
"""

import json
import logging
import threading
from typing import Any

import numpy as np

from tso_robotics_sockets import (
    CompressionType,
    InferenceRequestKey,
    InferenceResponseKey,
    ServerRoute,
    ServerStatus,
    SocketServer,
    TransportKey,
    compress_array,
)
from versatil_constants.libero import LiberoCamera, LiberoProprioKey
from versatil_constants.shared import ActionComponent, ObsKey

from versatil_inference.environment import Environment
from versatil_inference.socket_flags import DEFAULT_CLIENT_NAME, TaskSuiteName

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LiberoServer(SocketServer):
    """ZMQ-based server for running Libero simulation environments.

    Routes client requests to vectorized environments and returns
    observations, status updates, and reset signals.
    """

    def __init__(
        self,
        ip_address: str = "0.0.0.0",
        port: int = 5555,
        resolution: int = 256,
        compression_type: str = CompressionType.RAW.value,
        task_suite_name: str = TaskSuiteName.LIBERO_OBJECT.value,
        seed: int = 0,
        num_steps_wait: int = 20,
        num_trials_per_task: int = 10,
        output_folder: str = "",
        max_parallel_envs: int = 10,
        record_wrist_camera: bool = False,
    ):
        """Initialize the server and start environment creation in background.

        Args:
            ip_address: Address to bind the ZMQ socket.
            port: Port to bind the ZMQ socket.
            resolution: Image observation resolution (height and width).
            compression_type: Image compression method for responses.
            task_suite_name: Libero task suite to evaluate.
            seed: Random seed for environment initialization.
            num_steps_wait: Idle steps after each environment reset.
            num_trials_per_task: Number of evaluation episodes per task.
            output_folder: Directory for rollout recordings.
            max_parallel_envs: Maximum environments running simultaneously.
            record_wrist_camera: Whether to record wrist camera video.
        """
        super().__init__(ip_address=ip_address, port=port)
        self.resolution = resolution
        self.compression_type = compression_type
        self.task_suite_name = task_suite_name
        self.seed = seed
        self.num_steps_wait = num_steps_wait
        self.num_trials_per_task = num_trials_per_task
        self.output_folder = output_folder
        self.environment = Environment(
            task_suite_name=self.task_suite_name,
            seed=self.seed,
            resolution=self.resolution,
            num_steps_wait=self.num_steps_wait,
            num_trials_per_task=self.num_trials_per_task,
            output_folder=self.output_folder,
            max_parallel_envs=max_parallel_envs,
            record_wrist_camera=record_wrist_camera,
        )
        self._register_routes()
        thread = threading.Thread(
            target=self.environment.initialize, daemon=True
        )
        thread.start()

    def _register_routes(self) -> None:
        """Register all request routes with the socket server."""
        self.add_route(
            ServerRoute.GET_OBSERVATION.value,
            self.handle_request,
            blocking=True,
        )
        self.add_route(
            ServerRoute.SEND_ACTION.value,
            self.handle_request,
            blocking=True,
        )
        self.add_route(
            ServerRoute.REGISTER_CLIENT.value,
            self.handle_request,
            blocking=True,
        )

    def _handle_register_client(
        self, request_data: dict
    ) -> tuple[bool, dict]:
        """Store client name on the environment and return status."""
        client_name = request_data.get(
            InferenceRequestKey.CLIENT_NAME.value, DEFAULT_CLIENT_NAME
        )
        self.environment.set_client_name(client_name)
        logging.info(f"Client connected: {client_name}")
        return True, {
            TransportKey.STATUS.value: (
                self.environment.current_status
            ),
        }

    def _handle_get_observation(
        self, request_data: dict
    ) -> tuple[bool, dict]:
        """Build multi-env observation response with only the requested keys."""
        environment = self.environment
        if environment.current_status != ServerStatus.WAITING_ACTION.value:
            return True, {
                TransportKey.STATUS.value: environment.current_status,
            }
        latest_observation = environment.get_latest_observation()
        # All active envs are in wait mode (NO_OP settling steps).
        # Step internally so wait counters decrement and evaluation can exit.
        while not latest_observation:
            environment.step(actions={})
            if environment.current_status != ServerStatus.WAITING_ACTION.value:
                return True, {
                    TransportKey.STATUS.value: (
                        environment.current_status
                    ),
                }
            latest_observation = environment.get_latest_observation()
        requested_keys = request_data.get(
            InferenceRequestKey.REQUESTED_KEYS.value, []
        )
        compression_type = request_data.get(
            InferenceRequestKey.COMPRESSION_TYPE.value,
            self.compression_type,
        )
        requested_keys_set = set(requested_keys)
        response: dict[str, Any] = {
            TransportKey.STATUS.value: environment.current_status,
            InferenceResponseKey.IMAGE_HEIGHT.value: int(
                environment.resolution
            ),
            InferenceResponseKey.IMAGE_WIDTH.value: int(
                environment.resolution
            ),
            InferenceResponseKey.RESET_ENVIRONMENT_INDICES.value: (
                environment.consume_reset_indices()
            ),
            InferenceResponseKey.TIMESTEP.value: {
                environment_index: latest_observation[environment_index][
                    InferenceResponseKey.TIMESTEP.value
                ]
                for environment_index in latest_observation
            },
        }
        for requested_key in requested_keys_set:
            match requested_key:
                case LiberoCamera.AGENTVIEW.value:
                    compressed_images = {}
                    for environment_index in latest_observation:
                        agentview = latest_observation[environment_index].get(
                            LiberoCamera.AGENTVIEW.value
                        )
                        if agentview is not None:
                            if agentview.dtype != np.uint8:
                                agentview = (agentview * 255).astype(
                                    np.uint8
                                )
                            compressed_images[environment_index] = (
                                compress_array(
                                    agentview,
                                    method=compression_type,
                                    as_base64=True,
                                )
                            )
                    response[LiberoCamera.AGENTVIEW.value] = (
                        compressed_images
                    )
                case LiberoCamera.EYE_IN_HAND.value:
                    compressed_images = {}
                    for environment_index in latest_observation:
                        eye_in_hand = latest_observation[
                            environment_index
                        ].get(LiberoCamera.EYE_IN_HAND.value)
                        if eye_in_hand is not None:
                            if eye_in_hand.dtype != np.uint8:
                                eye_in_hand = (eye_in_hand * 255).astype(
                                    np.uint8
                                )
                            compressed_images[environment_index] = (
                                compress_array(
                                    eye_in_hand,
                                    method=compression_type,
                                    as_base64=True,
                                )
                            )
                    response[LiberoCamera.EYE_IN_HAND.value] = (
                        compressed_images
                    )
                case LiberoProprioKey.EE_POS_ACTION.value:
                    positions = {}
                    for environment_index in latest_observation:
                        ee_pos = latest_observation[environment_index].get(
                            LiberoProprioKey.EE_POS_ACTION.value
                        )
                        if ee_pos is not None:
                            positions[environment_index] = ee_pos.tolist()
                    response[LiberoProprioKey.EE_POS_ACTION.value] = (
                        positions
                    )
                case LiberoProprioKey.EE_ORI_ACTION.value:
                    orientations = {}
                    for environment_index in latest_observation:
                        ee_ori = latest_observation[environment_index].get(
                            LiberoProprioKey.EE_ORI_ACTION.value
                        )
                        if ee_ori is not None:
                            orientations[environment_index] = ee_ori.tolist()
                    response[LiberoProprioKey.EE_ORI_ACTION.value] = (
                        orientations
                    )
                case LiberoProprioKey.GRIPPER_STATE_ACTION.value:
                    grippers = {}
                    for environment_index in latest_observation:
                        gripper = latest_observation[environment_index].get(
                            LiberoProprioKey.GRIPPER_STATE_ACTION.value
                        )
                        if gripper is not None:
                            grippers[environment_index] = gripper.tolist()
                    response[
                        LiberoProprioKey.GRIPPER_STATE_ACTION.value
                    ] = grippers
                case ObsKey.LANGUAGE.value:
                    language_instructions = {}
                    for environment_index in latest_observation:
                        language_instructions[environment_index] = (
                            latest_observation[environment_index].get(
                                ObsKey.LANGUAGE.value,
                                "",
                            )
                        )
                    response[
                        ObsKey.LANGUAGE.value
                    ] = language_instructions
        return True, response

    def _handle_send_action(self, request_data: dict) -> tuple[bool, dict]:
        """Forward actions to the environment."""
        environment = self.environment
        if environment.current_status != ServerStatus.WAITING_ACTION.value:
            return True, {
                TransportKey.STATUS.value: environment.current_status,
            }
        raw_actions = request_data.get(
            InferenceRequestKey.ACTIONS.value, {}
        )
        actions = {
            int(key): self._flatten_action(structured_action=value)
            for key, value in raw_actions.items()
        }
        environment.step(actions=actions)
        return True, {
            TransportKey.STATUS.value: environment.current_status,
        }

    @staticmethod
    def _flatten_action(structured_action: dict[str, list[float]]) -> list[float]:
        """Flatten a structured action dict into the gym action format.

        Args:
            structured_action: Dict mapping ActionComponent values to action arrays.

        Returns:
            Flat action list in gym order: position, orientation, gripper.
        """
        flat: list[float] = []
        for component in (
            ActionComponent.POSITION.value,
            ActionComponent.ORIENTATION.value,
            ActionComponent.GRIPPER.value,
        ):
            if component in structured_action:
                flat.extend(structured_action[component])
        return flat

    def handle_request(self, request_data: dict) -> tuple[bool, dict]:
        """Dispatch request to the appropriate handler based on route."""
        route_name = request_data.get(
            TransportKey.ROUTE_NAME.value, None
        )
        match route_name:
            case ServerRoute.GET_OBSERVATION.value:
                return self._handle_get_observation(request_data)
            case ServerRoute.SEND_ACTION.value:
                return self._handle_send_action(request_data)
            case ServerRoute.REGISTER_CLIENT.value:
                return self._handle_register_client(request_data)
            case _:
                return False, {
                    TransportKey.ERROR_MSG.value: (
                        f"Unknown route: {route_name}"
                    ),
                }

    def handle_client_request(self) -> dict:
        """Receive a client request, dispatch it, and send the response."""
        message = self.reply_socket.recv_string()
        request = json.loads(message)
        success, response = self.handle_request(request)
        if not success:
            response[TransportKey.STATUS.value] = (
                ServerStatus.ERROR.value
            )
        self.reply_socket.send_string(json.dumps(response))
        if (
            response.get(TransportKey.STATUS.value)
            == ServerStatus.FINISHED.value
        ):
            self.environment.close()
        return response

    def shutdown(self) -> None:
        """Close the environment and shut down the server."""
        logging.info("Shutting down LiberoServer...")
        self.environment.close()
        logging.info("LiberoServer shut down complete.")