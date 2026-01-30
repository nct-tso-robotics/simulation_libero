"""LIBERO Policy Server for running trained policies in LIBERO simulation.

This module provides a ZMQ-based server that runs LIBERO simulation environments
and receives actions from a remote policy client.
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from imitation_learning_toolkit.sockets.compression import CompressionType, compress_array
from imitation_learning_toolkit.sockets.server import SocketServer

from libero_socket_flags import (
    LiberoRequestKeys,
    LiberoResponseKeys,
    LiberoRoutes,
    LiberoStatus,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class EpisodeState:
    """Tracks the current episode state."""

    timestep: int = 0
    done: bool = False
    success: bool = False
    total_reward: float = 0.0
    max_timesteps: int = 300
    task_description: str = ""


class LiberoServer(SocketServer):
    """ZMQ-based server for running LIBERO simulation environments.
    
        The server receives
        actions from a remote policy client and returns observations from the LIBERO
        environment.
    """

    def __init__(
        self,
        task_suite_name: str = "libero_spatial",
        ip_address: str = "127.0.0.1",
        port: int = 5555,
        resolution: int = 256,
        num_steps_wait: int = 10,
        max_workers: int = 4,
        compression_type: str = CompressionType.RAW.value,
        seed: int = 0,
    ):
        """Initialize the LIBERO server.

        Args:
            task_suite_name: Name of the LIBERO benchmark suite
            ip_address: IP address to bind the ZMQ socket
            port: Port to bind the ZMQ socket
            resolution: Image resolution for environment rendering
            num_steps_wait: Steps to wait for objects to stabilize
            max_workers: Max workers for thread pool
            compression_type: Compression method for images
            seed: Random seed for environment
        """
        super().__init__(ip_address=ip_address, port=port, max_workers=max_workers)
        self.task_suite_name = task_suite_name
        self.resolution = resolution
        self.num_steps_wait = num_steps_wait
        self.compression_type = compression_type
        self.seed = seed
        self.env: OffScreenRenderEnv | None = None
        self.current_obs: dict | None = None
        self.episode_state = EpisodeState()
        self.task_suite = None
        self.current_task_idx: int = 0
        self.current_episode_idx: int = 0
        self.initial_states: list | None = None
        self.current_task = None
        self._init_benchmark()
        self._register_routes()
        logging.info(f"LiberoServer initialized on tcp://{ip_address}:{port}")
        logging.info(f"Task suite: {task_suite_name}, Resolution: {resolution}")


    def _register_routes(self) -> None:
        """Register all routes with the server."""
        self.add_route(LiberoRoutes.GET_OBSERVATION.value, self.handle_request, blocking=True)
        self.add_route(LiberoRoutes.SEND_ACTION.value, self.handle_request, blocking=True)
        self.add_route(LiberoRoutes.RESET_EPISODE.value, self.handle_request, blocking=True)


    def _init_benchmark(self) -> None:
        """Initialize the LIBERO benchmark and task suite."""
        benchmark_dict = benchmark.get_benchmark_dict()
        if self.task_suite_name not in benchmark_dict:
            available = list(benchmark_dict.keys())
            raise ValueError(
                f"Task suite '{self.task_suite_name}' not found. "
                f"Available: {available}"
            )
        self.task_suite = benchmark_dict[self.task_suite_name]()
        self.num_tasks = self.task_suite.n_tasks
        logging.info(f"Loaded task suite with {self.num_tasks} tasks")


    def _init_env_for_task(self, task_idx: int) -> str:
        """Initialize environment for a specific task.

        Args:
            task_idx: Index of the task in the suite

        Returns:
            Task description string
        """
        if self.env is not None:
            self.env.close()
        task = self.task_suite.get_task(task_idx)
        self.current_task = task
        self.current_task_idx = task_idx
        task_description = task.language
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.resolution,
            "camera_widths": self.resolution,
        }
        self.env = OffScreenRenderEnv(**env_args)
        self.env.seed(self.seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
        self.initial_states = self.task_suite.get_task_init_states(task_idx)
        self.episode_state.max_timesteps = TASK_SUITE_MAX_STEPS.get(
            self.task_suite_name, 300
        )
        self.episode_state.task_description = task_description
        logging.info(f"Initialized environment for task {task_idx}: {task_description}")
        return task_description


    def _reset_episode(self, episode_idx: int = 0) -> dict:
        """Reset the environment for a new episode.

        Args:
            episode_idx: Index of the episode (for initial state selection)

        Returns:
            Initial observation dictionary
        """
        if self.env is None:
            self._init_env_for_task(self.current_task_idx)
        self.current_episode_idx = episode_idx
        self.env.reset()
        if self.initial_states is not None and episode_idx < len(self.initial_states):
            self.current_obs = self.env.set_init_state(
                self.initial_states[episode_idx]
            )
        else:
            self.current_obs = self.env.reset()
        self.episode_state = EpisodeState(
            timestep=0,
            done=False,
            success=False,
            total_reward=0.0,
            max_timesteps=TASK_SUITE_MAX_STEPS.get(self.task_suite_name, 300),
            task_description=self.episode_state.task_description,
        )
        dummy_action = self._get_dummy_action()
        for _ in range(self.num_steps_wait):
            self.current_obs, _, _, _ = self.env.step(dummy_action)
            self.episode_state.timestep += 1
        logging.info(
            f"Reset episode {episode_idx} for task {self.current_task_idx}"
        )
        return self.current_obs


    def _get_dummy_action(self) -> list:
        """Get a no-op action for waiting periods."""
        return [0, 0, 0, 0, 0, 0, -1]


    def _step_env(self, action: list) -> tuple[dict, float, bool, dict]:
        """Step the environment with the given action.

        Args:
            action: 7D action [pos_delta(3), ori_delta(3), gripper(1)]

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset first.")
        obs, reward, done, info = self.env.step(action)
        self.current_obs = obs
        self.episode_state.timestep += 1
        self.episode_state.total_reward += float(reward)
        self.episode_state.done = bool(done)
        self.episode_state.success = bool(done)
        if self.episode_state.timestep >= self.episode_state.max_timesteps:
            self.episode_state.done = True
        return obs, reward, done, info


    def _build_observation_response(
        self, obs: dict, request_data: dict
    ) -> dict[str, Any]:
        """Build observation response from environment observation.

        Args:
            obs: Raw observation from LIBERO environment
            request_data: Request data containing which observations to include

        Returns:
            Response dictionary with requested observations
        """
        response: dict[str, Any] = {
            LiberoResponseKeys.STATUS.value: LiberoStatus.FINISHED.value,
            LiberoResponseKeys.TIMESTEP.value: int(self.episode_state.timestep),
            LiberoResponseKeys.MAX_TIMESTEPS.value: int(self.episode_state.max_timesteps),
            LiberoResponseKeys.DONE.value: bool(self.episode_state.done),
            LiberoResponseKeys.SUCCESS.value: bool(self.episode_state.success),
            LiberoResponseKeys.COMPRESSION_TYPE.value: self.compression_type,
            LiberoResponseKeys.IMAGE_HEIGHT.value: int(self.resolution),
            LiberoResponseKeys.IMAGE_WIDTH.value: int(self.resolution),
        }
        if request_data.get(LiberoRequestKeys.REQUEST_AGENTVIEW.value, True):
            agentview = obs.get("agentview_image", None)
            if agentview is not None:
                if agentview.dtype != np.uint8:
                    agentview = (agentview * 255).astype(np.uint8)
                response[LiberoResponseKeys.AGENTVIEW_RGB.value] = compress_array(
                    agentview, method=self.compression_type, as_base64=True
                )
        if request_data.get(LiberoRequestKeys.REQUEST_EYE_IN_HAND.value, True):
            eye_in_hand = obs.get("robot0_eye_in_hand_image", None)
            if eye_in_hand is not None:
                if eye_in_hand.dtype != np.uint8:
                    eye_in_hand = (eye_in_hand * 255).astype(np.uint8)
                response[LiberoResponseKeys.EYE_IN_HAND_RGB.value] = compress_array(
                    eye_in_hand, method=self.compression_type, as_base64=True
                )

        if request_data.get(LiberoRequestKeys.REQUEST_EE_POS.value, True):
            ee_pos = obs.get("robot0_eef_pos", None)
            if ee_pos is not None:
                response[LiberoResponseKeys.EE_POS.value] = ee_pos.tolist()
        if request_data.get(LiberoRequestKeys.REQUEST_EE_ORI.value, True):
            ee_quat = obs.get("robot0_eef_quat", None)
            if ee_quat is not None:
                ee_ori = self._quat_to_axis_angle(ee_quat)
                response[LiberoResponseKeys.EE_ORI.value] = ee_ori.tolist()
        if request_data.get(LiberoRequestKeys.REQUEST_EE_STATES.value, False):
            ee_pos = obs.get("robot0_eef_pos", None)
            ee_quat = obs.get("robot0_eef_quat", None)
            if ee_pos is not None and ee_quat is not None:
                ee_ori = self._quat_to_axis_angle(ee_quat)
                ee_states = np.concatenate([ee_pos, ee_ori])
                response[LiberoResponseKeys.EE_STATES.value] = ee_states.tolist()
        if request_data.get(LiberoRequestKeys.REQUEST_GRIPPER_STATES.value, True):
            gripper_qpos = obs.get("robot0_gripper_qpos", None)
            if gripper_qpos is not None:
                response[LiberoResponseKeys.GRIPPER_STATES.value] = gripper_qpos.tolist()

        if request_data.get(LiberoRequestKeys.REQUEST_JOINT_STATES.value, False):
            joint_pos = obs.get("robot0_joint_pos", None)
            if joint_pos is not None:
                response[LiberoResponseKeys.JOINT_STATES.value] = joint_pos.tolist()
        if request_data.get(LiberoRequestKeys.REQUEST_LANGUAGE_INSTRUCTION.value, True):
            response[LiberoResponseKeys.LANGUAGE_INSTRUCTION.value] = (
                self.episode_state.task_description
            )
        return response


    @staticmethod
    def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (x, y, z, w) to axis-angle representation.

        Args:
            quat: Quaternion array [x, y, z, w]

        Returns:
            Axis-angle representation [rx, ry, rz]
        """
        x, y, z, w = quat
        angle = 2 * np.arccos(np.clip(w, -1, 1))

        if np.abs(angle) < 1e-7:
            return np.zeros(3)

        s = np.sqrt(1 - w * w)
        if s < 1e-7:
            return np.array([x, y, z]) * angle

        axis = np.array([x, y, z]) / s
        return axis * angle


    def _handle_get_observation(self, request_data: dict) -> tuple[bool, dict]:
        """Handle GET_OBSERVATION route."""
        if self.current_obs is None:
            return False, {
                LiberoResponseKeys.ERROR_MSG.value: "No observation available. Reset episode first."
            }
        response = self._build_observation_response(self.current_obs, request_data)
        return True, response


    def _handle_send_action(self, request_data: dict) -> tuple[bool, dict]:
        """Handle SEND_ACTION route."""
        action = request_data.get(LiberoRequestKeys.ROBOT_ACTION.value, None)
        if action is None:
            return False, {
                LiberoResponseKeys.ERROR_MSG.value: "Missing 'robot_action' in request"
            }

        if len(action) != 7:
            return False, {
                LiberoResponseKeys.ERROR_MSG.value: f"Expected 7D action, got {len(action)}D"
            }
        obs, reward, done, info = self._step_env(action)
        response = self._build_observation_response(obs, request_data)
        response[LiberoResponseKeys.REWARD.value] = float(reward)
        response[LiberoResponseKeys.DONE.value] = bool(self.episode_state.done)
        response[LiberoResponseKeys.SUCCESS.value] = bool(self.episode_state.success)
        if self.episode_state.done:
            response[LiberoResponseKeys.STATUS.value] = (
                LiberoStatus.EPISODE_SUCCESS.value
                if self.episode_state.success
                else LiberoStatus.EPISODE_DONE.value
            )

        return True, response


    def _handle_reset_episode(self, request_data: dict) -> tuple[bool, dict]:
        """Handle RESET_EPISODE route."""
        task_idx = request_data.get(
            LiberoRequestKeys.TASK_IDX.value, self.current_task_idx
        )
        episode_idx = request_data.get(LiberoRequestKeys.EPISODE_IDX.value, 0)

        if task_idx != self.current_task_idx or self.env is None:
            self._init_env_for_task(task_idx)
        obs = self._reset_episode(episode_idx)
        response = self._build_observation_response(obs, request_data)
        response[LiberoResponseKeys.LANGUAGE_INSTRUCTION.value] = (
            self.episode_state.task_description
        )
        return True, response


    def handle_request(self, request_data: dict) -> tuple[bool, dict]:
        """Route incoming requests to appropriate handlers."""
        route_name = request_data.get(LiberoRequestKeys.ROUTE_NAME.value, None)
        if route_name == LiberoRoutes.GET_OBSERVATION.value:
            return self._handle_get_observation(request_data)
        elif route_name == LiberoRoutes.SEND_ACTION.value:
            return self._handle_send_action(request_data)
        elif route_name == LiberoRoutes.RESET_EPISODE.value:
            return self._handle_reset_episode(request_data)
        else:
            return False, {
                LiberoResponseKeys.ERROR_MSG.value: f"Unknown route: {route_name}"
            }


    def handle_client_request(self) -> tuple[str, bool, bool]:
        """Handle one request-response cycle with the client.
        
        Note: this method blocks waiting for client request (GET_OBSERVATION or SEND_ACTION), 
         calls appropriate handler and return info about what happened

        Returns:
            (route_name, episode_done, episode_success)
        """
        message = self.reply_socket.recv_string()
        request = json.loads(message)
        route = request.get(LiberoRequestKeys.ROUTE_NAME.value)
        success, response = self.handle_request(request)
        response["status"] = "FINISHED" if success else "ERROR"
        self.reply_socket.send_string(json.dumps(response))
        return route, self.episode_state.done, self.episode_state.success


    def shutdown(self) -> None:
        """Clean up resources."""
        logging.info("Shutting down LiberoServer...")
        if self.env is not None:
            self.env.close()
        self.executor.shutdown(wait=False)
        logging.info("LiberoServer shut down complete.")


