"""Vectorized Libero environment manager for benchmark evaluation."""

import csv
import datetime
import logging
import os
from pathlib import Path

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.venv import DummyVectorEnv, SubprocVectorEnv

from versatil_inference.episode_recorder import EpisodeRecorder
from versatil_inference.socket_flags import (
    DEFAULT_CLIENT_NAME,
    DEFAULT_MAX_TIMESTEPS,
    LIBERO_ALL_SUITES,
    TASK_SUITE_MAX_STEPS,
    LiberoGymKey,
    LiberoObservationKey,
    LiberoResponseKey,
    LiberoStatus,
    LiberoTrajectoryColumn,
    TaskSuiteName,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

NO_OP_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
ROBOSUITE_INTERNAL_HORIZON = 1000


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle representation.

    Args:
        quaternion: Quaternion array [x, y, z, w].

    Returns:
        Axis-angle representation [rx, ry, rz].
    """
    x, y, z, w = quaternion
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    if np.abs(angle) < 1e-7:
        return np.zeros(3)
    sine = np.sqrt(1 - w * w)
    if sine < 1e-7:
        return np.array([x, y, z]) * angle
    axis = np.array([x, y, z]) / sine
    return axis * angle


def _make_env_function(environment_args: dict) -> callable:
    """Create a factory function for OffScreenRenderEnv.

    Uses closure capture to avoid lambda late-binding issues.
    """
    def _make() -> OffScreenRenderEnv:
        return OffScreenRenderEnv(**environment_args)
    return _make


class Environment:
    """Manages vectorized Libero environments for benchmark evaluation.
    
    Note:
        Tasks are processed in batches of max_parallel_envs. Only one batch's
        vec_env is alive at a time. When all tasks in the current batch finish,
        the vec_env is closed and the next batch is created.
    """

    def __init__(
        self,
        task_suite_name: str,
        seed: int,
        resolution: int,
        num_steps_wait: int,
        num_trials_per_task: int,
        output_folder: str,
        max_parallel_envs: int = 10,
    ):
        self.task_suite_name = task_suite_name
        self.seed = seed
        self.resolution = resolution
        self.num_steps_wait = num_steps_wait
        self.num_trials_per_task = num_trials_per_task
        self.output_folder = output_folder
        self.max_parallel_envs = max_parallel_envs
        self.current_status = LiberoStatus.CREATING_ENV.value
        self.client_name = DEFAULT_CLIENT_NAME
        self._rollout_date = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        self.vectorized_environment = None
        self.latest_observation: dict[int, dict] = {}
        self.tasks: list = []
        self.num_envs: int = 0
        self.suite_name_per_task: list[str] = []
        self.max_timesteps_per_env: list[int] = []
        self.initial_states_per_task: list = []
        self.task_descriptions: list[str] = []
        self.active_environments: list[bool] = []
        self.steps_counts: list[int] = []
        self.number_of_resets: list[int] = []
        self.environments_successes: list[int] = []
        self.wait_steps_remaining: list[int] = []
        self.recorders: list[EpisodeRecorder | None] = []
        self.recently_reset_indices: list[int] = []
        self._batch_global_indices: list[int] = []
        self.trajectory_columns = [
            column.value for column in LiberoTrajectoryColumn
        ]
        self._init_benchmark()

    @property
    def rollout_directory(self) -> Path:
        """Build the structured rollout output path.

        If output_folder is set:
            {output_folder}/{safe_client_name}/{task_suite}/{date}/
        Default:
            {client_checkpoint_dir}/rollouts/{checkpoint_name}/{task_suite}/{date}/

        Returns:
            Rollout output directory path.
        """
        client_path = Path(self.client_name)
        if self.output_folder:
            safe_client_name = self.client_name.strip("/").replace(
                "/", "_"
            )
            return (
                Path(self.output_folder)
                / safe_client_name
                / self.task_suite_name
                / self._rollout_date
            )
        return (
            client_path.parent
            / "rollouts"
            / client_path.name
            / self.task_suite_name
            / self._rollout_date
        )

    def set_client_name(self, client_name: str) -> None:
        """Set client name used for the rollout directory path.

        Args:
            client_name: Identifier sent by the client at registration.
        """
        self.client_name = client_name

    def _init_benchmark(self) -> None:
        """Load tasks from the benchmark, handling LIBERO_ALL."""
        benchmark_dict = benchmark.get_benchmark_dict()
        if self.task_suite_name == TaskSuiteName.LIBERO_ALL.value:
            self._init_all_suites(benchmark_dict)
        else:
            self._init_single_suite(benchmark_dict)
        self.num_envs = len(self.tasks)
        self.active_environments = [False] * self.num_envs
        self.steps_counts = [0] * self.num_envs
        self.number_of_resets = [0] * self.num_envs
        self.environments_successes = [0] * self.num_envs
        self.wait_steps_remaining = [0] * self.num_envs
        self.recorders = [None] * self.num_envs
        logging.info(
            f"Loaded {self.num_envs} tasks for {self.task_suite_name}"
        )

    def _init_single_suite(self, benchmark_dict: dict) -> None:
        """Load tasks from a single benchmark suite."""
        if self.task_suite_name not in benchmark_dict:
            available = list(benchmark_dict.keys())
            raise ValueError(
                f"Task suite '{self.task_suite_name}' not found. "
                f"Available: {available}"
            )
        task_suite = benchmark_dict[self.task_suite_name]()
        max_timesteps = TASK_SUITE_MAX_STEPS.get(
            self.task_suite_name, DEFAULT_MAX_TIMESTEPS
        )
        for task_index in range(task_suite.n_tasks):
            task = task_suite.get_task(task_index)
            self.tasks.append(task)
            self.suite_name_per_task.append(self.task_suite_name)
            self.max_timesteps_per_env.append(max_timesteps)
            self.task_descriptions.append(task.language)
            self.initial_states_per_task.append(
                task_suite.get_task_init_states(task_index)
            )

    def _init_all_suites(self, benchmark_dict: dict) -> None:
        """Load and concatenate tasks from all 4 standard suites."""
        for suite_name in LIBERO_ALL_SUITES:
            if suite_name not in benchmark_dict:
                raise ValueError(
                    f"Suite '{suite_name}' not found in benchmark dict."
                )
            task_suite = benchmark_dict[suite_name]()
            max_timesteps = TASK_SUITE_MAX_STEPS.get(
                suite_name, DEFAULT_MAX_TIMESTEPS
            )
            for task_index in range(task_suite.n_tasks):
                task = task_suite.get_task(task_index)
                self.tasks.append(task)
                self.suite_name_per_task.append(suite_name)
                self.max_timesteps_per_env.append(max_timesteps)
                self.task_descriptions.append(task.language)
                self.initial_states_per_task.append(
                    task_suite.get_task_init_states(task_index)
                )

    def initialize(self) -> None:
        """Create the first batch of environments and perform initial setup.

        Intended to run in a background thread. Sets status to
        WAITING_ACTION when complete.
        """
        batch_size = min(self.max_parallel_envs, self.num_envs)
        self._batch_global_indices = list(range(batch_size))
        self._create_batch_vec_env()
        self.current_status = LiberoStatus.WAITING_ACTION.value

    def _create_batch_vec_env(self) -> None:
        """Create a SubprocVectorEnv for the current batch of tasks."""
        batch_size = len(self._batch_global_indices)
        logging.info(
            f"Creating batch: tasks {self._batch_global_indices[0]}-"
            f"{self._batch_global_indices[-1]} "
            f"({batch_size} parallel envs)"
        )
        env_functions = []
        for global_index in self._batch_global_indices:
            task = self.tasks[global_index]
            bddl_file = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )
            env_args = {
                "bddl_file_name": bddl_file,
                "camera_heights": self.resolution,
                "camera_widths": self.resolution,
            }
            env_functions.append(_make_env_function(env_args))
        if batch_size > 1:
            self.vectorized_environment = SubprocVectorEnv(env_functions)
        else:
            self.vectorized_environment = DummyVectorEnv(env_functions)
        self.vectorized_environment.seed(self.seed)
        for global_index in self._batch_global_indices:
            self.active_environments[global_index] = True
            self.steps_counts[global_index] = 0
            self.wait_steps_remaining[global_index] = 0
            self.recorders[global_index] = EpisodeRecorder(
                environment_id=self.suite_name_per_task[global_index],
                language_instruction=self.task_descriptions[global_index],
                trajectory_columns=self.trajectory_columns,
            )
        self.vectorized_environment.reset()
        initial_states = [
            self.initial_states_per_task[global_index][0]
            for global_index in self._batch_global_indices
        ]
        self.vectorized_environment.set_init_state(initial_states)
        self._perform_wait_steps()
        self.recently_reset_indices = list(range(batch_size))

    def _advance_to_next_batch(self) -> bool:
        """Close current vec_env and create the next batch.

        Returns:
            True if a new batch was created, False if all tasks are done.
        """
        self.vectorized_environment.close()
        self.vectorized_environment = None
        next_start = self._batch_global_indices[-1] + 1
        if next_start >= self.num_envs:
            return False
        end = min(next_start + self.max_parallel_envs, self.num_envs)
        self._batch_global_indices = list(range(next_start, end))
        self._create_batch_vec_env()
        return True

    def _perform_wait_steps(self) -> None:
        """Step batch envs with no-op actions for physics settling."""
        batch_size = len(self._batch_global_indices)
        all_actions = np.tile(NO_OP_ACTION, (batch_size, 1))
        for _ in range(self.num_steps_wait):
            observations, _, _, _ = self.vectorized_environment.step(all_actions)
            for local_index in range(batch_size):
                global_index = self._batch_global_indices[local_index]
                self.steps_counts[global_index] += 1
                frame = observations[local_index][
                    LiberoGymKey.AGENTVIEW_IMAGE.value
                ]
                frame = np.ascontiguousarray(frame[::-1, ::-1])
                self.recorders[global_index].add_observation(
                    frame=frame,
                    trajectory_row=self._build_trajectory_row(
                        observation=observations[local_index]
                    ),
                    reward=0.0,
                    output_directory=self.rollout_directory,
                )
        self.latest_observation = self._unbatch_observation(observations)

    def _get_observable_environment_indices(self) -> list[int]:
        """Return batch-local indices of active envs not in wait mode."""
        return [
            local_index
            for local_index, global_index
            in enumerate(self._batch_global_indices)
            if self.active_environments[global_index]
            and self.wait_steps_remaining[global_index] == 0
        ]

    def _unbatch_observation(self, observations) -> dict[int, dict]:
        """Convert vectorized observations to per-env dicts with VersatIL keys.

        Args:
            observations: Object array of per-env observation dicts.

        Returns:
            Dict mapping batch-local env index to observation dict.
        """
        observable_indices = self._get_observable_environment_indices()
        result = {}
        for local_index in observable_indices:
            global_index = self._batch_global_indices[local_index]
            env_obs = observations[local_index]
            per_env = {}
            agentview = env_obs.get(LiberoGymKey.AGENTVIEW_IMAGE.value)
            if agentview is not None:
                per_env[LiberoObservationKey.AGENTVIEW.value] = agentview
            eye_in_hand = env_obs.get(
                LiberoGymKey.EYE_IN_HAND_IMAGE.value
            )
            if eye_in_hand is not None:
                per_env[LiberoObservationKey.EYE_IN_HAND.value] = (
                    eye_in_hand
                )
            ee_pos = env_obs.get(LiberoGymKey.EE_POS.value)
            if ee_pos is not None:
                per_env[LiberoObservationKey.EE_POS_ACTION.value] = ee_pos
            ee_quat = env_obs.get(LiberoGymKey.EE_QUAT.value)
            if ee_quat is not None:
                per_env[LiberoObservationKey.EE_ORI_ACTION.value] = (
                    quat_to_axis_angle(ee_quat)
                )
            gripper = env_obs.get(LiberoGymKey.GRIPPER_QPOS.value)
            if gripper is not None:
                per_env[LiberoObservationKey.GRIPPER_STATE_ACTION.value] = (
                    gripper
                )
            per_env[LiberoObservationKey.LANGUAGE_INSTRUCTION.value] = (
                self.task_descriptions[global_index]
            )
            per_env[LiberoResponseKey.TIMESTEP.value] = (
                self.steps_counts[global_index]
            )
            result[local_index] = per_env
        return result

    def _build_trajectory_row(
        self, observation: dict
    ) -> dict[str, float]:
        """Build a trajectory row dict from a single-env observation.

        Args:
            observation: Raw gym observation dict for one env.

        Returns:
            Dict mapping trajectory column names to float values.
        """
        ee_pos = observation[LiberoGymKey.EE_POS.value]
        ee_quat = observation[LiberoGymKey.EE_QUAT.value]
        gripper = observation[LiberoGymKey.GRIPPER_QPOS.value]
        return {
            LiberoTrajectoryColumn.EE_POS_X.value: float(ee_pos[0]),
            LiberoTrajectoryColumn.EE_POS_Y.value: float(ee_pos[1]),
            LiberoTrajectoryColumn.EE_POS_Z.value: float(ee_pos[2]),
            LiberoTrajectoryColumn.EE_QUAT_X.value: float(ee_quat[0]),
            LiberoTrajectoryColumn.EE_QUAT_Y.value: float(ee_quat[1]),
            LiberoTrajectoryColumn.EE_QUAT_Z.value: float(ee_quat[2]),
            LiberoTrajectoryColumn.EE_QUAT_W.value: float(ee_quat[3]),
            LiberoTrajectoryColumn.GRIPPER_QPOS_0.value: float(gripper[0]),
            LiberoTrajectoryColumn.GRIPPER_QPOS_1.value: float(gripper[1]),
        }

    def get_latest_observation(self) -> dict[int, dict]:
        """Return the latest per-env observations."""
        return self.latest_observation

    def consume_reset_indices(self) -> list[int]:
        """Return and clear the list of recently reset environment indices."""
        indices = self.recently_reset_indices
        self.recently_reset_indices = []
        return indices

    def step(self, actions: dict[int, list[float]]) -> None:
        """Step all batch environments with given actions.

        Args:
            actions: Mapping from batch-local index to action list.
        """
        if self.current_status == LiberoStatus.FINISHED.value:
            return
        batch_size = len(self._batch_global_indices)
        # Inactive envs accumulate NO_OP steps after finishing all trials.
        # Reset before hitting robosuite's internal 1000-step horizon.
        for local_index in range(batch_size):
            global_index = self._batch_global_indices[local_index]
            if not self.active_environments[global_index]:
                self.steps_counts[global_index] += 1
                if (
                    self.steps_counts[global_index]
                    >= ROBOSUITE_INTERNAL_HORIZON - 1
                ):
                    self.vectorized_environment.reset(id=[local_index])
                    self.steps_counts[global_index] = 0
        all_actions = []
        for local_index in range(batch_size):
            global_index = self._batch_global_indices[local_index]
            if not self.active_environments[global_index]:
                all_actions.append(NO_OP_ACTION)
            elif self.wait_steps_remaining[global_index] > 0:
                all_actions.append(NO_OP_ACTION)
            else:
                all_actions.append(
                    actions.get(local_index, NO_OP_ACTION)
                )
        all_actions = np.array(all_actions)
        observations, rewards, dones, infos = self.vectorized_environment.step(
            all_actions
        )
        self.recently_reset_indices = []
        rollout_dir = self.rollout_directory
        for local_index in range(batch_size):
            global_index = self._batch_global_indices[local_index]
            if not self.active_environments[global_index]:
                continue
            if self.wait_steps_remaining[global_index] > 0:
                self.wait_steps_remaining[global_index] -= 1
                self.steps_counts[global_index] += 1
                frame = observations[local_index][
                    LiberoGymKey.AGENTVIEW_IMAGE.value
                ]
                frame = np.ascontiguousarray(frame[::-1, ::-1])
                self.recorders[global_index].add_observation(
                    frame=frame,
                    trajectory_row=self._build_trajectory_row(
                        observation=observations[local_index]
                    ),
                    reward=0.0,
                    output_directory=rollout_dir,
                )
                if self.wait_steps_remaining[global_index] == 0:
                    self.recently_reset_indices.append(local_index)
                continue
            self.steps_counts[global_index] += 1
            frame = observations[local_index][
                LiberoGymKey.AGENTVIEW_IMAGE.value
            ]
            frame = np.ascontiguousarray(frame[::-1, ::-1])
            self.recorders[global_index].add_observation(
                frame=frame,
                trajectory_row=self._build_trajectory_row(
                    observation=observations[local_index]
                ),
                reward=float(rewards[local_index]),
                output_directory=rollout_dir,
            )
            episode_success = bool(dones[local_index])
            episode_done = (
                episode_success
                or self.steps_counts[global_index]
                >= self.max_timesteps_per_env[global_index]
            )
            if episode_done:
                self.recorders[global_index].save(
                    was_success=episode_success,
                    output_directory=rollout_dir,
                )
                self.number_of_resets[global_index] += 1
                if episode_success:
                    self.environments_successes[global_index] += 1
                logging.info(
                    f"Env {global_index} "
                    f"({self.task_descriptions[global_index]}): "
                    f"episode done, success={episode_success}, "
                    f"trials={self.number_of_resets[global_index]}"
                    f"/{self.num_trials_per_task}"
                )
                if (
                    self.number_of_resets[global_index]
                    < self.num_trials_per_task
                ):
                    self._reset_single_environment(
                        local_index=local_index,
                        global_index=global_index,
                        trial_index=self.number_of_resets[global_index],
                    )
                else:
                    self.active_environments[global_index] = False
                    self.recorders[global_index] = None
                    self.vectorized_environment.reset(id=[local_index])
                    self.steps_counts[global_index] = 0
        self.latest_observation = self._unbatch_observation(observations)
        batch_active = any(
            self.active_environments[global_index]
            for global_index in self._batch_global_indices
        )
        if batch_active:
            self.current_status = LiberoStatus.WAITING_ACTION.value
        else:
            has_more = self._advance_to_next_batch()
            if has_more:
                self.current_status = LiberoStatus.WAITING_ACTION.value
            else:
                self._write_results_csv()
                self.current_status = LiberoStatus.FINISHED.value

    def _reset_single_environment(
        self,
        local_index: int,
        global_index: int,
        trial_index: int,
    ) -> None:
        """Reset a single env for its next trial.

        Args:
            local_index: Batch-local index for vec_env calls.
            global_index: Global task index for state arrays.
            trial_index: Index into initial states for this task.
        """
        self.vectorized_environment.reset(id=[local_index])
        init_state = self.initial_states_per_task[global_index]
        state_index = trial_index % len(init_state)
        self.vectorized_environment.set_init_state(
            init_state=[init_state[state_index]],
            id=[local_index],
        )
        self.steps_counts[global_index] = 0
        self.wait_steps_remaining[global_index] = self.num_steps_wait
        self.recorders[global_index].reset()

    def _write_results_csv(self) -> None:
        """Write per-task, per-suite, and overall results to CSV."""
        output_directory = self.rollout_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        csv_path = output_directory / "results.csv"
        environment_data = []
        for i in range(self.num_envs):
            successes = self.environments_successes[i]
            trials = self.number_of_resets[i]
            success_rate = successes / trials if trials > 0 else 0.0
            environment_data.append(
                (self.task_descriptions[i], successes, trials, success_rate)
            )
        suite_results = {}
        unique_suites = list(dict.fromkeys(self.suite_name_per_task))
        if len(unique_suites) > 1:
            for suite_name in unique_suites:
                matching = [
                    (description, successes, trials, success_rate)
                    for j, (description, successes, trials, success_rate)
                    in enumerate(environment_data)
                    if self.suite_name_per_task[j] == suite_name
                ]
                if matching:
                    total_successes = sum(s for _, s, _, _ in matching)
                    total_trials = sum(t for _, _, t, _ in matching)
                    mean_rate = (
                        sum(r for _, _, _, r in matching) / len(matching)
                    )
                    suite_results[suite_name] = (
                        total_successes,
                        total_trials,
                        mean_rate,
                    )
        total_successes = sum(s for _, s, _, _ in environment_data)
        total_trials = sum(t for _, _, t, _ in environment_data)
        overall_rate = (
            sum(r for _, _, _, r in environment_data) / len(environment_data)
            if environment_data
            else 0.0
        )
        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["task_description", "num_success/num_trials", "success_rate"]
            )
            for description, successes, trials, success_rate in (
                environment_data
            ):
                writer.writerow(
                    [
                        description,
                        f"{successes}/{trials}",
                        f"{success_rate:.4f}",
                    ]
                )
            if suite_results:
                writer.writerow([])
                for suite_name, (
                    suite_successes,
                    suite_trials,
                    suite_rate,
                ) in suite_results.items():
                    writer.writerow(
                        [
                            suite_name,
                            f"{suite_successes}/{suite_trials}",
                            f"{suite_rate:.4f}",
                        ]
                    )
            writer.writerow([])
            writer.writerow(
                [
                    "overall",
                    f"{total_successes}/{total_trials}",
                    f"{overall_rate:.4f}",
                ]
            )
        logging.info(f"Results saved to {csv_path}")

    def close(self) -> None:
        """Close the vectorized environment."""
        if self.vectorized_environment is not None:
            self.vectorized_environment.close()
            self.vectorized_environment = None