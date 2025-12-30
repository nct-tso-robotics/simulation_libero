"""Runs a model in a LIBERO simulation environment via ZMQ communication with a policy client."""
import datetime
import enum
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import draccus
import imageio
import numpy as np
import tqdm
import wandb
import yaml

from libero_policy_server import LiberoServer, TASK_SUITE_MAX_STEPS
from libero_socket_flags import LiberoRoutes
import perturbation


DATE = datetime.datetime.now().strftime("%Y-%m-%d")
DATE_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


class TaskSuite(str, enum.Enum):
    """Register for temporary evaluation tasks."""
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"
    # Temp variants (multiple perturbations)
    LIBERO_GOAL_TEMP = "libero_goal_temp"
    LIBERO_SPATIAL_TEMP = "libero_spatial_temp"
    LIBERO_10_TEMP = "libero_10_temp"
    LIBERO_OBJECT_TEMP = "libero_object_temp"
    # Language perturbation
    LIBERO_GOAL_LAN = "libero_goal_lan"
    LIBERO_SPATIAL_LAN = "libero_spatial_lan"
    LIBERO_10_LAN = "libero_10_lan"
    LIBERO_OBJECT_LAN = "libero_object_lan"
    # Object perturbation
    LIBERO_GOAL_OBJECT = "libero_goal_object"
    LIBERO_SPATIAL_OBJECT = "libero_spatial_object"
    LIBERO_10_OBJECT = "libero_10_object"
    LIBERO_OBJECT_OBJECT = "libero_object_object"
    # Swap perturbation
    LIBERO_GOAL_SWAP = "libero_goal_swap"
    LIBERO_SPATIAL_SWAP = "libero_spatial_swap"
    LIBERO_10_SWAP = "libero_10_swap"
    LIBERO_OBJECT_SWAP = "libero_object_swap"
    # Task perturbation
    LIBERO_GOAL_TASK = "libero_goal_task"
    LIBERO_SPATIAL_TASK = "libero_spatial_task"
    LIBERO_10_TASK = "libero_10_task"
    LIBERO_OBJECT_TASK = "libero_object_task"
    # Environment perturbation
    LIBERO_GOAL_ENV = "libero_goal_env"
    LIBERO_SPATIAL_ENV = "libero_spatial_env"
    LIBERO_10_ENV = "libero_10_env"
    LIBERO_OBJECT_ENV = "libero_object_env"


# Extended max steps for all task suite variants
TASK_MAX_STEPS_EXTENDED = {
    **TASK_SUITE_MAX_STEPS,
    "libero_goal_temp": 300,
    "libero_spatial_temp": 220,
    "libero_10_temp": 520,
    "libero_object_temp": 280,
    "libero_goal_lan": 300,
    "libero_spatial_lan": 220,
    "libero_10_lan": 520,
    "libero_object_lan": 280,
    "libero_goal_object": 300,
    "libero_spatial_object": 220,
    "libero_10_object": 520,
    "libero_object_object": 280,
    "libero_goal_swap": 300,
    "libero_spatial_swap": 220,
    "libero_10_swap": 520,
    "libero_object_swap": 280,
    "libero_goal_task": 300,
    "libero_spatial_task": 220,
    "libero_10_task": 520,
    "libero_object_task": 280,
    "libero_goal_env": 300,
    "libero_spatial_env": 220,
    "libero_10_env": 520,
    "libero_object_env": 280,
}


@dataclass
class EvalConfig:
    """Configuration for LIBERO evaluation."""
    # Task suite configuration
    task_suite_name: str = "libero_spatial"
    evaluation_config_path: str = "./evaluation_config.yaml"

    # Server configuration
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    resolution: int = 256
    ip_address: str = "127.0.0.1"
    port: int = 5555
    compression_type: str = "raw"

    # Logging configuration
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "libero-eval"
    wandb_entity: str = ""
    seed: int = 7


def setup_perturbations(cfg: EvalConfig) -> str:
    """Setup perturbations based on evaluation config YAML.

    Returns:
        Updated task_suite_name with perturbation suffix if applicable.
    """
    if cfg.evaluation_config_path is None:
        return cfg.task_suite_name

    with open(cfg.evaluation_config_path, "r", encoding="utf-8") as f:
        evaluation_cfg = yaml.safe_load(f)

    evaluation_cfg["bddl_files_path"] = evaluation_cfg.get("bddl_files_path", "") + "/" + cfg.task_suite_name
    evaluation_cfg["task_suite_name"] = cfg.task_suite_name

    use_swap = evaluation_cfg.get("use_swap", False)
    use_object = evaluation_cfg.get("use_object", False)
    use_language = evaluation_cfg.get("use_language", False)
    use_task = evaluation_cfg.get("use_task", False)
    use_environment = evaluation_cfg.get("use_environment", False)

    task_suite_name = cfg.task_suite_name

    # Check if multiple perturbation flags are True
    if sum([use_swap, use_object, use_language, use_task, use_environment]) > 1:
        # If more than one flag is True, use the temp environment
        bddl_file_path = evaluation_cfg.get("bddl_files_path", "") + cfg.task_suite_name + "_temp/"
        init_file_path = evaluation_cfg.get("init_file_dir", "") + cfg.task_suite_name + "_temp/"

        # Check if the directories exist and the log.txt file contents match
        if not os.path.exists(bddl_file_path) or not os.path.exists(init_file_path):
            os.makedirs(init_file_path, exist_ok=True)
            os.makedirs(bddl_file_path, exist_ok=True)

            log_content = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
            with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                log_file.write(log_content)

            perturbation.create_env(configs=evaluation_cfg)
        else:
            with open(os.path.join(bddl_file_path, "log.txt"), "r") as log_file:
                log_contents = log_file.read().strip()

            expected_log = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"

            if log_contents != expected_log:
                for folder in [bddl_file_path, init_file_path]:
                    for root, dirs, files in os.walk(folder, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))

                os.makedirs(init_file_path, exist_ok=True)
                os.makedirs(bddl_file_path, exist_ok=True)

                with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                    log_file.write(expected_log)

                perturbation.create_env(configs=evaluation_cfg)

        task_suite_name = cfg.task_suite_name + "_temp"

    # Handle the case when only one perturbation flag is True
    elif sum([use_swap, use_object, use_language, use_task, use_environment]) == 1:
        if use_swap:
            perturb_key = "use_swap"
        elif use_object:
            perturb_key = "use_object"
        elif use_language:
            perturb_key = "use_language"
        elif use_task:
            perturb_key = "use_task"
        elif use_environment:
            perturb_key = "use_environment"
        else:
            perturb_key = None

        if perturb_key:
            suffix = evaluation_cfg.get("perturbation_mapping", {}).get(perturb_key, "")
            init_file_path = evaluation_cfg.get("init_file_dir", "") + cfg.task_suite_name + "_" + suffix

            if not os.path.exists(init_file_path):
                perturbation.create_env(configs=evaluation_cfg)

            task_suite_name = cfg.task_suite_name + "_" + suffix

    return task_suite_name


class LiberoEvaluator:
    """Evaluator that uses LiberoServer for env and ZMQ communication."""

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.task_suite_name = setup_perturbations(cfg)
        self.server = LiberoServer(
            task_suite_name=self.task_suite_name,
            ip_address=cfg.ip_address,
            port=cfg.port,
            resolution=cfg.resolution,
            num_steps_wait=cfg.num_steps_wait,
            compression_type=cfg.compression_type,
        )
        self.log_file = None
        self.run_id = None


    def _setup_logging(self):
        self.run_id = f"EVAL-{self.task_suite_name}-{DATE_TIME}"
        if self.cfg.run_id_note:
            self.run_id += f"--{self.cfg.run_id_note}"
        os.makedirs(self.cfg.local_log_dir, exist_ok=True)
        local_log_filepath = os.path.join(self.cfg.local_log_dir, self.run_id + ".txt")
        self.log_file = open(local_log_filepath, "w")
        print(f"Logging to: {local_log_filepath}")
        if self.cfg.use_wandb:
            wandb.init(entity=self.cfg.wandb_entity, project=self.cfg.wandb_project, name=self.run_id)


    def run_evaluation(self):
        self._setup_logging()
        print(f"Task suite: {self.task_suite_name}")
        print(f"Waiting for PolicyClient on tcp://{self.cfg.ip_address}:{self.cfg.port}")
        self.log_file.write(f"Task suite: {self.task_suite_name}\n")

        total_episodes, total_successes = 0, 0
        max_steps = TASK_MAX_STEPS_EXTENDED.get(self.task_suite_name, 300)

        for task_idx in tqdm.tqdm(range(self.server.num_tasks), desc="Tasks"):
            self.server._init_env_for_task(task_idx)
            task_description = self.server.episode_state.task_description

            task_episodes, task_successes = 0, 0

            for episode_idx in tqdm.tqdm(range(self.cfg.num_trials_per_task), desc="Episodes", leave=False):
                print(f"\nTask: {task_description}")
                self.log_file.write(f"\nTask: {task_description}\n")
                self.server._reset_episode(episode_idx)
                replay_images = []
                done = False
                t = 0
                print(f"Starting episode {task_episodes + 1}...")
                self.log_file.write(f"Starting episode {task_episodes + 1}...\n")

                while t < max_steps and not done:
                    obs = self.server.current_obs
                    agentview = obs.get("agentview_image")
                    if agentview is not None:
                        if agentview.dtype != np.uint8:
                            agentview = (agentview * 255).astype(np.uint8)
                        replay_images.append(cv2.cvtColor(agentview, cv2.COLOR_BGR2RGB))

                    route, done, success = self.server.handle_client_request()
                    if route == LiberoRoutes.SEND_ACTION.value:
                        t += 1
                        if t % 20 == 0:
                            print(f"  Step {t}/{max_steps}")

                task_episodes += 1
                total_episodes += 1
                success = self.server.episode_state.success

                if success:
                    task_successes += 1
                    total_successes += 1

                save_rollout_video(replay_images, total_episodes, success, task_description, self.log_file)

                sr = total_successes / total_episodes * 100
                print(f"Success: {success} | Episodes: {total_episodes} | Success rate: {sr:.1f}%")
                self.log_file.write(f"Success: {success}\n")
                self.log_file.write(f"Episodes: {total_episodes}, Successes: {total_successes} ({sr:.1f}%)\n")
                self.log_file.flush()

            task_sr = task_successes / task_episodes
            print(f"Task success rate: {task_sr:.2f}")
            self.log_file.write(f"Task success rate: {task_sr:.2f}\n")

            if self.cfg.use_wandb:
                wandb.log({f"success_rate/{task_description}": task_sr, f"num_episodes/{task_description}": task_episodes})

        self.log_file.close()
        final_sr = total_successes / total_episodes
        print(f"\nFinal success rate: {final_sr * 100:.1f}%")

        if self.cfg.use_wandb:
            wandb.log({"success_rate/total": final_sr, "num_episodes/total": total_episodes})


    def shutdown(self):
        self.server.shutdown()
        if self.log_file and not self.log_file.closed:
            self.log_file.close()


@draccus.wrap()
def eval_libero(cfg: EvalConfig) -> None:
    evaluator = LiberoEvaluator(cfg)
    evaluator.run_evaluation()
    evaluator.shutdown()


if __name__ == "__main__":
    eval_libero()