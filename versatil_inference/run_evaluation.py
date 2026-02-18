"""Runs Libero evaluation via ZMQ communication with a policy client."""

import datetime
import os
from dataclasses import dataclass
from typing import Optional

import draccus
import wandb
import yaml

from versatil_inference.server import LiberoServer
from versatil_inference.socket_flags import (
    TASK_SUITE_MAX_STEPS,
    LiberoResponseKey,
    LiberoStatus,
    TaskSuiteName,
)

import perturbation

DATE_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

PERTURBATION_EXTENDED_MAX_STEPS: dict[str, int] = {
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
    """Configuration for Libero evaluation."""

    task_suite_name: str = TaskSuiteName.LIBERO_OBJECT.value
    evaluation_config_path: str = "./evaluation_config.yaml"
    num_steps_wait: int = 20
    num_trials_per_task: int = 10
    resolution: int = 128
    ip_address: str = "0.0.0.0"
    port: int = 5556
    compression_type: str = "raw"
    output_folder: str = ""
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = True
    wandb_project: str = "libero-eval"
    wandb_entity: str = ""
    seed: int = 7
    max_parallel_envs: int = 10
    record_wrist_camera: bool = False


def setup_perturbations(config: EvalConfig) -> str:
    """Apply perturbation setup based on evaluation config.

    Args:
        config: Evaluation configuration.

    Returns:
        Updated task_suite_name with perturbation suffix if applicable.
    """
    if config.evaluation_config_path is None:
        return config.task_suite_name
    with open(config.evaluation_config_path, "r", encoding="utf-8") as file:
        evaluation_config = yaml.safe_load(file)
    evaluation_config["bddl_files_path"] = (
        evaluation_config.get("bddl_files_path", "")
        + "/"
        + config.task_suite_name
    )
    evaluation_config["task_suite_name"] = config.task_suite_name
    use_swap = evaluation_config.get("use_swap", False)
    use_object = evaluation_config.get("use_object", False)
    use_language = evaluation_config.get("use_language", False)
    use_task = evaluation_config.get("use_task", False)
    use_environment = evaluation_config.get("use_environment", False)
    perturbation_flags = [
        use_swap,
        use_object,
        use_language,
        use_task,
        use_environment,
    ]
    active_count = sum(perturbation_flags)
    if active_count > 1:
        return _setup_multi_perturbation(
            config=config, evaluation_config=evaluation_config
        )
    elif active_count == 1:
        return _setup_single_perturbation(
            config=config, evaluation_config=evaluation_config
        )
    return config.task_suite_name


def _setup_multi_perturbation(
    config: EvalConfig, evaluation_config: dict
) -> str:
    """Set up environment when multiple perturbation flags are active."""
    use_swap = evaluation_config.get("use_swap", False)
    use_object = evaluation_config.get("use_object", False)
    use_language = evaluation_config.get("use_language", False)
    use_task = evaluation_config.get("use_task", False)
    use_environment = evaluation_config.get("use_environment", False)
    bddl_file_path = (
        evaluation_config.get("bddl_files_path", "")
        + config.task_suite_name
        + "_temp/"
    )
    init_file_path = (
        evaluation_config.get("init_file_dir", "")
        + config.task_suite_name
        + "_temp/"
    )
    log_content = (
        f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
    )
    if not os.path.exists(bddl_file_path) or not os.path.exists(
        init_file_path
    ):
        os.makedirs(init_file_path, exist_ok=True)
        os.makedirs(bddl_file_path, exist_ok=True)
        with open(
            os.path.join(bddl_file_path, "log.txt"), "w"
        ) as log_file:
            log_file.write(log_content)
        perturbation.create_env(configs=evaluation_config)
    else:
        with open(
            os.path.join(bddl_file_path, "log.txt"), "r"
        ) as log_file:
            existing_log = log_file.read().strip()
        if existing_log != log_content:
            for folder in [bddl_file_path, init_file_path]:
                for root, dirs, files in os.walk(folder, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
            os.makedirs(init_file_path, exist_ok=True)
            os.makedirs(bddl_file_path, exist_ok=True)
            with open(
                os.path.join(bddl_file_path, "log.txt"), "w"
            ) as log_file:
                log_file.write(log_content)
            perturbation.create_env(configs=evaluation_config)
    return config.task_suite_name + "_temp"


def _setup_single_perturbation(
    config: EvalConfig, evaluation_config: dict
) -> str:
    """Set up environment when exactly one perturbation flag is active."""
    perturbation_keys = [
        ("use_swap", evaluation_config.get("use_swap", False)),
        ("use_object", evaluation_config.get("use_object", False)),
        ("use_language", evaluation_config.get("use_language", False)),
        ("use_task", evaluation_config.get("use_task", False)),
        ("use_environment", evaluation_config.get("use_environment", False)),
    ]
    active_key = None
    for key, is_active in perturbation_keys:
        if is_active:
            active_key = key
            break
    if active_key is None:
        return config.task_suite_name
    suffix = evaluation_config.get("perturbation_mapping", {}).get(
        active_key, ""
    )
    init_file_path = (
        evaluation_config.get("init_file_dir", "")
        + config.task_suite_name
        + "_"
        + suffix
    )
    if not os.path.exists(init_file_path):
        perturbation.create_env(configs=evaluation_config)
    return config.task_suite_name + "_" + suffix


def run_evaluation(config: EvalConfig) -> None:
    """Create the server and run the evaluation loop.

    Args:
        config: Evaluation configuration.
    """
    if config.task_suite_name == TaskSuiteName.LIBERO_ALL.value:
        task_suite_name = config.task_suite_name
    else:
        task_suite_name = setup_perturbations(config)
    run_id = f"EVAL-{task_suite_name}-{DATE_TIME}"
    if config.run_id_note:
        run_id += f"--{config.run_id_note}"
    os.makedirs(config.local_log_dir, exist_ok=True)
    log_filepath = os.path.join(config.local_log_dir, run_id + ".txt")
    if config.use_wandb:
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=run_id,
        )
    server = LiberoServer(
        task_suite_name=task_suite_name,
        ip_address=config.ip_address,
        port=config.port,
        resolution=config.resolution,
        num_steps_wait=config.num_steps_wait,
        num_trials_per_task=config.num_trials_per_task,
        output_folder=config.output_folder,
        seed=config.seed,
        compression_type=config.compression_type,
        max_parallel_envs=config.max_parallel_envs,
        record_wrist_camera=config.record_wrist_camera,
    )
    print(
        f"Task suite: {task_suite_name}, "
        f"Waiting for client on tcp://{config.ip_address}:{config.port}"
    )
    try:
        while True:
            response = server.handle_client_request()
            if (
                response.get(LiberoResponseKey.STATUS.value)
                == LiberoStatus.FINISHED.value
            ):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        server.shutdown()
    rollout_dir = server.environment.rollout_directory
    rollout_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = str(rollout_dir / "log.txt")
    _log_results(
        server=server,
        config=config,
        task_suite_name=task_suite_name,
        log_filepath=log_filepath,
    )
    print(f"Log saved to: {log_filepath}")


def _log_results(
    server: LiberoServer,
    config: EvalConfig,
    task_suite_name: str,
    log_filepath: str,
) -> None:
    """Write per-task and overall results to local log file and wandb.

    Args:
        server: Server containing the evaluated environment.
        config: Evaluation configuration.
        task_suite_name: Resolved task suite name.
        log_filepath: Path for the local log file.
    """
    environment = server.environment
    if config.use_wandb:
        wandb.config.update({
            "client_name": environment.client_name,
            "task_suite_name": task_suite_name,
        })
    total_episodes = sum(environment.number_of_resets)
    total_successes = sum(environment.environments_successes)
    overall_rate = (
        total_successes / total_episodes if total_episodes > 0 else 0.0
    )
    with open(log_filepath, "w") as log_file:
        log_file.write(f"Task suite: {task_suite_name}\n")
        log_file.write(f"Total: {total_successes}/{total_episodes}\n")
        log_file.write(f"Overall success rate: {overall_rate:.4f}\n\n")
        for index in range(len(environment.task_descriptions)):
            description = environment.task_descriptions[index]
            successes = environment.environments_successes[index]
            episodes = environment.number_of_resets[index]
            task_rate = successes / episodes if episodes > 0 else 0.0
            log_file.write(
                f"{description}: {successes}/{episodes} ({task_rate:.4f})\n"
            )
            if config.use_wandb:
                wandb.log(
                    {
                        f"success_rate/{description}": task_rate,
                        f"num_episodes/{description}": episodes,
                    }
                )
        unique_suites = list(dict.fromkeys(environment.suite_name_per_task))
        if len(unique_suites) > 1:
            log_file.write("\n")
            for suite_name in unique_suites:
                suite_successes = sum(
                    environment.environments_successes[i]
                    for i in range(len(environment.suite_name_per_task))
                    if environment.suite_name_per_task[i] == suite_name
                )
                suite_episodes = sum(
                    environment.number_of_resets[i]
                    for i in range(len(environment.suite_name_per_task))
                    if environment.suite_name_per_task[i] == suite_name
                )
                suite_rate = (
                    suite_successes / suite_episodes
                    if suite_episodes > 0
                    else 0.0
                )
                log_file.write(
                    f"{suite_name}: {suite_successes}/{suite_episodes} "
                    f"({suite_rate:.4f})\n"
                )
                if config.use_wandb:
                    wandb.log(
                        {
                            f"success_rate/{suite_name}": suite_rate,
                            f"num_episodes/{suite_name}": suite_episodes,
                        }
                    )
    if config.use_wandb:
        wandb.log(
            {
                "success_rate/total": overall_rate,
                "num_episodes/total": total_episodes,
            }
        )
    print(f"\nFinal success rate: {overall_rate * 100:.1f}%")


@draccus.wrap()
def eval_libero(config: EvalConfig) -> None:
    """Entry point for Libero evaluation."""
    run_evaluation(config=config)


if __name__ == "__main__":
    eval_libero()