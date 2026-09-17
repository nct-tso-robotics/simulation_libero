"""Check the files required for LIBERO evaluation before starting a rollout."""

import argparse
from pathlib import Path

from libero.libero import (
    benchmark,
    config_file,
    get_libero_path,
)
from versatil_inference.socket_flags import LIBERO_ALL_SUITES, TaskSuiteName

ASSET_PATTERNS = (
    "scenes/*.xml",
    "textures/*.png",
    "articulated_objects/*.xml",
    "stable_scanned_objects/*/*.xml",
    "stable_hope_objects/*/*.xml",
    "turbosquid_objects/*/*.xml",
)
PRO_DOWNLOAD_URL = "https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro"


def check_assets(task_suite_name: str) -> None:
    """Validate simulator asset groups and every selected task's data files.

    Args:
        task_suite_name: Registered suite name, or libero_all.

    Raises:
        ValueError: If the suite name is unknown or path configuration is invalid.
        FileNotFoundError: If asset groups, BDDL files or initial states are missing.

    Note:
        Simulator assets, task definitions and initial states use
        config_libero_pro.yaml.
    """
    suites = (
        LIBERO_ALL_SUITES
        if task_suite_name == TaskSuiteName.LIBERO_ALL.value
        else [task_suite_name]
    )
    benchmarks = benchmark.get_benchmark_dict()
    for suite in suites:
        if suite not in benchmarks:
            raise ValueError(f"Unknown LIBERO task suite: {suite}")

    assets = Path(get_libero_path(query_key="assets"))
    bddl_files = Path(get_libero_path(query_key="bddl_files"))
    init_states = Path(get_libero_path(query_key="init_states"))
    print(f"LIBERO-Pro path configuration: {config_file}")
    print(f"Simulator assets: {assets}")
    print(f"BDDL task definitions: {bddl_files}")
    print(f"Initial states: {init_states}")

    missing = [
        str(assets / pattern)
        for pattern in ASSET_PATTERNS
        if not any(path.is_file() for path in assets.glob(pattern))
    ]
    task_count = 0
    for suite in suites:
        tasks = benchmarks[suite]().tasks
        if not tasks:
            raise ValueError(f"LIBERO task suite contains no tasks: {suite}")
        for task in tasks:
            task_count += 1
            for path in (
                bddl_files / task.problem_folder / task.bddl_file,
                init_states / task.problem_folder / task.init_states_file,
            ):
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(path))

    if missing:
        details = "\n".join(f"  {path}" for path in missing[:10])
        if len(missing) > 10:
            details += f"\n  ... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"LIBERO evaluation files are missing or empty:\n{details}\n"
            "Download the evaluation files using the README's asset setup steps.\n"
            f"Set assets, bddl_files and init_states in {config_file} "
            "to the downloaded directories under robotics_assets.\n"
            f"Additional LIBERO-PRO task variants are available at {PRO_DOWNLOAD_URL}."
        )
    print(f"Evaluation files found for {task_count} tasks ({', '.join(suites)}).")


def main() -> None:
    """Check one suite's evaluation files from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_suite_name", default=TaskSuiteName.LIBERO_10.value)
    args = parser.parse_args()
    check_assets(task_suite_name=args.task_suite_name)


if __name__ == "__main__":
    main()
