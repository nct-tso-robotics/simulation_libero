"""Tests for versatil_inference.run_evaluation."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import call, patch

import pytest
import yaml

from versatil_inference.run_evaluation import (
    EvalConfig,
    run_evaluation,
    setup_perturbations,
)


@pytest.fixture
def evaluation_config_factory(tmp_path: Path) -> Callable[..., EvalConfig]:
    def factory(use_language: bool) -> EvalConfig:
        path = tmp_path / "evaluation_config.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "bddl_files_path": "./libero/libero/bddl_files",
                    "init_file_dir": "./libero/libero/init_files",
                    "script_path": "./notebooks/generate_init_states.py",
                    "ood_task_configs": {"language": "./libero_ood/language.yaml"},
                    "use_language": use_language,
                }
            ),
            encoding="utf-8",
        )
        return EvalConfig(task_suite_name="libero_10", evaluation_config_path=str(path))

    return factory


@pytest.mark.unit
@pytest.mark.parametrize("perturbed_suite", [None, "libero_10_lan"])
def test_missing_files_stop_startup_before_wandb_and_server(
    perturbed_suite: str | None,
) -> None:
    error = FileNotFoundError("Missing evaluation files")
    validation_results = [error] if perturbed_suite is None else [None, error]
    config = EvalConfig(task_suite_name="libero_10", use_wandb=True)

    with (
        patch(
            "versatil_inference.run_evaluation.check_assets",
            side_effect=validation_results,
        ) as validate,
        patch(
            "versatil_inference.run_evaluation.setup_perturbations",
            return_value=perturbed_suite,
        ) as perturb,
        patch("versatil_inference.run_evaluation.LiberoServer") as server,
        patch("versatil_inference.run_evaluation.wandb.init") as wandb_init,
        pytest.raises(FileNotFoundError, match="^Missing evaluation files$"),
    ):
        run_evaluation(config=config)

    expected = [call(task_suite_name="libero_10")]
    if perturbed_suite is not None:
        expected.append(call(task_suite_name=perturbed_suite))
        perturb.assert_called_once_with(config)
    else:
        perturb.assert_not_called()
    assert validate.call_args_list == expected
    server.assert_not_called()
    wandb_init.assert_not_called()


@pytest.mark.integration
def test_disabled_perturbations_require_no_generation_paths(
    evaluation_config_factory: Callable[..., EvalConfig],
) -> None:
    config = evaluation_config_factory(use_language=False)
    Path(config.evaluation_config_path).write_text(
        "use_language: false\n", encoding="utf-8"
    )

    assert setup_perturbations(config=config) == "libero_10"


@pytest.mark.integration
def test_perturbation_paths_resolve_beside_config(
    evaluation_config_factory: Callable[..., EvalConfig],
    path_config_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    config = evaluation_config_factory(use_language=True)
    root = Path(config.evaluation_config_path).parent
    data = tmp_path / "robotics_assets/libero_pro"
    path_config_factory(
        values={
            "bddl_files": str(data / "bddl_files"),
            "init_states": str(data / "init_files"),
        }
    )
    with patch(
        "versatil_inference.run_evaluation._setup_single_perturbation",
        return_value="libero_10_lan",
    ) as setup:
        suite = setup_perturbations(config=config)

    assert suite == "libero_10_lan"
    resolved = setup.call_args.kwargs["evaluation_config"]
    assert resolved["bddl_files_path"] == str(data / "bddl_files/libero_10")
    assert resolved["init_file_dir"] == str(data / "init_files") + "/"
    assert resolved["script_path"] == str(root / "notebooks/generate_init_states.py")
    assert resolved["ood_task_configs"]["language"] == str(
        root / "libero_ood/language.yaml"
    )
