"""Tests for versatil_inference.check_assets."""

from collections.abc import Callable
from pathlib import Path

import pytest

from versatil_inference.check_assets import check_assets
from versatil_inference.socket_flags import LIBERO_ALL_SUITES

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("suite", ["libero_10", "libero_all", "libero_10_lan"])
def test_complete_evaluation_files_pass_without_training_demonstrations(
    evaluation_files_factory: Callable[..., dict[str, Path]],
    capsys: pytest.CaptureFixture,
    suite: str,
) -> None:
    suites = tuple(LIBERO_ALL_SUITES) if suite == "libero_all" else (suite,)
    paths = evaluation_files_factory(suites=suites)

    check_assets(task_suite_name=suite)

    output = capsys.readouterr().out
    assert f"Evaluation files found for {len(suites)} tasks" in output
    for path in paths.values():
        assert str(path) in output
    assert not (paths["assets"].parent / "datasets").exists()


@pytest.mark.parametrize(
    "key,filename",
    [
        ("assets", "textures/example.png"),
        ("bddl_files", "libero_10/pick_up_object.bddl"),
        ("init_states", "libero_10/pick_up_object.pruned_init"),
    ],
)
def test_missing_evaluation_files_report_paths_and_setup_instructions(
    evaluation_files_factory: Callable[..., dict[str, Path]], key: str, filename: str
) -> None:
    paths = evaluation_files_factory(suites=("libero_10",))
    missing = paths[key] / filename
    missing.unlink()

    with pytest.raises(FileNotFoundError) as error:
        check_assets(task_suite_name="libero_10")

    message = str(error.value)
    assert str(missing.parent) in message
    assert "README's asset setup steps" in message
    assert "config_libero_pro.yaml" in message
    assert "https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro" in message


def test_empty_initial_state_file_fails_before_loading_torch_checkpoint(
    evaluation_files_factory: Callable[..., dict[str, Path]],
) -> None:
    paths = evaluation_files_factory(suites=("libero_10",))
    initial_state = paths["init_states"] / "libero_10/pick_up_object.pruned_init"
    initial_state.write_bytes(b"")

    with pytest.raises(FileNotFoundError) as error:
        check_assets(task_suite_name="libero_10")

    assert str(initial_state) in str(error.value)


def test_unknown_suite_fails_explicitly(
    evaluation_files_factory: Callable[..., dict[str, Path]],
) -> None:
    evaluation_files_factory(suites=("libero_10",))
    with pytest.raises(ValueError, match="^Unknown LIBERO task suite: libero_1$"):
        check_assets(task_suite_name="libero_1")
