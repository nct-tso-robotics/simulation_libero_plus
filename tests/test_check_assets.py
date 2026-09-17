"""Tests for versatil_inference.check_assets."""

from collections.abc import Callable
from pathlib import Path

import pytest

from versatil_inference.check_assets import check_assets

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "task_name,bddl_name",
    [
        ("pick_up_object_view_1", "pick_up_object_view_1"),
        ("pick_up_object_view_1_initstate_10", "pick_up_object"),
    ],
)
def test_external_files_pass_with_shared_initial_states(
    evaluation_files_factory: Callable[..., dict[str, Path]],
    capsys: pytest.CaptureFixture,
    task_name: str,
    bddl_name: str,
) -> None:
    paths = evaluation_files_factory(task_name=task_name, bddl_name=bddl_name)
    check_assets(task_suite_name="libero_plus_10")
    output = capsys.readouterr().out
    for path in paths.values():
        assert str(path) in output
    assert "Evaluation files found for 1 tasks" in output


@pytest.mark.parametrize(
    "key,filename",
    [
        ("assets", "new_objects/example/example/usd/MJCF/example.xml"),
        ("bddl_files", "libero_10/pick_up_object_view_1.bddl"),
        ("init_states", "libero_10/pick_up_object.pruned_init"),
    ],
)
def test_missing_files_report_external_paths_and_configuration(
    evaluation_files_factory: Callable[..., dict[str, Path]], key: str, filename: str
) -> None:
    paths = evaluation_files_factory(task_name="pick_up_object_view_1")
    missing = paths[key] / filename
    missing.unlink()
    with pytest.raises(FileNotFoundError) as error:
        check_assets(task_suite_name="libero_plus_10")
    assert str(paths[key]) in str(error.value)
    assert "config_libero_plus.yaml" in str(error.value)


def test_unknown_suite_reports_its_name() -> None:
    with pytest.raises(ValueError, match="^Unknown LIBERO-plus task suite: libero_1$"):
        check_assets(task_suite_name="libero_1")
