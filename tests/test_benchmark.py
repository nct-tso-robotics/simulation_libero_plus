"""Tests for LIBERO-plus initial-state path resolution."""

from collections.abc import Callable
from pathlib import Path

import pytest

from libero.libero.benchmark import get_task_init_states_path

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "suffix,relative_path",
    [
        ("", "libero_10/pick_up_object.pruned_init"),
        ("_view_1", "libero_10/pick_up_object.pruned_init"),
        ("_language_2", "libero_10/pick_up_object.pruned_init"),
        ("_light_3", "libero_10/pick_up_object.pruned_init"),
        ("_table_4", "libero_10/pick_up_object.pruned_init"),
        ("_add_1", "libero_newobj/libero_10/pick_up_object_add_1.pruned_init"),
    ],
)
def test_variants_resolve_their_saved_state_files(
    path_config_factory: Callable[..., Path],
    tmp_path: Path,
    suffix: str,
    relative_path: str,
) -> None:
    root = tmp_path / "external/init_files"
    path_config_factory(values={"init_states": str(root)})
    assert get_task_init_states_path(
        task_name="pick_up_object" + suffix, suite_name="libero_10"
    ) == str(root / relative_path)
