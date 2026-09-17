"""Tests for rendering LIBERO-plus variants with external evaluation data."""

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from libero.libero import benchmark, get_default_path_dict
from libero.libero.envs import OffScreenRenderEnv

pytestmark = pytest.mark.integration


@pytest.fixture
def environment_factory(
    path_config_factory: Callable[..., Path],
) -> Iterator[Callable[..., OffScreenRenderEnv]]:
    data_root = os.environ.get("LIBERO_TEST_DATA_ROOT")
    if data_root is None:
        pytest.skip("Set LIBERO_TEST_DATA_ROOT to run external-asset rendering tests.")
    path_config_factory(values=get_default_path_dict(custom_location=data_root))
    with ExitStack() as cleanup:

        def factory(task_name: str) -> OffScreenRenderEnv:
            suite = benchmark.get_benchmark_dict()["libero_10"]()
            task_index = suite.get_task_names().index(task_name)
            environment = OffScreenRenderEnv(
                bddl_file_name=suite.get_task_bddl_file_path(task_index),
                camera_heights=64,
                camera_widths=64,
            )
            cleanup.callback(environment.close)
            environment.reset()
            environment.set_init_state(suite.get_task_init_states(task_index)[0])
            return environment

        yield factory


@pytest.mark.parametrize(
    "task_name",
    [
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_add_14",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_1",
    ],
)
def test_variants_render_and_step_with_external_assets(
    environment_factory: Callable[..., OffScreenRenderEnv], task_name: str
) -> None:
    environment = environment_factory(task_name=task_name)
    observation, reward, done, info = environment.step(
        np.zeros(7, dtype=np.float32)  # (7,)
    )
    for camera in ("agentview_image", "robot0_eye_in_hand_image"):
        image = observation[camera]  # (64, 64, 3)
        assert image.shape == (64, 64, 3)
        assert np.isfinite(image).all()
        assert image.max() > image.min()
    assert Path(environment.env.custom_asset_dir).is_relative_to(
        Path(os.environ["LIBERO_TEST_DATA_ROOT"])
    )
