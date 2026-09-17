"""Tests for versatil_inference.run_evaluation startup checks."""

from collections.abc import Callable
from unittest.mock import patch

import pytest

from versatil_inference.run_evaluation import EvalConfig, run_evaluation


@pytest.fixture
def evaluation_config_factory() -> Callable[..., EvalConfig]:
    def factory(task_suite_name: str, use_wandb: bool) -> EvalConfig:
        return EvalConfig(task_suite_name=task_suite_name, use_wandb=use_wandb)

    return factory


@pytest.mark.unit
def test_missing_assets_stop_startup_before_wandb_and_server(
    evaluation_config_factory: Callable[..., EvalConfig],
) -> None:
    config = evaluation_config_factory(task_suite_name="libero_plus_10", use_wandb=True)
    with (
        patch(
            "versatil_inference.run_evaluation.check_assets",
            side_effect=FileNotFoundError("Missing evaluation files"),
        ) as check,
        patch("versatil_inference.run_evaluation.LiberoServer") as server,
        patch("versatil_inference.run_evaluation.wandb.init") as wandb_init,
        pytest.raises(FileNotFoundError, match="^Missing evaluation files$"),
    ):
        run_evaluation(config=config)
    check.assert_called_once_with(task_suite_name="libero_plus_10")
    server.assert_not_called()
    wandb_init.assert_not_called()
