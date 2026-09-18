"""Fixtures for external evaluation data and simulator path configuration."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

import libero.libero as libero
from libero.libero import get_default_path_dict
from libero.libero.benchmark import Benchmark, get_benchmark_dict
from versatil_inference import check_assets


@pytest.fixture
def external_benchmark_factory(
    path_config_factory: Callable[..., Path],
) -> Callable[..., Benchmark]:
    def factory(suite_name: str) -> Benchmark:
        data_root = os.environ.get("LIBERO_TEST_DATA_ROOT")
        if data_root is None:
            pytest.skip("Set LIBERO_TEST_DATA_ROOT to run external-asset tests.")
        path_config_factory(values=get_default_path_dict(custom_location=data_root))
        return get_benchmark_dict()[suite_name]()

    return factory


@pytest.fixture
def path_config_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., Path]:
    path = tmp_path / "config" / "config_libero_plus.yaml"
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(path.parent))
    monkeypatch.setattr(libero, "config_file", str(path))
    monkeypatch.setattr(check_assets, "config_file", str(path))

    def factory(values: dict | None = None) -> Path:
        if values is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(values), encoding="utf-8")
        return path

    return factory


@pytest.fixture
def evaluation_files_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_config_factory: Callable[..., Path],
) -> Callable[..., dict[str, Path]]:
    def factory(
        task_name: str = "pick_up_object_view_1",
        bddl_name: str = "pick_up_object_view_1",
    ) -> dict[str, Path]:
        root = tmp_path / "robotics_assets/libero_plus"
        paths = {
            "assets": root / "assets",
            "bddl_files": root / "bddl_files",
            "init_states": root / "init_files",
        }
        path_config_factory(values={key: str(path) for key, path in paths.items()})
        for pattern in check_assets.ASSET_PATTERNS:
            path = paths["assets"] / pattern.replace("*", "example")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"asset")
        for path in (
            paths["bddl_files"] / "libero_10" / f"{bddl_name}.bddl",
            paths["init_states"] / "libero_10/pick_up_object.pruned_init",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"task data")
        monkeypatch.setattr(check_assets, "libero_task_map", {"libero_10": [task_name]})
        return paths

    return factory
