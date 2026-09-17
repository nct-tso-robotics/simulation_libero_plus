"""Fixtures for external evaluation data and simulator path configuration."""

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

import libero.libero as libero
from versatil_inference import check_assets


@pytest.fixture
def path_config_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., Path]:
    path = tmp_path / "config" / "config_libero_plus.yaml"
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
