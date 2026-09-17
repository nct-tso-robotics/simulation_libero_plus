"""Check external LIBERO-plus evaluation files before starting a rollout."""

import argparse
from pathlib import Path

from libero.libero import config_file, get_libero_path
from libero.libero.benchmark import get_task_init_states_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from versatil_inference.constants import (
    LIBERO_ALL_SUITES,
    SUITE_TO_BENCHMARK_KEY,
    TaskSuiteName,
)

ASSET_PATTERNS = (
    "scenes/*.xml",
    "textures/*.png",
    "articulated_objects/*.xml",
    "stable_scanned_objects/*/*.xml",
    "stable_hope_objects/*/*.xml",
    "turbosquid_objects/*/*.xml",
    "new_objects/*/*/usd/MJCF/*.xml",
)


def check_assets(task_suite_name: str) -> None:
    """Validate asset groups and the selected tasks' definitions and saved states.

    Args:
        task_suite_name: Evaluation suite name, such as libero_plus_10.

    Raises:
        ValueError: If the suite name or path configuration is invalid.
        FileNotFoundError: If required evaluation files are missing or empty.
    """
    suites = (
        LIBERO_ALL_SUITES
        if task_suite_name == TaskSuiteName.LIBERO_ALL.value
        else [task_suite_name]
    )
    for suite in suites:
        if suite not in SUITE_TO_BENCHMARK_KEY:
            raise ValueError(f"Unknown LIBERO-plus task suite: {suite}")

    assets = Path(get_libero_path(query_key="assets"))
    bddl_files = Path(get_libero_path(query_key="bddl_files"))
    init_states = Path(get_libero_path(query_key="init_states"))
    print(f"LIBERO-plus path configuration: {config_file}")
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
        benchmark_key = SUITE_TO_BENCHMARK_KEY[suite]
        for task_name in libero_task_map[benchmark_key]:
            task_count += 1
            bddl_name = task_name
            if "_view_" in task_name and "_initstate_" in task_name:
                bddl_name = task_name.split("_view_", maxsplit=1)[0]
            for path in (
                bddl_files / benchmark_key / f"{bddl_name}.bddl",
                Path(
                    get_task_init_states_path(
                        task_name=task_name, suite_name=benchmark_key
                    )
                ),
            ):
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(path))
    if missing:
        details = "\n".join(f"  {path}" for path in missing[:10])
        if len(missing) > 10:
            details += f"\n  ... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"LIBERO-plus evaluation files are missing or empty:\n{details}\n"
            "Download the evaluation files using the README's asset setup steps.\n"
            f"Set assets, bddl_files and init_states in {config_file} "
            "to the downloaded directories under robotics_assets."
        )
    print(f"Evaluation files found for {task_count} tasks ({', '.join(suites)}).")


def main() -> None:
    """Check an evaluation suite from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_suite_name", default=TaskSuiteName.LIBERO_10.value)
    arguments = parser.parse_args()
    check_assets(task_suite_name=arguments.task_suite_name)


if __name__ == "__main__":
    main()
