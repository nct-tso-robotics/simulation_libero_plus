"""Runs Libero-plus evaluation via ZMQ communication with a policy client."""

import logging as _logging
import os

# Redirect robosuite log to user-writable path (it hardcodes /tmp/robosuite.log)
_OrigFileHandler = _logging.FileHandler


class _SafeFileHandler(_OrigFileHandler):
    def __init__(self, filename, *args, **kwargs):
        if "robosuite" in str(filename):
            filename = os.path.expanduser("~/robosuite.log")
        super().__init__(filename, *args, **kwargs)


_logging.FileHandler = _SafeFileHandler

import datetime
from dataclasses import dataclass
from typing import Optional

import draccus
import wandb

from versatil_inference.server import LiberoServer
from versatil_inference.socket_flags import (
    LiberoResponseKey,
    LiberoStatus,
    TaskSuiteName,
)

DATE_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class EvalConfig:
    """Configuration for Libero-plus evaluation.

    LIBERO-plus uses 1 trial per task because each perturbation variant is
    already a separate task (2400+ per suite). Max episode length is 600
    steps matching the official evaluation config.
    """

    task_suite_name: str = TaskSuiteName.LIBERO_OBJECT.value
    num_steps_wait: int = 5
    num_trials_per_task: int = 1
    resolution: int = 256
    ip_address: str = "0.0.0.0"
    port: int = 5556
    compression_type: str = "raw"
    output_folder: str = ""
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = True
    wandb_project: str = "libero-plus-eval"
    wandb_entity: str = ""
    seed: int = 7
    max_parallel_envs: int = 10
    render_gpu_device_id: int = -1
    record_wrist_camera: bool = False


def run_evaluation(config: EvalConfig) -> None:
    """Create the server and run the evaluation loop.

    Args:
        config: Evaluation configuration.
    """
    task_suite_name = config.task_suite_name
    run_id = f"EVAL-{task_suite_name}-{DATE_TIME}"
    if config.run_id_note:
        run_id += f"--{config.run_id_note}"
    os.makedirs(config.local_log_dir, exist_ok=True)
    if config.use_wandb:
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=run_id,
        )
    server = LiberoServer(
        task_suite_name=task_suite_name,
        ip_address=config.ip_address,
        port=config.port,
        resolution=config.resolution,
        num_steps_wait=config.num_steps_wait,
        num_trials_per_task=config.num_trials_per_task,
        output_folder=config.output_folder,
        seed=config.seed,
        compression_type=config.compression_type,
        max_parallel_envs=config.max_parallel_envs,
        render_gpu_device_id=config.render_gpu_device_id,
        record_wrist_camera=config.record_wrist_camera,
    )
    print(
        f"Task suite: {task_suite_name}, "
        f"Waiting for client on tcp://{config.ip_address}:{config.port}"
    )
    try:
        while True:
            response = server.handle_client_request()
            if (
                response.get(LiberoResponseKey.STATUS.value)
                == LiberoStatus.FINISHED.value
            ):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        server.shutdown()
    rollout_dir = server.environment.rollout_directory
    rollout_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = str(rollout_dir / "log.txt")
    _log_results(
        server=server,
        config=config,
        task_suite_name=task_suite_name,
        log_filepath=log_filepath,
    )
    print(f"Log saved to: {log_filepath}")


def _log_results(
    server: LiberoServer,
    config: EvalConfig,
    task_suite_name: str,
    log_filepath: str,
) -> None:
    """Write per-task and overall results to local log file and wandb.

    Args:
        server: Server containing the evaluated environment.
        config: Evaluation configuration.
        task_suite_name: Resolved task suite name.
        log_filepath: Path for the local log file.
    """
    environment = server.environment
    if config.use_wandb:
        wandb.config.update({
            "client_name": environment.client_name,
            "task_suite_name": task_suite_name,
        })
    total_episodes = sum(environment.number_of_resets)
    total_successes = sum(environment.environments_successes)
    overall_rate = (
        total_successes / total_episodes if total_episodes > 0 else 0.0
    )
    with open(log_filepath, "w") as log_file:
        log_file.write(f"Task suite: {task_suite_name}\n")
        log_file.write(f"Total: {total_successes}/{total_episodes}\n")
        log_file.write(f"Overall success rate: {overall_rate:.4f}\n\n")
        for index in range(len(environment.task_descriptions)):
            description = environment.task_descriptions[index]
            successes = environment.environments_successes[index]
            episodes = environment.number_of_resets[index]
            task_rate = successes / episodes if episodes > 0 else 0.0
            log_file.write(
                f"{description}: {successes}/{episodes} ({task_rate:.4f})\n"
            )
            if config.use_wandb:
                wandb.log(
                    {
                        f"success_rate/{description}": task_rate,
                        f"num_episodes/{description}": episodes,
                    }
                )
        unique_suites = list(dict.fromkeys(environment.suite_name_per_task))
        if len(unique_suites) > 1:
            log_file.write("\n")
            for suite_name in unique_suites:
                suite_successes = sum(
                    environment.environments_successes[i]
                    for i in range(len(environment.suite_name_per_task))
                    if environment.suite_name_per_task[i] == suite_name
                )
                suite_episodes = sum(
                    environment.number_of_resets[i]
                    for i in range(len(environment.suite_name_per_task))
                    if environment.suite_name_per_task[i] == suite_name
                )
                suite_rate = (
                    suite_successes / suite_episodes
                    if suite_episodes > 0
                    else 0.0
                )
                log_file.write(
                    f"{suite_name}: {suite_successes}/{suite_episodes} "
                    f"({suite_rate:.4f})\n"
                )
                if config.use_wandb:
                    wandb.log(
                        {
                            f"success_rate/{suite_name}": suite_rate,
                            f"num_episodes/{suite_name}": suite_episodes,
                        }
                    )
    if config.use_wandb:
        wandb.log(
            {
                "success_rate/total": overall_rate,
                "num_episodes/total": total_episodes,
            }
        )
    print(f"\nFinal success rate: {overall_rate * 100:.1f}%")


@draccus.wrap()
def eval_libero(config: EvalConfig) -> None:
    """Entry point for Libero-plus evaluation."""
    run_evaluation(config=config)


if __name__ == "__main__":
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    eval_libero()
