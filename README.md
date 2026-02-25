# LIBERO-Plus with Socket Server Integration

This repository integrates the **LIBERO-Plus** simulation framework with a ZMQ-based Socket Server that enables remote policy evaluation. The server can be used together with a Policy Client from the [VersatIL library](https://gitlab.com/nct_tso_public/versatil), allowing the policy learning library to remain independent of the simulation engine.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Machine A (GPU Server)                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     VersatIL Policy Client                            │  │
│  │  - Loads trained checkpoint                                           │  │
│  │  - Receives observations via ZMQ                                      │  │
│  │  - Computes actions using the policy                                  │  │
│  │  - Sends actions back to simulation server                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                              ZMQ Socket
                           (TCP connection)
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Machine B (Simulation Server)                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                   LIBERO-Plus Policy Server                           │  │
│  │  - Runs LIBERO-Plus simulation environment                            │  │
│  │  - Handles environment resets and stepping                            │  │
│  │  - Sends observations (images, proprioception)                        │  │
│  │  - Receives and executes actions                                      │  │
│  │  - Tracks episode success/failure                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

This repository uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

- Python 3.10+
- Conda/Mamba (for environment management)
- Git credentials for private GitLab repositories (see below)

### Setup Steps

1. **Configure git credentials** (required for private dependencies):
   ```bash
   git config --global credential.helper store
   ```
   Then authenticate once to GitLab (e.g., by cloning a private repo). This stores credentials so `uv` can fetch the `imitation-learning-toolkit` dependency.

2. **Create a mamba environment with system dependencies and uv:**
   ```bash
   mamba create -n libero_plus python=3.10 libexpat fontconfig imagemagick cmake -c conda-forge -y
   mamba activate libero_plus
   mamba install uv -c conda-forge -y
   ```

3. **Install dependencies using uv with the conda prefix:**
   ```bash
   CMAKE_POLICY_VERSION_MINIMUM=3.5 UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync
   ```

### Assets

Download assets from [LIBERO-Plus on HuggingFace](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main) and unzip `assets.zip` to `libero/libero/`:

```text
LIBERO-plus/
└── libero/
    └── libero/
        └── assets/
            ├── articulated_objects/
            ├── new_objects/
            ├── scenes/
            ├── stable_hope_objects/
            ├── stable_scanned_objects/
            ├── textures/
            ├── turbosquid_objects/
            ├── serving_region.xml
            ├── wall_frames.stl
            └── wall.xml
```

### Configuration

If you have LIBERO installed, uninstall it first. Verify the config path at `~/.libero/config.yaml` points to this repo (see `libero/libero/__init__.py`).

### Private Dependencies

This project depends on [imitation-learning-toolkit](https://gitlab.com/nct_tso_public/imitation-learning-toolkit), a shared library for socket communication. The dependency is specified in `pyproject.toml` as a git source:

```toml
[tool.uv.sources]
imitation-learning-toolkit = { git = "https://gitlab.com/nct_tso_public/imitation-learning-toolkit.git" }
```

## Running Evaluation

The evaluation uses a **server-client architecture** where the simulation server and the policy client communicate over ZMQ. The server runs environments in **parallel batches** (`max_parallel_envs` at a time) for fast evaluation.

### Step 1: Start the LIBERO-Plus Simulation Server

On the simulation machine, run:

```bash
python -m versatil_inference.run_evaluation \
    --task_suite_name libero_plus_spatial \
    --output_folder ./results
```

**Configuration options:**
- `--task_suite_name`: LIBERO-Plus benchmark suite (`libero_plus_spatial`, `libero_plus_object`, `libero_plus_goal`, `libero_plus_10`, `libero_90`, `libero_plus_all`)
- `--ip_address`: IP to bind the server (default: `0.0.0.0`)
- `--port`: Port for ZMQ communication (default: `5556`)
- `--num_trials_per_task`: Number of episodes per task (default: `1`). **NOTE:** LIBERO-Plus uses 1 trial per task because each perturbation variant is already a separate task (2400+ per suite)
- `--max_parallel_envs`: Maximum environments to run in parallel per batch (default: `10`)
- `--resolution`: Image resolution (default: `256`)
- `--compression_type`: Image compression (`raw`, `jpeg`, `png`)
- `--output_folder`: Base directory for rollout videos, trajectory CSVs, and results. Output is structured as `{output_folder}/{client_name}/{task_suite}/{date}/`. If not set, defaults to `{client_checkpoint_dir}/rollouts/{checkpoint_name}/{task_suite}/{date}/`
- `--seed`: Random seed for reproducibility (default: `7`)
- `--use_wandb`: Enable WandB logging (default: `True`)
- `--record_wrist_camera`: Record wrist camera observations (default: `False`)

### Step 2: Run the Policy Client (VersatIL)

On the policy machine (can be the same or different), use the VersatIL simulation client:

```bash
python -m versatil.endpoints.test \
    --checkpoint_path "/path/to/checkpoints" \
    --checkpoint_name "best.ckpt" \
    --server_address <server_ip> \
    --server_port 5556
```

The client connects to the server, receives observations, runs inference, and sends actions back. Results (per-task success rates, rollout videos, trajectory CSVs) are saved on the server side.

For the full VersatIL client documentation, see: https://gitlab.com/nct_tso_public/versatil

---

# Original LIBERO-Plus README

*The following is the original README from the LIBERO-Plus benchmark for reference.*

---

<h1 align="center">
LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models
</h1>

<p align="center">
  📄 <a href="https://arxiv.org/pdf/2510.13626"><strong>Paper</strong></a> |
  🏗️ <a href="https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main"><strong>Assets</strong></a> |
  🌐 <a href="https://sylvestf.github.io/LIBERO-plus"><strong>Website</strong></a> |
  🤗 <a href="https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata"><strong>Model</strong></a> |
  📁 <a href="https://huggingface.co/datasets/Sylvest/libero_plus_rlds"><strong>RldsDataset</strong></a>
  📁 <a href="https://huggingface.co/datasets/Sylvest/libero_plus_lerobot"><strong>LerobotDataset</strong></a>
</p>

![libero-plus](./static/images/libero-plus.jpg)

## 🔥 Overview
This repository contains the official implementation and benchmark for our paper "In-depth Robustness Analysis for Vision-Language-Action Models". We systematically expose the hidden vulnerabilities of contemporary VLA models through comprehensive robustness evaluation across seven perturbation dimensions. You can simply replace the original `libero` with a `pip install -e .` without modifying your code.

## 🚀 Key Findings
- **Significant Fragility**: VLA models exhibit extreme sensitivity to camera viewpoints and robot initial states, with performance dropping from 95% to below 30% under modest perturbations
- **Language Ignorance**: Models largely ignore language instructions, functioning more like Vision-Action models
- **Negative Compositional Generalization**: Combined perturbations reveal complex interaction effects beyond independent factors

## 📊 LIBERO-plus Benchmark

### 7 Perturbation Dimensions
We introduce **LIBERO-plus**, a comprehensive benchmark with 10,030 tasks spanning:

1. **Objects Layout** - Confounding objects and target object displacement
2. **Camera Viewpoints** - Position, orientation, and field-of-view changes
3. **Robot Initial States** - Manipulator initial pose variations
4. **Language Instructions** - LLM-based instruction rewriting
5. **Light Conditions** - Intensity, direction, color, and shadow variations
6. **Background Textures** - Scene and surface appearance changes
7. **Sensor Noise** - Photometric distortions and image degradation

### Evaluated Models
- OpenVLA and variants (OFT, OFT_w, OFT_m)
- π₀ and π₀-fast
- Nora, WorldVLA, UniVLA, RIPT-VLA

## 🔧 Evaluation
The evaluation method is almost identical to `LIBERO`. The only required modification is adjusting `num_trials_per_task` from 50 to 1 in your configuration.

The mapping between task IDs, perturbation categories, and difficulty levels is provided in `.libero/libero/benchmark/task_classification.json`.

## 📊 LIBERO-Plus Benchmark Leaderboard
| Model | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|-------|--------|-------|----------|-------|------------|-------|--------|-------|
| [OpenVLA](https://github.com/openvla/openvla) | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| [OpenVLA-OFT_w](https://github.com/moojink/openvla-oft) | 10.4 | 38.7 | 70.5 | 76.8 | 93.6 | 49.9 | 69.9 | 55.8 |
| [NORA](https://github.com/declare-lab/nora) | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| [WorldVLA](https://github.com/alibaba-damo-academy/WorldVLA) | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| [UniVLA](https://github.com/OpenDriveLab/UniVLA) | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 43.9 |
| [π₀](https://github.com/Physical-Intelligence/openpi) | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| [π₀-Fast](https://github.com/Physical-Intelligence/openpi) | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| [RIPT-VLA](https://github.com/Ariostgx/ript-vla) | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| [OpenVLA-OFT_m](https://github.com/moojink/openvla-oft) | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 67.9 |
| **[OpenVLA-OFT+ (Ours)](https://github.com/moojink/openvla-oft)** | **92.8** | **30.3** | **85.8** | **94.9** | **93.9** | **89.3** | **77.6** | **79.6** |

- **OpenVLA-OFT+** shows the performance of [OpenVLA-OFT with a mix-sft on LIBERO-plus dataset](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata/tree/main).
- **OpenVLA-OFT_w** shows the performance of [OpenVLA-OFT without wrist observation input](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-without-wrist).
- **OpenVLA-OFT_m** shows the performance of [OpenVLA-OFT with a mix-sft](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10).


### Origin LIBERO Benchmark Leaderboard

To make it easier to get all the results in one place, we've compiled the evaluation results of current VLA models on the original LIBERO benchmark in this [table](./libero_res.md).


### Research Works Using LIBERO-Plus

The following projects have adopted **LIBERO-Plus** for evaluating their VLA models. We appreciate the community’s support and will continue to maintain and expand the benchmark.

If your project uses LIBERO-Plus, please submit a PR to add your work.

- AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention. (Avg 74.7) [[pdf]](https://arxiv.org/pdf/2511.18960)
- MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent. (Avg 72.2) [[pdf]](https://arxiv.org/pdf/2511.18810) [[github]](https://github.com/MergeVLA/MergeVLA) 
- SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models. (Avg 82.1) [[pdf]](https://arxiv.org/pdf/2511.15605) [[github]](https://github.com/sii-research/siiRL) [[hf-model]](https://huggingface.co/collections/Sylvest/srpo) 

🌱 ***More works are being added…***


## Citation
If you find this work useful for your research, please cite our paper:
```bibtex
@article{fei25libero-plus,
    title={LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models},
    author={Senyu Fei and Siyin Wang and Junhao Shi and Zihao Dai and Jikun Cai and Pengfang Qian and Li Ji and Xinzhe He and Shiduo Zhang and Zhaoye Fei and Jinlan Fu and Jingjing Gong and Xipeng Qiu},
    journal = {arXiv preprint arXiv:2510.13626},
    year={2025},
}
```
