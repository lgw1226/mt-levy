# MT-Lévy: Leveraging Temporally Extended Behavior Sharing for Multi-task Reinforcement Learning

This repository contains the official implementation for the paper: **Leveraging Temporally Extended Behavior Sharing for Multi-task Reinforcement Learning**.

**Authors:** Gawon Lee, Daesol Cho, H. Jin Kim

**Conference:** Accepted for publication in the proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

Link to paper: coming soon.

## Overview

Applying multi-task reinforcement learning (MTRL) to robotics is often hindered by the high cost of collecting diverse data. To address this sample inefficiency, we propose **MT-Lévy**, a novel exploration strategy that combines three key ideas to enhance learning in MTRL environments:

-   **Behavior Sharing:** Leverages policies from related, already-solved tasks to guide exploration toward useful states.
-   **Temporally Extended Exploration:** Executes exploratory actions for extended durations, with the duration sampled from a Lévy flight distribution. This encourages the agent to make significant, long-range "jumps" across the state space.
-   **Adaptive Exploration:** Automatically adjusts the intensity and frequency of exploration based on task success rates, balancing the need to explore with the need to exploit learned knowledge.

Our experiments on the MT10 multi-task manipulation benchmark show that MT-Lévy significantly improves sample efficiency and asymptotic performance in both dense and sparse reward settings.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/gawon-lee/mt-levy.git
    cd mt-levy
    ```

2.  **Set up the environment and install dependencies:**

    This project is managed with [`uv`](https://docs.astral.sh/uv/). You can install it with the
    official installer, which ensures the CLI is available on your `PATH` without polluting the
    project environment:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Once `uv` is installed, synchronize the environment (this creates `.venv/` automatically and
    installs the locked dependencies from `uv.lock`):

    ```bash
    uv sync
    ```

    If you prefer to work inside the virtual environment directly, activate it after syncing:

    ```bash
    source .venv/bin/activate
    ```

    > **Note:** You do **not** need to activate the virtual environment to run commands with
    > `uv`. All examples below can be prefixed with `uv run ...` to execute them inside the managed
    > environment without manual activation.

    Alternatively, if you cannot install `uv`, you can create and populate a standard virtual
    environment instead:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install .
    ```

## Usage

To start training, run the `main.py` script. The project uses [Hydra](https://hydra.cc/) for configuration, allowing you to easily modify parameters from the command line. With `uv`, execute the commands using `uv run` to avoid manual activation of the virtual environment. If you activated `.venv/` manually, you can drop the `uv run` prefix and call `python` directly.

### Training with MT-Lévy

To train an agent using the **MT-Lévy** exploration strategy, run:

```bash
uv run python main.py exploration_strategy=mtlevy
```

Evaluation is performed automatically at the end of each training epoch.

### Other Exploration Strategies

You can experiment with other exploration strategies:

-   **Standard SAC (no special exploration):**
    ```bash
    uv run python main.py exploration_strategy=base
    ```
-   **Quality-Metric-based Planning (QMP):**
    ```bash
    uv run python main.py exploration_strategy=qmp
    ```

### Overriding Configuration

You can override any parameter from the configuration files (`conf/*.yaml`) directly from the command line.

For example, to train for more epochs with a different seed:
```bash
uv run python main.py training.num_epochs=500 seed=42
```

To run on a different GPU:
```bash
uv run python main.py gpu_index=1
```

Refer to the files in the `conf/` directory for a full list of configurable parameters.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{lee2025mtlevy,
  title={Leveraging Temporally Extended Behavior Sharing for Multi-task Reinforcement Learning},
  author={Lee, Gawon and Cho, Daesol and Kim, H. Jin},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025},
  organization={IEEE}
}
```
