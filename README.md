<div align="center">
    <div style="margin-bottom: 20px">
        <!-- <h1 style="display: inline;"> -->
        <h1>
            Guided Flow Policy: </br>
            Learning from High-Value Actions in Offline Reinforcement Learning
        </h1>
        <!-- <span style="font-size: 1.4em"> -->
        <h2>
            <!-- &emsp;&emsp; -->
            <a href="https://arxiv.org/abs/2512.03973">
                ArXiv
            </a>
            &emsp;&emsp;
            <a href="https://hal.science/hal-05400311">
                HAL
            </a>
            &emsp;&emsp;
            <a href="https://simple-robotics.github.io/publications/guided-flow-policy">
                Webpage
            </a>
        </h2>
        <!-- </span> -->
    </div>
</div>

**Guided Flow Policy (GFP)** is an offline RL method based on flow matching. 
It couples a multi-step flow-matching policy trained with value-aware behavior cloning and a distilled one-step actor through a bidirectional guidance mechanism. 
This enables GFP to achieve state-of-the-art performance across 144 state and pixel-based tasks from the OGBench, Minari, and D4RL benchmarks, with substantial gains on suboptimal datasets and challenging tasks.

<div align="center">
    <img src="figures/figure-gfp-overview.png" width="90%">
</div>

# Features
This repository was forked from [FQL](https://github.com/seohongpark/fql), keeping the overall structure. Compared to the original code base:
- We added Guided Flow Policy in the [agent folder](agents/gfp.py).
- We changed the config management system to [Hydra](https://hydra.cc/), for very convenient configs and command line overrides, see [the usage section](#usage)
- In the [results](results/) folder, we share **csv files of all our benchmarking results**. In particular, it includes the 144 tasks GFP was evaluated on (see [gfp_results](results/per_task/gfp-actor_per_task_results.csv)), the extensive reevaluation of existing baselines (e.g. [rebrac_results](results/per_task/rebrac_per_task_results.csv)) on [OGBench](https://github.com/seohongpark/ogbench), and the first evaluation of GFP and FQL on [Minari](https://minari.farama.org/).
- In the [hyperparams](hyperparams/) folder, we provide the exact hyperparameters used to generate these results in csv files (e.g. for [gfp](hyperparams/gfp_params.csv)). For ease of reproduction, our [main script](main.py) recovers from these csv files the best hyperparameters for each task and method. Likewise, for pixel-based environments, it automatically selects the *encoder*, *p_aug* and *frame_stack*.
- We added an option to fuse several training steps together in the `jax.jit` compilation, thereby reducing the overhead (for this, `dataset_on_gpu=True` is needed). By default, `n_fused_steps=4`.
- To reduce the evaluation overhead during training, we implemented a parallel evaluation function, using parallel Gym environments.

# Installation
Create a Python environment if needed, for instance with conda:
```shell
conda create -n gfp python=3.12
conda activate gfp
```
Install the Jax version suited to your platform following [Jax installation guide](https://docs.jax.dev/en/latest/installation.html), for instance:
```shell
pip install "jax[cuda13]"
```
Install the other requirements (by default works with OGBench and Minari)
```shell
pip install -r requirements.txt
```

# Usage
We use [Hydra](https://hydra.cc/) to manage configs and command lines overrides. Given an env_name and an agent, the best hyperparameters are recovered from the [hyperparams folder](hyperparams/) if available. Here are some example commands:
```shell
# By default: env_name=cube-double-play ; agent=gfp
python main.py

# To precise the environment and the agent
python main.py env_name=antmaze-large-navigate-singletask-task1-v0 agent=gfp

# For a Minari task, and some hyperparameters overrides
python main.py env_name='D4RL/pen/expert-v2' agent.alpha=0.3 agent.eta_temperature=0.000001 offline_steps=200000
```

## Testing different configs/tasks in parallel with Hydra
Using [Hydra Multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) one can sweep over environments, agents or any hyperparameter. Greatly helping hyperparameter search and any tests. Note: for this a hydra launcher may be needed, see [Additional options/launcher](#additional-options).
```shell
# Sweep over two environments and two agents => launch 4 jobs
python main.py -m env_name=cube-triple-noisy-singletask-task1-v0,humanoidmaze-medium-navigate-singletask-task1-v0 agent=gfp,fql


# === Hyper parameter search ===
# Sweep over the alpha hyperparameter, using 2 seeds => 8 jobs
python main.py -m agent=gfp env_name=cube-double-play-singletask-v0 agent.alpha=3,1,0.3,0.1 seed=$RANDOM,$RANDOM
# Then sweep over the eta hyperparameter, using 2 seeds => 8 jobs
python main.py -m agent=gfp env_name=cube-double-play-singletask-v0 agent.eta_temperature=0.1,0.01,0.001,0.0001 seed=$RANDOM,$RANDOM agent.alpha=1


# Run all 5 sub tasks, each over 8 runs, to collect the final result
python main.py -m agent=gfp env_name=antmaze-large-navigate-singletask-task1-v0,antmaze-large-navigate-singletask-task2-v0,antmaze-large-navigate-singletask-task3-v0,antmaze-large-navigate-singletask-task4-v0,antmaze-large-navigate-singletask-task5-v0 seed=$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM is_final=True
```

## Additional options
We provide some optional hydra groups: [launcher](config/launcher/) and [opt](config/opt/):
- **Launcher** may be used to override *hydra.launcher* for multi-runs. We provide an example of how to launch multiple jobs using slurm on a cluster (needs `pip install hydra-submitit-launcher --upgrade`). Hydra optional parameters are added with a `+`:
```shell
python main.py +launcher=our_slurm
```
- **Opt** provides convenient shortcuts for overriding several arguments. For example, [light_log.yaml](config/opt/light_log.yaml) speeds up computation time by reporting only the final success rate:
```shell
python main.py +opt=light_log
```
Combining both, a quick hyperparameter search on a cluster can be done using:
```shell
python main.py -m +launcher=our_slurm +opt=light_log agent=gfp env_name=cube-double-play-singletask-v0 agent.alpha=3,1,0.3
```

# Miscellaneous  
## News & Updates
- 🟢 **2025-12-03** - Release of the paper on ArXiv 
- 🟢 **2026-01-26** - Paper accepted at ICLR
- 🟢 **2026-03-09** - Code released

## Citing Guided Flow Policy
```bibtex
@article{tiofack2025guided,
  title   = {Guided Flow Policy: Learning from High-Value Actions in Offline Reinforcement Learning},
  author  = {Nguimatsia Tiofack, Franki and Le Hellard, Th{\'e}otime and Schramm, Fabian and Perrin-Gilbert, Nicolas and Carpentier, Justin},
  journal = {arXiv preprint arXiv:2512.03973},
  year    = {2025}
}
```
