# TEKO Vision-Based Docking System

This repository is part of the master's thesis project
**"Adaptive Cooperation in Agricultural Robot Swarms: Reinforcement Learning and Evolutionary Algorithms for Modular Docking"**.

The work is being carried out at the **University of Hohenheim**,
Department of Artificial Intelligence in Agricultural Engineering,
under the supervision of **Prof. Dr. Anthony Stein** and **Dr. David Reiser**.

---

## 1. Project Overview

The TEKO project studies how small agricultural robots can **physically dock** to form modular units capable of performing tasks that would traditionally require larger, more complex machines.

This repository implements a **vision-only autonomous docking system**:

* A mobile TEKO robot must **locate, align, and connect** to a static TEKO goal robot.
* Perception is based purely on **RGB images** from a rear-mounted camera.
* Control is learned with **reinforcement learning (PPO)** in **NVIDIA Isaac Lab 0.47.1 / Isaac Sim 5.0**.
* The setup is designed to be later transferable to **real TEKO hardware**.

The code covers the full pipeline: CAD → USD/URDF models → simulated environment → RL training → logging and analysis.

---

## 2. Research Motivation

Modern agriculture faces:

* Labour shortages and high labour costs
* Increased climate variability and production risks
* Pressure to reduce inputs and environmental impact

Traditional solutions rely on **large, monolithic machines**, which are expensive and not always suitable for smaller or more diversified farms.

**Swarm and modular robotics** offer an alternative: many small, affordable units that can act **individually or cooperatively**. A key technical challenge is enabling **robust physical cooperation**, such as mechanical docking and resource sharing.

This project focuses on the **autonomous docking behaviour** itself, using vision-based RL to make two small robots **connect reliably without handcrafted docking sequences**.

---

## 3. Main Objectives

1. **Design and model** a docking-capable TEKO robot in simulation (USD/URDF, CAD-based geometry).
2. **Implement a realistic docking arena**, including a goal robot with an ArUco marker and a well-defined docking interface.
3. **Train a reinforcement learning agent** to perform docking using **only RGB input** from a rear camera.
4. **Introduce a multi-stage curriculum** that gradually increases task difficulty (distance, lateral offset, orientation).
5. **Prepare the pipeline for evolutionary hyperparameter optimisation** and later **sim-to-real transfer**.

---

## 4. System Overview

### 4.1 Simulation and Robot Models

The simulation is implemented in **NVIDIA Isaac Lab 0.47.1** (Isaac Sim 5.0):

* TEKO robot exported from **Fusion 360** as meshes and assembled into USD/URDF.
* The robot includes:
  * Chassis, four differential-drive wheels, body, roof and sensor mounts.
  * Back-mounted **camera module** emulating the Raspberry Pi Camera Module 2.
  * A **rear connector** (male/female) used for mechanical docking.
* A separate **static TEKO goal** is spawned as the docking target, equipped with:
  * An **ArUco marker** in front of the connector.
  * Spheres on both robots to define a geometric **docking distance**.

The arena (`stage_arena.usd`) defines walls and floor, constraining the robot to a controlled region.

### 4.2 Docking Geometry and Ground Truth

Docking quality is measured using **virtual spheres** placed in the connectors of both robots. The environment computes:

* **3D distance** between the connector spheres,
* **Projected XY distance** on the ground plane (`surface_xy`),

and uses these distances for:

* **Reward shaping** (distance, progress, proximity),
* **Success detection** (dock if `surface_xy < 0.03 m`),
* **Collision detection** (too fast / too close → heavy penalty).

This keeps the learning signal **geometric and consistent**, independent of the camera artefacts.

---

## 5. Reinforcement Learning Setup

### 5.1 Observations

* **Modality:** RGB images from the **rear camera** of the mobile robot.
* **Resolution:** `640 × 480` (3 channels, `float32` in `[0, 1]`).
* **Viewpoint:** The rear camera looks toward the docking interface and ArUco marker when the robot is correctly positioned.

### 5.2 Action Space

The policy outputs a **2D continuous action vector**:

* `v_cmd` – forward/backward command (linear component)
* `w_cmd` – turning command (angular component)

These commands are then **mapped inside the environment** to **wheel torques** for the left and right wheel pairs.

Previously, the agent directly produced `[left_torque, right_torque]`. While this is also 2D, it implicitly allowed many unstructured combinations (weak left / strong right, asymmetric strong torques, etc.), and the network had to discover the underlying structure on its own.

The new `[v_cmd, w_cmd]` parameterisation:

* Encodes the **natural structure of differential drive**,
* Makes the control space **more interpretable**,
* Encourages **smoother and more consistent** docking behaviour,
* Still remains a **continuous** action space.

### 5.3 Reward Function

The reward (see `reward_functions.py`) combines:

1. **Distance reward** – linear penalty with respect to connector distance.
2. **Progress reward** – positive when the robot reduces distance to the goal.
3. **Alignment reward** – based on the **rear of the robot** facing the goal; the yaw of the rear connector is aligned to the vector from robot to goal.
4. **Velocity penalty** – discourages excessive speed in the plane.
5. **Oscillation penalty** – penalises large changes in actions between timesteps.
6. **Collision penalty** – large negative reward when the robot approaches too fast and "crashes" instead of docking (anti-exploit).
7. **Boundary penalty** – large negative reward when leaving the arena.
8. **Success bonus** – strong positive reward when docking is successful (`surface_xy < 3 cm`).
9. **Proximity bonus** – extra reward when very close but not yet docked.
10. **Survival bonus** – small per-step reward to make "surviving and trying" better than crashing early.

This design encourages the agent to dock **quickly but safely**, rather than exploiting collisions or walls.

### 5.4 Policy Network and Visual Encoder

The policy is implemented in pure **PyTorch** (no external RL frameworks at runtime):

* **Encoder:** configurable CNN defined in `cnn_model.py`.
* **Actor head:** MLP mapping visual features → `[v_cmd, w_cmd]` (Tanh).
* **Critic head:** MLP mapping visual features → state value estimate.

Two encoder options exist:

1. **SimpleCNN (default)**
   * Lightweight, custom CNN with several Conv2D + pooling blocks.
   * Automatically adapts to input size (`480 × 640`).
   * Initialised with Kaiming/Xavier schemes.
   * Designed for **stability and low memory usage** with many parallel environments.

2. **MobileNetV3-Small (optional)**
   * Pretrained on ImageNet via `torchvision.models.mobilenet_v3_small`.
   * Provides strong visual feature extraction out of the box.
   * More memory-intensive; useful for **transfer-learning experiments** or encoder comparisons.

> **Why SimpleCNN as default?**
> Earlier tests with a pretrained MobileNetV3-Small backbone significantly increased GPU memory usage and occasionally led to out-of-memory errors when training with many parallel environments on the RTX 3090. The SimpleCNN keeps the model compact, avoids memory issues with 16 environments, and still provides sufficiently rich features for learning the docking behaviour. MobileNetV3-Small remains available in `cnn_model.py` for future experiments targeting transfer learning or detailed encoder comparisons.

### 5.5 PPO Algorithm

The training loop in `scripts/skrl/train_curriculum.py` implements **Proximal Policy Optimization (PPO)** with:

* **GAE(λ)** advantage estimation,
* Clipped policy objective,
* Optional value clipping,
* Entropy regularisation,
* Gradient clipping,
* Support for **checkpointing** and **TensorBoard logging**.

Core hyperparameters (γ, λ, clipping, entropy/value coefficients, etc.) are centralised in a `HYPERPARAMS` dictionary, simplifying **future genetic/evolutionary optimisation**.

---

## 6. Curriculum Learning

Docking is trained via a **16-stage ultra-gradual curriculum** (`curriculum_manager.py`):

* Early stages: robot starts **very close and aligned**, learning basic backward motion into the connector.
* Intermediate stages: gradually increase **distance**, **lateral offset**, and **yaw misalignment**.
* Later stages: include **180° misalignment** and **far starting positions**, requiring turning, search and re-alignment.
* Final stage: **"Full Autonomy"** – robot starts anywhere in a large area around the goal with random yaw.

Stage progression:

* The trainer tracks success rate per stage.
* A stage is eligible for transition when:
  * A **minimum number of steps** has been executed in the current stage (e.g. 15,000 steps), **and**
  * The stage success rate reaches a predefined threshold (e.g. 85 %).

Curriculum control is implemented jointly by:

* `curriculum_manager.py` – defines spawn distributions per stage.
* `train_curriculum.py` – implements stage advancement logic based on metrics.

---

## 7. Repository Structure

Aligned with the current project tree:
```text
TEKO/
├── README.md                     ← Project documentation (this file)
├── _cam_out/                     ← Camera debug output (if used)
├── debug_frames/
│   └── verification_frame.png    ← Docking geometry & camera sanity checks
├── documents/
│   ├── Aruco/                    ← ArUco marker textures
│   ├── CAD/                      ← Fusion 360 exports (meshes + URDF)
│   │   ├── Other_Formats/
│   │   │   ├── stage_1/          ← Stage arena assets
│   │   │   └── teko/             ← TEKO robot meshes & URDF
│   │   └── USD/
│   │       ├── stage_arena.usd   ← Docking arena
│   │       ├── teko.usd          ← TEKO robot (mobile)
│   │       └── teko_goal.usd     ← TEKO goal (static)
│   ├── error.txt                 ← Misc. notes / debug info
│   └── pictures/                 ← Figures for thesis & documentation
│
├── scripts/
│   ├── dockin_aruco.py           ← ArUco-based docking experiments (utility)
│   ├── straight.py               ← Simple motion tests
│   ├── test_env.py               ← Environment sanity checks
│   ├── visualize_docking_points.py ← Visualisation of connector spheres
│   └── skrl/                     ← RL training scripts (custom PPO)
│       ├── debug.py
│       ├── red_dots.py
│       ├── train_curriculum.py   ← MAIN training entrypoint (16-stage curriculum)
│       ├── train_curriculum_until_s4.py ← Early-stage tests
│       ├── train_manual.py       ← Manual / non-curriculum experiments
│       └── train_production.py   ← Older production-style training script
│
├── source/
│   └── teko/
│       ├── config/
│       │   └── extension.toml    ← Isaac Lab extension config
│       ├── docs/
│       │   └── CHANGELOG.rst
│       ├── pyproject.toml
│       ├── setup.py
│       ├── teko/
│       │   ├── __init__.py
│       │   ├── ui_extension_example.py
│       │   ├── tasks/
│       │   │   └── direct/
│       │   │       └── teko/
│       │   │           ├── teko_env.py         ← Environment implementation
│       │   │           ├── teko_env_cfg.py     ← Environment configuration (camera, robots, sim)
│       │   │           ├── curriculum/
│       │   │           │   ├── __init__.py
│       │   │           │   └── curriculum_manager.py ← 16-stage curriculum
│       │   │           ├── rewards/
│       │   │           │   ├── __init__.py
│       │   │           │   └── reward_functions.py   ← Sphere-based docking rewards
│       │   │           ├── penalties/
│       │   │           │   ├── __init__.py
│       │   │           │   └── penalties.py          ← Legacy/experimental penalties (currently unused)
│       │   │           ├── robots/
│       │   │           │   ├── __init__.py
│       │   │           │   ├── teko.py               ← Dynamic TEKO articulation configuration
│       │   │           │   └── teko_static.py        ← Static TEKO goal configuration
│       │   │           ├── sensors/
│       │   │           │   ├── __init__.py
│       │   │           │   ├── camera.py
│       │   │           │   ├── imu.py
│       │   │           │   └── lidar.py
│       │   │           ├── teko_brain/
│       │   │           │   ├── __init__.py
│       │   │           │   └── cnn_model.py          ← SimpleCNN + MobileNetV3 encoders
│       │   │           └── utils/
│       │   │               ├── __init__.py
│       │   │               ├── geometry_utils.py     ← Sphere distances, transforms, etc.
│       │   │               └── logging_utils.py      ← Reward component logging helpers
│       └── teko.egg-info/   ← Python package metadata (generated)
│
├── teko_curriculum/             ← TensorBoard logs & checkpoints
│   ├── YYYYMMDD_HHMMSS/
│   │   ├── events.out.tfevents.*   ← Training logs
│   │   └── final.pt                ← Final model checkpoint
│   └── ...
│
├── spawn_positions_12stage.png  ← Visualisation of spawn positions (older design)
├── test_env_basic.py            ← Minimal environment smoke test
└── train_pid.txt                ← PID experiments / notes (if used)
```

---

## 8. Training Workflow

### 8.1 Launching Isaac Lab (headless PPO training)

From the repository root:
```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/skrl/train_curriculum.py \
  --num_envs 16 \
  --steps 2000000 \
  --headless
```

Common options:

* `--num_envs` : number of parallel environments (e.g. 8, 16).
* `--steps`    : total environment steps to train.
* `--lr`       : learning rate (default `1e-4`).
* `--rollout_len` : rollout horizon per PPO update (default `64`).
* `--epochs`   : PPO epochs per update (default `8`).
* `--batch_size` : minibatch size for PPO.
* `--checkpoint` : path to a `.pt` checkpoint to resume from.

### 8.2 Monitoring Training

TensorBoard logs are written under `teko_curriculum/`:
```bash
tensorboard --logdir teko_curriculum
```

Main metrics:

* `train/reward` – mean episode reward
* `train/episode_length` – mean episode length
* `train/success_rate` – recent docking success rate
* `train/stage_success` – success rate within the current curriculum stage
* `train/curriculum_stage` – current stage index (0–15)
* Policy/value losses and entropy

### 8.3 Checkpoints

The trainer periodically saves:

* `ckpt_*.pt` – intermediate checkpoints with:
  * Policy weights
  * Optimiser state
  * Current training step
  * Curriculum level and steps in current stage

* `final.pt` – final model at the end of training.

These checkpoints can be used both for **evaluation** scripts and for **fine-tuning / continued training**.

---

## 9. Future Work

* **Evolutionary hyperparameter optimisation** (e.g. genetic algorithms) using the centralised `HYPERPARAMS` dictionary.
* Detailed comparison between **SimpleCNN** and **pretrained MobileNetV3** encoders in terms of sample efficiency, robustness and sim-to-real transfer.
* **Sim-to-Real** deployment on physical TEKO robots in the research hall, including:
  * Domain randomisation,
  * Sensor noise,
  * Real lighting and material effects.
* Extension from **single docking** to **multi-robot cooperative docking** and chained configurations.
* Integration with **ROS 2** for real-time control and logging.

---

## 10. Contact

**Alexandre Schleier Neves da Silva**  
M.Sc. Environmental Protection and Agricultural Food Production  
University of Hohenheim

📧 [alexandre.schleiernevesdasilva@uni-hohenheim.de](mailto:alexandre.schleiernevesdasilva@uni-hohenheim.de)

---

This repository provides the experimental framework for the **TEKO Vision-Based Docking System**, forming the core of the master's thesis on **adaptive cooperation in modular agricultural robot swarms**.