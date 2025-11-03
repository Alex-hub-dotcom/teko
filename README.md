# TEKO Docking System

This repository is part of the master’s thesis project “Adaptive Cooperation in Agricultural Robot Swarms: Reinforcement Learning and Evolutionary Algorithms for Modular Docking".

The project is being developed at the University of Hohenheim, within the Department of Artificial Intelligence in Agricultural Engineering,  
under the supervision of **Prof. Dr. Anthony Stein** and **Dr. David Reiser**.

---

## 1. Project Overview

The project investigates how small-scale agricultural robots can physically connect to form modular units capable of performing tasks that would otherwise require larger and more complex machines.  
This work proposes and validates an **end-to-end framework** for autonomous robot docking, using **vision-based perception** and **reinforcement learning** within simulation and physical hardware setups.

The study combines robotic design, simulation, and machine learning into a unified experimental pipeline — from CAD models and simulated environments to training algorithms and real-world transfer.  

---

## 2. Research Motivation

Modern agriculture faces growing challenges such as labor shortages, climate stress, and the need for sustainable production. Traditional monolithic machinery is expensive and often impractical for smaller farms.  
**Swarm and modular robotics** offer a scalable alternative: many small, affordable robots that can act cooperatively.  
However, reliable **physical cooperation** — such as docking and power/data sharing — remains an open research challenge.

The TEKO project aims to address this by developing small modular robots that can autonomously **locate, align, and connect** using reinforcement learning and evolutionary algorithms for parameter optimization.

---

## 3. Objectives

- **Design and implement** a reliable docking mechanism between modular TEKO robots.  
- **Develop a reinforcement learning agent** capable of learning the physical coupling behaviour based solely on visual input.  
- **Explore hybrid and evolutionary algorithms** to improve training efficiency, robustness, and exploration.  
- **Validate the system** in NVIDIA Isaac Lab with photorealistic simulation.  
- **Transfer the learned policies** to real robots for evaluation in the experimental arena of the research hall.  

---

## 4. Research Approach

### 4.1 Simulation and Testbed
The simulation is implemented in **NVIDIA Isaac Lab 0.47.1**, using accurate physics, camera sensors, and custom CAD models of the TEKO robot and docking arena.  
Each TEKO model includes fully defined USD and URDF configurations, physically realistic connectors, and sensor placements.

### 4.2 Learning Framework
- **Algorithm:** Proximal Policy Optimization (PPO) via the *SKRL* library.  
- **Input:** RGB camera images (640×480), emulating the Raspberry Pi Camera Module 2, from the rear-mounted camera of the mobile robot.  
- **Output:** Differential wheel velocities `[left, right]`.  
- **Reward:** Weighted combination of distance, alignment, success, and efficiency terms.  
- **Network:** Convolutional encoder (4 Conv2D layers + pooling) feeding actor/critic MLPs (~30 M parameters).  
- **Parallel environments:** Up to (number of environments to be defined) concurrent rollouts for GPU efficiency.  

### 4.3 Planned Extensions
1. **Evolutionary Reinforcement Learning (ERL):**  
   Use evolutionary algorithms to improve PPO’s exploration and hyperparameter optimization.
2. **Sim-to-Real Transfer:**  
   Transfer trained models to physical TEKO robots operating in the laboratory arena.  
   Investigate robustness under sensor noise, real lighting, and mechanical inaccuracies.

---

## 5. Repository Structure

```
TEKO/
├── README.md                     ← Project documentation
├── documents/                    ← CAD, USD, and URDF robot models
│   ├── Aruco/                    ← ArUco marker textures
│   ├── CAD/                      ← Fusion 360 exports
│   ├── USD/                      ← Isaac-compatible stage and robot assets
│   └── pictures/                 ← Simulation and docking results
│
├── scripts/
│   ├── camera_tes.py             ← Camera streaming tests
│   ├── fluid_movement.py         ← Motion prototype
│   └── skrl/                     ← Reinforcement Learning framework
│       ├── train_ppo.py
│       ├── eval_policy.py
│       ├── visualize_results.py
│       ├── quickstart.sh
│       └── requirements.txt
│
├── source/
│   └── teko/
│       ├── teko/tasks/direct/teko/
│       │   ├── teko_env.py       ← Environment definition
│       │   ├── teko_env_cfg.py   ← Environment configuration
│       │   ├── utils/rewards.py  ← Reward computation
│       │   └── agents/           ← PPO models and configuration
│       │       ├── cnn_model.py
│       │       ├── ppo_policy.py
│       │       └── skrl_ppo_cfg.yaml
│       └── sensors/              ← Camera, LIDAR, and IMU modules
│
└── logs/                         ← Training logs and checkpoints
```

---

## 6. Training Workflow

1. **Setup and Dependencies**  
   ```bash
   pip install -r scripts/skrl/requirements.txt
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. **Train the PPO Agent**  
   ```bash
   cd scripts/skrl
   python train_ppo.py --num-envs 4
   ```
3. **Monitor Learning**  
   ```bash
   tensorboard --logdir runs/
   ```
4. **Evaluate and Visualize Results**  
   ```bash
   python eval_policy.py --model runs/.../final_model.pt --render
   python visualize_results.py --experiment-dir runs/.../
   ```

Training progress is logged automatically and includes mean reward, episode length, policy/value loss, and KL divergence.

---

## 7. Performance Benchmarks (RTX 3090)

| Environments | VRAM | Steps/min | Time (100k steps) |
|--------------|------|------------|-------------------|
| 1 | ~3 GB | 1,700 | 60 min |
| 4 | ~8 GB | 5,000 | 20 min |
| 8 | ~15 GB | 8,000 | 13 min |

Docking typically converges after 100k–500k steps with >80 % success rate.

---

## 8. Evaluation Metrics

| Metric | Target  |
|--------|---------|
| Success Rate     | ≥ 80 % (good), ≥ 90 % (excellent) |
| Mean Reward      | Positive and increasing           |
| Final Distance   | 0.475 m ± 0.01 m                  |
| Episode Length   | Shorter = more efficient          |

Additional analysis includes collision rate, alignment accuracy, and stability under randomized conditions.

---

## 9. Technologies Used

- **NVIDIA Isaac Lab 0.47.1** — simulation and physics engine  
- **PyTorch** — deep learning and model training  
- **SKRL** — PPO implementation for Isaac Lab  
- **TensorBoard** — logging and visualization  
- **USD / URDF** — robot and environment modeling  
- **ROS 2 (planned)** — deployment and real-time control  

---

## 10. Implementation Notes

### Coordinate System
The TEKO model was exported from *Fusion 360*, which uses **z-up** coordinates.  
Simulation randomizes orientation around the **z-axis** to preserve upright posture.  
If using **y-up** (Omniverse-native) models, modify rotation accordingly.

---

## 11. Future Work

- Integration of **Evolutionary Algorithms** to form an *Evolutionary Reinforcement Learning (ERL)* system, improving exploration and hyperparameter adaptation.  
- **Sim-to-Real Transfer**: deploy and validate trained models on physical TEKO robots within the research hall arena.  
- Expansion to **multi-robot docking** and cooperative task execution.  
- Integration of **shared energy and data buses** between coupled modules.  
- Development of **real-time monitoring tools** for experimental evaluation.

---

## 12. Contact

**Alexandre Schleier Neves da Silva**  
M.Sc. Student Environmental Protection and Agricultural Food Production  
University of Hohenheim  
📧 alexandre.schleiernevesdasilva@uni-hohenheim.de  

---

This repository provides the full framework for simulation, learning, and analysis of the **TEKO Vision-Based Docking System**, forming the experimental foundation of the author’s master’s thesis on **adaptive cooperation in modular agricultural robotic swarms**.

To begin training:

```bash
python train_ppo.py --num-envs 1
```

Good luck with your experiments. 🚀