# Milestone 1 Report — Vision–Language–Action Behavioral Cloning

## 1. System Overview
This milestone implements the baseline behavioral cloning pipeline for a Franka Panda manipulation policy powered by a Vision–Language–Action (VLA) architecture. Demonstrations are consumed in the LeRobot format and used to train a DINOv2 + BERT model that predicts joint velocity commands conditioned on RGB observations, natural-language instructions, and proprioceptive state.

```mermaid
graph TD
    A[RGB Image (DINOv2)] --> C[Fusion Transformer]
    B[Language Instruction (BERT)] --> C
    D[Proprioception] --> C
    C --> E[Action Head]
    E --> F[Joint Velocity Command]
```

## 2. Dataset Summary
* Source: MuJoCo simulation (Franka Panda tabletop manipulation).
* Format: LeRobot episodes containing synchronized RGB frames, proprioceptive states, action trajectories, and natural-language instructions.
* Scale: 500 demonstrations (400 static, 100 hindered) planned.
* Preprocessing: Images normalized with DINOv2 image processor, proprioception scaled to [-1, 1].

## 3. Model Architecture
* **Vision encoder:** `facebook/dinov2-base` (frozen by default).
* **Language encoder:** `bert-base-uncased` with tokenizer-managed padding/truncation.
* **Proprio branch:** Two-layer MLP projecting 7-DoF joint configuration to fusion space.
* **Fusion module:** Multi-layer transformer encoder operating over vision, language, and proprio tokens.
* **Action head:** LayerNorm + linear map to 7-D joint velocity output.

## 4. Training Configuration
* Loss: Mean squared error between predicted and demonstrated joint velocities.
* Optimizer: AdamW (`lr=1e-4`, weight decay `1e-2`).
* Scheduler: Optional cosine annealing with warm restarts disabled.
* Regularization: Gradient clipping, optional mixed precision, deterministic seeding utilities.
* Checkpoints: Best + last snapshots saved under `runs/` alongside serialized configs.

## 5. Evaluation Protocol
* Environment: MuJoCo Franka pick-and-place with static and hindered variants.
* Metrics: Success rate, cumulative reward per episode, CSV logs, comparison bar chart.
* Episodes: 50 static + 50 hindered (configurable).

## 6. Results Placeholder
Evaluation is executed via `evaluate_bc_mujoco.py`. Populate the tables below after running the evaluation script.

| Mode      | Success Rate | Average Reward |
|-----------|--------------|----------------|
| Static    | TBD          | TBD            |
| Hindered  | TBD          | TBD            |

## 7. Challenges & Next Steps
* Large-model memory footprint requires careful batch sizing and potential layer freezing.
* Dataset schema validation and preprocessing must be robust to missing modalities.
* Next milestone: integrate PPO fine-tuning using `rl/ppo_config.yaml`, leveraging BC policy for initialization and continuing evaluation pipeline.

## 8. Reproducibility Checklist
- [x] Deterministic seeding via `utils/seed.py`.
- [x] YAML-driven configs with override support.
- [x] Training/evaluation scripts log to console and persist results.
- [x] Directory structure prepared for checkpoints, results, and reports.
