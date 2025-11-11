# RL Fine-tuning for VLA Policy

This document describes the RL fine-tuning implementation based on the [RL4VLA](https://github.com/gen-robot/RL4VLA) approach for improving Vision-Language-Action (VLA) models through reinforcement learning.

## Overview

The RL fine-tuning pipeline follows a **two-stage approach**:

1. **Stage 1 - Behavioral Cloning (BC) Warmup**: Train the VLA model on demonstration data using supervised learning
2. **Stage 2 - PPO Refinement**: Fine-tune the BC-trained model using Proximal Policy Optimization (PPO) in the environment

This approach allows the model to:
- Bootstrap from human demonstrations (BC)
- Optimize beyond behavioral cloning through environmental interaction (PPO)
- Discover better strategies through trial and error
- Generalize to scenarios not covered in the demonstration data

## Architecture

### Model Extensions

The base `VLADinoV2Policy` has been extended with **actor-critic** capabilities:

- **Actor (Policy Head)**: Predicts actions from multimodal inputs (already present in BC model)
- **Critic (Value Head)**: Predicts state values for advantage estimation (new for RL)

Key methods added:
- `get_value()`: Get state value predictions
- `get_action_and_value()`: Sample actions and get values (for rollout collection)
- `evaluate_actions()`: Evaluate actions under current policy (for PPO updates)

### PPO Algorithm

The implementation includes:

1. **Rollout Collection**: Collect trajectories using current policy
2. **Advantage Estimation**: Compute advantages using Generalized Advantage Estimation (GAE)
3. **PPO Updates**: Update policy using clipped surrogate objective
4. **Value Function Learning**: Train critic to predict returns

**PPO Loss Components**:
```
Total Loss = Policy Loss + Value Coefficient × Value Loss + Entropy Coefficient × Entropy Loss
```

- **Policy Loss**: Clipped surrogate objective for policy improvement
- **Value Loss**: MSE between predicted and actual returns (clipped)
- **Entropy Loss**: Encourages exploration

## File Structure

```
COMS4733_RL_Finetuning/
├── rl/
│   ├── ppo_config.yaml           # PPO hyperparameters and training config
│   └── ppo_trainer.py            # PPO algorithm implementation
├── models/
│   └── vla_dinov2.py             # VLA model with actor-critic heads
├── train_ppo.py                  # Main PPO training script
├── evaluate_ppo.py               # Evaluation script for RL-trained models
└── RL_TRAINING.md                # This document
```

## Usage

### 1. Train BC Model (Warmup)

First, train a BC model on demonstration data:

```bash
python train_bc.py --config configs/openvla_dinov2_bc.yaml --epochs 35
```

This creates a checkpoint: `runs/dinov2_bc_best.pt`

### 2. Fine-tune with PPO

Fine-tune the BC model using PPO:

```bash
python train_ppo.py --config rl/ppo_config.yaml --checkpoint runs/dinov2_bc_best.pt
```

**Optional flags**:
- `--render`: Visualize environment during training (slower but useful for debugging)

The training will:
1. Load the BC checkpoint (actor weights)
2. Initialize the value head randomly
3. Collect rollouts using the current policy
4. Update the policy with PPO
5. Save checkpoints to `runs/ppo/`

### 3. Evaluate RL Model

Evaluate the trained RL model:

```bash
python evaluate_ppo.py --checkpoint runs/ppo/ppo_final.pt --num-episodes 20 --deterministic
```

**Flags**:
- `--checkpoint`: Path to PPO checkpoint
- `--num-episodes`: Number of episodes to evaluate (default: 10)
- `--deterministic`: Use mean actions instead of sampling
- `--render`: Visualize evaluation
- `--output`: Save results to JSON file

## Configuration

### PPO Hyperparameters (`rl/ppo_config.yaml`)

Key hyperparameters:

```yaml
policy:
  # Learning
  learning_rate: 3e-6          # Lower LR for fine-tuning
  clip_range: 0.2              # PPO clipping parameter
  action_std: 0.1              # Exploration noise

  # Loss coefficients
  value_coef: 0.5              # Weight for value loss
  entropy_coef: 0.01           # Weight for entropy bonus

  # Training schedule
  num_epochs: 100              # Total training epochs
  rollout_length: 2048         # Steps per rollout
  ppo_epochs: 4                # PPO updates per rollout
  batch_size: 64               # Minibatch size

  # Discount and GAE
  gamma: 0.99                  # Discount factor
  gae_lambda: 0.95             # GAE lambda
```

### Tuning Tips

**If policy is not improving:**
- Increase `rollout_length` to collect more data
- Decrease `learning_rate` to make updates more conservative
- Increase `entropy_coef` to encourage more exploration

**If training is unstable:**
- Decrease `learning_rate`
- Decrease `clip_range` to make updates more conservative
- Increase `batch_size` for more stable gradient estimates

**If policy is too exploratory:**
- Decrease `action_std`
- Decrease `entropy_coef`

## Implementation Details

### Gaussian Policy

Actions are sampled from a Gaussian distribution:
```
π(a|s) = N(μ(s), σ²)
```
where:
- `μ(s)` is the mean action predicted by the actor head
- `σ` is the fixed standard deviation (`action_std`)

### Generalized Advantage Estimation (GAE)

Advantages are computed using GAE:
```
A_t = Σ(γλ)^l δ_{t+l}
```
where:
- `δ_t = r_t + γV(s_{t+1}) - V(s_t)` is the TD error
- `γ` is the discount factor
- `λ` is the GAE lambda parameter

### PPO Clipped Objective

The policy loss uses the clipped surrogate objective:
```
L^{CLIP} = E[min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]
```
where:
- `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)` is the probability ratio
- `ε` is the clip range
- `Â_t` is the normalized advantage

### Action History

The model uses **temporal context** through action history:
- Maintains a buffer of the last 5 actions
- Encodes action history into a feature vector
- Fuses with vision, language, and proprioception

This enables:
- Closed-loop control
- Temporal reasoning
- Better handling of dynamic tasks

## Monitoring Training

### TensorBoard

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir runs/ppo/tensorboard
```

**Key metrics to monitor**:

**Rollout Metrics**:
- `rollout/mean_reward`: Average reward per episode
- `rollout/success_rate`: Success rate on task
- `rollout/mean_length`: Average episode length

**Policy Metrics**:
- `policy/approx_kl`: KL divergence (should stay < 0.02)
- `policy/clip_fraction`: Fraction of clipped updates
- `policy/entropy`: Policy entropy (exploration)

**Loss Metrics**:
- `loss/policy`: Policy gradient loss
- `loss/value`: Value function loss
- `loss/total`: Combined loss

### Expected Behavior

**Initial epochs (1-20)**:
- High exploration (high entropy)
- Variable rewards
- Model learns basic behaviors

**Middle epochs (20-60)**:
- Rewards should increase
- Success rate improves
- Policy stabilizes (lower KL)

**Late epochs (60-100)**:
- Rewards plateau
- High success rate
- Low exploration (low entropy)

## Differences from RL4VLA

While inspired by RL4VLA, this implementation has some differences:

**Similarities**:
- Two-stage training (BC warmup → PPO refinement)
- PPO algorithm for policy gradient updates
- GAE for advantage estimation

**Differences**:
- **No LoRA**: We fine-tune all parameters, not just low-rank adapters
  - LoRA can be added for memory efficiency if needed
- **Simpler environment**: Pick-and-place task vs. multiple SimulatED tasks
- **Single environment**: No multi-environment parallelization (can be added)
- **Different base model**: DINOv2 + BERT vs. OpenVLA

## Troubleshooting

### No BC checkpoint found

**Error**: `FileNotFoundError: runs/dinov2_bc_best.pt`

**Solution**: Train a BC model first:
```bash
python train_bc.py --config configs/openvla_dinov2_bc.yaml
```

### CUDA out of memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `batch_size` in config
2. Reduce `rollout_length` in config
3. Use gradient accumulation (modify training script)
4. Use a smaller model (reduce fusion layers/dims)

### Policy not learning

**Symptoms**: Rewards not increasing, success rate stays low

**Solutions**:
1. Check BC model quality - it should have >50% success rate
2. Reduce learning rate
3. Increase rollout length for more data
4. Check reward function is providing good signal
5. Verify action denormalization is correct

### Training is slow

**Solutions**:
1. Disable rendering (`--render` flag)
2. Reduce `rollout_length`
3. Reduce `ppo_epochs`
4. Use mixed precision training (add to script)
5. Profile to find bottlenecks

## Next Steps

### Potential Improvements

1. **Multi-environment parallelization**: Collect rollouts from multiple environments in parallel
2. **LoRA adaptation**: Use low-rank adaptation for memory-efficient fine-tuning
3. **Curriculum learning**: Gradually increase task difficulty
4. **Hindsight Experience Replay (HER)**: Learn from failed attempts
5. **Adaptive action std**: Decay exploration noise over training
6. **Recurrent policy**: Use LSTM/GRU for better temporal reasoning
7. **Intrinsic motivation**: Add curiosity-driven exploration

### Evaluation Metrics

Consider tracking:
- Success rate on held-out test scenarios
- Generalization to new object positions
- Robustness to visual perturbations
- Sample efficiency (episodes to reach 80% success)

## References

- [RL4VLA Repository](https://github.com/gen-robot/RL4VLA)
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [GAE Paper](https://arxiv.org/abs/1506.02438) - Schulman et al., 2016
- [OpenVLA](https://openvla.github.io/) - Kim et al., 2024

## Citation

If you use this implementation, please cite:

```bibtex
@misc{rl4vla2024,
  title={Reinforcement Learning Fine-tuning for Vision-Language-Action Models},
  author={Your Name},
  year={2024},
  howpublished={GitHub Repository},
  url={https://github.com/ekanshgupta92/COMS4733_RL_Finetuning}
}
```
