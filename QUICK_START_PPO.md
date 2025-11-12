# Quick Start Guide - PPO Training

## Quick Test Run (Smoke Test - 5-10 minutes)

**IMPORTANT: This is a smoke test to verify the pipeline works, not for actual training!**

The quick test will:
- Verify all code runs without crashes
- Validate environment, model, and PPO integration
- Confirm checkpoints save correctly
- Check for numerical stability (no NaN values)
- NOT train a useful policy (only 3 epochs with reduced rollouts)

**Run the smoke test:**

```bash
python -m train_ppo --checkpoint runs/dinov2_bc_best.pt --quick-test
```

**Expected outcome:**
- Training completes without errors
- Success rate likely stays at 0% (expected - need full training)
- Checkpoints saved to `runs/ppo_test/`
- Runtime: ~5-10 minutes

**This is NOT a bug!** The quick test uses minimal training (3 epochs, 256 steps) just to verify everything works.

## Full Training (~10 hours on M4 Max)

Once quick test passes, run full training:

```bash
python -m train_ppo --checkpoint runs/dinov2_bc_best.pt
```

## Monitor Training

```bash
# In separate terminal
tensorboard --logdir=runs/ppo

# Open browser to: http://localhost:6006
```

## Evaluate Results

```bash
python -m evaluate_ppo --checkpoint runs/ppo/ppo_final.pt --num-episodes 20 --deterministic
```

## Configuration

### Quick Test Config
- File: `rl/ppo_config_quick_test.yaml`
- Epochs: 3
- Rollout length: 256
- Runtime: ~5-10 minutes

### Full Training Config
- File: `rl/ppo_config.yaml`
- Epochs: 100
- Rollout length: 2048
- Runtime: ~10 hours
- Reward type: shaped (dense guidance + success bonus)

## Reward Types

Change in `rl/ppo_config.yaml`:

```yaml
environment:
  reward_type: shaped  # Options: dense, sparse, shaped
```

- **dense**: `-distance` (easiest, continuous feedback)
- **sparse**: `1.0 if success else 0.0` (hardest, realistic)
- **shaped**: `-distance + 10*success` (recommended, best of both)

## Troubleshooting

**Error: "No module named 'env'"**
```bash
# Run from project root with -m flag
python -m train_ppo --quick-test
```

**Error: "BC checkpoint not found"**
```bash
# Train BC model first
python -m train_bc --config configs/openvla_dinov2_bc.yaml --epochs 5
```

**Training too slow?**
- Reduce `rollout_length` in config (e.g., 512 instead of 2048)
- Reduce `num_epochs` (e.g., 25 instead of 100)
- Check MPS is available: `python -c "import torch; print(torch.backends.mps.is_available())"`

## What to Expect

### Quick Test (Smoke Test)

**Purpose: Pipeline validation, NOT training**

Expected results after quick test:
- Code completes without crashes
- Checkpoints saved successfully
- No NaN or Inf values in losses
- Success rate: **0% is NORMAL** (only 3 epochs)
- Mean reward: **Negative is NORMAL** (minimal training)

**This is validation, not training!** Don't expect improvements.

**If quick test fails:**
- Check error messages carefully
- Verify BC checkpoint exists: `ls runs/dinov2_bc_best.pt`
- Ensure running from project root with `-m` flag

### Full Training (Actual Learning)

**After 100 epochs, expect:**
- Success rate: 20-40% (if BC baseline is good)
- Success rate: 0-10% (if BC baseline is weak, like 0%)
- Mean reward: Gradually increasing
- Training curves in TensorBoard showing improvement

**Metrics to watch:**
- `rollout/success_rate`: Should increase over time (if BC is good)
- `rollout/mean_reward`: Should increase (less negative for dense/shaped)
- `loss/policy`: Should decrease initially then stabilize
- `policy/approx_kl`: Should stay < 0.01 (if not, reduces learning rate automatically)

**Good signs (Full Training):**
- Success rate increases from BC baseline
- Mean reward improves steadily
- No NaN values in losses
- Policy loss stabilizes (not growing)

**Bad signs:**
- Success rate decreases from BC baseline
- Losses become NaN
- KL divergence consistently > 0.05
- Reward gets progressively worse

## Understanding Performance

**If BC baseline is 0%:**
- PPO will struggle to improve (no good starting point)
- Consider: Retraining BC with more demos or better hyperparameters
- Or: Run PPO longer (200+ epochs) with high exploration

**If BC baseline is >10%:**
- PPO should improve from there
- Expect gradual improvement over 50-100 epochs
- Look for 2-5% absolute improvement in success rate

**Quick test vs Full training:**
- Quick test: 3 epochs, 256 steps = **smoke test only**
- Full training: 100 epochs, 2048 steps = **actual learning**
- Quick test taking ~5-10 mins confirms full training will take ~10 hours
