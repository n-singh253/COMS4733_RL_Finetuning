"""Quick test script to verify RL implementation works."""
import torch
import numpy as np
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from rl.ppo_trainer import RolloutBuffer, PPOTrainer, RolloutBatch

def test_model_extensions():
    """Test that model has RL extensions."""
    print("Testing model extensions...")

    config = VLADinoV2Config(
        fusion_hidden_dim=512,
        fusion_layers=2,
        action_dim=7,
        history_length=5,
        freeze_vision=False,
        freeze_language=False,
    )

    model = VLADinoV2Policy(config)
    model.eval()

    # Create dummy inputs
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 224, 224)
    instructions = ["test instruction"] * batch_size
    proprio = torch.randn(batch_size, 8)
    action_history = torch.randn(batch_size, 5, 7)

    with torch.no_grad():
        # Test forward (actor)
        print("  Testing forward (actor)...")
        actions = model(rgb, instructions, proprio, action_history)
        assert actions.shape == (batch_size, 7), f"Expected (2, 7), got {actions.shape}"
        print(f"    ✓ Forward pass works: {actions.shape}")

        # Test value head (critic)
        print("  Testing get_value (critic)...")
        values = model.get_value(rgb, instructions, proprio, action_history)
        assert values.shape == (batch_size, 1), f"Expected (2, 1), got {values.shape}"
        print(f"    ✓ Value prediction works: {values.shape}")

        # Test get_action_and_value
        print("  Testing get_action_and_value...")
        action, log_prob, value = model.get_action_and_value(
            rgb, instructions, proprio, action_history, action_std=0.1
        )
        assert action.shape == (batch_size, 7)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size, 1)
        print(f"    ✓ Action sampling works: action={action.shape}, log_prob={log_prob.shape}, value={value.shape}")

        # Test evaluate_actions
        print("  Testing evaluate_actions...")
        dummy_actions = torch.randn(batch_size, 7)
        log_prob, value, entropy = model.evaluate_actions(
            rgb, instructions, proprio, dummy_actions, action_history, action_std=0.1
        )
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size, 1)
        assert entropy.shape == (batch_size,)
        print(f"    ✓ Action evaluation works: log_prob={log_prob.shape}, value={value.shape}, entropy={entropy.shape}")

    print("✅ Model extensions test passed!\n")
    return model


def test_rollout_buffer():
    """Test rollout buffer."""
    print("Testing rollout buffer...")

    device = torch.device("cpu")
    buffer = RolloutBuffer(
        buffer_size=100,
        action_dim=7,
        history_length=5,
        device=device,
    )

    # Add some dummy data
    print("  Adding dummy transitions...")
    for i in range(10):
        buffer.add(
            rgb=torch.randn(3, 224, 224),
            instruction="test",
            proprio=torch.randn(8),
            action_history=torch.randn(5, 7),
            action=torch.randn(7),
            log_prob=torch.randn(1),
            value=torch.randn(1, 1),
            reward=float(i),
            done=False,
        )

    assert len(buffer) == 10, f"Expected 10, got {len(buffer)}"
    print(f"    ✓ Added {len(buffer)} transitions")

    # Compute returns and advantages
    print("  Computing returns and advantages...")
    last_value = torch.randn(1, 1)
    advantages, returns = buffer.compute_returns_and_advantages(
        last_value=last_value,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages.shape == (10,)
    assert returns.shape == (10,)
    print(f"    ✓ GAE computation works: advantages={advantages.shape}, returns={returns.shape}")

    # Get batch
    print("  Creating batch...")
    batch = buffer.get(advantages, returns)
    assert isinstance(batch, RolloutBatch)
    assert batch.rgb_static.shape == (10, 3, 224, 224)
    assert batch.actions.shape == (10, 7)
    assert len(batch.instructions) == 10
    print(f"    ✓ Batch creation works: rgb={batch.rgb_static.shape}, actions={batch.actions.shape}")

    print("✅ Rollout buffer test passed!\n")
    return buffer


def test_ppo_trainer(model):
    """Test PPO trainer."""
    print("Testing PPO trainer...")

    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = PPOTrainer(
        policy=model,
        optimizer=optimizer,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        action_std=0.1,
        target_kl=0.01,
    )

    # Create dummy batch
    print("  Creating dummy batch...")
    batch_size = 16
    batch = RolloutBatch(
        rgb_static=torch.randn(batch_size, 3, 224, 224),
        instructions=["test"] * batch_size,
        proprio=torch.randn(batch_size, 8),
        action_history=torch.randn(batch_size, 5, 7),
        actions=torch.randn(batch_size, 7),
        log_probs=torch.randn(batch_size),
        values=torch.randn(batch_size, 1),
        rewards=torch.randn(batch_size),
        dones=torch.zeros(batch_size),
        advantages=torch.randn(batch_size),
        returns=torch.randn(batch_size),
    )

    # Test PPO loss computation
    print("  Computing PPO loss...")
    loss, info, approx_kl = trainer.compute_ppo_loss(batch, batch.log_probs)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert "loss/policy" in info
    assert "loss/value" in info
    assert "loss/total" in info
    print(f"    ✓ Loss computation works: total_loss={loss.item():.4f}")

    # Test training step
    print("  Running training step...")
    metrics = trainer.train_step(batch, num_epochs=1, batch_size=8)
    assert "loss/policy" in metrics
    assert "loss/value" in metrics
    assert "loss/total" in metrics
    print(f"    ✓ Training step works: policy_loss={metrics['loss/policy']:.4f}")

    print("✅ PPO trainer test passed!\n")


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")

    try:
        from env.mujoco_env import FrankaPickPlaceEnv
        print("  ✓ Environment import works")
    except Exception as e:
        print(f"  ⚠ Environment import failed (expected if MuJoCo not available): {e}")

    try:
        from utils.config import load_config
        from utils.logging import get_logger
        from utils.seed import seed_everything
        print("  ✓ Utility imports work")
    except Exception as e:
        print(f"  ✗ Utility import failed: {e}")
        raise

    print("✅ Import test passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("RL IMPLEMENTATION TEST")
    print("="*60)
    print()

    try:
        # Test imports
        test_imports()

        # Test model extensions
        model = test_model_extensions()

        # Test rollout buffer
        test_rollout_buffer()

        # Test PPO trainer
        test_ppo_trainer(model)

        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
