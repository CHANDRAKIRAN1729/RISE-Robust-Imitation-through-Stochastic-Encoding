#!/usr/bin/env python3
"""
Quick demo: Test trajectory encoding and dataset loading.
Verifies the pipeline components are properly integrated.
"""
import sys
from pathlib import Path

# Add parent to path
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

import torch
import numpy as np


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from trajectory_dataset import TrajectoryImageDataset, CompactTrajectoryImageDataset
        from image_generation_models import ConditionalVAE, vae_loss
        from vision_policy_models import EndToEndVisionPolicy, VisionEnhancedPolicy
        print("  ✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_models():
    """Test that models can be instantiated."""
    print("\nTesting model instantiation...")
    try:
        from image_generation_models import ConditionalVAE
        from vision_policy_models import EndToEndVisionPolicy
        
        # Create CVAE
        cvae = ConditionalVAE(traj_dim=17, latent_dim=128, img_channels=3, img_size=128)
        print(f"  ✓ CVAE created: {sum(p.numel() for p in cvae.parameters()):,} parameters")
        
        # Create vision policy
        policy = EndToEndVisionPolicy(
            state_dim=3, goal_dim=2, img_feature_dim=128, action_dim=2
        )
        print(f"  ✓ Vision policy created: {sum(p.numel() for p in policy.parameters()):,} parameters")
        
        # Test forward pass
        device = torch.device('cpu')
        cvae.to(device)
        policy.to(device)
        
        dummy_traj = torch.randn(2, 17)
        dummy_state = torch.randn(2, 3)
        dummy_goal = torch.randn(2, 2)
        
        with torch.no_grad():
            img = cvae.generate(dummy_traj, deterministic=True)
            action = policy(dummy_state, dummy_goal, img)
        
        print(f"  ✓ Forward pass successful: img shape {img.shape}, action shape {action.shape}")
        return True
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_trajectory_encoding():
    """Test trajectory encoding logic."""
    print("\nTesting trajectory encoding...")
    try:
        # Simulate a simple trajectory
        trajectory = {
            'states': np.random.randn(20, 3).astype(np.float32),
            'actions': np.random.randn(20, 2).astype(np.float32),
            'goal': np.array([2.5, 0.5], dtype=np.float32),
            'obstacle_seq': np.random.randn(20, 5).astype(np.float32),
        }
        
        # Compact encoding (17D)
        initial_state = trajectory['states'][0]
        final_state = trajectory['states'][-1]
        mean_action = trajectory['actions'].mean(axis=0)
        std_action = trajectory['actions'].std(axis=0)
        final_obstacle = trajectory['obstacle_seq'][-1]
        
        compact_encoding = np.concatenate([
            initial_state, final_state, trajectory['goal'],
            mean_action, std_action, final_obstacle
        ])
        
        print(f"  ✓ Compact encoding shape: {compact_encoding.shape} (expected: (17,))")
        assert compact_encoding.shape == (17,), f"Wrong shape: {compact_encoding.shape}"
        
        return True
        
    except Exception as e:
        print(f"  ✗ Encoding test failed: {e}")
        return False


def test_environment():
    """Test that the environment can be imported and reset."""
    print("\nTesting environment...")
    try:
        from trial_Unicycle_Env import TrialUnicycleEnv
        
        env = TrialUnicycleEnv()
        obs, info = env.reset(seed=42)
        
        print(f"  ✓ Environment created and reset")
        print(f"    State shape: {env.state.shape}")
        print(f"    Goal: {env.goal}")
        print(f"    Obstacle pos: {env.obstacle_position}")
        
        # Take a random action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  ✓ Step successful, terminated: {terminated}, truncated: {truncated}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        return False


def main():
    print("="*60)
    print("RISE ImageGen Pipeline - Component Tests")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Models", test_models()))
    results.append(("Encoding", test_trajectory_encoding()))
    results.append(("Environment", test_environment()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {name}")
        all_passed = all_passed and passed
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("  1. Run './run_pipeline.sh' to execute the full pipeline")
        print("  2. Or manually run scripts step-by-step (see README.md)")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
