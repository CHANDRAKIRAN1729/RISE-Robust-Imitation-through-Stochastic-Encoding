#!/usr/bin/env python3
"""
Test policy trained using PC-BC approach (no vision/images).
Tests (state, goal, obstacle_params) -> action mapping.
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import pygame
import imageio.v2 as imageio
from tqdm import tqdm

# Setup paths
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from trial_Unicycle_Env import TrialUnicycleEnv
from train_vision_policy import SimplePolicyNetwork


def build_obstacle_vector(env: TrialUnicycleEnv) -> np.ndarray:
    """Build 5-dim obstacle vector."""
    return np.concatenate([
        env.obstacle_position.astype(np.float32),
        env.obstacle_velocity.astype(np.float32),
        np.array([env.obstacle_radius], dtype=np.float32)
    ])


def run_episode_with_policy(env: TrialUnicycleEnv,
                            policy: SimplePolicyNetwork,
                            device: torch.device,
                            max_steps: int = 300,
                            writer=None,
                            episode_idx: int = 0) -> dict:
    """
    Run one episode using simple policy (state, goal, obstacle) -> action.
    """
    obs, _ = env.reset()
    
    terminated = False
    truncated = False
    steps = 0
    collided = False
    reached = False
    
    while not (terminated or truncated) and steps < max_steps:
        # Build inputs
        state = torch.tensor(env.state, dtype=torch.float32, device=device).unsqueeze(0)
        goal = torch.tensor(env.goal, dtype=torch.float32, device=device).unsqueeze(0)
        obstacle = torch.tensor(build_obstacle_vector(env), dtype=torch.float32, device=device).unsqueeze(0)
        
        # Predict action
        with torch.no_grad():
            action = policy(state, goal, obstacle).squeeze(0).cpu().numpy()
        
        # Step environment
        _, _, terminated, truncated, info = env.step(action)
        
        # Render frame for video
        if writer is not None:
            env.render()
            frame_surface = env.screen
            if frame_surface is not None:
                arr = np.transpose(np.array(pygame.surfarray.array3d(frame_surface)), (1, 0, 2))
                try:
                    import cv2
                    cv2.putText(arr, f'Ep {episode_idx}, Step {steps}', (10, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                except:
                    pass
                writer.append_data(arr)
        
        steps += 1
        collided = collided or bool(info.get('collision', False))
        reached = reached or bool(info.get('reached_goal', False))
    
    return {
        'steps': steps,
        'collision': collided,
        'reached': reached and not collided
    }


def main():
    parser = argparse.ArgumentParser(description='Test policy trained with PC-BC approach')
    parser.add_argument('--policy', type=str, default='models/vision_policy_best.pth',
                        help='Path to trained policy')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--video-out', type=str, default='vision_policy_test.mp4')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    # Create or load policy
    print("\nInitializing policy...")
    policy = SimplePolicyNetwork(
        state_dim=3,
        goal_dim=2,
        obstacle_dim=5,
        action_dim=2,
        hidden_dim=128
    ).to(device)
    
    policy_path = Path(args.policy) if os.path.isabs(args.policy) else (_FILE_DIR / args.policy).resolve()
    if policy_path.exists():
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        print(f"✓ Loaded trained policy from {policy_path}")
        policy.eval()
    else:
        print(f"✗ WARNING: Policy not found at {policy_path}")
        print("  Using randomly initialized policy (agent will NOT move properly)")
        print("  Please train the policy first using:")
        print(f"    cd {_FILE_DIR}")
        print("    python train_vision_policy.py")
        print()
        user_input = input("Continue anyway with random policy? (y/N): ")
        if user_input.lower() != 'y':
            print("Exiting. Please train the policy first.")
            sys.exit(1)
        policy.eval()
    
    # Run episodes
    env = TrialUnicycleEnv()
    
    video_path = Path(args.video_out)
    video_path.parent.mkdir(exist_ok=True, parents=True)
    writer = imageio.get_writer(str(video_path), fps=30)
    
    violations = 0
    reaches = 0
    
    print(f"\nRunning {args.episodes} episodes...")
    for ep in tqdm(range(args.episodes), desc='Episodes'):
        res = run_episode_with_policy(
            env, policy, device,
            max_steps=args.max_steps,
            writer=writer,
            episode_idx=ep
        )
        
        if res['collision']:
            violations += 1
        elif res['reached']:
            reaches += 1
        
        # Add gap between episodes
        if writer is not None:
            env.render()
            h, w = env.screen.get_height(), env.screen.get_width()
            gap = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(3):
                writer.append_data(gap)
    
    writer.close()
    env.close()
    
    # Report results
    safety_rate = 100.0 * (args.episodes - violations) / args.episodes
    reach_rate = 100.0 * reaches / args.episodes
    
    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"Episodes: {args.episodes}")
    print(f"Safety Rate: {safety_rate:.2f}%")
    print(f"Reach Rate: {reach_rate:.2f}%")
    print(f"Video saved to: {video_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
