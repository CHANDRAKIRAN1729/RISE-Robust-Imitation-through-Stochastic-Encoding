#!/usr/bin/env python3
"""
Collect successful trajectory-image pairs from both BC and VAE-latent policies.
Only saves episodes where goal is reached without collision.

Output: trajectory_image_dataset.pkl containing list of dicts:
{
  'trajectory': {
    'states': [(x,y,theta), ...],  # T timesteps
    'actions': [(v,w), ...],       # T timesteps
    'goal': (gx, gy),
    'obstacle_seq': [(ox,oy,vx,vy,r), ...],  # T timesteps (obstacle position over time)
  },
  'images': [img1, img2, ...],  # sampled frames as (H,W,3) uint8 arrays
  'metadata': {
    'policy_type': 'bc' or 'vae',
    'episode_length': int,
    'final_state': (x,y,theta),
    'camera_config': {...}
  }
}
"""
import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import pygame
from tqdm import tqdm

# Make parent directory importable
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from trial_Unicycle_Env import TrialUnicycleEnv

# Import policy models
sys.path.append(str(_REPO_ROOT / 'Behavior_Cloning'))
sys.path.append(str(_REPO_ROOT / 'VAE_Policy'))
from pc_bc_models import PolicyNetwork as BCPolicyNetwork
from rise import VAE, PolicyNetwork as VAEPolicyNetwork


def build_obstacle_vector(env: TrialUnicycleEnv) -> np.ndarray:
    """Build 5-dim obstacle vector: [pos_x, pos_y, vel_x, vel_y, radius]"""
    return np.concatenate([
        env.obstacle_position.astype(np.float32),
        env.obstacle_velocity.astype(np.float32),
        np.array([env.obstacle_radius], dtype=np.float32)
    ])


def render_to_numpy(env: TrialUnicycleEnv, resolution: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """Render environment to numpy array and resize to target resolution."""
    env.render()
    if env.screen is None:
        return np.zeros((*resolution, 3), dtype=np.uint8)
    
    # Get pygame surface as array
    arr = np.transpose(np.array(pygame.surfarray.array3d(env.screen)), (1, 0, 2))
    
    # Resize if needed
    if arr.shape[:2] != resolution:
        try:
            import cv2
            arr = cv2.resize(arr, resolution, interpolation=cv2.INTER_AREA)
        except ImportError:
            # Fallback: simple downsampling (not ideal but works)
            from scipy.ndimage import zoom
            h_ratio = resolution[0] / arr.shape[0]
            w_ratio = resolution[1] / arr.shape[1]
            arr = zoom(arr, (h_ratio, w_ratio, 1), order=1).astype(np.uint8)
    
    return arr.astype(np.uint8)


def collect_episode_bc(env: TrialUnicycleEnv, 
                       policy: BCPolicyNetwork, 
                       device: torch.device,
                       max_steps: int = 300,
                       image_resolution: Tuple[int, int] = (128, 128),
                       frame_skip: int = 5) -> Dict:
    """Run one episode with BC policy, return trajectory+images if successful."""
    obs, _ = env.reset()
    
    states = []
    actions = []
    obstacle_seq = []
    images = []
    
    terminated = False
    truncated = False
    steps = 0
    
    while not (terminated or truncated) and steps < max_steps:
        # Record state
        states.append(env.state.copy())
        obstacle_seq.append(build_obstacle_vector(env))
        
        # Capture image every frame_skip steps
        if steps % frame_skip == 0:
            img = render_to_numpy(env, image_resolution)
            images.append(img)
        
        # Compute action
        state = torch.tensor(env.state, dtype=torch.float32, device=device).unsqueeze(0)
        goal = torch.tensor(env.goal, dtype=torch.float32, device=device).unsqueeze(0)
        c = torch.tensor(build_obstacle_vector(env), dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            action = policy(state, goal, c).squeeze(0).cpu().numpy()
        
        actions.append(action.copy())
        
        # Step environment
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
    
    # Capture final frame
    img = render_to_numpy(env, image_resolution)
    images.append(img)
    
    # Check if successful (reached goal without collision)
    reached = info.get('reached_goal', False)
    collision = info.get('collision', False)
    
    if not reached or collision:
        return None  # Failed episode
    
    return {
        'trajectory': {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'goal': env.goal.copy(),
            'obstacle_seq': np.array(obstacle_seq, dtype=np.float32),
        },
        'images': np.array(images, dtype=np.uint8),
        'metadata': {
            'policy_type': 'bc',
            'episode_length': steps,
            'final_state': env.state.copy(),
            'camera_config': {
                'resolution': image_resolution,
                'frame_skip': frame_skip,
            }
        }
    }


def collect_episode_vae(env: TrialUnicycleEnv,
                        vae: VAE,
                        policy: VAEPolicyNetwork,
                        device: torch.device,
                        max_steps: int = 300,
                        image_resolution: Tuple[int, int] = (128, 128),
                        frame_skip: int = 5) -> Dict:
    """Run one episode with VAE-latent policy, return trajectory+images if successful."""
    obs, _ = env.reset()
    
    states = []
    actions = []
    obstacle_seq = []
    images = []
    
    terminated = False
    truncated = False
    steps = 0
    
    while not (terminated or truncated) and steps < max_steps:
        # Record state
        states.append(env.state.copy())
        obstacle_seq.append(build_obstacle_vector(env))
        
        # Capture image every frame_skip steps
        if steps % frame_skip == 0:
            img = render_to_numpy(env, image_resolution)
            images.append(img)
        
        # Compute action using VAE encoder
        state = torch.tensor(env.state, dtype=torch.float32, device=device).unsqueeze(0)
        goal = torch.tensor(env.goal, dtype=torch.float32, device=device).unsqueeze(0)
        c = torch.tensor(build_obstacle_vector(env), dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            mu, logvar = vae.encode(c)
            z = vae.reparameterize(mu, logvar)
            action = policy(state, goal, z).squeeze(0).cpu().numpy()
        
        actions.append(action.copy())
        
        # Step environment
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
    
    # Capture final frame
    img = render_to_numpy(env, image_resolution)
    images.append(img)
    
    # Check if successful
    reached = info.get('reached_goal', False)
    collision = info.get('collision', False)
    
    if not reached or collision:
        return None  # Failed episode
    
    return {
        'trajectory': {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'goal': env.goal.copy(),
            'obstacle_seq': np.array(obstacle_seq, dtype=np.float32),
        },
        'images': np.array(images, dtype=np.uint8),
        'metadata': {
            'policy_type': 'vae',
            'episode_length': steps,
            'final_state': env.state.copy(),
            'camera_config': {
                'resolution': image_resolution,
                'frame_skip': frame_skip,
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Collect trajectory-image pairs from successful episodes')
    parser.add_argument('--bc-policy', type=str, default='models/pc_bc_policy.pth')
    parser.add_argument('--vae', type=str, default='models/vae.pth')
    parser.add_argument('--vae-policy', type=str, default='models/policy.pth')
    parser.add_argument('--episodes-per-policy', type=int, default=150, 
                        help='Target successful episodes per policy')
    parser.add_argument('--max-attempts', type=int, default=500,
                        help='Max attempts per policy to collect successful episodes')
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--resolution', type=int, default=128, help='Image resolution (square)')
    parser.add_argument('--frame-skip', type=int, default=5, help='Capture image every N steps')
    parser.add_argument('--out', type=str, default='trajectory_image_dataset.pkl')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    resolution = (args.resolution, args.resolution)
    
    # Resolve paths relative to repo root
    bc_policy_path = Path(args.bc_policy) if os.path.isabs(args.bc_policy) else (_REPO_ROOT / args.bc_policy).resolve()
    vae_path = Path(args.vae) if os.path.isabs(args.vae) else (_REPO_ROOT / args.vae).resolve()
    vae_policy_path = Path(args.vae_policy) if os.path.isabs(args.vae_policy) else (_REPO_ROOT / args.vae_policy).resolve()
    out_path = Path(args.out) if os.path.isabs(args.out) else (_FILE_DIR / args.out).resolve()
    
    print(f"Device: {device}")
    print(f"Image resolution: {resolution}")
    print(f"Frame skip: {args.frame_skip}")
    
    # Load BC policy
    print("\nLoading BC policy...")
    bc_policy = BCPolicyNetwork(state_dim=3, goal_dim=2, cond_dim=5, action_dim=2).to(device)
    bc_policy.load_state_dict(torch.load(bc_policy_path, map_location=device))
    bc_policy.eval()
    
    # Load VAE + policy
    print("Loading VAE-latent policy...")
    vae = VAE(input_dim=5, latent_dim=3).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    
    vae_policy = VAEPolicyNetwork(state_dim=3, goal_dim=2, latent_dim=3, action_dim=2).to(device)
    vae_policy.load_state_dict(torch.load(vae_policy_path, map_location=device))
    vae_policy.eval()
    
    env = TrialUnicycleEnv()
    dataset = []
    
    # Collect from BC policy
    print(f"\n=== Collecting from BC policy (target: {args.episodes_per_policy} successful) ===")
    bc_successes = 0
    bc_attempts = 0
    with tqdm(total=args.episodes_per_policy, desc="BC episodes") as pbar:
        while bc_successes < args.episodes_per_policy and bc_attempts < args.max_attempts:
            result = collect_episode_bc(env, bc_policy, device, args.max_steps, resolution, args.frame_skip)
            bc_attempts += 1
            if result is not None:
                dataset.append(result)
                bc_successes += 1
                pbar.update(1)
    
    print(f"BC: {bc_successes}/{bc_attempts} successful episodes ({100*bc_successes/bc_attempts:.1f}%)")
    
    # Collect from VAE-latent policy
    print(f"\n=== Collecting from VAE-latent policy (target: {args.episodes_per_policy} successful) ===")
    vae_successes = 0
    vae_attempts = 0
    with tqdm(total=args.episodes_per_policy, desc="VAE episodes") as pbar:
        while vae_successes < args.episodes_per_policy and vae_attempts < args.max_attempts:
            result = collect_episode_vae(env, vae, vae_policy, device, args.max_steps, resolution, args.frame_skip)
            vae_attempts += 1
            if result is not None:
                dataset.append(result)
                vae_successes += 1
                pbar.update(1)
    
    print(f"VAE: {vae_successes}/{vae_attempts} successful episodes ({100*vae_successes/vae_attempts:.1f}%)")
    
    env.close()
    
    # Save dataset
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Total successful episodes: {len(dataset)}")
    print(f"BC episodes: {bc_successes}")
    print(f"VAE episodes: {vae_successes}")
    print(f"Saved to: {out_path}")
    
    # Quick stats
    total_images = sum(len(d['images']) for d in dataset)
    avg_traj_len = np.mean([d['metadata']['episode_length'] for d in dataset])
    print(f"Total images: {total_images}")
    print(f"Avg trajectory length: {avg_traj_len:.1f} steps")


if __name__ == '__main__':
    main()
