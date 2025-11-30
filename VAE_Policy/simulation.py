#!/usr/bin/env python3
"""
Simulation and evaluation script for RISE.
Loads trained VAE encoder and PolicyNetwork, runs 100 episodes in TrialUnicycleEnv,
records a single video with all episodes, and prints Safety and Reach Rates.
"""
import os
import argparse
import json
import csv
import numpy as np
import pygame
import torch
import imageio.v2 as imageio
from tqdm import tqdm

# Make parent directory importable for env module regardless of CWD
import sys
from pathlib import Path
_FILE_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _FILE_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.append(str(_PARENT_DIR))

from trial_Unicycle_Env import TrialUnicycleEnv
from rise import VAE, PolicyNetwork


def build_obstacle_vector(env: TrialUnicycleEnv) -> np.ndarray:
    return np.concatenate([
        env.obstacle_position.astype(np.float32),
        env.obstacle_velocity.astype(np.float32),
        np.array([env.obstacle_radius], dtype=np.float32)
    ]).astype(np.float32)


def run_episode(env: TrialUnicycleEnv,
                device: torch.device,
                vae: VAE,
                policy: PolicyNetwork,
                writer,
                episode_index: int,
                max_steps: int = 300,
                overlay: bool = True,
                log_rows=None) -> dict:
    obs, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0
    collided = False
    reached = False

    # initial frame
    env.render()
    frame_surface = env.screen
    if frame_surface is not None:
        arr = np.transpose(np.array(pygame.surfarray.array3d(frame_surface)), (1, 0, 2))
        if overlay:
            arr = draw_overlay(arr, episode_index, steps)
        writer.append_data(arr)

    while not (terminated or truncated) and steps < max_steps:
        # Build inputs
        state = torch.tensor(env.state, dtype=torch.float32, device=device).unsqueeze(0)
        goal = torch.tensor(env.goal, dtype=torch.float32, device=device).unsqueeze(0)
        c = torch.tensor(build_obstacle_vector(env), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, logvar = vae.encode(c)
            z = vae.reparameterize(mu, logvar)
            action = policy(state, goal, z).squeeze(0).cpu().numpy().astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        frame_surface = env.screen
        if frame_surface is not None:
            arr = np.transpose(np.array(pygame.surfarray.array3d(frame_surface)), (1, 0, 2))
            if overlay:
                arr = draw_overlay(arr, episode_index, steps)
            writer.append_data(arr)
        if log_rows is not None:
            log_rows.append([
                episode_index, steps,
                *env.state.tolist(),
                *env.goal.tolist(),
                *env.obstacle_position.tolist(),
                *env.obstacle_velocity.tolist(),
                env.obstacle_radius,
                action[0], action[1],
                float(reached), float(collided)
            ])

        steps += 1
        collided = collided or bool(info.get('collision', False))
        reached = reached or bool(info.get('reached_goal', False))

    return {
        'steps': steps,
        'collision': collided,
        'reached': reached and not collided
    }


def draw_overlay(arr: np.ndarray, ep: int, step: int) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return arr
    # Ensure the array is contiguous and uint8 for OpenCV operations.
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    try:
        cv2.putText(arr, f'Episode {ep}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(arr, f'Step {step}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    except Exception:
        # If OpenCV fails for any reason, return the original frame without overlay.
        return arr
    return arr


def main():
    import pygame  # ensure pygame present for rendering
    parser = argparse.ArgumentParser(description='Simulate trained RISE policy')
    parser.add_argument('--vae', type=str, default='models/vae.pth')
    parser.add_argument('--policy', type=str, default='models/policy.pth')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--out', type=str, default='simulation_results.mp4')
    parser.add_argument('--csv', type=str, default='simulation_log.csv', help='Per-step CSV log path')
    parser.add_argument('--no-overlay', action='store_true', help='Disable text overlay in video')
    parser.add_argument('--config', type=str, default=None, help='Optional JSON config to override args')
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    # Resolve output paths relative to this script's directory if not absolute
    if not os.path.isabs(args.out):
        args.out = str((_FILE_DIR / args.out).resolve())
    if args.csv and not os.path.isabs(args.csv):
        args.csv = str((_FILE_DIR / args.csv).resolve())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    vae = VAE(input_dim=5, latent_dim=3).to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device))
    vae.eval()

    policy = PolicyNetwork(state_dim=3, goal_dim=2, latent_dim=3, action_dim=2).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device))
    policy.eval()

    env = TrialUnicycleEnv()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    writer = imageio.get_writer(args.out, fps=30)
    violations = 0
    reaches = 0

    log_rows = []
    for ep in tqdm(range(args.episodes), desc='Simulating'):
        res = run_episode(env, device, vae, policy, writer,
                          episode_index=ep,
                          max_steps=args.max_steps,
                          overlay=not args.no_overlay,
                          log_rows=log_rows)
        if res['collision']:
            violations += 1
        elif res['reached']:
            reaches += 1
        # add a short black gap between episodes
        env.render()
        h, w = env.screen.get_height(), env.screen.get_width()
        gap = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(3):
            writer.append_data(gap)
        print(f"Episode {ep+1}/{args.episodes}: steps={res['steps']}, collision={res['collision']}, reached={res['reached']}")

    writer.close()
    env.close()

    if args.csv:
        header = ['episode','step','x','y','theta','goal_x','goal_y','obs_x','obs_y','obs_vx','obs_vy','obs_radius','action_v','action_w','reached','collision']
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in log_rows:
                w.writerow(row)
        print(f'Saved CSV log to {args.csv}')

    safety_rate = 100.0 * (args.episodes - violations) / args.episodes
    reach_rate = 100.0 * reaches / args.episodes
    print(f"Safety Rate %: {safety_rate:.2f}")
    print(f"Reach Rate %: {reach_rate:.2f}")


if __name__ == '__main__':
    main()
