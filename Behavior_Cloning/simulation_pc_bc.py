#!/usr/bin/env python3
"""
Simulation and evaluation for PC-BC baseline.
Loads PolicyNetwork trained with raw obstacle parameters and evaluates in TrialUnicycleEnv.
Produces a combined video (simulation_pc_bc_results.mp4) and prints Safety/Reach rates.
"""
import os
import csv
import argparse
from typing import Dict

import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm
import pygame

# Ensure we can import modules from the repo root regardless of CWD
import sys
from pathlib import Path
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from trial_Unicycle_Env import TrialUnicycleEnv
from pc_bc_models import PolicyNetwork


def build_obstacle_vector(env: TrialUnicycleEnv) -> np.ndarray:
    return np.concatenate([
        env.obstacle_position.astype(np.float32),
        env.obstacle_velocity.astype(np.float32),
        np.array([env.obstacle_radius], dtype=np.float32)
    ]).astype(np.float32)


essential_overlay = True

def draw_overlay(arr: np.ndarray, ep: int, step: int) -> np.ndarray:
    if not essential_overlay:
        return arr
    try:
        import cv2
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        cv2.putText(arr, f'Episode {ep}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(arr, f'Step {step}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    except Exception:
        return arr
    return arr


def run_episode(env: TrialUnicycleEnv, device: torch.device, policy: PolicyNetwork, writer, episode_index: int,
                max_steps: int = 300) -> Dict:
    obs, _ = env.reset()
    terminated = False
    truncated = False
    collided = False
    reached = False
    steps = 0

    env.render()
    frame_surface = env.screen
    if frame_surface is not None:
        arr = np.transpose(np.array(pygame.surfarray.array3d(frame_surface)), (1, 0, 2))
        arr = draw_overlay(arr, episode_index, steps)
        writer.append_data(arr)

    while not (terminated or truncated) and steps < max_steps:
        state = torch.tensor(env.state, dtype=torch.float32, device=device).unsqueeze(0)
        goal = torch.tensor(env.goal, dtype=torch.float32, device=device).unsqueeze(0)
        c = torch.tensor(build_obstacle_vector(env), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = policy(state, goal, c).squeeze(0).cpu().numpy().astype(np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        frame_surface = env.screen
        if frame_surface is not None:
            arr = np.transpose(np.array(pygame.surfarray.array3d(frame_surface)), (1, 0, 2))
            arr = draw_overlay(arr, episode_index, steps)
            writer.append_data(arr)

        steps += 1
        collided = collided or bool(info.get('collision', False))
        reached = reached or bool(info.get('reached_goal', False))

    return {
        'steps': steps,
        'collision': collided,
        'reached': reached and not collided,
    }


def main():
    parser = argparse.ArgumentParser(description='Simulate PC-BC policy')
    # default paths relative to repo root
    default_policy = str((_REPO_ROOT / 'models' / 'pc_bc_policy.pth').as_posix())
    # Save outputs inside the Behavior_Cloning folder by default
    default_out = str((_FILE_DIR / 'simulation_pc_bc_results.mp4').as_posix())
    default_csv = str((_FILE_DIR / 'simulation_pc_bc_log.csv').as_posix())

    parser.add_argument('--policy', type=str, default=default_policy)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--out', type=str, default=default_out)
    parser.add_argument('--csv', type=str, default=default_csv)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    policy = PolicyNetwork(state_dim=3, goal_dim=2, cond_dim=5, action_dim=2).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device))
    policy.eval()

    env = TrialUnicycleEnv()

    # Resolve output paths relative to this script's directory if not absolute
    if not os.path.isabs(args.out):
        args.out = str((_FILE_DIR / args.out).resolve())
    if args.csv and not os.path.isabs(args.csv):
        args.csv = str((_FILE_DIR / args.csv).resolve())

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    writer = imageio.get_writer(args.out, fps=30)

    violations = 0
    reaches = 0

    for ep in tqdm(range(args.episodes), desc='Simulating PC-BC'):
        res = run_episode(env, device, policy, writer, ep, max_steps=args.max_steps)
        if res['collision']:
            violations += 1
        elif res['reached']:
            reaches += 1

        # Add small gap between episodes
        env.render()
        h, w = env.screen.get_height(), env.screen.get_width()
        gap = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(3):
            writer.append_data(gap)
        print(f"Episode {ep+1}/{args.episodes}: steps={res['steps']}, collision={res['collision']}, reached={res['reached']}")

    writer.close()
    env.close()

    safety_rate = 100.0 * (args.episodes - violations) / args.episodes
    reach_rate = 100.0 * reaches / args.episodes
    print(f"Safety Rate %: {safety_rate:.2f}")
    print(f"Reach Rate %: {reach_rate:.2f}")


if __name__ == '__main__':
    main()
