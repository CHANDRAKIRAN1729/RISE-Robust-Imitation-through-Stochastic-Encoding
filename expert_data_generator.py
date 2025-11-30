#!/usr/bin/env python3
"""
Generate expert demonstration dataset using a simple oracle controller in TrialUnicycleEnv.
Saves a list of transitions into expert_data.pkl via pickle.
Each item: {
  'state': [x,y,theta],
  'goal': [gx,gy],
  'obstacle': {'pos':[ox,oy], 'vel':[vx,vy], 'radius': r},
  'action': [v,w]
}
"""
import os
import pickle
import numpy as np
from typing import List, Dict

from trial_Unicycle_Env import TrialUnicycleEnv


def oracle_controller(state: np.ndarray, goal: np.ndarray, obstacle_pos: np.ndarray, obstacle_radius: float) -> np.ndarray:
    """Proportional go-to-goal with simple obstacle avoidance (repulsive turn)."""
    x, y, theta = state.astype(np.float32)
    dx = goal[0] - x
    dy = goal[1] - y
    desired_theta = np.arctan2(dy, dx)
    # angle error
    ang_err = (desired_theta - theta + np.pi) % (2*np.pi) - np.pi
    dist = float(np.hypot(dx, dy))

    # base control
    v = np.clip(1.2 * dist, -1.0, 1.0)
    w = np.clip(2.5 * ang_err, -1.0, 1.0)

    # obstacle reactive steer: if too close, steer away by adding to w
    d_obs = float(np.linalg.norm(np.array([x, y]) - obstacle_pos))
    if d_obs < (obstacle_radius + 0.25):
        away_theta = np.arctan2(y - obstacle_pos[1], x - obstacle_pos[0])
        steer = (away_theta - theta + np.pi) % (2*np.pi) - np.pi
        w += np.clip(2.0 * steer, -0.8, 0.8)

    return np.array([v, w], dtype=np.float32)


def collect_expert_data(num_episodes: int = 100, max_steps: int = 200, seed: int = 0) -> List[Dict]:
    env = TrialUnicycleEnv()
    data: List[Dict] = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated) and steps < max_steps:
            state = env.state.copy()
            action = oracle_controller(state, env.goal, env.obstacle_position, env.obstacle_radius)
            next_obs, reward, terminated, truncated, info = env.step(action)
            data.append({
                'state': state.astype(np.float32),
                'goal': env.goal.astype(np.float32),
                'obstacle': {
                    'pos': env.obstacle_position.astype(np.float32),
                    'vel': env.obstacle_velocity.astype(np.float32),
                    'radius': float(env.obstacle_radius)
                },
                'action': action.astype(np.float32),
            })
            steps += 1
    env.close()
    return data


def main():
    os.makedirs('dataset', exist_ok=True)
    out_path = 'dataset/expert_data.pkl'
    data = collect_expert_data(num_episodes=200, max_steps=150, seed=42)
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} transitions to {out_path}")


if __name__ == '__main__':
    main()
