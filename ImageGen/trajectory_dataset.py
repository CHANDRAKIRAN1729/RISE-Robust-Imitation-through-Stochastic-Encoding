#!/usr/bin/env python3
"""
Dataset wrapper for trajectory→image pairs.
Provides trajectory encodings and corresponding target images for training.
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional


class TrajectoryImageDataset(Dataset):
    """
    Dataset for trajectory→image generation.
    
    Returns:
        trajectory_encoding: flattened/encoded trajectory vector
        target_image: corresponding RGB image as (C, H, W) tensor
        metadata: dict with auxiliary info
    """
    
    def __init__(self, 
                 data_path: str,
                 max_trajectory_len: int = 60,
                 use_final_frame_only: bool = False,
                 normalize_states: bool = True):
        """
        Args:
            data_path: path to trajectory_image_dataset.pkl
            max_trajectory_len: pad/truncate trajectories to this length
            use_final_frame_only: if True, only use final image; else random sample
            normalize_states: normalize state coordinates to [-1, 1]
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.max_traj_len = max_trajectory_len
        self.use_final_frame = use_final_frame_only
        self.normalize_states = normalize_states
        
        # Compute normalization stats if needed
        if self.normalize_states:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute mean/std for state normalization."""
        all_states = []
        for d in self.data:
            all_states.append(d['trajectory']['states'])
        all_states = np.concatenate(all_states, axis=0)  # (N, 3)
        
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0) + 1e-8
        
        # Also normalize goal and obstacle positions similarly
        # For simplicity, use same stats for x,y coords
        self.pos_mean = self.state_mean[:2]
        self.pos_std = self.state_std[:2]
    
    def _encode_trajectory(self, traj_dict: Dict) -> np.ndarray:
        """
        Encode trajectory as a fixed-length vector.
        
        Encoding strategy:
        - Pad/truncate states and actions to max_traj_len
        - Concatenate: [states, actions, goal, final_obstacle_state]
        
        Returns: (D,) array where D = max_traj_len * (3+2) + 2 + 5
        """
        states = traj_dict['states']  # (T, 3)
        actions = traj_dict['actions']  # (T, 2)
        goal = traj_dict['goal']  # (2,)
        obstacle_seq = traj_dict['obstacle_seq']  # (T, 5)
        
        T = len(states)
        
        # Normalize states if enabled
        if self.normalize_states:
            states = (states - self.state_mean) / self.state_std
            goal = (goal - self.pos_mean) / self.pos_std
            obstacle_seq[:, :2] = (obstacle_seq[:, :2] - self.pos_mean) / self.pos_std
        
        # Pad or truncate to max_traj_len
        if T < self.max_traj_len:
            pad_len = self.max_traj_len - T
            states = np.concatenate([states, np.zeros((pad_len, 3), dtype=np.float32)], axis=0)
            actions = np.concatenate([actions, np.zeros((pad_len, 2), dtype=np.float32)], axis=0)
            obstacle_seq = np.concatenate([obstacle_seq, np.zeros((pad_len, 5), dtype=np.float32)], axis=0)
        else:
            states = states[:self.max_traj_len]
            actions = actions[:self.max_traj_len]
            obstacle_seq = obstacle_seq[:self.max_traj_len]
        
        # Flatten and concatenate: states, actions, goal, final obstacle
        states_flat = states.reshape(-1)  # (max_traj_len * 3,)
        actions_flat = actions.reshape(-1)  # (max_traj_len * 2,)
        final_obstacle = obstacle_seq[-1]  # (5,)
        
        encoding = np.concatenate([states_flat, actions_flat, goal, final_obstacle], axis=0)
        return encoding.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        item = self.data[idx]
        
        # Encode trajectory
        traj_encoding = self._encode_trajectory(item['trajectory'])
        
        # Select image
        images = item['images']
        if self.use_final_frame:
            img = images[-1]  # final frame
        else:
            # Random sample from available frames
            img_idx = np.random.randint(0, len(images))
            img = images[img_idx]
        
        # Convert image to tensor (H, W, 3) -> (3, H, W) and normalize to [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        traj_tensor = torch.from_numpy(traj_encoding)
        
        metadata = {
            'policy_type': item['metadata']['policy_type'],
            'episode_length': item['metadata']['episode_length'],
        }
        
        return traj_tensor, img_tensor, metadata


class CompactTrajectoryImageDataset(Dataset):
    """
    More compact encoding using summary statistics instead of full trajectory.
    Useful for faster training with smaller models.
    """
    
    def __init__(self,
                 data_path: str,
                 use_final_frame_only: bool = False):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.use_final_frame = use_final_frame_only
    
    def _encode_trajectory_compact(self, traj_dict: Dict) -> np.ndarray:
        """
        Compact trajectory encoding using summary statistics:
        - Initial state (3)
        - Final state (3)
        - Goal (2)
        - Mean action (2)
        - Std action (2)
        - Final obstacle state (5)
        Total: 17 dimensions
        """
        states = traj_dict['states']
        actions = traj_dict['actions']
        goal = traj_dict['goal']
        obstacle_seq = traj_dict['obstacle_seq']
        
        initial_state = states[0]
        final_state = states[-1]
        mean_action = actions.mean(axis=0)
        std_action = actions.std(axis=0)
        final_obstacle = obstacle_seq[-1]
        
        encoding = np.concatenate([
            initial_state,
            final_state,
            goal,
            mean_action,
            std_action,
            final_obstacle
        ], axis=0)
        
        return encoding.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        item = self.data[idx]
        
        traj_encoding = self._encode_trajectory_compact(item['trajectory'])
        
        images = item['images']
        if self.use_final_frame:
            img = images[-1]
        else:
            img_idx = np.random.randint(0, len(images))
            img = images[img_idx]
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        traj_tensor = torch.from_numpy(traj_encoding)
        
        metadata = {
            'policy_type': item['metadata']['policy_type'],
            'episode_length': item['metadata']['episode_length'],
        }
        
        return traj_tensor, img_tensor, metadata
