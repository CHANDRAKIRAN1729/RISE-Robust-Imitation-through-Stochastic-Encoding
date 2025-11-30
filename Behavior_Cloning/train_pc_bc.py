#!/usr/bin/env python3
"""
Train the PC-BC baseline policy using expert_data.pkl.
Behavior cloning with MSE loss on actions: a_hat = pi(s, g, c).
Saves weights to pc_bc_policy.pth by default.
"""
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pc_bc_models import PolicyNetwork


class ExpertDataset(Dataset):
    """Dataset providing (state, goal, obstacle_params, action)."""
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for d in raw:
            s = np.asarray(d['state'], dtype=np.float32)          # (3,)
            g = np.asarray(d['goal'], dtype=np.float32)           # (2,)
            o = np.concatenate([
                np.asarray(d['obstacle']['pos'], dtype=np.float32),
                np.asarray(d['obstacle']['vel'], dtype=np.float32),
                np.asarray([d['obstacle']['radius']], dtype=np.float32),
            ], axis=0)                                            # (5,)
            a = np.asarray(d['action'], dtype=np.float32)         # (2,)
            self.samples.append((s, g, o, a))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s, g, o, a = self.samples[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(g, dtype=torch.float32),
            torch.tensor(o, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
        )


def train(
    data_path: str = 'dataset/expert_data.pkl',
    device_str: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    save_path: str = 'models/pc_bc_policy.pth',
) -> None:
    device = torch.device(device_str)
    dataset = ExpertDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PolicyNetwork(state_dim=3, goal_dim=2, cond_dim=5, action_dim=2).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for s, g, o, a in loader:
            s = s.to(device)
            g = g.to(device)
            o = o.to(device)
            a = a.to(device)

            pred = model(s, g, o)
            loss = F.mse_loss(pred, a, reduction='mean')

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * s.size(0)
        avg = total_loss / len(dataset)
        print(f"[PC-BC] Epoch {ep+1}/{epochs} Loss: {avg:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved PC-BC policy to {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train PC-BC baseline policy')
    parser.add_argument('--data', type=str, default='dataset/expert_data.pkl')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', type=str, default='models/pc_bc_policy.pth')
    args = parser.parse_args()

    train(
        data_path=args.data,
        device_str=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save,
    )
