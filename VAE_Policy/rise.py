#!/usr/bin/env python3
"""
Two-phase pipeline:
1) Train a VAE (SafetyEncoder + SafetyDecoder) on safety parameters only.
2) Load the trained encoder, freeze it, and use it (with reparameterization sampling)
   inside the LatentSafeImitationPolicy while training the policy and predictor.
"""

import os
import pickle
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# This file now only contains model definitions and training stages for RISE.

# =======================
# FIFO Memory Buffer
# =======================
class ObstacleDataset(Dataset):
    """Stage 1 dataset: obstacle configuration parameters only."""
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)
        # obstacle params: pos(x,y), vel(x,y), radius -> 5-dim
        self.obstacles = [
            np.concatenate([
                d['obstacle']['pos'],
                d['obstacle']['vel'],
                [d['obstacle']['radius']]
            ]).astype(np.float32)
            for d in raw
        ]

    def __len__(self):
        return len(self.obstacles)

    def __getitem__(self, idx):
        return torch.tensor(self.obstacles[idx], dtype=torch.float32)


class ExpertDataset(Dataset):
    """Stage 2 dataset: (state, goal, obstacle_params, expert_action)."""
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)
        self.samples = []
        for d in raw:
            state = d['state']  # 3
            goal = d['goal']    # 2
            obst = np.concatenate([
                d['obstacle']['pos'],
                d['obstacle']['vel'],
                [d['obstacle']['radius']]
            ])  # 5
            action = d['action']  # 2
            self.samples.append((state.astype(np.float32), goal.astype(np.float32), obst.astype(np.float32), action.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s, g, o, a = self.samples[idx]
        return (torch.tensor(s), torch.tensor(g), torch.tensor(o), torch.tensor(a))


# =======================
# VAE: Encoder + Decoder (+ wrapper)
# =======================
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_fc1(x))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, goal_dim: int, latent_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = state_dim + goal_dim + latent_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor, goal: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, goal, z], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.out(h)

# =======================
# Obstacle Predictor, Policy, and Latent-Safe Policy wrapper
# =======================
class ObstaclePredictor(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_dim=64):
        super().__init__()
        input_dim = seq_len * feat_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
    def forward(self, seq_buffer):
        return self.net(seq_buffer)

class ImitationPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

#############################
# Loss helpers
#############################
def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


#############################
# Stage 1: Train VAE
#############################
def train_vae(data_path: str, device: torch.device, epochs: int = 50, batch_size: int = 256,
              lr: float = 1e-3, latent_dim: int = 3, save_path: str = 'models/vae.pth') -> VAE:
    dataset = ObstacleDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = 5  # pos(x,y), vel(x,y), radius
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    opt = optim.Adam(vae.parameters(), lr=lr)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    for ep in range(epochs):
        vae.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar, _ = vae(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        avg = total_loss / len(dataset)
        print(f"[Stage1 VAE] Epoch {ep+1}/{epochs} Loss: {avg:.6f}")
    torch.save(vae.state_dict(), save_path)
    print(f"Saved VAE to {save_path}")
    return vae


#############################
# Stage 2: Train Policy
#############################
def train_policy(data_path: str, device: torch.device, vae: VAE, epochs: int = 50, batch_size: int = 256,
                 lr: float = 1e-3, latent_samples: int = 5, save_path: str = 'models/policy.pth') -> PolicyNetwork:
    dataset = ExpertDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    state_dim, goal_dim, latent_dim, action_dim = 3, 2, vae.enc_mu.out_features, 2
    policy = PolicyNetwork(state_dim, goal_dim, latent_dim, action_dim).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    for ep in range(epochs):
        policy.train()
        total_loss = 0.0
        for state, goal, obst, action in loader:
            state = state.to(device)
            goal = goal.to(device)
            obst = obst.to(device)
            action = action.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(obst)
            # Monte Carlo over latent samples
            preds = []
            for _ in range(latent_samples):
                z = vae.reparameterize(mu, logvar)
                pred = policy(state, goal, z)
                preds.append(pred)
            preds = torch.stack(preds, dim=0).mean(dim=0)
            loss = F.mse_loss(preds, action, reduction='mean')
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * state.size(0)
        avg = total_loss / len(dataset)
        print(f"[Stage2 Policy] Epoch {ep+1}/{epochs} Loss: {avg:.6f}")
    torch.save(policy.state_dict(), save_path)
    print(f"Saved Policy to {save_path}")
    return policy


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train RISE components (Stage 1 & 2).')
    parser.add_argument('--data', type=str, default='dataset/expert_data.pkl')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--vae-epochs', type=int, default=30)
    parser.add_argument('--policy-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=3)
    parser.add_argument('--latent-samples', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='models')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    print('Device:', device)
    vae = train_vae(args.data, device, epochs=args.vae_epochs, batch_size=args.batch_size,
                    latent_dim=args.latent_dim, save_path=os.path.join(args.out_dir, 'vae.pth'))
    policy = train_policy(args.data, device, vae, epochs=args.policy_epochs, batch_size=args.batch_size,
                          latent_samples=args.latent_samples, save_path=os.path.join(args.out_dir, 'policy.pth'))
    print('Training complete.')


if __name__ == '__main__':
    main()

