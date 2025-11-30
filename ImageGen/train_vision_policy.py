#!/usr/bin/env python3
"""
Train policy using behavior cloning on expert data (PC-BC approach).
Uses (state, goal, obstacle_params) -> action mapping.
No vision/images involved - this is the same as PC-BC baseline.
"""
import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))


class SimplePolicyNetwork(nn.Module):
    """
    MLP policy: (state, goal, obstacle_params) -> action
    Same as PC-BC architecture.
    """
    def __init__(self, state_dim=3, goal_dim=2, obstacle_dim=5, action_dim=2, hidden_dim=128):
        super().__init__()
        input_dim = state_dim + goal_dim + obstacle_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        
        nn.init.uniform_(self.out.weight, -0.01, 0.01)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, state, goal, obstacle):
        x = torch.cat([state, goal, obstacle], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.out(h)


class SimpleDataset(Dataset):
    """Dataset for (state, goal, obstacle) -> action."""
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.samples = []
        for d in raw_data:
            traj = d['trajectory']
            states = np.array(traj['states'], dtype=np.float32)
            actions = np.array(traj['actions'], dtype=np.float32)
            goal = np.array(traj['goal'], dtype=np.float32)
            obstacle_seq = np.array(traj['obstacle_seq'], dtype=np.float32)
            
            for t in range(len(states)):
                self.samples.append({
                    'state': states[t],
                    'goal': goal,
                    'obstacle': obstacle_seq[t],
                    'action': actions[t]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s['state'], dtype=torch.float32),
            torch.tensor(s['goal'], dtype=torch.float32),
            torch.tensor(s['obstacle'], dtype=torch.float32),
            torch.tensor(s['action'], dtype=torch.float32)
        )


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for state, goal, obstacle, action in tqdm(dataloader, desc='Training'):
        state, goal, obstacle, action = state.to(device), goal.to(device), obstacle.to(device), action.to(device)
        
        pred_action = model(state, goal, obstacle)
        loss = F.mse_loss(pred_action, action)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * state.size(0)
    
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for state, goal, obstacle, action in tqdm(dataloader, desc='Validation'):
            state, goal, obstacle, action = state.to(device), goal.to(device), obstacle.to(device), action.to(device)
            pred_action = model(state, goal, obstacle)
            loss = F.mse_loss(pred_action, action)
            total_loss += loss.item() * state.size(0)
    
    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description='Train policy with behavior cloning (PC-BC)')
    parser.add_argument('--data', type=str, default='trajectory_image_dataset.pkl')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='models')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    save_dir = Path(args.save_dir) if os.path.isabs(args.save_dir) else (_FILE_DIR / args.save_dir).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)
    data_path = Path(args.data) if os.path.isabs(args.data) else (_FILE_DIR / args.data).resolve()
    
    print("="*60)
    print("Training Policy (PC-BC)")
    print("="*60)
    print(f"Device: {device}")
    print(f"Data: {data_path}\n")
    
    full_dataset = SimpleDataset(str(data_path))
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = SimplePolicyNetwork().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')
    save_path = save_dir / 'vision_policy.pth'
    best_path = save_dir / 'vision_policy_best.pth'
    
    print("Starting training...")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        
        torch.save(model.state_dict(), save_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"✓ New best (val_loss: {val_loss:.6f})")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best model: {best_path}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print("="*60)


if __name__ == '__main__':
    main()
