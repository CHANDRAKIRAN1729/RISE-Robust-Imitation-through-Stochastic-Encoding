#!/usr/bin/env python3
"""
Training script for Conditional VAE: trajectory → image generation.
Trains a model to generate realistic scene images from trajectory vectors.
"""
import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

# Add path setup
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent

from trajectory_dataset import TrajectoryImageDataset, CompactTrajectoryImageDataset
from image_generation_models import ConditionalVAE, vae_loss


def save_sample_images(model: ConditionalVAE,
                       dataset: torch.utils.data.Dataset,
                       device: torch.device,
                       save_path: str,
                       num_samples: int = 8):
    """Generate and save sample images for visualization."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            traj, target_img, _ = dataset[idx]
            traj = traj.unsqueeze(0).to(device)
            
            # Generate image (deterministic)
            generated_img = model.generate(traj, deterministic=True)
            
            # Convert to numpy for display
            target_np = target_img.permute(1, 2, 0).cpu().numpy()
            generated_np = generated_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Plot
            axes[i, 0].imshow(target_np)
            axes[i, 0].set_title('Target')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(generated_np)
            axes[i, 1].set_title('Generated')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample images to {save_path}")


def train_epoch(model: ConditionalVAE,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                kld_weight: float = 0.001) -> dict:
    """Train for one epoch."""
    model.train()
    total_losses = {'total': 0.0, 'recon': 0.0, 'kld': 0.0}
    
    for traj, target_img, _ in dataloader:
        traj = traj.to(device)
        target_img = target_img.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_img, mu, logvar = model(traj)
        
        # Compute loss
        loss, loss_dict = vae_loss(recon_img, target_img, mu, logvar, kld_weight)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key] * traj.size(0)
    
    # Average losses
    num_samples = len(dataloader.dataset)
    for key in total_losses:
        total_losses[key] /= num_samples
    
    return total_losses


def validate(model: ConditionalVAE,
             dataloader: DataLoader,
             device: torch.device,
             kld_weight: float = 0.001) -> dict:
    """Validate model."""
    model.eval()
    total_losses = {'total': 0.0, 'recon': 0.0, 'kld': 0.0}
    
    with torch.no_grad():
        for traj, target_img, _ in dataloader:
            traj = traj.to(device)
            target_img = target_img.to(device)
            
            recon_img, mu, logvar = model(traj)
            loss, loss_dict = vae_loss(recon_img, target_img, mu, logvar, kld_weight)
            
            for key in total_losses:
                total_losses[key] += loss_dict[key] * traj.size(0)
    
    num_samples = len(dataloader.dataset)
    for key in total_losses:
        total_losses[key] /= num_samples
    
    return total_losses


def main():
    parser = argparse.ArgumentParser(description='Train Conditional VAE for trajectory→image generation')
    parser.add_argument('--data', type=str, default='trajectory_image_dataset.pkl',
                        help='Path to trajectory-image dataset')
    parser.add_argument('--compact', action='store_true',
                        help='Use compact trajectory encoding (17D instead of full sequence)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--kld-weight', type=float, default=0.001,
                        help='Weight for KL divergence term')
    parser.add_argument('--max-traj-len', type=int, default=60,
                        help='Max trajectory length (only for full encoding)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--save-samples-every', type=int, default=10,
                        help='Save sample images every N epochs')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    # Resolve save directory relative to ImageGen directory (where script runs)
    save_dir = Path(args.save_dir) if os.path.isabs(args.save_dir) else (_FILE_DIR / args.save_dir).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)
    
    samples_dir = _FILE_DIR / 'sample_images'
    samples_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Device: {device}")
    print(f"Compact encoding: {args.compact}")
    
    # Load dataset
    print("\nLoading dataset...")
    if args.compact:
        full_dataset = CompactTrajectoryImageDataset(
            args.data,
            use_final_frame_only=True
        )
        traj_dim = 17  # compact encoding dimension
    else:
        full_dataset = TrajectoryImageDataset(
            args.data,
            max_trajectory_len=args.max_traj_len,
            use_final_frame_only=True,
            normalize_states=True
        )
        # traj_dim = max_traj_len * (3+2) + 2 + 5
        traj_dim = args.max_traj_len * 5 + 7
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Trajectory dimension: {traj_dim}")
    
    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Create model
    print("\nCreating Conditional VAE model...")
    model = ConditionalVAE(
        traj_dim=traj_dim,
        latent_dim=args.latent_dim,
        img_channels=3,
        img_size=128
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, args.kld_weight)
        
        # Validate
        val_losses = validate(model, val_loader, device, args.kld_weight)
        
        # Update scheduler
        scheduler.step(val_losses['total'])
        
        # Log
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Total: {train_losses['total']:.6f}, Recon: {train_losses['recon']:.6f}, KLD: {train_losses['kld']:.6f}")
        print(f"  Val   - Total: {val_losses['total']:.6f}, Recon: {val_losses['recon']:.6f}, KLD: {val_losses['kld']:.6f}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_path = save_dir / 'cvae_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args),
            }, save_path)
            print(f"  → Saved best model (val_loss: {best_val_loss:.6f})")
        
        # Save sample images periodically
        if (epoch + 1) % args.save_samples_every == 0:
            sample_path = samples_dir / f'samples_epoch_{epoch+1:03d}.png'
            save_sample_images(model, val_dataset, device, str(sample_path), num_samples=8)
    
    # Save final model
    final_path = save_dir / 'cvae_final.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
