#!/usr/bin/env python3
"""
Conditional VAE and GAN models for trajectory→image generation.
Models learn to generate realistic scene images conditioned on trajectory vectors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TrajectoryEncoder(nn.Module):
    """Encode trajectory vector to a latent representation."""
    
    def __init__(self, traj_dim: int, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(traj_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(traj))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class ConvDecoder(nn.Module):
    """Decode latent vector to RGB image using transposed convolutions."""
    
    def __init__(self, latent_dim: int = 128, img_channels: int = 3, img_size: int = 128):
        super().__init__()
        self.img_size = img_size
        
        # Initial projection: latent -> (latent_dim, 8, 8)
        self.fc = nn.Linear(latent_dim, latent_dim * 8 * 8)
        
        # Transposed convolutions to upsample
        # 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.deconv5 = nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Project and reshape
        h = self.fc(z)
        h = h.view(-1, 128, 8, 8)
        
        # Upsample through deconv layers
        h = F.relu(self.bn1(self.deconv1(h)))  # 16x16
        h = F.relu(self.bn2(self.deconv2(h)))  # 32x32
        h = F.relu(self.bn3(self.deconv3(h)))  # 64x64
        h = F.relu(self.bn4(self.deconv4(h)))  # 128x128
        h = torch.sigmoid(self.deconv5(h))     # (B, 3, 128, 128)
        
        return h


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for trajectory→image generation.
    Encodes trajectory to latent space, then decodes to image.
    """
    
    def __init__(self, 
                 traj_dim: int,
                 latent_dim: int = 128,
                 img_channels: int = 3,
                 img_size: int = 128):
        super().__init__()
        
        self.traj_encoder = TrajectoryEncoder(traj_dim, latent_dim * 2)  # outputs mean + logvar
        self.decoder = ConvDecoder(latent_dim, img_channels, img_size)
        self.latent_dim = latent_dim
        
    def encode(self, traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode trajectory to latent mean and logvar."""
        h = self.traj_encoder(traj)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)
    
    def forward(self, traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: trajectory -> image."""
        mu, logvar = self.encode(traj)
        z = self.reparameterize(mu, logvar)
        img = self.decode(z)
        return img, mu, logvar
    
    def generate(self, traj: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Generate image from trajectory (optionally deterministic using mean)."""
        with torch.no_grad():
            mu, logvar = self.encode(traj)
            if deterministic:
                z = mu
            else:
                z = self.reparameterize(mu, logvar)
            img = self.decode(z)
        return img


class ConditionalGAN(nn.Module):
    """
    Conditional GAN for higher quality image generation.
    Generator takes trajectory embedding + noise -> image.
    """
    
    class Generator(nn.Module):
        def __init__(self, traj_dim: int, noise_dim: int = 64, img_channels: int = 3):
            super().__init__()
            
            # Embed trajectory
            self.traj_embed = nn.Sequential(
                nn.Linear(traj_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            
            # Combine with noise
            combined_dim = 128 + noise_dim
            self.fc = nn.Linear(combined_dim, 256 * 8 * 8)
            
            # Decoder
            self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
            self.bn1 = nn.BatchNorm2d(128)
            self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
            self.bn2 = nn.BatchNorm2d(64)
            self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
            self.bn3 = nn.BatchNorm2d(32)
            self.deconv4 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
            self.bn4 = nn.BatchNorm2d(16)
            self.deconv5 = nn.Conv2d(16, img_channels, 3, 1, 1)
        
        def forward(self, traj: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Embed trajectory
            traj_emb = self.traj_embed(traj)
            
            # Sample noise if not provided
            if noise is None:
                noise = torch.randn(traj.size(0), 64, device=traj.device)
            
            # Combine
            combined = torch.cat([traj_emb, noise], dim=1)
            h = self.fc(combined).view(-1, 256, 8, 8)
            
            # Decode
            h = F.relu(self.bn1(self.deconv1(h)))
            h = F.relu(self.bn2(self.deconv2(h)))
            h = F.relu(self.bn3(self.deconv3(h)))
            h = F.relu(self.bn4(self.deconv4(h)))
            h = torch.sigmoid(self.deconv5(h))
            
            return h
    
    class Discriminator(nn.Module):
        def __init__(self, traj_dim: int, img_channels: int = 3):
            super().__init__()
            
            # Image encoder
            self.conv1 = nn.Conv2d(img_channels, 32, 4, 2, 1)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
            self.bn4 = nn.BatchNorm2d(256)
            
            # Trajectory encoder
            self.traj_fc = nn.Sequential(
                nn.Linear(traj_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )
            
            # Combined classifier
            self.fc = nn.Sequential(
                nn.Linear(256 * 8 * 8 + 256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 1)
            )
        
        def forward(self, img: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
            # Encode image
            h = F.leaky_relu(self.conv1(img), 0.2)
            h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
            h = F.leaky_relu(self.bn3(self.conv3(h)), 0.2)
            h = F.leaky_relu(self.bn4(self.conv4(h)), 0.2)
            h = h.view(h.size(0), -1)
            
            # Encode trajectory
            t = self.traj_fc(traj)
            
            # Combine and classify
            combined = torch.cat([h, t], dim=1)
            return self.fc(combined)
    
    def __init__(self, traj_dim: int, noise_dim: int = 64, img_channels: int = 3):
        super().__init__()
        self.generator = self.Generator(traj_dim, noise_dim, img_channels)
        self.discriminator = self.Discriminator(traj_dim, img_channels)
        self.noise_dim = noise_dim


def vae_loss(recon_img: torch.Tensor, 
             target_img: torch.Tensor,
             mu: torch.Tensor,
             logvar: torch.Tensor,
             kld_weight: float = 0.001) -> Tuple[torch.Tensor, dict]:
    """
    VAE loss = reconstruction loss + KL divergence.
    
    Args:
        recon_img: reconstructed image (B, C, H, W)
        target_img: target image (B, C, H, W)
        mu, logvar: latent distribution parameters
        kld_weight: weight for KL divergence term
    
    Returns:
        total_loss, loss_dict
    """
    # Reconstruction loss (MSE or BCE)
    recon_loss = F.mse_loss(recon_img, target_img, reduction='mean')
    
    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + kld_weight * kld
    
    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'kld': kld.item(),
    }
    
    return total_loss, loss_dict


def gan_loss(discriminator: nn.Module,
             real_imgs: torch.Tensor,
             fake_imgs: torch.Tensor,
             traj: torch.Tensor,
             label_smoothing: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    GAN loss for generator and discriminator.
    
    Returns:
        gen_loss, disc_loss, loss_dict
    """
    # Discriminator loss
    real_pred = discriminator(real_imgs, traj)
    fake_pred = discriminator(fake_imgs.detach(), traj)
    
    # Label smoothing for stability
    real_label = 1.0 - label_smoothing
    fake_label = 0.0
    
    d_loss_real = F.binary_cross_entropy_with_logits(
        real_pred, torch.full_like(real_pred, real_label)
    )
    d_loss_fake = F.binary_cross_entropy_with_logits(
        fake_pred, torch.full_like(fake_pred, fake_label)
    )
    d_loss = d_loss_real + d_loss_fake
    
    # Generator loss (fool discriminator)
    fake_pred_gen = discriminator(fake_imgs, traj)
    g_loss = F.binary_cross_entropy_with_logits(
        fake_pred_gen, torch.ones_like(fake_pred_gen)
    )
    
    loss_dict = {
        'd_real': d_loss_real.item(),
        'd_fake': d_loss_fake.item(),
        'd_total': d_loss.item(),
        'g_total': g_loss.item(),
    }
    
    return g_loss, d_loss, loss_dict
