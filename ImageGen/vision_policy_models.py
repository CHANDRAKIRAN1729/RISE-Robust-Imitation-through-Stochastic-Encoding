#!/usr/bin/env python3
"""
Vision-enhanced policy models that use both state vectors and generated images.
Combines trajectory→image generator with policy network for image-aware navigation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ImageEncoder(nn.Module):
    """Encode RGB image to compact feature vector using CNN."""
    
    def __init__(self, img_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        
        # Convolutional encoder (similar to discriminator)
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)  # 64x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 16x16
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 8x8
        self.bn4 = nn.BatchNorm2d(256)
        
        # Flatten and project to feature_dim
        self.fc = nn.Linear(256 * 8 * 8, feature_dim)
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 3, H, W) image tensor, values in [0, 1]
        
        Returns:
            features: (B, feature_dim)
        """
        h = F.relu(self.conv1(img))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = h.view(h.size(0), -1)
        features = self.fc(h)
        return features


class VisionEnhancedPolicy(nn.Module):
    """
    Policy network that uses both state vectors and visual features.
    Input: state (x,y,theta), goal (gx,gy), image_features
    Output: action (v, w)
    """
    
    def __init__(self,
                 state_dim: int = 3,
                 goal_dim: int = 2,
                 img_feature_dim: int = 128,
                 action_dim: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        
        input_dim = state_dim + goal_dim + img_feature_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, action_dim)
        
        # Small init for stability
        nn.init.uniform_(self.out.weight, -0.01, 0.01)
        nn.init.zeros_(self.out.bias)
    
    def forward(self,
                state: torch.Tensor,
                goal: torch.Tensor,
                img_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
            goal: (B, goal_dim)
            img_features: (B, img_feature_dim)
        
        Returns:
            action: (B, action_dim)
        """
        x = torch.cat([state, goal, img_features], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.out(h)


class EndToEndVisionPolicy(nn.Module):
    """
    End-to-end model combining image encoder and policy.
    Can be trained jointly or with frozen encoder.
    """
    
    def __init__(self,
                 state_dim: int = 3,
                 goal_dim: int = 2,
                 img_channels: int = 3,
                 img_feature_dim: int = 128,
                 action_dim: int = 2,
                 hidden_dim: int = 256,
                 freeze_encoder: bool = False):
        super().__init__()
        
        self.image_encoder = ImageEncoder(img_channels, img_feature_dim)
        self.policy = VisionEnhancedPolicy(state_dim, goal_dim, img_feature_dim, action_dim, hidden_dim)
        
        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
    
    def forward(self,
                state: torch.Tensor,
                goal: torch.Tensor,
                img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
            goal: (B, goal_dim)
            img: (B, 3, H, W)
        
        Returns:
            action: (B, action_dim)
        """
        img_features = self.image_encoder(img)
        action = self.policy(state, goal, img_features)
        return action
    
    def freeze_encoder(self):
        """Freeze image encoder weights."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze image encoder weights."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True


class HybridVisionPolicy(nn.Module):
    """
    Hybrid policy that can use both raw obstacle parameters AND visual features.
    Useful for ablation studies and comparing modalities.
    """
    
    def __init__(self,
                 state_dim: int = 3,
                 goal_dim: int = 2,
                 obstacle_dim: int = 5,
                 img_feature_dim: int = 128,
                 action_dim: int = 2,
                 hidden_dim: int = 256,
                 use_vision: bool = True,
                 use_obstacle_params: bool = True):
        super().__init__()
        
        self.use_vision = use_vision
        self.use_obstacle_params = use_obstacle_params
        
        # Image encoder (optional)
        if use_vision:
            self.image_encoder = ImageEncoder(3, img_feature_dim)
        
        # Compute input dimension based on what's enabled
        input_dim = state_dim + goal_dim
        if use_vision:
            input_dim += img_feature_dim
        if use_obstacle_params:
            input_dim += obstacle_dim
        
        # Policy network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, action_dim)
        
        nn.init.uniform_(self.out.weight, -0.01, 0.01)
        nn.init.zeros_(self.out.bias)
    
    def forward(self,
                state: torch.Tensor,
                goal: torch.Tensor,
                img: torch.Tensor = None,
                obstacle_params: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
            goal: (B, goal_dim)
            img: (B, 3, H, W) - optional if use_vision=True
            obstacle_params: (B, obstacle_dim) - optional if use_obstacle_params=True
        
        Returns:
            action: (B, action_dim)
        """
        inputs = [state, goal]
        
        if self.use_vision:
            if img is None:
                raise ValueError("img required when use_vision=True")
            img_features = self.image_encoder(img)
            inputs.append(img_features)
        
        if self.use_obstacle_params:
            if obstacle_params is None:
                raise ValueError("obstacle_params required when use_obstacle_params=True")
            inputs.append(obstacle_params)
        
        x = torch.cat(inputs, dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.out(h)


class GenerativeVisionPolicy(nn.Module):
    """
    Complete pipeline: trajectory encoder → image generator → policy.
    Uses CVAE to generate images on-the-fly from trajectory context.
    """
    
    def __init__(self,
                 cvae_model,
                 state_dim: int = 3,
                 goal_dim: int = 2,
                 img_feature_dim: int = 128,
                 action_dim: int = 2,
                 hidden_dim: int = 256,
                 freeze_generator: bool = True):
        super().__init__()
        
        self.cvae = cvae_model
        self.image_encoder = ImageEncoder(3, img_feature_dim)
        self.policy = VisionEnhancedPolicy(state_dim, goal_dim, img_feature_dim, action_dim, hidden_dim)
        
        if freeze_generator:
            for param in self.cvae.parameters():
                param.requires_grad = False
    
    def forward(self,
                state: torch.Tensor,
                goal: torch.Tensor,
                trajectory_encoding: torch.Tensor,
                use_real_img: bool = False,
                real_img: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, state_dim)
            goal: (B, goal_dim)
            trajectory_encoding: (B, traj_dim) for CVAE
            use_real_img: if True, use real_img instead of generating
            real_img: (B, 3, H, W) - optional real image
        
        Returns:
            action: (B, action_dim)
            generated_img: (B, 3, H, W) - the generated or used image
        """
        if use_real_img and real_img is not None:
            img = real_img
        else:
            # Generate image from trajectory
            with torch.no_grad() if self.cvae.training == False else torch.enable_grad():
                img = self.cvae.generate(trajectory_encoding, deterministic=True)
        
        # Encode image
        img_features = self.image_encoder(img)
        
        # Predict action
        action = self.policy(state, goal, img_features)
        
        return action, img
