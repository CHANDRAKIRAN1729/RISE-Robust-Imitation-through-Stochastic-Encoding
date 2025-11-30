# Trajectory → Image Generation Pipeline

This module extends the RISE project to generate realistic RGB images from robot trajectories, enabling vision-based policy learning without requiring an existing image dataset.

## Overview

The pipeline consists of:

1. **Trajectory Collection**: Run trained BC and VAE-latent policies to collect successful episodes (goal reached, no collisions) with corresponding rendered images
2. **Conditional VAE Training**: Learn to generate realistic scene images conditioned on trajectory vectors
3. **Vision-Enhanced Policy**: Train policies that use both state vectors and visual features from generated images
4. **Evaluation**: Measure image quality (PSNR, SSIM) and policy performance

## Architecture

```
┌──────────────┐
│  Trajectory  │ (state sequence, actions, goal, obstacles)
│   Encoding   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Conditional  │ trajectory → latent → RGB image
│     VAE      │ (encoder-decoder with reparameterization)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Generated  │ (128x128x3 RGB)
│    Image     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Image     │ CNN → feature vector
│   Encoder    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Vision-Enhanced│ (state, goal, image_features) → action
│    Policy     │
└──────────────┘
```

## Dataset Format

`trajectory_image_dataset.pkl` contains:
```python
[
  {
    'trajectory': {
      'states': np.ndarray,      # (T, 3) - [x, y, theta] over time
      'actions': np.ndarray,     # (T, 2) - [v, w] over time
      'goal': np.ndarray,        # (2,) - [goal_x, goal_y]
      'obstacle_seq': np.ndarray # (T, 5) - [ox, oy, vx, vy, radius]
    },
    'images': np.ndarray,        # (N, 128, 128, 3) - sampled frames
    'metadata': {
      'policy_type': str,        # 'bc' or 'vae'
      'episode_length': int,
      'final_state': np.ndarray,
      'camera_config': dict
    }
  },
  ...
]
```

## Usage

### Step 1: Train Base Policies

First, ensure BC and VAE-latent policies are trained:

```bash
# Train BC policy
cd ../Behavior_Cloning
python train_pc_bc.py --epochs 50

# Train VAE-latent policy
cd ../VAE_Policy
python rise.py --vae-epochs 30 --policy-epochs 50
```

### Step 2: Collect Trajectory-Image Pairs

Collect successful episodes with rendered images:

```bash
cd ImageGen
python collect_trajectory_image_pairs.py \
  --episodes-per-policy 150 \
  --resolution 128 \
  --frame-skip 5 \
  --out trajectory_image_dataset.pkl
```

**Parameters:**
- `--episodes-per-policy`: Target number of successful episodes per policy type
- `--resolution`: Image resolution (square)
- `--frame-skip`: Capture image every N steps (reduces dataset size)
- `--max-attempts`: Maximum episodes to try per policy (default: 500)

**Expected output:**
```
BC: 150/200 successful episodes (75.0%)
VAE: 150/180 successful episodes (83.3%)
Total successful episodes: 300
Total images: 3600
Saved to: trajectory_image_dataset.pkl
```

### Step 3: Train Conditional VAE

Train the trajectory→image generation model:

```bash
python train_cvae.py \
  --data trajectory_image_dataset.pkl \
  --compact \
  --epochs 100 \
  --batch-size 32 \
  --latent-dim 128 \
  --kld-weight 0.001 \
  --save-dir models
```

**Key arguments:**
- `--compact`: Use compact trajectory encoding (17D summary statistics) instead of full sequence
- `--kld-weight`: Weight for KL divergence term (lower = more diverse images, higher = more faithful reconstruction)
- `--save-samples-every`: Generate sample images every N epochs

**Outputs:**
- `models/cvae_best.pth`: Best model (lowest validation loss)
- `models/cvae_final.pth`: Final model after all epochs
- `sample_images/`: Generated sample images during training

**Training tips:**
- Start with `--kld-weight 0.0001` for diverse images, increase to 0.001-0.01 for higher fidelity
- Monitor sample images to ensure quality improves
- Typical training time: 1-2 hours on GPU for 100 epochs

### Step 4: Evaluate Image Quality

Measure generation quality metrics:

```bash
python evaluate.py \
  --data trajectory_image_dataset.pkl \
  --cvae models/cvae_best.pth \
  --num-samples 500 \
  --out evaluation_results.json
```

**Metrics:**
- **MSE**: Mean Squared Error (lower is better)
- **PSNR**: Peak Signal-to-Noise Ratio in dB (>25 dB is good, >30 dB is excellent)
- **SSIM**: Structural Similarity Index (0-1, higher is better, >0.8 is good)

**Sample output:**
```
Image Quality Metrics:
  MSE:  0.001234 ± 0.000456
  PSNR: 28.45 ± 3.21 dB
  SSIM: 0.8234 ± 0.0567
  
Interpretation:
  ✓ Good PSNR (>25 dB) - images are reasonably high quality
  ✓ High SSIM (>0.8) - good structural similarity
```

### Step 5: Integration Test

Test the complete pipeline with vision-enhanced policy:

```bash
python test_integration.py \
  --cvae models/cvae_best.pth \
  --episodes 50 \
  --use-generated \
  --video-out vision_policy_test.mp4
```

**Flags:**
- `--use-generated`: Use CVAE-generated images (omit to use real rendered images for comparison)
- `--policy`: Path to pre-trained vision policy (optional, uses random init if not provided)

**Output:**
```
Integration Test Results
===========================================================
Episodes: 50
Using generated images: True
Safety Rate: 72.00%
Reach Rate: 64.00%
Video saved to: vision_policy_test.mp4
===========================================================
```

## Model Architectures

### Conditional VAE
- **Trajectory Encoder**: MLP (traj_dim → 256 → 256 → latent_dim*2)
- **Image Decoder**: TransposedConv (latent → 8x8 → 16x16 → 32x32 → 64x64 → 128x128)
- **Loss**: Reconstruction (MSE) + KL Divergence

### Vision-Enhanced Policy
- **Image Encoder**: CNN (128x128x3 → 64x64 → 32x32 → 16x16 → 8x8 → 128D features)
- **Policy Network**: MLP (state + goal + img_features → 256 → 256 → 128 → action)

## Advanced Usage

### Training Vision-Enhanced Policy (End-to-End)

Train a policy that uses generated images:

```python
from vision_policy_models import EndToEndVisionPolicy
from image_generation_models import ConditionalVAE
import torch

# Load pre-trained CVAE
cvae = ConditionalVAE(traj_dim=17, latent_dim=128).to(device)
cvae.load_state_dict(torch.load('models/cvae_best.pth')['model_state_dict'])
cvae.eval()

# Create vision policy
policy = EndToEndVisionPolicy(
    state_dim=3, goal_dim=2, img_feature_dim=128, action_dim=2,
    freeze_encoder=False  # Allow fine-tuning of image encoder
).to(device)

# Training loop would go here (imitation learning on expert data)
# ...
```

### Ablation Studies

Compare different input modalities:

```python
from vision_policy_models import HybridVisionPolicy

# Vision only
policy_vision = HybridVisionPolicy(
    use_vision=True, use_obstacle_params=False
)

# Obstacle params only (baseline)
policy_params = HybridVisionPolicy(
    use_vision=False, use_obstacle_params=True
)

# Both (fusion)
policy_hybrid = HybridVisionPolicy(
    use_vision=True, use_obstacle_params=True
)
```

## File Structure

```
ImageGen/
├── collect_trajectory_image_pairs.py  # Data collection script
├── trajectory_dataset.py              # Dataset classes
├── image_generation_models.py         # CVAE and GAN models
├── vision_policy_models.py            # Vision-enhanced policies
├── train_cvae.py                      # Training script
├── evaluate.py                        # Evaluation metrics
├── test_integration.py                # Integration testing
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── trajectory_image_dataset.pkl       # Collected data (generated)
├── models/                            # Saved model checkpoints
│   ├── cvae_best.pth
│   ├── cvae_final.pth
│   └── vision_policy.pth
└── sample_images/                     # Generated samples during training
    ├── samples_epoch_010.png
    └── ...
```

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pygame>=2.5.0
imageio>=2.31.0
matplotlib>=3.7.0
tqdm>=4.65.0
opencv-python>=4.8.0  # Optional: for image resizing and overlay
scipy>=1.11.0         # Fallback for image ops
torchmetrics>=1.0.0   # Optional: for SSIM metric
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Low PSNR/SSIM scores
- **Increase training epochs**: Try 150-200 epochs
- **Adjust KLD weight**: Lower values (0.0001) give more diverse but less accurate images
- **Increase latent dimension**: Try 256 or 512
- **Check dataset quality**: Ensure collected episodes have good coverage of scenarios

### CVAE generates blurry images
- **Reduce KLD weight**: Try 0.0001 instead of 0.001
- **Use GAN**: Switch to ConditionalGAN model (more complex but sharper)
- **Add perceptual loss**: Use VGG features instead of pixel MSE

### Integration test shows poor policy performance
- **This is expected**: Random policy initialization won't navigate well
- **Train vision policy**: Use behavior cloning on expert data with generated images
- **Verify image quality first**: Ensure PSNR > 25 dB before policy training

### Out of memory during training
- **Reduce batch size**: Try 16 or 8
- **Use compact encoding**: Add `--compact` flag
- **Reduce image resolution**: Modify dataset collection to use 64x64 images

## Citation

If you use this trajectory→image generation pipeline in your research, please cite:

```bibtex
@misc{rise_trajectory_to_image,
  title={Trajectory-Conditioned Image Generation for Vision-Based Robot Navigation},
  author={RISE Project},
  year={2025}
}
```

## Future Improvements

- [ ] Implement GAN training for sharper images
- [ ] Add temporal consistency (video generation from trajectory sequences)
- [ ] Object detection-based semantic evaluation
- [ ] Multi-view image generation from different camera angles
- [ ] Behavior cloning training script for vision-enhanced policy
- [ ] Real-world image domain adaptation
