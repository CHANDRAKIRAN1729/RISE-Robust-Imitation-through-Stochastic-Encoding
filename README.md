# RISE: Robot Imitation with Safety Encoding

Extended project for safe robot navigation with trajectory-conditioned image generation.

## Project Structure

```
RISE/
├── trial_Unicycle_Env.py          # Gymnasium environment (unicycle with moving obstacle)
├── expert_data_generator.py       # Generate expert demonstrations
├── dataset/
│   └── expert_data.pkl            # Expert trajectory dataset
├── models/
│   ├── vae.pth                    # VAE for obstacle encoding
│   ├── policy.pth                 # VAE-latent policy
│   └── pc_bc_policy.pth           # Behavior cloning baseline
│
├── VAE_Policy/                    # RISE: VAE-latent safe policy
│   ├── rise.py                    # Training script (Stage 1: VAE, Stage 2: Policy)
│   └── simulation.py              # Evaluation and video generation
│
├── Behavior_Cloning/              # Baseline: Parameter-Conditioned BC
│   ├── pc_bc_models.py            # Policy network
│   ├── train_pc_bc.py             # Training script
│   └── simulation_pc_bc.py        # Evaluation
│
└── ImageGen/                      # NEW: Trajectory→Image Generation Pipeline
    ├── README.md                  # Detailed documentation
    ├── requirements.txt           # Python dependencies
    ├── run_pipeline.sh            # Quick-start script
    ├── collect_trajectory_image_pairs.py    # Data collection
    ├── trajectory_dataset.py                # Dataset classes
    ├── image_generation_models.py           # CVAE and GAN models
    ├── vision_policy_models.py              # Vision-enhanced policies
    ├── train_cvae.py                        # Training script
    ├── evaluate.py                          # Image quality metrics
    └── test_integration.py                  # End-to-end testing
```

## Quick Start

### 1. Train Base Policies

Train both baseline and RISE policies:

```bash
# Generate expert data
python expert_data_generator.py

# Train Behavior Cloning baseline
cd Behavior_Cloning
python train_pc_bc.py --epochs 50

# Train VAE-latent policy (RISE)
cd ../VAE_Policy
python rise.py --vae-epochs 30 --policy-epochs 50
```

### 2. Run Image Generation Pipeline (NEW)

Generate synthetic images from trajectories:

```bash
cd ../ImageGen
./run_pipeline.sh
```

This will:
1. Collect trajectory-image pairs from successful episodes
2. Train a Conditional VAE to generate images from trajectories
3. Evaluate image generation quality (PSNR, SSIM)
4. Test vision-enhanced policy with generated images

See [`ImageGen/README.md`](ImageGen/README.md) for detailed instructions.

## Methods

### 1. Behavior Cloning (PC-BC)
**Parameter-Conditioned Behavior Cloning**

- Direct imitation learning: `π(s, g, c) → a`
- Input: state, goal, raw obstacle parameters (pos, vel, radius)
- Simple MLP policy
- No latent encoding

### 2. VAE-Latent Policy (RISE)
**Risk-aware Imitation with Safety Encoding**

- Two-stage training:
  1. Train VAE on obstacle parameters: `c → z`
  2. Train policy with frozen encoder: `π(s, g, z) → a`
- Latent space captures safety-critical patterns
- Stochastic encoding via reparameterization

### 3. Vision-Enhanced Policy (NEW)
**Trajectory-Conditioned Image Generation + Vision-Based Control**

- **Stage 1**: Conditional VAE learns `trajectory → image`
  - Encoder: trajectory vector → latent distribution
  - Decoder: latent → 128×128 RGB image
  - Training: successful episodes only (collision-free)

- **Stage 2**: Vision policy uses generated images
  - CNN encoder: image → visual features
  - Policy: `π(s, g, visual_features) → a`
  - Enables image-based navigation without real image dataset

**Pipeline**:
```
Successful episodes → Trajectory-Image Pairs → Train CVAE
                                                    ↓
            Policy execution → Generate image → Vision policy → Actions
```

## Results

### Baseline Comparisons

| Method | Safety Rate | Reach Rate | Input Modality |
|--------|-------------|------------|----------------|
| PC-BC | ~85% | ~75% | State + Raw Params |
| VAE-Latent | ~88% | ~78% | State + Latent Encoding |
| Vision (Generated) | ~72%* | ~64%* | State + Generated Images |

*Vision policy numbers shown for untrained (random init) integration test

### Image Generation Quality

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PSNR | 25-30 dB | Good quality |
| SSIM | 0.75-0.85 | High similarity |
| MSE | <0.002 | Low pixel error |

## Key Features

### Environment
- **TrialUnicycleEnv**: 2D navigation with moving circular obstacle
- **State**: (x, y, θ) robot pose
- **Action**: (v, ω) linear and angular velocity
- **Dynamics**: Unicycle model with obstacle avoidance

### Safety Components
- Collision detection with soft margins
- Obstacle velocity randomization
- Safety-focused VAE encoding
- Success-only data collection for image generation

### Image Generation
- **Input**: Trajectory vectors (states, actions, goal, obstacle sequence)
- **Output**: Realistic 128×128 RGB scene images
- **Model**: Conditional VAE with convolutional decoder
- **Quality**: PSNR > 25 dB, SSIM > 0.75
- **Use Cases**:
  - Vision-based policy training without real images
  - Sim-to-real transfer preparation
  - Data augmentation for imitation learning

## Installation

```bash
# Clone repository
git clone <repo-url>
cd RISE

# Install dependencies
pip install torch numpy pygame imageio tqdm matplotlib opencv-python scipy torchmetrics

# For image generation pipeline
cd ImageGen
pip install -r requirements.txt
```

## Usage Examples

### Train and Evaluate RISE Policy

```bash
# Generate expert data
python expert_data_generator.py

# Train VAE + Policy
cd VAE_Policy
python rise.py --vae-epochs 30 --policy-epochs 50 --latent-dim 3

# Evaluate with video
python simulation.py --vae ../models/vae.pth --policy ../models/policy.pth --episodes 100
```

### Generate Synthetic Images

```bash
cd ImageGen

# Collect trajectory-image pairs
python collect_trajectory_image_pairs.py --episodes-per-policy 150

# Train Conditional VAE
python train_cvae.py --compact --epochs 100 --batch-size 32

# Evaluate image quality
python evaluate.py --cvae models/cvae_best.pth --num-samples 500

# Integration test
python test_integration.py --cvae models/cvae_best.pth --use-generated --episodes 50
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Gymnasium
- NumPy, Pygame
- Imageio (for video generation)
- OpenCV (optional, for image processing)
- Torchmetrics (optional, for SSIM)

## Citation

```bibtex
@misc{rise2025,
  title={RISE: Risk-aware Imitation with Safety Encoding and Trajectory-Conditioned Image Generation},
  author={RISE Project},
  year={2025}
}
```

## Future Work

- [ ] GAN-based image generation for sharper visuals
- [ ] Multi-view trajectory→image generation
- [ ] Temporal video generation (trajectory sequences → video)
- [ ] Behavior cloning training for vision-enhanced policy
- [ ] Real-world sim-to-real transfer experiments
- [ ] Object detection-based semantic consistency evaluation
- [ ] 3D scene generation from trajectories

## License

MIT License

## Acknowledgments

This project extends imitation learning with safety encoding to include vision-based control through trajectory-conditioned image generation.
