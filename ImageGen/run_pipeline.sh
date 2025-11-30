#!/bin/bash
# Quick-start script for trajectory→image generation pipeline

set -e  # Exit on error

echo "=========================================="
echo "RISE Trajectory→Image Pipeline Setup"
echo "=========================================="

# Navigate to ImageGen directory
cd "$(dirname "$0")"

echo ""
echo "Step 1: Installing dependencies..."
echo "Note: Checking numpy version compatibility..."
pip install -q 'numpy>=1.24.0,<2.0.0' || echo "Warning: numpy version conflicts detected"
pip install -q -r requirements.txt || echo "Warning: Some dependency conflicts exist but continuing..."

echo ""
echo "Step 2: Checking for trained policies..."
if [ ! -f "../models/pc_bc_policy.pth" ]; then
    echo "⚠️  BC policy not found. Training BC policy first..."
    cd ../Behavior_Cloning
    python train_pc_bc.py --epochs 50 --batch-size 256
    cd ../ImageGen
else
    echo "✓ BC policy found"
fi

if [ ! -f "../models/vae.pth" ] || [ ! -f "../models/policy.pth" ]; then
    echo "⚠️  VAE-latent policy not found. Training VAE policy first..."
    cd ../VAE_Policy
    python rise.py --vae-epochs 30 --policy-epochs 50
    cd ../ImageGen
else
    echo "✓ VAE-latent policy found"
fi

echo ""
echo "Step 3: Collecting trajectory-image pairs..."
if [ ! -f "trajectory_image_dataset.pkl" ]; then
    python collect_trajectory_image_pairs.py \
        --episodes-per-policy 150 \
        --resolution 128 \
        --frame-skip 5 \
        --max-attempts 500
else
    echo "✓ Dataset already exists (trajectory_image_dataset.pkl)"
    read -p "Regenerate dataset? (y/N): " regenerate
    if [ "$regenerate" = "y" ] || [ "$regenerate" = "Y" ]; then
        python collect_trajectory_image_pairs.py \
            --episodes-per-policy 150 \
            --resolution 128 \
            --frame-skip 5 \
            --max-attempts 500
    fi
fi

echo ""
echo "Step 4: Training Conditional VAE..."
if [ ! -f "models/cvae_best.pth" ]; then
    mkdir -p models sample_images
    python train_cvae.py \
        --data trajectory_image_dataset.pkl \
        --compact \
        --epochs 100 \
        --batch-size 32 \
        --latent-dim 128 \
        --kld-weight 0.001 \
        --save-samples-every 10
else
    echo "✓ CVAE model already trained (models/cvae_best.pth)"
    read -p "Retrain CVAE? (y/N): " retrain
    if [ "$retrain" = "y" ] || [ "$retrain" = "Y" ]; then
        python train_cvae.py \
            --data trajectory_image_dataset.pkl \
            --compact \
            --epochs 100 \
            --batch-size 32 \
            --latent-dim 128 \
            --kld-weight 0.001 \
            --save-samples-every 10
    fi
fi

echo ""
echo "Step 5: Training policy (PC-BC approach)..."
if [ ! -f "models/vision_policy_best.pth" ]; then
    mkdir -p models
    python train_vision_policy.py \
        --data trajectory_image_dataset.pkl \
        --epochs 50 \
        --batch-size 64 \
        --lr 1e-3 \
        --save-dir models
else
    echo "✓ Policy already trained (models/vision_policy_best.pth)"
    read -p "Retrain policy? (y/N): " retrain_policy
    if [ "$retrain_policy" = "y" ] || [ "$retrain_policy" = "Y" ]; then
        python train_vision_policy.py \
            --data trajectory_image_dataset.pkl \
            --epochs 50 \
            --batch-size 64 \
            --lr 1e-3 \
            --save-dir models
    fi
fi

echo ""
echo "Step 6: Evaluating image generation quality..."
python evaluate.py \
    --data trajectory_image_dataset.pkl \
    --num-samples 500 \
    --out evaluation_results.json

echo ""
echo "Step 7: Running integration test with trained policy..."
python test_integration.py \
    --episodes 50 \
    --policy models/vision_policy_best.pth \
    --video-out vision_policy_test.mp4

echo ""
echo "=========================================="
echo "✓ Pipeline complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  • Dataset: trajectory_image_dataset.pkl"
echo "  • CVAE model: models/cvae_best.pth"
echo "  • Policy model: models/vision_policy_best.pth"
echo "  • Sample images: sample_images/"
echo "  • Evaluation: evaluation_results.json"
echo "  • Test video: vision_policy_test.mp4"
echo ""
echo "Next steps:"
echo "  1. Review sample_images/ to verify image quality"
echo "  2. Check evaluation_results.json for metrics"
echo "  3. Watch vision_policy_test.mp4 to see trained policy in action"
echo "  4. Compare reach rates with BC and VAE-latent baselines"
echo ""
