#!/bin/bash
# Quick script to train only the vision-enhanced policy
# Assumes CVAE and trajectory dataset are already available

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Training Vision-Enhanced Policy"
echo "=========================================="

# Check prerequisites
if [ ! -f "trajectory_image_dataset.pkl" ]; then
    echo "✗ Error: trajectory_image_dataset.pkl not found"
    echo "  Please run ./run_pipeline.sh first to collect dataset"
    exit 1
fi

if [ ! -f "models/cvae_best.pth" ]; then
    echo "✗ Error: CVAE model not found at models/cvae_best.pth"
    echo "  Please train CVAE first using ./run_pipeline.sh"
    exit 1
fi

echo "✓ Prerequisites found"
echo ""

# Train vision policy
python train_vision_policy.py \
    --data trajectory_image_dataset.pkl \
    --cvae models/cvae_best.pth \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --save-dir models

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "=========================================="
echo ""
echo "Model saved to: models/vision_policy_best.pth"
echo ""
echo "Test the trained policy:"
echo "  python test_integration.py --policy models/vision_policy_best.pth --episodes 100"
echo ""
