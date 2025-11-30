# Trajectory → Image Generation: Quick Reference

## Complete Pipeline in 5 Steps

### Step 1: Verify Setup
```bash
cd RISE/ImageGen
python test_components.py
```
Expected: All tests pass ✓

### Step 2: Collect Data (if policies are trained)
```bash
python collect_trajectory_image_pairs.py \
  --episodes-per-policy 150 \
  --resolution 128 \
  --frame-skip 5
```
**Output**: `trajectory_image_dataset.pkl` (~300 successful episodes)

**Time**: 15-30 minutes depending on policy success rates

### Step 3: Train Image Generator
```bash
python train_cvae.py \
  --data trajectory_image_dataset.pkl \
  --compact \
  --epochs 100 \
  --batch-size 32 \
  --latent-dim 128 \
  --save-samples-every 10
```
**Output**: 
- `models/cvae_best.pth` (best model)
- `sample_images/` (training samples)

**Time**: 1-2 hours on GPU, 5-8 hours on CPU

**Monitor**: Check `sample_images/` every 10 epochs to verify quality improves

### Step 4: Evaluate Quality
```bash
python evaluate.py \
  --cvae models/cvae_best.pth \
  --num-samples 500
```
**Expected Metrics**:
- PSNR: > 25 dB (good) or > 30 dB (excellent)
- SSIM: > 0.7 (acceptable) or > 0.8 (good)

**Output**: `evaluation_results.json`

### Step 5: Integration Test
```bash
python test_integration.py \
  --cvae models/cvae_best.pth \
  --episodes 50 \
  --use-generated \
  --video-out vision_test.mp4
```
**Output**: `vision_test.mp4` showing policy using generated images

**Note**: Policy is randomly initialized (untrained), so performance will be poor. This just tests integration.

---

## Automated Pipeline

Run everything automatically:
```bash
./run_pipeline.sh
```

This executes all 5 steps with sensible defaults.

---

## File Outputs

| File | Description | Size (approx) |
|------|-------------|---------------|
| `trajectory_image_dataset.pkl` | Collected trajectory-image pairs | 1-5 GB |
| `models/cvae_best.pth` | Best CVAE checkpoint | 10-20 MB |
| `models/cvae_final.pth` | Final CVAE checkpoint | 10-20 MB |
| `sample_images/samples_epoch_*.png` | Training samples | 1-2 MB each |
| `evaluation_results.json` | Quality metrics | 1-5 KB |
| `vision_test.mp4` | Integration test video | 5-50 MB |

---

## Troubleshooting

### Issue: Low image quality (PSNR < 20 dB)

**Solutions**:
1. Train longer: `--epochs 150`
2. Lower KLD weight: `--kld-weight 0.0001`
3. Increase latent dim: `--latent-dim 256`
4. Collect more data: `--episodes-per-policy 300`

### Issue: Blurry images

**Solutions**:
1. Reduce KLD weight to 0.0001
2. Add perceptual loss (modify `vae_loss` in `image_generation_models.py`)
3. Try GAN instead of VAE (use `ConditionalGAN` class)

### Issue: Out of memory

**Solutions**:
1. Reduce batch size: `--batch-size 16`
2. Use compact encoding: `--compact` (already default)
3. Reduce image resolution in data collection: `--resolution 64`

### Issue: Dataset collection fails (low success rate)

**Solutions**:
1. Increase max attempts: `--max-attempts 1000`
2. Verify policies are properly trained
3. Check policy paths: `--bc-policy`, `--vae`, `--vae-policy`

### Issue: Training is slow

**Solutions**:
1. Use GPU: Script auto-detects CUDA
2. Reduce workers: `num_workers=0` in dataloaders
3. Train on subset first to verify: `--epochs 20`

---

## Model Hyperparameters

### Recommended for Quick Testing (30 min)
```bash
python train_cvae.py --compact --epochs 30 --batch-size 64 --latent-dim 64
```

### Recommended for Good Quality (2 hours)
```bash
python train_cvae.py --compact --epochs 100 --batch-size 32 --latent-dim 128
```

### Recommended for Best Quality (4+ hours)
```bash
python train_cvae.py --compact --epochs 200 --batch-size 32 --latent-dim 256 --kld-weight 0.001
```

---

## Architecture Summary

```
Trajectory Encoding (17D):
├─ Initial state (3): [x₀, y₀, θ₀]
├─ Final state (3): [x_f, y_f, θ_f]
├─ Goal (2): [g_x, g_y]
├─ Mean action (2): [v̄, ω̄]
├─ Std action (2): [σ_v, σ_ω]
└─ Final obstacle (5): [o_x, o_y, v_x, v_y, r]

         ↓

Conditional VAE:
├─ Encoder: 17 → 256 → 256 → 256 (mu + logvar)
├─ Latent: 128D
└─ Decoder: 128 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128 RGB

         ↓

Vision Policy (optional):
├─ Image Encoder: CNN (128×128×3 → 128D features)
└─ Policy: [state, goal, features] → [v, ω]
```

---

## Next Steps After Pipeline

1. **Train Vision-Enhanced Policy**:
   - Collect expert demonstrations with generated images
   - Use behavior cloning: `(state, goal, generated_image) → action`
   - Compare performance to baseline PC-BC

2. **Ablation Studies**:
   - Vision only vs. params only vs. hybrid
   - Generated images vs. real rendered images
   - Different trajectory encoding strategies

3. **Improve Image Quality**:
   - Train GAN instead of VAE
   - Add perceptual loss (VGG features)
   - Multi-scale architecture
   - Temporal consistency (video generation)

4. **Real-World Transfer**:
   - Domain adaptation to real images
   - Fine-tune on small real dataset
   - Test in physical robot

---

## Citation

If you use this pipeline, please cite:

```bibtex
@misc{rise_trajectory_image_2025,
  title={Trajectory-Conditioned Image Generation for Vision-Based Robot Navigation},
  author={RISE Project},
  year={2025},
  howpublished={https://github.com/...}
}
```

---

## Support

For issues or questions:
1. Check `ImageGen/README.md` for detailed documentation
2. Review `test_components.py` output for integration problems
3. Inspect `sample_images/` during training for quality checks
4. Check `evaluation_results.json` for quantitative metrics
