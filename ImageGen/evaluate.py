#!/usr/bin/env python3
"""
Evaluation script for trajectory→image generation pipeline.
Measures:
1. Image generation quality (MSE, SSIM, PSNR)
2. Semantic consistency (obstacle/robot position accuracy)
3. Policy performance with generated vs. real images
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import pickle
import json

_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from image_generation_models import ConditionalVAE
from trajectory_dataset import CompactTrajectoryImageDataset
from torch.utils.data import DataLoader


def compute_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute Mean Squared Error between two images."""
    return torch.mean((img1 - img2) ** 2).item()


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute Structural Similarity Index (simplified version).
    For proper SSIM, use torchmetrics or skimage.
    """
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(img1.device)
        # Ensure batch dimension
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        return ssim(img1, img2).item()
    except (ImportError, RuntimeError, Exception):
        # Fallback: correlation-based similarity
        img1_flat = img1.view(-1)
        img2_flat = img2.view(-1)
        corr = torch.corrcoef(torch.stack([img1_flat, img2_flat]))[0, 1]
        return corr.item()


def evaluate_image_quality(cvae: ConditionalVAE,
                           dataloader: DataLoader,
                           device: torch.device,
                           num_samples: int = None) -> dict:
    """Evaluate image generation quality metrics."""
    cvae.eval()
    
    metrics = {
        'mse': [],
        'psnr': [],
        'ssim': [],
    }
    
    count = 0
    with torch.no_grad():
        for traj, target_img, _ in tqdm(dataloader, desc='Evaluating image quality'):
            traj = traj.to(device)
            target_img = target_img.to(device)
            
            # Generate image (deterministic)
            generated_img = cvae.generate(traj, deterministic=True)
            
            # Compute metrics per sample
            for i in range(traj.size(0)):
                gen = generated_img[i]
                tgt = target_img[i]
                
                metrics['mse'].append(compute_mse(gen, tgt))
                metrics['psnr'].append(compute_psnr(gen, tgt))
                metrics['ssim'].append(compute_ssim(gen, tgt))
                
                count += 1
                if num_samples is not None and count >= num_samples:
                    break
            
            if num_samples is not None and count >= num_samples:
                break
    
    # Aggregate
    results = {
        'mse_mean': np.mean(metrics['mse']),
        'mse_std': np.std(metrics['mse']),
        'psnr_mean': np.mean(metrics['psnr']),
        'psnr_std': np.std(metrics['psnr']),
        'ssim_mean': np.mean(metrics['ssim']),
        'ssim_std': np.std(metrics['ssim']),
        'num_samples': count,
    }
    
    return results


def evaluate_semantic_consistency(dataset_path: str,
                                  cvae: ConditionalVAE,
                                  device: torch.device,
                                  num_samples: int = 100) -> dict:
    """
    Evaluate semantic consistency: do generated images preserve obstacle/goal positions?
    This is a placeholder - requires object detection or manual annotation.
    """
    # Placeholder: would need to implement object detection or pose estimation
    # to extract obstacle/robot positions from generated images and compare
    # with ground truth trajectory parameters.
    
    print("Semantic consistency evaluation requires object detection (not implemented)")
    return {
        'obstacle_position_error': None,
        'robot_position_error': None,
        'note': 'Requires object detector for full evaluation'
    }


def compare_policy_performance(real_stats: dict, generated_stats: dict) -> dict:
    """Compare policy performance with real vs generated images."""
    comparison = {
        'real_reach_rate': real_stats['reach_rate'],
        'generated_reach_rate': generated_stats['reach_rate'],
        'real_safety_rate': real_stats['safety_rate'],
        'generated_safety_rate': generated_stats['safety_rate'],
        'reach_rate_diff': generated_stats['reach_rate'] - real_stats['reach_rate'],
        'safety_rate_diff': generated_stats['safety_rate'] - real_stats['safety_rate'],
    }
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Evaluate trajectory→image generation pipeline')
    parser.add_argument('--data', type=str, default='trajectory_image_dataset.pkl')
    parser.add_argument('--cvae', type=str, default='ImageGen/models/cvae_best.pth')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=str, default='evaluation_results.json')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"Device: {device}")
    print(f"Evaluating CVAE: {args.cvae}")
    
    # Resolve paths
    cvae_path = Path(args.cvae) if os.path.isabs(args.cvae) else (_REPO_ROOT / args.cvae).resolve()
    data_path = Path(args.data) if os.path.isabs(args.data) else (_FILE_DIR / args.data).resolve()
    
    # Load CVAE
    print("\nLoading CVAE model...")
    checkpoint = torch.load(cvae_path, map_location=device)
    cvae_args = checkpoint['args']
    
    traj_dim = 17  # compact encoding
    cvae = ConditionalVAE(
        traj_dim=traj_dim,
        latent_dim=cvae_args.get('latent_dim', 128),
        img_channels=3,
        img_size=128
    ).to(device)
    cvae.load_state_dict(checkpoint['model_state_dict'])
    cvae.eval()
    print(f"Model loaded (epoch {checkpoint['epoch']})")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = CompactTrajectoryImageDataset(str(data_path), use_final_frame_only=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Dataset size: {len(dataset)}")
    
    # Evaluate image quality
    print("\n" + "="*60)
    print("Evaluating Image Generation Quality")
    print("="*60)
    
    quality_results = evaluate_image_quality(cvae, dataloader, device, args.num_samples)
    
    print(f"\nImage Quality Metrics:")
    print(f"  MSE:  {quality_results['mse_mean']:.6f} ± {quality_results['mse_std']:.6f}")
    print(f"  PSNR: {quality_results['psnr_mean']:.2f} ± {quality_results['psnr_std']:.2f} dB")
    print(f"  SSIM: {quality_results['ssim_mean']:.4f} ± {quality_results['ssim_std']:.4f}")
    print(f"  Samples evaluated: {quality_results['num_samples']}")
    
    # Evaluate semantic consistency (placeholder)
    print("\n" + "="*60)
    print("Evaluating Semantic Consistency")
    print("="*60)
    semantic_results = evaluate_semantic_consistency(args.data, cvae, device, args.num_samples or 100)
    
    # Compile all results
    all_results = {
        'model_checkpoint': args.cvae,
        'dataset': args.data,
        'num_samples_evaluated': quality_results['num_samples'],
        'image_quality': quality_results,
        'semantic_consistency': semantic_results,
        'model_info': {
            'epoch': checkpoint['epoch'],
            'val_loss': checkpoint.get('val_loss', None),
            'latent_dim': cvae_args.get('latent_dim'),
            'kld_weight': cvae_args.get('kld_weight'),
        }
    }
    
    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete. Results saved to: {out_path}")
    print(f"{'='*60}")
    
    # Interpretation
    print("\nInterpretation:")
    if quality_results['psnr_mean'] > 25:
        print("  ✓ Good PSNR (>25 dB) - images are reasonably high quality")
    elif quality_results['psnr_mean'] > 20:
        print("  ~ Moderate PSNR (20-25 dB) - acceptable quality")
    else:
        print("  ✗ Low PSNR (<20 dB) - images may need improvement")
    
    if quality_results['ssim_mean'] > 0.8:
        print("  ✓ High SSIM (>0.8) - good structural similarity")
    elif quality_results['ssim_mean'] > 0.6:
        print("  ~ Moderate SSIM (0.6-0.8) - acceptable similarity")
    else:
        print("  ✗ Low SSIM (<0.6) - structural differences present")


if __name__ == '__main__':
    main()
