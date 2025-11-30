#!/usr/bin/env python3
"""Quick script to verify all paths are correctly set up."""
import sys
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent

print("Path Verification")
print("=" * 60)
print(f"ImageGen directory: {_FILE_DIR}")
print(f"RISE root directory: {_REPO_ROOT}")
print()

# Check for required models
print("Checking for trained models:")
bc_policy = _REPO_ROOT / 'models' / 'pc_bc_policy.pth'
vae_model = _REPO_ROOT / 'models' / 'vae.pth'
policy_model = _REPO_ROOT / 'models' / 'policy.pth'

print(f"  BC policy: {bc_policy}")
print(f"    Exists: {bc_policy.exists()}")
print(f"  VAE model: {vae_model}")
print(f"    Exists: {vae_model.exists()}")
print(f"  Policy model: {policy_model}")
print(f"    Exists: {policy_model.exists()}")
print()

# Check for expert data
expert_data = _REPO_ROOT / 'dataset' / 'expert_data.pkl'
print(f"Expert data: {expert_data}")
print(f"  Exists: {expert_data.exists()}")
print()

# Check directories
print("Directory structure:")
print(f"  RISE root exists: {_REPO_ROOT.exists()}")
print(f"  models/ exists: {(_REPO_ROOT / 'models').exists()}")
print(f"  dataset/ exists: {(_REPO_ROOT / 'dataset').exists()}")
print(f"  ImageGen/ exists: {_FILE_DIR.exists()}")
print()

if all([bc_policy.exists(), vae_model.exists(), policy_model.exists(), expert_data.exists()]):
    print("✓ All required files found! Ready to collect trajectory-image pairs.")
    sys.exit(0)
else:
    print("✗ Missing required files.")
    print("\nTo generate missing files:")
    if not expert_data.exists():
        print("  1. Generate expert data:")
        print(f"     cd {_REPO_ROOT}")
        print("     python expert_data_generator.py")
    if not bc_policy.exists():
        print("  2. Train BC policy:")
        print(f"     cd {_REPO_ROOT / 'Behavior_Cloning'}")
        print("     python train_pc_bc.py")
    if not vae_model.exists() or not policy_model.exists():
        print("  3. Train VAE-latent policy:")
        print(f"     cd {_REPO_ROOT / 'VAE_Policy'}")
        print("     python rise.py")
    sys.exit(1)
