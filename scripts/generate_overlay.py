# --- scripts/generate_overlay.py ---
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def create_clinical_overlay(scan_path: str, mask_path: str, output_path: str):
    """Generates a professional 3-plane medical overlay of the scan and the predicted mask."""
    print(f"Loading Scan: {scan_path}")
    scan = nib.load(scan_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    # Ensure mask is binary for clean overlay
    mask = (mask > 0).astype(float)
    # Mask out zeros so they become transparent
    mask[mask == 0] = np.nan 

    # Find the center of the mask to ensure we slice right through the teeth
    coords = np.argwhere(mask == 1)
    if len(coords) > 0:
        mid_x, mid_y, mid_z = coords.mean(axis=0).astype(int)
    else:
        # Fallback to absolute center if mask is empty
        mid_x, mid_y, mid_z = [dim // 2 for dim in scan.shape]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("DentalVision AI: 3D CBCT Segmentation Overlay", fontsize=16, fontweight='bold', color='white')
    fig.patch.set_facecolor('#0f172a') # Dark clinical theme

    # 1. Sagittal Plane
    axes[0].imshow(np.rot90(scan[mid_x, :, :]), cmap='gray')
    axes[0].imshow(np.rot90(mask[mid_x, :, :]), cmap='autumn', alpha=0.5)
    axes[0].set_title(f'Sagittal Plane (X={mid_x})', color='white')
    
    # 2. Coronal Plane
    axes[1].imshow(np.rot90(scan[:, mid_y, :]), cmap='gray')
    axes[1].imshow(np.rot90(mask[:, mid_y, :]), cmap='autumn', alpha=0.5)
    axes[1].set_title(f'Coronal Plane (Y={mid_y})', color='white')
    
    # 3. Axial Plane
    axes[2].imshow(np.rot90(scan[:, :, mid_z]), cmap='gray')
    axes[2].imshow(np.rot90(mask[:, :, mid_z]), cmap='autumn', alpha=0.5)
    axes[2].set_title(f'Axial Plane (Z={mid_z})', color='white')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=300)
    print(f"[SUCCESS] High-resolution overlay saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=str, required=True, help="Path to original .nii.gz")
    parser.add_argument("--mask", type=str, required=True, help="Path to predicted .nii.gz")
    parser.add_argument("--out", type=str, default="data/processed/clinical_overlay.png")
    args = parser.parse_args()
    
    create_clinical_overlay(args.scan, args.mask, args.out)