# --- scripts/inspect_data.py ---
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage


def inspect_nifti_volume(file_path: str | Path, output_image_path: str = "inspection_output.png") -> None:
    """
    Loads a 3D NIfTI volume, prints its medical metadata, and saves a mid-slice visualization.
    
    Args:
        file_path (str | Path): Path to the .nii.gz file.
        output_image_path (str): Path to save the output visualization plot.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    print(f"--- Inspecting Volume: {file_path.name} ---")
    
    # 1. Load Image and Metadata using MONAI
    loader = LoadImage(image_only=False, ensure_channel_first=True)
    image_tensor, metadata = loader(file_path)
    
    # Convert to numpy for inspection
    volume = image_tensor.numpy()[0]  # Strip the channel dimension for pure 3D array
    
    # 2. Extract Metadata
    shape = volume.shape
    spacing = metadata.get("pixdim", [1.0, 1.0, 1.0, 1.0])[1:4]  # Voxel spacing in mm
    affine = metadata.get("affine", np.eye(4))
    
    # 3. Calculate Intensity Statistics (Hounsfield Units for CT/CBCT)
    min_val, max_val = volume.min(), volume.max()
    mean_val, std_val = volume.mean(), volume.std()

    # 4. Print Engineering Report
    print("\n[Metadata]")
    print(f"Volume Dimensions (X, Y, Z): {shape}")
    print(f"Voxel Spacing (mm):          [{spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}]")
    print(f"Orientation (Affine matrix): \n{affine}")
    
    print("\n[Intensity Statistics (HU)]")
    print(f"Min: {min_val:.2f} | Max: {max_val:.2f}")
    print(f"Mean: {mean_val:.2f} | Std Dev: {std_val:.2f}")
    
    # 5. Visual Inspection: Extract middle slices across 3 axes
    mid_x = shape[0] // 2
    mid_y = shape[1] // 2
    mid_z = shape[2] // 2
    
    slice_sagittal = volume[mid_x, :, :]
    slice_coronal = volume[:, mid_y, :]
    slice_axial = volume[:, :, mid_z]

    # Plot and save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(np.rot90(slice_sagittal), cmap="gray")
    axes[0].set_title(f"Sagittal (X={mid_x})")
    axes[0].axis("off")
    
    axes[1].imshow(np.rot90(slice_coronal), cmap="gray")
    axes[1].set_title(f"Coronal (Y={mid_y})")
    axes[1].axis("off")
    
    axes[2].imshow(np.rot90(slice_axial), cmap="gray")
    axes[2].set_title(f"Axial (Z={mid_z})")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"\n[Success] Visualization saved to {output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a 3D medical volume.")
    parser.add_argument(
        "--file", 
        type=str, 
        required=True, 
        help="Path to the .nii.gz file to inspect"
    )
    args = parser.parse_args()
    
    inspect_nifti_volume(args.file)
    