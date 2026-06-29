from typing import List, Dict 
import torch 
from monai.transforms import(
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    Orientationd, 
    Spacingd, 
    ScaleIntensityRanged,
    Resized, 
    Lambdad, 
    ToTensord,
)

def get_preprocessing_transforms(
    spatial_size: tuple = (96, 96, 96), 
    is_training: bool = True
) -> Compose:
    """
    Constructs the standard preprocessing pipeline for 3D dental CBCT scans.
    
    Args:
        spatial_size (tuple): The target (X, Y, Z) dimensions to resize the volume.
                              Crucial for batching and avoiding GPU OOM errors.
        is_training (bool): Flag to determine if we should include training-specific 
                            augmentations (handled in Phase 7).
                            
    Returns:
        monai.transforms.Compose: The chained sequence of operations.
    """
    
    # 1. Base operations applied to both Train and Validation/Inference
    base_transforms = [
        # Load NIfTI files (Image and Mask/Label)
        LoadImaged(keys=["image", "label"]),
        
        # PyTorch expects (Channel, Depth, Height, Width). This adds the Channel dim.
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Standardize anatomical orientation to Right-Anterior-Superior (RAS)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Standardize voxel spacing to 1x1x1 mm (Isotropic)
        # Image uses bilinear interpolation (smooth), Label uses nearest (preserves integers 0/1)
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear", "nearest")
        ),
        
        # Clinical HU Windowing: Focus on bone/teeth density, normalize to [0, 1]
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=-200,    # Lower bound (soft tissue / air ignored)
            a_max=1500,    # Upper bound (dense bone / enamel)
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        
        # Binary Label Mapping: Convert any multiclass label > 0 into 1 (Teeth class)
        Lambdad(
            keys=["label"], 
            func=lambda x: (x > 0).to(torch.float32)
        ),
        
        # Resize to a fixed spatial size so we can stack them into batches
        Resized(
            keys=["image", "label"], 
            spatial_size=spatial_size, 
            mode=("trilinear", "nearest")
        ),
        
        # Convert NumPy arrays to PyTorch Tensors
        ToTensord(keys=["image", "label"])
    ]
    
    return Compose(base_transforms)


if __name__ == "__main__":
    print("Preprocessing module initialized. Ready to import into DataLoaders.")
