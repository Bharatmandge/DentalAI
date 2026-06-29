from typing import Tuple
from monai.transforms import (
    Compose,
    RandRotated,
    RandFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd
)
# Fixed relative import for the src package structure
from .preprocess import get_preprocessing_transforms


def get_train_transforms(spatial_size: Tuple[int, int, int] = (96, 96, 96)) -> Compose:
    """
    Constructs the training pipeline: Base Preprocessing + Stochastic Augmentations.
    
    Args:
        spatial_size: Target (X, Y, Z) dimensions.
        
    Returns:
        monai.transforms.Compose: The complete training transform pipeline.
    """
    
    base_pipeline = get_preprocessing_transforms(spatial_size=spatial_size)
    base_transforms = list(base_pipeline.transforms)
    
    augmentations = [
        # Random 3D Rotation
        RandRotated(
            keys=["image", "label"],
            range_x=0.3, 
            range_y=0.3, 
            range_z=0.3, 
            prob=0.5,
            mode=("bilinear", "nearest")
        ),
        
        # Random Zoom
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.5,
            mode=("trilinear", "nearest")
        ),
        
        # Random Flip
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=0, 
            prob=0.5
        ),
        
        # Add random static noise (simulates lower-dose radiation scans)
        RandGaussianNoised(
            keys=["image"], 
            prob=0.2, 
            mean=0.0, 
            std=0.05
        ),
        
        # Shift the brightness up or down globally
        RandShiftIntensityd(
            keys=["image"], 
            offsets=0.1, 
            prob=0.5
        ),
        
        # Scale the contrast of the image
        RandScaleIntensityd(
            keys=["image"], 
            factors=0.1, 
            prob=0.5
        )
    ]
    
    # We insert augmentations before the final ToTensor operation
    tensor_transform = base_transforms.pop(-1)
    final_transforms = base_transforms + augmentations + [tensor_transform]
    
    return Compose(final_transforms)


def get_val_transforms(spatial_size: Tuple[int, int, int] = (96, 96, 96)) -> Compose:
    """
    Constructs the validation/inference pipeline. 
    Strictly Base Preprocessing ONLY. No random augmentations.
    """
    return get_preprocessing_transforms(spatial_size=spatial_size, is_training=False)


if __name__ == "__main__":
    train_pipeline = get_train_transforms()
    val_pipeline = get_val_transforms()
    print(f"Training pipeline contains {len(train_pipeline.transforms)} operations.")
    print(f"Validation pipeline contains {len(val_pipeline.transforms)} operations.")
    print("Data Augmentation module initialized successfully.")
