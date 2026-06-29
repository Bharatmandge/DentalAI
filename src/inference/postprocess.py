import numpy as np 
import torch 
from scipy import ndimage
from skimage import morphology 

def clean_prediction_mask(
    prediction: torch.Tensor | np.ndarray, 
    threshold: float = 0.5, 
    min_object_size: int = 500
) -> np.ndarray:
    """
    Applies mathematical morphological operations to clean raw neural network probabilities 
    into clinical-grade 3D binary masks.
    
    Args:
        prediction: The raw probability output from the model (shape: Z, Y, X or C, Z, Y, X).
        threshold: The probability cutoff for binarization.
        min_object_size: The minimum volume (in voxels) for a predicted object to be kept.
                         Any isolated cluster smaller than this is considered noise and deleted.
                         
    Returns:
        np.ndarray: The cleaned, binary 3D volume.
    """
    # 1. Standardize input to NumPy array and strip the channel dimension if present
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
        
    if prediction.ndim == 4 and prediction.shape[0] == 1:
        mask = prediction[0]  # Convert (1, Z, Y, X) to (Z, Y, X)
    else:
        mask = prediction

    # 2. Thresholding: Convert probabilities to strict binary (0 or 1)
    binary_mask = mask >= threshold

    # 3. Morphological Closing: Smooth edges and bridge tiny gaps
    # We use a 3D structural element (a small 3x3x3 cube of 1s) to slide across the volume.
    structure_element = ndimage.generate_binary_structure(rank=3, connectivity=1)
    binary_mask = ndimage.binary_closing(binary_mask, structure=structure_element)

    # 4. Hole Filling: Fill enclosed empty spaces within solid structures (like the tooth pulp)
    binary_mask = ndimage.binary_fill_holes(binary_mask)

    # 5. Connected Components & Small Object Removal:
    # This algorithm scans the 3D grid, finds all disconnected "islands" of 1s, 
    # calculates their volume, and deletes islands smaller than 'min_object_size'.
    binary_mask = morphology.remove_small_objects(
        binary_mask, 
        min_size=min_object_size, 
        connectivity=1
    )

    # Convert boolean array back to integers (0, 1) for standardized saving/inference
    final_mask = binary_mask.astype(np.uint8)
    
    return final_mask


# Quick execution test block
if __name__ == "__main__":
    print("Testing Post-Processing Pipeline...")
    
    # Create a dummy 3D volume (64x64x64) initialized with background (0.0)
    dummy_pred = np.zeros((64, 64, 64), dtype=np.float32)
    
    # Inject a "Tooth" (A solid 10x10x10 block of high probability)
    dummy_pred[20:30, 20:30, 20:30] = 0.9
    
    # Inject a "Hole" in the middle of the tooth (low probability)
    dummy_pred[24:26, 24:26, 24:26] = 0.1
    
    # Inject "Noise" (A tiny 2x2x2 block far away)
    dummy_pred[50:52, 50:52, 50:52] = 0.8
    
    print(f"Original Volume Sum (Raw probabilities): {dummy_pred.sum():.2f}")
    
    # Run the pipeline
    cleaned_mask = clean_prediction_mask(dummy_pred, threshold=0.5, min_object_size=50)
    
    # Evaluate Results
    print(f"Cleaned Volume Sum (Total Voxels): {cleaned_mask.sum()}")
    
    # The solid block was 1000 voxels. The hole was 8 voxels. The noise was 8 voxels.
    # Without post-processing, binary sum would be 1000 - 8 + 8 = 1000.
    # WITH post-processing, the hole is filled (+8) and the noise is removed (-8).
    # Total sum should perfectly equal the solid 1000 block.
    if cleaned_mask.sum() == 1000:
        print("[Success] Morphological Closing, Hole Filling, and Small Object Removal worked perfectly.")
    else:
        print("[Failed] Post-processing did not achieve expected voxel count.")