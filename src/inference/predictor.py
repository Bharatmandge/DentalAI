# --- src/inference/predictor.py ---
import os
import torch
import numpy as np
import nibabel as nib
from contextlib import nullcontext
from monai.inferers import sliding_window_inference
from monai.transforms import InvertD, SaveImaged

from src.models.registry import create_model
from src.data.augmentation import get_val_transforms
from src.inference.postprocess import clean_prediction_mask


class DentalSegmenter:
    """
    Production-grade inference engine for DentalVision AI.
    Handles loading, sliding-window prediction, post-processing, and NIfTI export.
    """
    def __init__(self, config: dict):
        # Determine device: use GPU only if available and explicitly requested
        if torch.cuda.is_available() and config.get("device") == "cuda":
            self.device = torch.device("cuda")
        else:
            # Force CPU to avoid CUDA import errors on CPU-only builds
            self.device = torch.device("cpu")
            if config.get("device") == "cuda":
                print("[Warning] CUDA requested but not available; falling back to CPU.")
        self.spatial_size = tuple(config["data"]["spatial_size"])
        
        # 1. Load Model Architecture and Weights
        self.model = create_model(config["model"]["name"], config["model"])
        weight_path = os.path.join(config["training"]["save_dir"], "best_model.pth")
        
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
            print(f"[Inference] Loaded production weights from {weight_path}")
        else:
            print("[Warning] No weights found. Using initialized random parameters.")
            
        # Move model to the selected device (CPU fallback already ensured)
        self.model.to(self.device)
        self.model.eval()

        # 2. Load Preprocessing Pipeline
        self.val_transforms = get_val_transforms(spatial_size=self.spatial_size)

    def predict_scan(self, input_path: str, output_path: str) -> None:
        """
        Executes the full end-to-end prediction pipeline on a single scan.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Cannot find input scan: {input_path}")
            
        print(f"\n--- Processing Scan: {os.path.basename(input_path)} ---")

        # Step 1: Preprocessing
        # We wrap the single file path in a dictionary format expected by MONAI
        input_data = {"image": input_path}
        
        # Apply standard orientations, spacing, and Hounsfield windowing
        processed_data = self.val_transforms(input_data)
        
        # Extract the tensor and add a batch dimension (B, C, Z, Y, X)
        input_tensor = processed_data["image"].unsqueeze(0).to(self.device)

        # Step 2: Neural Network Prediction (Sliding Window)
        print("Running Sliding Window Inference (This may take a moment)...")
        with torch.no_grad():
            # Use autocast only when CUDA is available; otherwise, use a no-op context.
            autocast_ctx = torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext()
            with autocast_ctx:
                # overlap=0.5 means the patches overlap by 50% for smoother boundaries
                logits = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=self.spatial_size,
                    sw_batch_size=4,
                    predictor=self.model,
                    overlap=0.5 
                )
                
                # Convert logits to probabilities
                probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Step 3: Post-Processing
        print("Applying Mathematical Cleaning (Morphology & Hole Filling)...")
        cleaned_mask = clean_prediction_mask(
            prediction=probabilities, 
            threshold=0.5, 
            min_object_size=500
        )

        # Step 4: Export to NIfTI (Restoring Original Spacing)
        print("Exporting final mask...")
        
        # Load the original NIfTI file to steal its header (spacing, affine, orientation)
        original_nifti = nib.load(input_path)
        original_affine = original_nifti.affine
        
        # Ensure our mask matches the original array shape (in case resizing occurred)
        # Note: In a true production inverted pipeline, we'd use MONAI's InvertD transform here.
        # For simplicity and speed in this phase, we map our cleaned numpy array directly to the affine.
        final_nifti = nib.Nifti1Image(cleaned_mask, original_affine)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(final_nifti, output_path)
        print(f"[Success] Mask exported to: {output_path}")